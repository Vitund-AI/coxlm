"""Packed model exports: 4-bit codebook weights unpacked to bf16 at load time (int8wo exports: see coxlm.int8; on the
CPU, Apple MPS and MLX their int8 tensors are dequantized by the same code as the int8 embedding below).

A packed export is a normal export (see checkpoint.is_export) whose cox_config.json has
``"weights": {"format": "codebook4", "version": 1, "group_size": G, "embed": "bf16" | "int8"}``. Its download is about
half the size (cox-1-3b: 2.9 GB instead of 5.7 GB), and the model computes in bf16 exactly as the unpacked export, so
answers and speed are unchanged; only the file is smaller.

In model.safetensors, a packed weight W (out, in) is stored as ``<name>.idx`` (uint8, out x in/2, two 4-bit indices per
byte, low nibble first) and ``<name>.codebook`` (bf16, out x in/G x 16): each group of G consecutive weights in a row
indexes its own 16 values. An int8 embedding is ``<name>.q8`` (int8) with ``<name>.scale`` (bf16, rows x 1). Every other
tensor is stored as it is.
"""
from __future__ import annotations

import json
import os

FORMAT = "codebook4"
FORMATS = {"codebook4": 1, "int8wo": 1}  # weights formats this coxlm reads, with their version (int8wo: coxlm.int8)


def packed_info(path: str) -> dict | None:
    """The export's weights settings (codebook4 or int8wo), or None for an ordinary bf16 export. Raises on a format this
    coxlm cannot read."""
    try:
        cfg = json.load(open(os.path.join(path, "cox_config.json")))
    except (OSError, ValueError):
        return None
    w = cfg.get("weights")
    if not w:
        return None
    if FORMATS.get(w.get("format")) != int(w.get("version", 0)):
        raise ValueError(f"{path}: weights format {w.get('format')!r} v{w.get('version')} is newer than this coxlm; "
                         "upgrade coxlm")
    return w


def _files(path: str) -> list[str]:
    return [os.path.join(path, f) for f in sorted(os.listdir(path)) if f.startswith("model") and f.endswith(".safetensors")]


def check_format_version(path: str, cfg: dict) -> None:
    """cox_config.json "format_version": 1 for bf16 and codebook4 exports, 2 for int8wo exports (coxlm.int8)."""
    v = int(cfg.get("format_version", 0))
    fmt = (cfg.get("weights") or {}).get("format")
    if v == 1 and fmt != "int8wo":
        return
    if v == 2 and fmt == "int8wo":
        return
    if v in (1, 2):
        raise ValueError(f"{path}: export format {v} with weights format {fmt!r} is not a valid combination")
    raise ValueError(f"{path}: export format {cfg.get('format_version')!r} is newer than this coxlm; upgrade coxlm")


def iter_unpacked_torch(path: str, info: dict, device: str = "cpu"):
    """Yield (name, bf16 tensor) one backbone weight at a time, unpacked on `device` (codebook4 indices, or int8 rows
    times their scales). Streaming keeps peak memory at the model plus one weight (a loader copies each into place and
    drops it)."""
    import torch
    from safetensors import safe_open

    g = int(info.get("group_size") or 0)  # codebook4 only
    for f in _files(path):
        with safe_open(f, "pt", device=device) as h:
            keys = set(h.keys())
            for k in sorted(keys):
                if k.endswith(".idx"):
                    name = k[: -len(".idx")]
                    packed, book = h.get_tensor(k), h.get_tensor(name + ".codebook")
                    o, half = packed.shape
                    idx = torch.stack([packed & 15, packed >> 4], dim=-1).reshape(o, half * 2 // g, g)
                    yield name, torch.gather(book, 2, idx.long()).reshape(o, half * 2)
                elif k.endswith(".q8"):
                    name = k[: -len(".q8")]
                    yield name, (h.get_tensor(k).float() * h.get_tensor(name + ".scale").float()).to(torch.bfloat16)
                elif not k.endswith((".codebook", ".scale")):
                    yield k, h.get_tensor(k)


def unpack_mlx(path: str, info: dict) -> dict:
    """{name: bf16 mx.array} for the whole backbone, each weight evaluated as it is unpacked (MLX is lazy: without that,
    every packed tensor and its unpacked copy would be alive at once)."""
    import mlx.core as mx

    raw: dict = {}
    for f in _files(path):
        raw.update(mx.load(f))
    g = int(info.get("group_size") or 0)  # codebook4 only
    out = {}
    for k in sorted(raw):
        if k.endswith(".idx"):
            name = k[: -len(".idx")]
            packed, book = raw.pop(k), raw.pop(name + ".codebook")
            o, half = packed.shape
            idx = mx.stack([packed & 15, mx.right_shift(packed, 4)], axis=-1).reshape(o, half * 2 // g, g)
            w = mx.take_along_axis(book, idx.astype(mx.uint32), axis=-1).reshape(o, half * 2)
            mx.eval(w)
            out[name] = w
            del packed, book, idx
        elif k.endswith(".q8"):
            name = k[: -len(".q8")]
            q, scale = raw.pop(k), raw.pop(name + ".scale")
            w = (q.astype(mx.float32) * scale.astype(mx.float32)).astype(mx.bfloat16)
            mx.eval(w)
            out[name] = w
    out.update({k: v for k, v in raw.items() if not k.endswith((".codebook", ".scale"))})
    return out
