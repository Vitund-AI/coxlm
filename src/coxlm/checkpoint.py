"""Reading a coxlm checkpoint into a built model (inference only).

A checkpoint is a ``torch.save``d state dict. Besides the weights it may carry a
metadata entry under ``META_KEY``: the backbone it was trained on, the build
settings (readout mode, option isolation, ...) and the schema features it was
trained to understand. ``load_state`` applies those settings before loading the
weights, so a checkpoint is always read the way it was trained.

Two kinds of checkpoint exist: adapter checkpoints (LoRA tensors + readout head;
the frozen base weights come from the Hub) and full-weight checkpoints (every
backbone weight, no LoRA tensors). ``checkpoint_lora_r`` tells them apart.
"""
from __future__ import annotations

import os

import torch

from .model import AmortizedDecisionModel

META_KEY = "__cox_meta__"


HUB_FILES = ["config.json", "*.safetensors", "*.safetensors.index.json", "README.md", "LICENSE"]


def resolve(path: str | os.PathLike, revision: str | None = None) -> str:
    """A local checkpoint (model.pt, a .safetensors file or a release folder), or a Hugging Face repo id such as
    "Vitund/cox-4b-research", downloaded on first use and cached (needs huggingface_hub; private repos use your
    `hf auth login`). ``revision`` pins a tag, branch or commit of a Hub repo."""
    import re

    path = os.fspath(path)
    if os.path.exists(path):
        return path
    if re.fullmatch(r"[\w.-]+/[\w.-]+", path):
        try:
            from huggingface_hub import snapshot_download
        except ImportError as e:  # pragma: no cover - depends on the environment
            raise ImportError("loading from the Hugging Face Hub needs huggingface_hub: pip install huggingface_hub") from e
        return snapshot_download(repo_id=path, revision=revision, allow_patterns=HUB_FILES)
    raise FileNotFoundError(f"{path}: no such checkpoint file or folder, and not a Hugging Face repo id (owner/name)")


def read_checkpoint(path: str | os.PathLike) -> dict:
    """The checkpoint's state dict, with its metadata under META_KEY. Reads a torch model.pt (memory-mapped where
    possible, so a large checkpoint is never held in RAM twice), or a release folder / .safetensors file with its
    config.json beside it (the Hugging Face release format, written by the research repo's hf_release.py)."""
    path = os.fspath(path)
    if os.path.isdir(path) or path.endswith(".safetensors"):
        return _read_release(path)
    try:
        return torch.load(path, map_location="cpu", mmap=True)
    except (RuntimeError, ValueError):  # legacy (non-zipfile) checkpoints cannot be memory-mapped
        return torch.load(path, map_location="cpu")


def _read_release(path: str) -> dict:
    import json

    from safetensors.torch import load_file

    folder = path if os.path.isdir(path) else os.path.dirname(path)
    config_path = os.path.join(folder, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"{folder}: no config.json beside the weights")
    config = json.load(open(config_path))
    if int(config.get("coxlm_format", 0)) != 1:
        raise ValueError(f"{config_path}: not a coxlm release (coxlm_format {config.get('coxlm_format')!r}); "
                         "upgrade coxlm if it is newer")
    index = os.path.join(folder, "model.safetensors.index.json")
    if os.path.isdir(path) and os.path.exists(index):  # sharded weights
        files = sorted(set(json.load(open(index))["weight_map"].values()))
    elif os.path.isdir(path):
        files = ["model.safetensors"]
    else:
        files = [os.path.basename(path)]
    state: dict = {}
    for f in files:
        state.update(load_file(os.path.join(folder, f)))
    state[META_KEY] = {"encoder": config.get("encoder"), "build": config.get("build") or {},
                       "features": config.get("features") or []}
    return state


def checkpoint_meta(state: dict) -> dict:
    return dict(state.get(META_KEY) or {})


def checkpoint_lora_r(state: dict) -> int:
    """LoRA rank of an adapter checkpoint, read off its lora_A tensors; 0 for a full-weight checkpoint."""
    for k, v in state.items():
        if ".lora_A." in k and hasattr(v, "shape"):
            return int(v.shape[0])
    return 0  # full backbone weights, or a head-only checkpoint on a frozen backbone


def _quantize_fp8(model) -> None:
    """Simulated FP8 W8A8 (e4m3, per-row scales), numerically what an FP8 deployment does, at bf16 speed: each backbone
    Linear's weight is quantized to float8 and back once (per output row), and its input is quantized and back per token
    on every call. (Real torchao FP8 kernels recompile for every new input shape, which our variable-length batches
    make impractically slow; the rounding is the same, only the matmul's accumulation differs.) Readout head untouched."""
    import torch
    import torch.nn as nn
    fmax = torch.finfo(torch.float8_e4m3fn).max

    def qdq(x, dim):  # quantize to e4m3 with a scale per row along `dim`, then back to x's dtype
        scale = x.detach().abs().amax(dim=dim, keepdim=True).float().clamp(min=1e-12) / fmax
        return ((x.float() / scale).to(torch.float8_e4m3fn).float() * scale).to(x.dtype)

    n = 0
    for m in model.encoder.modules():
        if isinstance(m, nn.Linear):
            with torch.no_grad():
                m.weight.copy_(qdq(m.weight, dim=1))
            m.register_forward_pre_hook(lambda mod, args: (qdq(args[0], dim=-1),) + tuple(args[1:]))
            n += 1
    print(f"FP8 W8A8 (simulated, e4m3, per-row): {n} backbone linear layers", flush=True)


def _quantize_nvfp4(model, activations: bool) -> None:
    """Simulated NVFP4 (as on Blackwell): values in fp4 e2m1 {0, 0.5, 1, 1.5, 2, 3, 4, 6}, blocks of 16 along the input
    dimension, each block's scale stored in fp8 e4m3 (with a per-tensor fp32 scale so block scales fit e4m3's range).
    Weights are rounded once; with activations=True every Linear input is rounded per call as well (W4A4), otherwise
    inputs stay bf16 (W4A16). Exact rounding, bf16 speed; the 4090 has no FP4 hardware, so this measures accuracy only."""
    import torch
    import torch.nn as nn
    grid = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    e4m3_max = torch.finfo(torch.float8_e4m3fn).max

    def qdq(x):
        shape, dt = x.shape, x.dtype
        k = shape[-1]
        if k % 16:
            return x
        xb = x.float().reshape(-1, k // 16, 16)
        amax = xb.abs().amax(-1, keepdim=True)
        tensor_scale = (amax.amax() / (6.0 * e4m3_max)).clamp(min=1e-12)  # maps the largest block scale into e4m3
        bscale = (amax / 6.0 / tensor_scale).clamp(min=2 ** -9).to(torch.float8_e4m3fn).float() * tensor_scale
        y = xb / bscale
        g = grid.to(y.device)
        mag = y.abs().clamp(max=6.0)
        idx = torch.bucketize(mag, (g[1:] + g[:-1]) / 2)  # round to the nearest grid value
        return (torch.sign(y) * g[idx] * bscale).reshape(shape).to(dt)

    n = 0
    for m in model.encoder.modules():
        if isinstance(m, nn.Linear) and m.in_features % 16 == 0:
            with torch.no_grad():
                m.weight.copy_(qdq(m.weight))
            if activations:  # rows in chunks: the fp32 temporaries of a long input would not fit next to the model
                def act(mod, args):
                    x = args[0]
                    flat = x.reshape(-1, x.shape[-1])
                    out = torch.cat([qdq(flat[i:i + 2048]) for i in range(0, flat.shape[0], 2048)]) if flat.shape[0] else flat
                    return (out.reshape(x.shape),) + tuple(args[1:])
                m.register_forward_pre_hook(act)
            n += 1
    print(f"NVFP4 {'W4A4' if activations else 'W4A16'} (simulated, blocks of 16, e4m3 block scales): {n} backbone linear layers", flush=True)


# Build settings a checkpoint may record, and the values coxlm implements. Anything else was a research setting
# that never shipped; such a checkpoint is refused by name rather than read wrongly. (key: (default if absent, allowed))
_SUPPORTED = {
    "question_mode": ("packed", {"packed"}),
    "prompt_format": ("plain", {"xml"}),
    "packed_state_attention": ("native", {"native"}),
    "read_layer": (None, {None}),
    "truncate": (False, {False}),
    "option_block": (0, {0}),
    "task_meta_pos": ("off", {"off"}),
    "state_tag": ("state", {"state"}),
    "question_prefix": ("", {""}),
    "state_preamble": ("", {""}),
    "span_pool": ("last", {"last"}),
    "answer_sentinel": ("none", {"none"}),
    "lora_scope": ("all", {"all"}),
    "readout_mode": ("slot", {"slot", "pointer"}),
    "option_contrast": (0, {0}),
    "recycle": (0, {0}),
    "pause_tokens": (0, {0}),
    "state_norm": ("standardize", {"standardize", "off", False, None}),
}
# head weights of components the packed layout never uses (older checkpoints save them)
_UNUSED_HEAD = ("readout.", "readout_norm.", "readout_stack.", "span_attn.")


def check_supported(build: dict) -> None:
    """Raise ValueError naming every recorded build setting coxlm does not implement."""
    bad = [f"{k}={build.get(k, d)!r}" for k, (d, ok) in _SUPPORTED.items() if build.get(k, d) not in ok]
    if build.get("readout_mode") == "pointer" and build.get("pointer_yesno") is False and build.get("yesno_layout", "pointer") != "slot":
        bad.append(f"yesno_layout={build.get('yesno_layout', 'pointer')!r}")
    if bad:
        raise ValueError("this checkpoint uses build settings coxlm does not support: " + ", ".join(bad))


def load_state(model: AmortizedDecisionModel, path: str | dict) -> AmortizedDecisionModel:
    """Load a checkpoint (a path, or a state dict already read with ``read_checkpoint``) into ``model``,
    applying the build settings recorded in its metadata. ``COX_QUANT`` = fp8 / fp4w / fp4 simulates
    quantized inference after the weights are in (accuracy only; no speed-up)."""
    state = read_checkpoint(path) if isinstance(path, (str, os.PathLike)) else dict(path)
    meta = state.pop(META_KEY, None) or {}
    build = meta.get("build") or {}
    check_supported(build)
    # a checkpoint with no metadata predates every optional schema feature
    model.trained_features = set(meta.get("features", []))
    model.checkpoint_meta = meta
    if build.get("readout_mode", "slot") != "slot":  # create the pointer head before its weights are loaded
        model.set_readout(build["readout_mode"])
    model.option_isolation = bool(build.get("option_isolation", False))
    model.pointer_yesno = bool(build.get("pointer_yesno", True))
    model.clear_cache()
    state = {k: v for k, v in state.items() if not k.startswith(_UNUSED_HEAD)}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"checkpoint has keys the model lacks (wrong backbone / lora settings?): {unexpected[:5]}")
    head_missing = [k for k in missing if not k.startswith("encoder.")]
    if head_missing:
        raise RuntimeError(f"checkpoint is missing readout weights: {head_missing[:5]}")
    q = os.environ.get("COX_QUANT")
    if q == "fp8":  # after the weights are in: FP8 weights + dynamic FP8 activations
        _quantize_fp8(model)
    elif q in ("fp4w", "fp4"):  # NVFP4 weights (fp4w: activations stay bf16; fp4: activations NVFP4 too)
        _quantize_nvfp4(model, activations=(q == "fp4"))
    return model
