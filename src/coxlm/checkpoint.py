"""Reading a coxlm checkpoint into a built model (inference only).

A checkpoint is a ``torch.save``d state dict. Besides the weights it may carry a
metadata entry under ``META_KEY``: the backbone it was trained on, the build
settings (prompt format, readout mode, ...) and the schema features it was
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


def read_checkpoint(path: str | os.PathLike) -> dict:
    """The checkpoint's state dict, memory-mapped where possible: tensors are paged in as they are
    copied into the model, so a large full-weight checkpoint is never held in RAM twice."""
    try:
        return torch.load(path, map_location="cpu", mmap=True)
    except (RuntimeError, ValueError):  # legacy (non-zipfile) checkpoints cannot be memory-mapped
        return torch.load(path, map_location="cpu")


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


def load_state(model: AmortizedDecisionModel, path: str | dict) -> AmortizedDecisionModel:
    """Load a checkpoint (a path, or a state dict already read with ``read_checkpoint``) into ``model``,
    applying the build settings recorded in its metadata. ``COX_QUANT`` = fp8 / fp4w / fp4 simulates
    quantized inference after the weights are in (accuracy only; no speed-up)."""
    state = read_checkpoint(path) if isinstance(path, (str, os.PathLike)) else dict(path)
    meta = state.pop(META_KEY, None)
    # a checkpoint with no metadata predates every optional schema feature
    model.trained_features = set((meta or {}).get("features", []))
    model.checkpoint_meta = meta or {}
    # A checkpoint must be read in the prompt format it was trained in. Older
    # checkpoints (no metadata, or metadata predating the format option) were all
    # plain, so that is the default; this overrides the constructor default so a
    # loaded checkpoint is always read correctly whatever the build/flag said.
    model.prompt_format = ((meta or {}).get("build") or {}).get("prompt_format", "plain")
    model.task_meta_pos = ((meta or {}).get("build") or {}).get("task_meta_pos", "off")
    model.state_tag = ((meta or {}).get("build") or {}).get("state_tag", "state")
    model.question_prefix = ((meta or {}).get("build") or {}).get("question_prefix", "")
    model.state_preamble = ((meta or {}).get("build") or {}).get("state_preamble", "")
    model.span_pool = ((meta or {}).get("build") or {}).get("span_pool", "last")
    model.answer_sentinel = ((meta or {}).get("build") or {}).get("answer_sentinel", "none")
    build = (meta or {}).get("build") or {}
    if build.get("lora_scope", "all") != getattr(model, "lora_scope", "all"):  # adapters on question tokens only
        model.lora_scope = build["lora_scope"]
        model.install_lora_scope()
    if build.get("letter_codes", "letters") != "letters":
        model.letter_codes = build["letter_codes"]
    if build.get("readout_mode", "slot") != "slot":  # create the pointer / letter head before its weights are loaded
        model.set_readout(build["readout_mode"])
    model.option_isolation = bool(build.get("option_isolation", False))
    model.option_contrast = int(build.get("option_contrast", 0))
    model.pointer_yesno = bool(build.get("pointer_yesno", True))
    model.yesno_layout = build.get("yesno_layout", "pointer")
    if build.get("recycle", 0) != 0:  # recreate the recycle projection before its weights are loaded
        model.set_recycle(int(build["recycle"]))
    if build.get("pause_tokens"):  # recreate the pause vectors before their weights are loaded
        model.set_pause(int(build["pause_tokens"]), build.get("pause_mode", "learned"))
    model.clear_cache()
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"checkpoint has keys the model lacks (wrong backbone / lora settings?): {unexpected[:5]}")
    head_missing = [k for k in missing if not k.startswith("encoder.")]
    if model.span_pool != "attn":  # span_attn is unused unless span_pool=="attn"; ok if a pre-span_attn checkpoint lacks it
        head_missing = [k for k in head_missing if not k.startswith("span_attn.")]
    if head_missing:
        raise RuntimeError(f"checkpoint is missing readout weights: {head_missing[:5]}")
    q = os.environ.get("COX_QUANT")
    if q == "fp8":  # after the weights are in: FP8 weights + dynamic FP8 activations
        _quantize_fp8(model)
    elif q in ("fp4w", "fp4"):  # NVFP4 weights (fp4w: activations stay bf16; fp4: activations NVFP4 too)
        _quantize_nvfp4(model, activations=(q == "fp4"))
    return model
