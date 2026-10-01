"""The local model: ``coxlm.load(path, encoder=...)`` runs a checkpoint in this process.

Needs the ``local`` extra (torch, transformers, peft). Imported lazily by
``coxlm.load`` so that ``import coxlm`` never imports torch.
"""
from __future__ import annotations

import os

# build settings recorded in a checkpoint's metadata that build_model takes; the
# rest (prompt format, readout mode, pauses, ...) are applied by load_state
_BUILD_KEYS = ("dtype", "state_norm", "question_mode", "packed_state_attention", "readout_layers", "pool",
               "read_layer", "truncate", "option_block", "prompt_format")
# what the published checkpoints use, for a checkpoint without metadata
_DEFAULTS = {"dtype": "bf16", "state_norm": "standardize", "question_mode": "packed"}


def load(path: str | os.PathLike, encoder: str | None = None, device: str | None = None, dtype: str | None = None,
         max_length: int = 2048):
    """Load a checkpoint for local inference. Returns the model; call ``model.decide(states, schema)``.

    ``encoder`` is the Hugging Face backbone the checkpoint was trained on (e.g. "Qwen/Qwen3.5-4B-Base");
    it defaults to the one recorded in the checkpoint. Adapter (LoRA) and full-weight checkpoints are told
    apart from the checkpoint itself. ``dtype`` overrides the backbone dtype (default: as trained, usually
    bf16); ``device`` defaults to CUDA when available. ``max_length`` is the state token budget.
    """
    import torch

    from .checkpoint import checkpoint_lora_r, checkpoint_meta, load_state, read_checkpoint
    from .model import build_model

    state = read_checkpoint(path)
    meta = checkpoint_meta(state)
    encoder = encoder or meta.get("encoder")
    if not encoder:
        raise ValueError(f"{path}: the checkpoint does not record its backbone; pass encoder=... (e.g. 'Qwen/Qwen3.5-4B-Base')")
    if meta.get("encoder") and meta["encoder"] != encoder:
        import warnings
        warnings.warn(f"checkpoint was trained on {meta['encoder']!r} but is being loaded on {encoder!r}", stacklevel=2)
    build = meta.get("build") or {}
    kwargs = {k: build[k] for k in _BUILD_KEYS if k in build}
    for k, v in _DEFAULTS.items():
        kwargs.setdefault(k, v)
    if dtype is not None:
        kwargs["dtype"] = dtype
    lora_r = checkpoint_lora_r(state)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if str(dev).startswith("cpu"):  # the hub causal-conv1d kernel is CUDA-only
        os.environ.setdefault("COX_HUB_CONV1D", "0")
    model = build_model(encoder, lora_r=lora_r, **kwargs).to(dev)
    load_state(model, state)
    del state
    model.eval()
    model.max_length = max_length
    model.encoder_name = encoder
    model.checkpoint_path = str(path)
    return model
