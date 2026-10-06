"""The local model: ``coxlm.load(path, encoder=...)`` runs a checkpoint in this process.

Needs the ``local`` extra (torch, transformers, peft). Imported lazily by
``coxlm.load`` so that ``import coxlm`` never imports torch.
"""
from __future__ import annotations

import os

# build settings recorded in a checkpoint's metadata that build_model takes; the readout settings are applied by
# load_state, which also refuses settings coxlm does not implement
_BUILD_KEYS = ("state_norm", "pool")


def load(path: str | os.PathLike, encoder: str | None = None, device: str | None = None, dtype: str | None = None,
         max_length: int = 2048):
    """Load a checkpoint for local inference. Returns the model; call ``model.decide(state, questions)``.

    ``encoder`` is the Hugging Face backbone the checkpoint was trained on (e.g. "Qwen/Qwen3.5-4B-Base");
    it defaults to the one recorded in the checkpoint. Adapter (LoRA) and full-weight checkpoints are told
    apart from the checkpoint itself. ``dtype`` is the backbone dtype (default bf16, the dtype models are
    trained and evaluated in; fp32 checkpoints are cast down); ``device`` defaults to CUDA when available. ``max_length`` is the state token budget.
    """
    import torch

    from .checkpoint import check_supported, checkpoint_lora_r, checkpoint_meta, load_state, read_checkpoint
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
    check_supported(build)  # before downloading a backbone for a checkpoint that cannot be read
    kwargs = {k: build[k] for k in _BUILD_KEYS if build.get(k) is not None}
    kwargs.setdefault("state_norm", "standardize")
    # bf16 unless asked otherwise, whatever the checkpoint recorded: models are trained (and every published number
    # measured) with a bf16 backbone, and full-weight checkpoints saved in fp32 would otherwise load at twice the memory
    kwargs["dtype"] = dtype or "bf16"
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
