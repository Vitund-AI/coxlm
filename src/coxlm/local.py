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
         max_length: int = 2048, revision: str | None = None):
    """Load a checkpoint for local inference. Returns the model; call ``model.decide(state, questions)``.

    ``encoder`` is the Hugging Face backbone the checkpoint was trained on (e.g. "Qwen/Qwen3.5-4B-Base");
    it defaults to the one recorded in the checkpoint. Adapter (LoRA) and full-weight checkpoints are told
    apart from the checkpoint itself. ``dtype`` is the backbone dtype (default bf16, the dtype models are
    trained and evaluated in; fp32 checkpoints are cast down); ``device`` defaults to CUDA when available. ``max_length`` is the state token budget.
    """
    import torch

    from .checkpoint import (check_supported, checkpoint_lora_r, checkpoint_meta, is_export, load_state, read_checkpoint,
                             resolve)
    from .model import build_model

    source = str(path)
    path = resolve(path, revision=revision)
    if is_export(path):
        return _load_export(path, source, device=device, dtype=dtype, max_length=max_length)
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
    # a full-weight checkpoint (every backbone weight present, no LoRA) needs only the backbone's configuration:
    # downloading and initialising the pretrained weights would be wasted, since all of them are overwritten
    full = lora_r == 0 and any(k.startswith("encoder.") for k in state)
    model = build_model(encoder, lora_r=lora_r, pretrained_weights=not full, **kwargs)
    if full:
        expected = {"encoder." + k for k in model.encoder.state_dict()}
        missing = sorted(expected - set(state))
        if missing:
            raise ValueError(f"{source}: full-weight checkpoint lacks {len(missing)} backbone weights, e.g. {missing[:3]}")
    model = model.to(dev)
    load_state(model, state)
    del state
    model.eval()
    model.max_length = max_length
    model.encoder_name = encoder
    model.checkpoint_path = source
    return _maybe_compile(model)


def _load_export(path: str, source: str, device: str | None = None, dtype: str | None = None, max_length: int = 2048):
    """Load a model export (see checkpoint.is_export): the backbone is the folder itself (only the blocks the head
    reads, text only), its final norm replaced by an identity when cox_config.json says so (the head reads the
    residual stream before that norm), then the head and its build settings."""
    import json

    import torch
    from safetensors.torch import load_file
    from torch import nn

    from .checkpoint import META_KEY, check_supported, load_state
    from .model import build_model

    cfg = json.load(open(os.path.join(path, "cox_config.json")))
    if int(cfg.get("format_version", 0)) != 1:
        raise ValueError(f"{path}: export format {cfg.get('format_version')!r} is newer than this coxlm; upgrade coxlm")
    build = dict((cfg.get("meta") or {}).get("build") or {})
    build.update({"read_layer": None, "truncate": False})  # the folder holds exactly the read blocks
    check_supported(build)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if str(dev).startswith("cpu"):  # the hub causal-conv1d kernel is CUDA-only
        os.environ.setdefault("COX_HUB_CONV1D", "0")
    model = build_model(path, lora_r=0, pretrained_weights=True, state_norm=build.get("state_norm") or "standardize",
                        dtype=dtype or "bf16")
    if cfg.get("final_norm") == "identity":
        text = next((m for m in model.encoder.modules()
                     if isinstance(m._modules.get("layers"), nn.ModuleList) and "norm" in m._modules), None)
        if text is None:
            raise RuntimeError(f"{path}: could not find the backbone's final norm")
        text.norm = nn.Identity()
    state = dict(load_file(os.path.join(path, "cox_head.safetensors")))
    features = list((cfg.get("meta") or {}).get("features") or [])
    if (cfg.get("capabilities") or {}).get("reads_instructions") and "channel_ablation" not in features:
        features.append("channel_ablation")  # the model reads a question's wording (no bare yes/no warning)
    state[META_KEY] = {"encoder": cfg.get("base_model"), "build": build, "features": features}
    model = model.to(dev)
    load_state(model, state)
    model.eval()
    model.max_length = max_length
    model.encoder_name = cfg.get("base_model")
    model.checkpoint_path = source
    return _maybe_compile(model)


def _cpu_notice(model):
    """A one-time hint when no GPU is in use: the model works on the CPU, but each answer takes seconds."""
    if not str(getattr(model, "device", "cpu")).startswith("cpu"):
        return
    import sys
    import torch
    why = ("this PyTorch build has no CUDA support" if not torch.backends.cuda.is_built()
           else "no CUDA GPU was found")
    print(f"coxlm: running on the CPU ({why}): expect a few seconds per answer. For GPU speed, install a CUDA build "
          "of PyTorch (https://pytorch.org/get-started/locally/).", file=sys.stderr, flush=True)


def _maybe_compile(model):
    """COXLM_COMPILE=1: compile each backbone block with torch.compile (dynamic shapes). Fuses the many small kernels of
    a block: about 15% faster on long texts on an RTX 4090 (6k tokens 353 -> 299 ms), no change on short requests, same
    answers within bf16 noise; costs ~40 s of compiling over the first requests. Off by default."""
    _cpu_notice(model)
    if os.environ.get("COXLM_COMPILE", "0") != "1" or not str(getattr(model, "device", "cpu")).startswith("cuda"):
        return model
    import torch
    from torch import nn
    import torch._dynamo as dynamo
    for name in ("recompile_limit", "cache_size_limit"):  # many shapes per block (states, questions, options)
        if hasattr(dynamo.config, name):
            setattr(dynamo.config, name, max(getattr(dynamo.config, name), 64))
    for m in model.encoder.modules():
        layers = m._modules.get("layers")
        if isinstance(layers, nn.ModuleList):
            for i, layer in enumerate(layers):
                layers[i] = torch.compile(layer, dynamic=True)
            break
    return model
