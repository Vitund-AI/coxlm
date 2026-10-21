"""Fine-tune adapters: a small LoRA (plus, optionally, trained head weights) on top of a model export.

An adapter is a folder:

    adapter_config.json        the PEFT LoraConfig fields (so PEFT tooling can read it), plus a "cox" entry: the base
                               model it was trained on (name, source, Hub revision, fingerprint), the coxlm version,
                               the re-fitted temperature, which head weights are included, and a training summary
    adapter_model.safetensors  the LoRA tensors under PEFT's names (base_model.model.<module>.lora_A.weight, ...), and
                               trained head weights under "cox_head.<name>" (e.g. cox_head.pointer_q.weight)
    report.json                (written by coxlm.finetune) the held-out before/after report

``coxlm.load(export, adapter=folder_or_repo_id)`` applies one; it refuses an adapter trained on another base. The
base's weights stay unmerged unless ``merge=True``.

PyTorch only: imported by coxlm.load / coxlm.finetune.
"""
from __future__ import annotations

import hashlib
import json
import os

ADAPTER_CONFIG = "adapter_config.json"
ADAPTER_WEIGHTS = "adapter_model.safetensors"
HEAD_PREFIX = "cox_head."
# head weights an adapter may carry: the pointer readout, the slot readout's projection and its temperature
HEAD_PARTS = ("pointer_q", "pointer_k", "proj", "log_temp")
_SAMPLES, _CHUNK = 64, 1 << 16  # model.safetensors: its header plus 64 evenly spaced 64 KiB chunks


def fingerprint(export_dir: str) -> str:
    """A content fingerprint of a model export: sha256 over cox_config.json, config.json, cox_head.safetensors and,
    for the (large) model.safetensors, its size, its header (every tensor's name, dtype, shape and offset) and 64
    evenly spaced 64 KiB chunks of its weights. Two exports with the same fingerprint are, for any practical purpose,
    the same model; it takes well under a second on a 6 GB export."""
    h = hashlib.sha256()
    for name in ("cox_config.json", "config.json", "cox_head.safetensors"):
        p = os.path.join(export_dir, name)
        h.update(name.encode() + b"\0")
        if os.path.exists(p):
            with open(p, "rb") as f:
                for block in iter(lambda: f.read(1 << 20), b""):
                    h.update(block)
    for name in sorted(n for n in os.listdir(export_dir) if n.startswith("model") and n.endswith(".safetensors")):
        p = os.path.join(export_dir, name)
        size = os.path.getsize(p)
        h.update(f"{name}\0{size}\0".encode())
        with open(p, "rb") as f:
            n = int.from_bytes(f.read(8), "little")
            h.update(f.read(min(n, 1 << 26)))
            start = 8 + n
            span = max(0, size - start - _CHUNK)
            for i in range(_SAMPLES):
                f.seek(start + (span * i) // max(1, _SAMPLES - 1))
                h.update(f.read(_CHUNK))
    return "sha256:" + h.hexdigest()


def _hub_revision(export_dir: str) -> str | None:
    """The commit hash of a Hub snapshot folder (.../snapshots/<commit>/), else None."""
    parts = os.path.normpath(export_dir).split(os.sep)
    if len(parts) >= 2 and parts[-2] == "snapshots" and len(parts[-1]) == 40:
        return parts[-1]
    return None


def base_identity(model) -> dict:
    """What an adapter records about the export ``model`` was loaded from (computed once per model)."""
    info = getattr(model, "_base_identity", None)
    if info is None:
        path = getattr(model, "export_path", None)
        if not path:
            raise ValueError("adapters work on model exports (such as Vitund/cox-1-3b); this model was loaded from "
                             "another checkpoint format")
        cfg = json.load(open(os.path.join(path, "cox_config.json")))
        info = {"name": cfg.get("name"), "source": getattr(model, "checkpoint_path", None),
                "revision": _hub_revision(path), "fingerprint": fingerprint(path)}
        model._base_identity = info
    return dict(info)


def lora_config(r: int = 8, alpha: int | None = None, dropout: float = 0.0):
    from peft import LoraConfig
    return LoraConfig(r=r, lora_alpha=alpha or 2 * r, lora_dropout=dropout, target_modules="all-linear")


def attach_lora(model, config) -> None:
    """Wrap the backbone in PEFT LoRA layers (zero-initialised B, so the outputs are unchanged); LoRA weights in fp32
    whatever the backbone's dtype."""
    from peft import get_peft_model

    if getattr(model, "adapter_info", None) is not None or hasattr(model.encoder, "peft_config"):
        raise ValueError("this model already carries an adapter; load the base export again to train or apply another")
    model.encoder = get_peft_model(model.encoder, config)
    for n, p in model.encoder.named_parameters():
        if "lora_" in n:
            p.data = p.data.float()
    model.clear_cache()


def save_adapter(model, out_dir: str, head_parts=(), temperature: float = 1.0, training: dict | None = None) -> str:
    """Write ``model``'s LoRA (and the listed head parts) as an adapter folder. Returns the folder."""
    import torch
    from peft.utils import get_peft_model_state_dict
    from safetensors.torch import save_file

    from . import __version__

    os.makedirs(out_dir, exist_ok=True)
    peft_cfg = model.encoder.peft_config["default"]
    tensors = {k: v.detach().to("cpu").contiguous() for k, v in get_peft_model_state_dict(model.encoder).items()}
    for part in head_parts:
        mod = getattr(model, part)
        items = [("", mod)] if isinstance(mod, torch.nn.Parameter) else [("." + n, p) for n, p in mod.named_parameters()]
        for n, p in items:
            tensors[f"{HEAD_PREFIX}{part}{n}"] = p.detach().float().to("cpu").contiguous().clone()
    save_file(tensors, os.path.join(out_dir, ADAPTER_WEIGHTS), metadata={"format": "pt"})
    cfg = peft_cfg.to_dict()
    for k, v in list(cfg.items()):
        if isinstance(v, set):
            cfg[k] = sorted(v)
    base = base_identity(model)
    cfg["base_model_name_or_path"] = base.get("source") or base.get("name")
    cfg["revision"] = base.get("revision")
    cfg["cox"] = {"format_version": 1, "coxlm_version": __version__, "base": base, "temperature": float(temperature),
                  "head_parts": list(head_parts), "training": training or {}}
    with open(os.path.join(out_dir, ADAPTER_CONFIG), "w") as f:
        json.dump(cfg, f, indent=2, default=str)
    return out_dir


def read_adapter_config(adapter_dir: str) -> dict:
    p = os.path.join(adapter_dir, ADAPTER_CONFIG)
    if not os.path.exists(p):
        raise FileNotFoundError(f"{adapter_dir}: no {ADAPTER_CONFIG}; not a coxlm adapter folder")
    cfg = json.load(open(p))
    cox = cfg.get("cox")
    if not isinstance(cox, dict):
        raise ValueError(f"{p}: a PEFT adapter without coxlm's 'cox' entry; coxlm cannot tell which base it was "
                         "trained on")
    if int(cox.get("format_version", 0)) != 1:
        raise ValueError(f"{p}: adapter format {cox.get('format_version')!r} is newer than this coxlm; upgrade coxlm")
    return cfg


def check_base(model, cfg: dict, adapter_dir: str) -> None:
    """Refuse an adapter trained on another base export."""
    want = cfg["cox"].get("base") or {}
    have = base_identity(model)
    if want.get("fingerprint") != have["fingerprint"]:
        def name(b):
            rev = f" @ {b['revision'][:12]}" if b.get("revision") else ""
            return f"{b.get('name') or '?'} ({b.get('source') or '?'}{rev}, {str(b.get('fingerprint'))[:19]}...)"
        raise ValueError(f"{adapter_dir}: this adapter was trained on {name(want)}, not on the loaded model "
                         f"{name(have)}. Load the base it was trained on (the same export, same revision).")


def apply_adapter(model, adapter_dir: str, merge: bool = False):
    """Apply an adapter folder to a model loaded from an export (in place); returns the model."""
    import torch
    from peft import LoraConfig
    from peft.utils import set_peft_model_state_dict
    from safetensors.torch import load_file

    cfg = read_adapter_config(adapter_dir)
    check_base(model, cfg, adapter_dir)
    cox = cfg["cox"]
    keys = set(LoraConfig.__dataclass_fields__)
    lcfg = LoraConfig(**{k: v for k, v in cfg.items() if k in keys and k not in ("peft_type", "task_type", "revision",
                                                                                 "base_model_name_or_path")})
    tensors = load_file(os.path.join(adapter_dir, ADAPTER_WEIGHTS))
    head = {k[len(HEAD_PREFIX):]: v for k, v in tensors.items() if k.startswith(HEAD_PREFIX)}
    lora = {k: v for k, v in tensors.items() if not k.startswith(HEAD_PREFIX)}
    attach_lora(model, lcfg)
    res = set_peft_model_state_dict(model.encoder, lora)
    unexpected = list(getattr(res, "unexpected_keys", []) or [])
    if unexpected:
        raise RuntimeError(f"{adapter_dir}: LoRA tensors the model lacks (another backbone?): {unexpected[:3]}")
    missing = [k for k in (getattr(res, "missing_keys", []) or []) if "lora_" in k]
    if missing:
        raise RuntimeError(f"{adapter_dir}: the adapter lacks LoRA tensors: {missing[:3]}")
    if head:
        own = dict(model.named_parameters())
        bad = sorted(k for k in head if k not in own)
        if bad:
            raise RuntimeError(f"{adapter_dir}: head weights the model lacks: {bad[:3]}")
        with torch.no_grad():
            for k, v in head.items():
                if own[k].shape != v.shape:
                    raise RuntimeError(f"{adapter_dir}: head weight {k} has shape {tuple(v.shape)}, the model "
                                       f"{tuple(own[k].shape)}")
                own[k].copy_(v.to(own[k].dtype))
    model.logit_temperature = float(cox.get("temperature", 1.0))
    model.adapter_info = {"path": adapter_dir, **cox}
    if merge:
        merge_adapter(model)
    model.clear_cache()
    model.eval()
    return model


def merge_adapter(model) -> None:
    """Fold the LoRA into the backbone's weights (in the backbone's dtype): one model, no LoRA overhead per call."""
    if hasattr(model.encoder, "merge_and_unload"):
        model.encoder = model.encoder.merge_and_unload()
        model.clear_cache()
