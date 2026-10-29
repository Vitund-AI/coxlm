"""INT8 weight-only model exports ("int8wo"): int8 weights that stay int8 in GPU memory.

An int8wo export is a model export (see checkpoint.is_export) whose cox_config.json has ``"format_version": 2`` and
``"weights": {"format": "int8wo", "version": 1, ...}`` (format_version 2 makes coxlm 0.4-0.6 stop with "upgrade
coxlm" instead of misreading the file). In model.safetensors every int8 tensor W (rows x cols) is stored as
``<name>.q8`` (int8) and ``<name>.scale`` (bf16, rows x 1): symmetric per-row quantization, W ~= q8 * scale (torchao's
Int8Tensor PerRow scheme). The backbone's Linear weights and the embedding table are int8; norms and the small
linear-attention parameters are bf16; the head file is unchanged. cox_config.json lists which tensors are which.

How the weights are held (``COXLM_INT8``):

- ``auto`` (default): on CUDA the int8 tensors stay int8 (a Linear computes ``(x @ q8.T) * scale`` in bf16, the
  int8 weight widened per product; the embedding gathers int8 rows and scales them), so the backbone takes about half
  the GPU memory of the bf16 export. On the CPU and Apple MPS they are dequantized to bf16 as the model loads (same
  answers within bf16 rounding, no memory saving there). The MLX backend always dequantizes.
- ``resident``: keep int8 on any device; ``dequant``: dequantize on any device.

On CUDA the Linear weights are torchao ``Int8Tensor`` weights when torchao is installed (``pip install torchao``),
which is what torch.compile and torchao's kernels work with; without torchao, a built-in module computes the same
products.
"""
from __future__ import annotations

import os

FORMAT = "int8wo"


def resident_for(device) -> bool:
    """Whether the int8 tensors stay int8 on `device` (COXLM_INT8 = auto | resident | dequant)."""
    mode = os.environ.get("COXLM_INT8", "auto").strip().lower()
    if mode not in ("auto", "resident", "dequant"):
        raise ValueError(f"COXLM_INT8={mode!r}: use auto, resident or dequant")
    if mode == "auto":
        return str(device).startswith("cuda")
    return mode == "resident"


def _torchao_int8():
    """torchao's Int8Tensor class, or None when torchao is missing or has another layout."""
    try:
        from torchao.quantization.quantize_.workflows.int8.int8_tensor import Int8Tensor
    except Exception:  # noqa: BLE001 - not installed, or an incompatible torch / torchao pair
        return None
    return Int8Tensor


def _modules():
    import torch
    import torch.nn.functional as F
    from torch import nn

    class Int8Embedding(nn.Module):
        """An embedding table as int8 rows + one bf16 scale per row: gathers int8 rows, then scales them (in fp32,
        rounded once to the output dtype), the same values as the dequantized table without building it."""

        def __init__(self, q8, scale, dtype, padding_idx=None):
            super().__init__()
            self.register_buffer("q8", q8)
            self.register_buffer("scale", scale)
            self.num_embeddings, self.embedding_dim = q8.shape
            self.padding_idx = padding_idx
            self.out_dtype = dtype

        def forward(self, ids):
            return (F.embedding(ids, self.q8).float() * F.embedding(ids, self.scale).float()).to(self.out_dtype)

        @property
        def weight(self):
            """The dequantized table, built on each access (for code that reads a few rows or the shape)."""
            return (self.q8.float() * self.scale.float()).to(self.out_dtype)

        def extra_repr(self):
            return f"{self.num_embeddings}, {self.embedding_dim}, int8 rows + bf16 scale"

    class Int8Linear(nn.Module):
        """A bias-free Linear with an int8 weight and a per-output-channel scale, computing what torchao's weight-only
        Int8Tensor computes: (x @ q8.T in x's dtype) * scale."""

        def __init__(self, q8, scale):
            super().__init__()
            self.register_buffer("q8", q8)
            self.register_buffer("scale", scale)
            self.out_features, self.in_features = q8.shape

        def forward(self, x):
            y = torch.mm(x.reshape(-1, x.shape[-1]), self.q8.t().to(x.dtype))
            y = y * self.scale.to(y.dtype).flatten()
            return y.reshape(*x.shape[:-1], self.out_features)

        def extra_repr(self):
            return f"in_features={self.in_features}, out_features={self.out_features}, int8 weight + bf16 scale"

    return Int8Embedding, Int8Linear


def load_resident(encoder, path: str, device, dtype=None, kernel: str | None = None) -> dict:
    """Fill `encoder` (built from the export's configuration, weights not loaded) from an int8wo export, keeping the
    int8 tensors int8, and move it to `device`. kernel: "torchao" | "builtin" | None (torchao when available).
    Returns what was loaded: {"int8": n, "bf16": n, "kernel": ..., "embedding": "int8" | "bf16"}."""
    import torch
    from safetensors import safe_open
    from torch import nn

    from .packed import _files

    dtype = dtype or torch.bfloat16
    Int8Embedding, Int8Linear = _modules()
    Int8Tensor = _torchao_int8() if kernel in (None, "torchao") else None
    if kernel == "torchao" and Int8Tensor is None:
        raise ImportError("COXLM int8 kernel 'torchao' asked for, but torchao is not importable (pip install torchao)")
    kernel = "torchao" if Int8Tensor is not None else "builtin"
    own = dict(encoder.state_dict())
    expected = set(own)
    seen, n_q, emb = set(), 0, "bf16"
    with torch.no_grad():
        for f in _files(path):
            with safe_open(f, "pt", device="cpu") as h:
                for k in list(h.keys()):
                    if k.endswith(".scale"):
                        continue
                    if not k.endswith(".q8"):
                        name = k[len("model."):] if k.startswith("model.") else k
                        if name not in own:
                            raise RuntimeError(f"{path}: weight {name!r} is not in the backbone")
                        own[name].copy_(h.get_tensor(k).to(own[name].dtype))
                        seen.add(name)
                        continue
                    full = k[: -len(".q8")]
                    name = full[len("model."):] if full.startswith("model.") else full
                    if name not in own or not name.endswith(".weight"):  # (or listed twice)
                        raise RuntimeError(f"{path}: int8 weight {name!r} is not in the backbone")
                    q8 = h.get_tensor(k).to(device)
                    scale = h.get_tensor(full + ".scale").to(device)
                    modname = name[: -len(".weight")]
                    mod = encoder.get_submodule(modname)
                    parent, _, attr = modname.rpartition(".")
                    holder = encoder.get_submodule(parent) if parent else encoder
                    if isinstance(mod, nn.Embedding):
                        setattr(holder, attr, Int8Embedding(q8, scale, dtype, padding_idx=mod.padding_idx))
                        emb = "int8"
                    elif isinstance(mod, nn.Linear) and mod.bias is None:
                        if Int8Tensor is not None:
                            t = Int8Tensor(q8, scale.float(), [1, q8.shape[1]], dtype,
                                           zero_point=torch.zeros_like(scale, dtype=torch.int8))
                            mod.weight = nn.Parameter(t, requires_grad=False)
                        else:
                            setattr(holder, attr, Int8Linear(q8, scale))
                    else:
                        raise RuntimeError(f"{path}: int8 weight {name!r} belongs to a {type(mod).__name__}, "
                                           "which this coxlm keeps in bf16 only")
                    del own[name]  # drop the reference to the replaced bf16 tensor
                    n_q += 1
                    seen.add(name)
    missing = sorted(expected - seen)
    if missing:
        raise RuntimeError(f"{path}: the int8 export lacks backbone weights, e.g. {missing[:3]}")
    del own
    encoder.to(device)
    return {"int8": n_q, "bf16": len(seen) - n_q, "kernel": kernel, "embedding": emb, "resident": True}
