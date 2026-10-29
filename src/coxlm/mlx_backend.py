"""The MLX backend: ``coxlm.load(path, backend="mlx")`` runs a model export natively on Apple silicon.

Needs the ``mlx`` extra (``pip install "coxlm[mlx]"``: mlx, mlx-lm, transformers for the tokenizer, safetensors);
no torch. The backbone is mlx-lm's Qwen3.5 text model (its gated delta rule has a Metal kernel, which is the point:
PyTorch on the Apple GPU runs the reference loop as thousands of small launches), loaded from the export folder with
its final norm replaced by an identity, and the decision head is a few matrix products here. The prompt layout and
the decide API are the PyTorch backend's own (``layout.PromptLayout``), so both read exactly the same tokens.

How each request is run mirrors ``AmortizedDecisionModel._forked_pointer_slots``:

  one pass   short states with few rows: every row [state | question row] as one right-padded sequence, no cache.
  forks      otherwise: each state is run once with the cache on, and the cache is copied per branch row (option
             lines, the answer tail, a yes/no block); a question with many options runs its head once and forks
             that (the nested mode). Rows are right-padded, which needs no mask: the backbone is causal, so padding
             after a row's last token cannot change it.

Only model exports (cox-1 and later, see ``checkpoint.is_export``) on Qwen3.5 backbones load here, in bf16 (as
trained) or with ``quantize=4`` / ``8`` (mlx's group-wise affine quantisation of the backbone's linear layers; the
embedding and the head stay as they are).
"""
from __future__ import annotations

import functools
import json
import os
from typing import Any

from .layout import PromptLayout
from .schema import Schema

# backbones this backend builds (mlx-lm's Qwen3.5 text model)
MLX_MODEL_TYPES = {"qwen3_5_text", "qwen3_5"}


def mlx_available() -> bool:
    """True when mlx and mlx-lm import (Apple silicon, or mlx's CPU build on Linux)."""
    import importlib.util
    if importlib.util.find_spec("mlx") is None or importlib.util.find_spec("mlx_lm") is None:
        return False
    try:
        import mlx.core  # noqa: F401  (the Linux wheel without a backend raises here)
    except Exception:
        return False
    return True


def load(path: str | os.PathLike, max_length: int = 2048, revision: str | None = None, quantize: int | None = None,
         dtype: str | None = None):
    """Load a model export with the MLX backend. ``dtype`` is the activation precision: "bf16" (default, as trained)
    or "fp32" (the weights stay bf16 in memory and are widened exactly in each product: the PyTorch fp32 result at the
    bf16 memory footprint; on Linux CPUs, where mlx has no fast bf16 products, also the faster one). ``quantize``:
    4 or 8 bits for the backbone's linear layers (group size 64), or None."""
    from .checkpoint import check_supported, is_export, resolve

    source = str(path)
    path = resolve(path, revision=revision)
    if not is_export(path):
        raise ValueError(f"{source}: the MLX backend loads model exports (a folder with cox_config.json, e.g. "
                         "Vitund/cox-1-3b); use the PyTorch backend for other checkpoints")
    cfg = json.load(open(os.path.join(path, "cox_config.json")))
    from .packed import check_format_version
    check_format_version(path, cfg)
    build = dict((cfg.get("meta") or {}).get("build") or {})
    build.update({"read_layer": None, "truncate": False})  # the folder holds exactly the read blocks
    check_supported(build)
    features = list((cfg.get("meta") or {}).get("features") or [])
    if (cfg.get("capabilities") or {}).get("reads_instructions") and "channel_ablation" not in features:
        features.append("channel_ablation")  # the model reads a question's wording (no bare yes/no warning)
    backbone = _load_backbone(path, final_norm=cfg.get("final_norm"), dtype=dtype or "bf16", quantize=quantize)
    tokenizer = _load_tokenizer(path)
    head = _load_head(os.path.join(path, "cox_head.safetensors"))
    model = MLXDecisionModel(backbone, tokenizer, head, build, features, max_length=max_length)
    model.encoder_name = cfg.get("base_model")
    model.checkpoint_path = source
    return model


# -- the backbone ---------------------------------------------------------------------------------------------------

def _load_tokenizer(path: str):
    from transformers import AutoTokenizer
    from transformers.utils import logging as hf_logging

    level = hf_logging.get_verbosity()
    hf_logging.set_verbosity_error()  # "PyTorch was not found": expected here, only the tokenizer is used
    try:
        tok = AutoTokenizer.from_pretrained(path)
    finally:
        hf_logging.set_verbosity(level)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def _load_backbone(path: str, final_norm: str | None, dtype: str = "bf16", quantize: int | None = None):
    """mlx-lm's Qwen3.5 text model with the export's weights. The zero-centred RMSNorms (weight 1 + w) keep their
    weights in fp32 and normalise in fp32, as the PyTorch model does (mlx-lm folds the 1 into a bf16 weight, which
    rounds it); the final norm becomes an identity when the export says so (the head reads the residual stream)."""
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs

    config = json.load(open(os.path.join(path, "config.json")))
    config = config.get("text_config", config)
    if config.get("model_type") not in MLX_MODEL_TYPES:
        raise NotImplementedError(f"the MLX backend runs Qwen3.5 backbones; this export is {config.get('model_type')!r}")
    model = TextModel(TextModelArgs.from_dict(config))
    from .packed import packed_info, unpack_mlx
    info = packed_info(path)
    if info is not None:  # a packed export (4-bit codebooks) or int8wo: unpacked to bf16 here, computed in bf16 as usual
        weights: dict[str, Any] = unpack_mlx(path, info)
    else:
        files = [os.path.join(path, f) for f in sorted(os.listdir(path)) if f.startswith("model") and f.endswith(".safetensors")]
        weights = {}
        for f in files:
            weights.update(mx.load(f))
    weights = {k.replace("model.language_model.", "model."): v for k, v in weights.items()
               if not k.startswith(("mtp.", "model.visual", "lm_head"))}
    norm_suffixes = (".input_layernorm.weight", ".post_attention_layernorm.weight", ".q_norm.weight", ".k_norm.weight")
    for k, v in list(weights.items()):
        if "conv1d.weight" in k and v.shape[-1] != 1:  # PyTorch (C, 1, K) -> MLX (C, K, 1)
            weights[k] = v.moveaxis(2, 1)
        elif k.endswith(norm_suffixes) or k == "model.norm.weight":
            weights[k] = v.astype(mx.float32) + 1.0
    if dtype not in ("bf16", "fp32"):
        raise ValueError(f"dtype {dtype!r}: the MLX backend runs bf16 (as trained) or fp32")

    _swap_norms(model)
    model.load_weights(list(weights.items()), strict=True)
    if final_norm == "identity":
        model.model.norm = _Identity()
    if dtype == "fp32":  # fp32 activations; the weights stay bf16 in memory and are widened (exactly) per product
        old = model.model.embed_tokens
        model.model.embed_tokens = _CastEmbedding(old.weight)
    if quantize:
        nn.quantize(model, group_size=64, bits=int(quantize),
                    class_predicate=lambda p, m: isinstance(m, nn.Linear) and "embed" not in p and m.weight.shape[-1] % 64 == 0)
    model.eval()
    mx.eval(model.parameters())
    return model


def _swap_norms(model) -> None:
    """Replace every zero-centred RMSNorm (decoder layers, attention q/k norms, final norm) with _PreciseRMSNorm."""
    import mlx.nn as nn

    def swap(parent, name):
        old = getattr(parent, name)
        if isinstance(old, nn.RMSNorm):
            setattr(parent, name, _PreciseRMSNorm(old.weight.shape[0], old.eps))

    for layer in model.model.layers:
        swap(layer, "input_layernorm")
        swap(layer, "post_attention_layernorm")
        if not layer.is_linear:
            swap(layer.self_attn, "q_norm")
            swap(layer.self_attn, "k_norm")
    swap(model.model, "norm")


@functools.lru_cache(maxsize=None)
def _module_classes():
    """The two small mlx modules this backend adds (defined lazily: importing coxlm never imports mlx)."""
    import mlx.core as mx
    import mlx.nn as nn

    class PreciseRMSNorm(nn.Module):
        """RMSNorm computed in fp32 with an fp32 weight (already 1 + w), cast back to the input's dtype."""

        def __init__(self, dims: int, eps: float):
            super().__init__()
            self.eps = eps
            self.weight = mx.ones((dims,), dtype=mx.float32)

        def __call__(self, x):
            return mx.fast.rms_norm(x.astype(mx.float32), self.weight, self.eps).astype(x.dtype)

    class Identity(nn.Module):
        """Stands in for the final norm."""

        def __call__(self, x):
            return x

    class CastEmbedding(nn.Module):
        """An embedding whose output is fp32 (its table stays as stored)."""

        def __init__(self, weight):
            super().__init__()
            self.weight = weight

        def __call__(self, x):
            return self.weight[x].astype(mx.float32)

        def as_linear(self, x):
            return x @ self.weight.T

    return PreciseRMSNorm, Identity, CastEmbedding


def _PreciseRMSNorm(dims: int, eps: float):
    return _module_classes()[0](dims, eps)


def _Identity():
    return _module_classes()[1]()


def _CastEmbedding(weight):
    return _module_classes()[2](weight)


def _load_head(path: str) -> dict:
    """The decision head's tensors as fp32 MLX arrays (keys as in cox_head.safetensors)."""
    import mlx.core as mx

    head = mx.load(path)
    return {k: v.astype(mx.float32) if v.dtype != mx.bool_ else v for k, v in head.items()}


# -- head maths (shared with the tests) ---------------------------------------------------------------------------

def head_view(hidden, mu, sd):
    """A backbone output as the head sees it: fp32, standardised per dimension."""
    import mlx.core as mx
    return (hidden.astype(mx.float32) - mu) / sd


def pointer_logits(slot, opts, wq, bq, wk, bk):
    """Bilinear pointer scores: slot (B, d), opts (B, k, d) -> (B, k), scaled by 1/sqrt(pointer width)."""
    import mlx.core as mx
    q = slot @ wq.T + bq
    k = opts @ wk.T + bk
    return mx.einsum("bd,bkd->bk", q, k) * (wq.shape[0] ** -0.5)


def slot_logits(slot, opts, proj, log_temp):
    """Slot readout: cosine between the projected answer vector (B, d) and each projected option encoding (k, d),
    over the learned temperature softplus(log_temp) + 1e-4 -> (B, k)."""
    import mlx.core as mx
    import mlx.nn as nn
    s = slot @ proj.T
    o = opts @ proj.T
    s = s / mx.maximum(mx.linalg.norm(s, axis=-1, keepdims=True), 1e-12)
    o = o / mx.maximum(mx.linalg.norm(o, axis=-1, keepdims=True), 1e-12)
    temp = nn.softplus(log_temp) + 1e-4
    return (s @ o.T) / temp


# -- the decision model ---------------------------------------------------------------------------------------------

class MLXDecisionModel(PromptLayout):
    """A Cox model export on MLX: ``decide`` / ``decide_batch`` / ``decide_iter`` / ``predict`` as with the
    PyTorch backend. Only the hybrid pointer layout (cox-1 exports: pointer readout, option isolation, yes/no
    questions scored by the slot readout) is implemented."""

    backend = "mlx"
    pool = "last"

    def __init__(self, backbone, tokenizer, head: dict, build: dict, features: list[str], max_length: int = 2048):
        self.backbone = backbone
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.head = head
        self.readout_mode = build.get("readout_mode", "slot")
        self.option_isolation = bool(build.get("option_isolation", False))
        self.pointer_yesno = bool(build.get("pointer_yesno", True))
        self.state_norm_mode = build.get("state_norm") or "standardize"
        if self.readout_mode != "pointer" or not self.option_isolation:
            raise NotImplementedError("the MLX backend runs pointer-readout exports with option isolation (cox-1)")
        if self.state_norm_mode not in ("standardize", "off"):
            raise NotImplementedError(f"state_norm={self.state_norm_mode!r}")
        self.trained_features = set(features)
        self.checkpoint_meta = {"build": build, "features": features}
        self._cache: dict = {}
        self.closed = False
        self.device = "mlx"

    # lifecycle, as the PyTorch model
    def close(self) -> None:
        self._cache.clear()
        self.__dict__.pop("_tok_memo", None)
        self.backbone = None
        self.head = {}
        self.closed = True

    def __enter__(self) -> "MLXDecisionModel":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def clear_cache(self) -> None:
        self._cache.clear()

    # -- backbone passes --------------------------------------------------------------------------------------------

    def _view(self, hidden):
        if self.state_norm_mode == "standardize":
            return head_view(hidden, self.head["state_mu"], self.head["state_sd"])
        import mlx.core as mx
        return hidden.astype(mx.float32)

    def _pad(self, rows: list[list[int]]):
        import mlx.core as mx
        pad = getattr(self.tokenizer, "pad_token_id", None) or 0
        L = max(len(r) for r in rows)
        return mx.array([r + [pad] * (L - len(r)) for r in rows], dtype=mx.int32)

    def _last_hidden(self, rows: list[list[int]], cache=None):
        """Head view (fp32, standardised) of each right-padded row's last real token: (n, d). With a cache (already
        holding n rows, or one row broadcast to all), every row continues from it."""
        import mlx.core as mx
        ids = self._pad(rows)
        h = self.backbone.model(ids, cache=cache)
        idx = mx.array([len(r) - 1 for r in rows])
        last = mx.take_along_axis(h, idx[:, None, None], axis=1)[:, 0]
        last = self._view(last)
        mx.eval(last)
        return last

    def _budget(self) -> int:
        return int(os.environ.get("COXLM_FORK_TOKENS", "16384"))

    def _run_rows(self, rows: list[list[int]], prefix_cache=None):
        """Last-token vectors of many rows (each continuing from `prefix_cache`, a one-row cache, or from nothing),
        in chunks of at most COXLM_FORK_TOKENS padded tokens, rows sorted by length to keep padding low."""
        import mlx.core as mx
        order = sorted(range(len(rows)), key=lambda j: len(rows[j]))
        budget = self._budget()
        out: list = [None] * len(rows)
        i = 0
        while i < len(order):
            # rows sorted ascending: the chunk's last row is its longest
            j = i + 1
            while j < len(order) and (j + 1 - i) * len(rows[order[j]]) <= budget:
                j += 1
            idx = order[i:j]
            chunk = [rows[t] for t in idx]
            cache = _fork(prefix_cache, len(chunk)) if prefix_cache is not None else None
            vecs = self._last_hidden(chunk, cache)
            for t, v in zip(idx, vecs):
                out[t] = v
            i = j
        return mx.stack(out)

    def _prefix_cache(self, ids: list[int], cache=None):
        """Run one unpadded row `ids` (continuing from `cache` if given, which is copied, not changed) and return the
        cache holding it."""
        import mlx.core as mx
        c = _fork(cache, 1) if cache is not None else self.backbone.make_cache()
        self.backbone.model(mx.array([ids], dtype=mx.int32), cache=c)
        mx.eval([_cache_arrays(x) for x in c])
        return c

    # -- the pointer layout -----------------------------------------------------------------------------------------

    def _option_embeddings(self, schema: Schema) -> list:
        """Slot readout: each yes/no field's options encoded on their own (last token), cached per schema."""
        if len(self._cache) > 512:
            self._cache.clear()
        key = ("options", schema)
        if key not in self._cache:
            out = []
            for f in schema:
                if f.kind == "yesno" and not self.pointer_yesno:
                    enc = self.tokenizer(self._option_texts(f), truncation=True, max_length=self.max_length)
                    out.append(self._run_rows([list(ids) for ids in enc["input_ids"]]))
                else:
                    out.append(None)
            self._cache[key] = out
        return self._cache[key]

    def _slots(self, states: list[str], schema: Schema):
        """(slots (B, F, d), per-field option vectors (B, k, d) or None for a slot-scored yes/no field)."""
        import mlx.core as mx
        q_ids = self._question_ids(schema)
        s_ids = self._state_ids(states)
        key = ("questions", schema, self.readout_mode, self.pointer_yesno)
        starts_all, ends_all = self._cache[("starts",) + key], self._cache[("ends",) + key]
        nb = len(s_ids)
        nested_min = int(os.environ.get("COXLM_NESTED_OPTIONS", "8"))

        rows, owner, plan = [], [], []
        for qi, (q, starts, ends) in enumerate(zip(q_ids, starts_all, ends_all)):
            if starts is None:  # a yes/no field in the slot layout: its whole block after the state
                plan.append(("plain", len(rows), None))
                for b in range(nb):
                    rows.append(q)
                    owner.append(b)
                continue
            head, tail = q[: starts[0]], q[ends[-1] + 1:]
            lines = [q[a: e + 1] for a, e in zip(starts, ends)]
            if len(lines) >= nested_min:
                plan.append(("nested", qi, (head, tail, lines)))
                continue
            plan.append(("pointer", len(rows), len(lines)))
            for b in range(nb):
                for r in [head + tail] + [head + line for line in lines]:
                    rows.append(r)
                    owner.append(b)

        state_len = [len(st) for st in s_ids]
        pass_cost = int(os.environ.get("COXLM_PASS_COST_TOKENS", "256"))
        single = bool(rows) and not any(kind == "nested" for kind, _, _ in plan) and \
            sum(state_len[b] for b in owner) - sum(state_len) <= pass_cost
        bases: dict = {}

        def base(b):  # the state's cache, computed once when a forked pass needs it
            if b not in bases:
                bases[b] = self._prefix_cache(s_ids[b])
            return bases[b]

        last = None
        if single:
            last = self._run_rows([s_ids[b] + r for b, r in zip(owner, rows)])
        elif rows:
            parts: list = [None] * len(rows)
            for b in range(nb):
                mine = [j for j in range(len(rows)) if owner[j] == b]
                vecs = self._run_rows([rows[j] for j in mine], base(b))
                for t, j in enumerate(mine):
                    parts[j] = vecs[t]
            last = mx.stack(parts)

        slots, opts = [], []
        for kind, start, k in plan:
            if kind == "plain":
                slots.append(last[start: start + nb])
                opts.append(None)
            elif kind == "pointer":
                blk = last[start: start + nb * (k + 1)].reshape(nb, k + 1, -1)
                slots.append(blk[:, 0])
                opts.append(blk[:, 1:])
            else:
                head, tail, lines = k
                blks = []
                for b in range(nb):  # the question head once per state, then each line forked from state + head
                    hc = self._prefix_cache(head, base(b))
                    blks.append(self._run_rows([tail] + lines, hc))
                blk = mx.stack(blks)
                slots.append(blk[:, 0])
                opts.append(blk[:, 1:])
        return mx.stack(slots, axis=1), opts

    # -- forward ----------------------------------------------------------------------------------------------------

    def forward(self, states: list[str], schema: Schema) -> list:
        """States -> a list (length F) of per-field logit arrays, each (B, k_f) (MLX arrays)."""
        h = self.head
        option_embs = self._option_embeddings(schema)
        slots, opts = self._slots(states, schema)
        logits = []
        for i, f in enumerate(schema):
            if not (f.kind == "yesno" and not self.pointer_yesno):
                if f.n_outputs != len(f.options):
                    raise NotImplementedError("pointer readout: options outside the listed ones (e.g. none_of_these)")
                logits.append(pointer_logits(slots[:, i], opts[i], h["pointer_q.weight"], h["pointer_q.bias"],
                                             h["pointer_k.weight"], h["pointer_k.bias"]))
            else:
                logits.append(slot_logits(slots[:, i], option_embs[i], h["proj.weight"], h["log_temp"]))
        return logits

    def predict(self, states: list[str], schema: Schema) -> list[dict[str, list[float]]]:
        """States -> list of {field_name: [probabilities]} per example."""
        import mlx.core as mx
        if self.closed:
            raise RuntimeError("this model was closed; load it again to use it")
        probs = [mx.softmax(l, axis=-1) for l in self.forward(states, schema)]
        mx.eval(probs)
        probs = [p.tolist() for p in probs]
        return [{schema.fields[i].name: probs[i][b] for i in range(len(schema))} for b in range(len(states))]


# -- caches ---------------------------------------------------------------------------------------------------------

def _cache_arrays(c) -> list:
    from mlx_lm.models.cache import KVCache
    if isinstance(c, KVCache):
        return [c.keys, c.values]
    return [x for x in c.cache if x is not None]


def _fork(cache: list, n: int) -> list:
    """A copy of a one-row cache with `n` rows (the original is left untouched): KV caches are cut to their filled
    length and repeated, the linear-attention conv and recurrent states repeated."""
    import mlx.core as mx
    from mlx_lm.models.cache import ArraysCache, KVCache

    def rep(x):
        if x is None:
            return None
        if x.shape[0] != 1:
            raise ValueError("_fork copies a one-row cache")
        return mx.repeat(x, n, axis=0) if n > 1 else x

    out = []
    for c in cache:
        if isinstance(c, KVCache):
            nc = KVCache()
            if c.keys is not None:
                nc.keys = rep(c.keys[..., : c.offset, :])
                nc.values = rep(c.values[..., : c.offset, :])
            nc.offset = c.offset
        elif isinstance(c, ArraysCache):
            nc = ArraysCache(len(c.cache))
            nc.cache = [rep(x) for x in c.cache]
        else:  # pragma: no cover - a cache type mlx-lm may add
            raise TypeError(f"cannot fork a {type(c).__name__}")
        out.append(nc)
    return out
