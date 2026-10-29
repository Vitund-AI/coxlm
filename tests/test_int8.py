"""INT8 weight-only exports (coxlm.int8) on the tiny random export: CPU only, no download."""
from __future__ import annotations

import json
import os
import shutil

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

import coxlm  # noqa: E402

from tiny_export import WORDS, make_int8_exports  # noqa: E402

STATES = ["my card was charged twice and I want my money back today",
          "the app crash when I open settings",
          " ".join(WORDS * 6)]


def _schema():
    from coxlm import choice, questions, score, yesno
    many = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
    return questions(team=choice(["billing", "support", "sales"], instructions="Which team handles this?"),
                     urgency=score(["low", "medium", "high"], instructions="How urgent is it?"),
                     refund=yesno("Is the customer asking for a refund?"),
                     number=choice(many, instructions="Which one is it?"))


def _probs(model):
    schema = _schema()
    return [[p[f.name] for f in schema] for p in model.predict(STATES, schema)]


def _max_diff(a, b):
    return max(abs(x - y) for ra, rb in zip(a, b) for pa, pb in zip(ra, rb) for x, y in zip(pa, pb))


@pytest.fixture(scope="module")
def int8_pair(tiny_export, tmp_path_factory):
    root = tmp_path_factory.mktemp("int8")
    return make_int8_exports(tiny_export, str(root / "int8"), str(root / "ref"))


def _load(path, monkeypatch, mode="auto", dtype="fp32", kernel=None, **kw):
    monkeypatch.setenv("COXLM_INT8", mode)
    if kernel:
        monkeypatch.setenv("COXLM_INT8_KERNEL", kernel)
    else:
        monkeypatch.delenv("COXLM_INT8_KERNEL", raising=False)
    return coxlm.load(path, backend="torch", device="cpu", dtype=dtype, **kw)


def test_cpu_dequantizes_to_the_reference(int8_pair, monkeypatch):
    """On the CPU (auto) the int8 tensors are dequantized: answers equal the export holding the dequantized values."""
    int8, ref = int8_pair
    m = _load(int8, monkeypatch)
    r = _load(ref, monkeypatch)
    assert m.weights_format == "int8wo" and m.int8_info == {"resident": False}
    assert _max_diff(_probs(m), _probs(r)) == 0.0


@pytest.mark.parametrize("kernel", ["builtin", "torchao"])
def test_resident_int8_stays_int8(int8_pair, monkeypatch, kernel):
    """COXLM_INT8=resident keeps the weights int8 (the CUDA default): the backbone's Linear weights and the embedding
    hold int8, and the answers match the dequantized model within bf16 rounding."""
    if kernel == "torchao":
        pytest.importorskip("torchao")
    int8, ref = int8_pair
    m = _load(int8, monkeypatch, mode="resident", dtype="bf16", kernel=kernel)
    r = _load(ref, monkeypatch, dtype="bf16")
    assert m.int8_info["resident"] and m.int8_info["kernel"] == kernel and m.int8_info["embedding"] == "int8"
    n_int8 = sum(t.numel() for t in m.encoder.state_dict().values()
                 if getattr(t, "dtype", None) == torch.int8 or type(t).__name__ == "Int8Tensor")
    n_total = sum(t.numel() for t in r.encoder.state_dict().values())
    assert n_int8 > 0.8 * n_total
    assert m.int8_info["int8"] == 1 + 3 * 8 + 7  # embedding + 3 linear-attention blocks x 8 + 1 attention block x 7
    assert _max_diff(_probs(m), _probs(r)) < 0.03
    emb = m.encoder.get_input_embeddings()
    assert emb.q8.dtype == torch.int8 and emb.embedding_dim == 64


def test_torchao_and_builtin_kernels_agree(int8_pair, monkeypatch):
    pytest.importorskip("torchao")
    int8, _ = int8_pair
    a = _load(int8, monkeypatch, mode="resident", dtype="bf16", kernel="torchao")
    b = _load(int8, monkeypatch, mode="resident", dtype="bf16", kernel="builtin")
    assert _max_diff(_probs(a), _probs(b)) == 0.0


def _with_config(src, tmp_path, **changes):
    d = tmp_path / "changed"
    shutil.copytree(src, d)
    cfg = json.load(open(d / "cox_config.json"))
    cfg.update(changes)
    json.dump(cfg, open(d / "cox_config.json", "w"))
    return str(d)


def test_unknown_formats_ask_for_an_upgrade(int8_pair, tmp_path):
    int8, _ = int8_pair
    newer = _with_config(int8, tmp_path, format_version=3)
    with pytest.raises(ValueError, match="upgrade coxlm"):
        coxlm.load(newer, backend="torch", device="cpu")
    shutil.rmtree(newer)
    other = _with_config(int8, tmp_path, weights={"format": "int8wo", "version": 2})
    with pytest.raises(ValueError, match="upgrade coxlm"):
        coxlm.load(other, backend="torch", device="cpu")
    shutil.rmtree(other)
    mixed = _with_config(int8, tmp_path, format_version=1)  # an int8 export must say format 2
    with pytest.raises(ValueError, match="not a valid combination"):
        coxlm.load(mixed, backend="torch", device="cpu")


def test_adapters_need_the_bf16_export(int8_pair, monkeypatch, tmp_path):
    int8, _ = int8_pair
    with pytest.raises(ValueError, match="bf16 export"):
        _load(int8, monkeypatch, adapter=str(tmp_path))
