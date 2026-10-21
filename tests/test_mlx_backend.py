"""The MLX backend against the PyTorch backend, on a tiny random Qwen3.5 export (no GPU, no download).

Skipped unless mlx and mlx-lm import (Apple silicon, or ``pip install "mlx[cpu]" mlx-lm`` on Linux); the parity tests
also need torch and transformers. Both backends run in fp32 here, so they must agree to float rounding.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from coxlm.mlx_backend import mlx_available

pytestmark = pytest.mark.skipif(not mlx_available(), reason="mlx / mlx-lm not installed")

from tiny_export import WORDS  # noqa: E402


def _schema():
    from coxlm import choice, questions, score, yesno
    many = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
    return questions(
        team=choice(["billing", "support", "sales"], instructions="Which team handles this?"),
        urgency=score(["low", "medium", "high"], instructions="How urgent is it?"),
        refund=yesno("Is the customer asking for a refund?"),
        number=choice(many, instructions="Which one is it?"),  # >= 8 options: the nested mode
    )


def _short_schema():
    """No many-option question, so short requests take the one-pass route."""
    from coxlm.schema import Schema
    return Schema(tuple(f for f in _schema() if f.name != "number"))


STATES = ["my card was charged twice and I want my money back today",
          "the app crash when I open settings",
          " ".join(WORDS * 6)]  # long: forked passes


def _probs(model, states, schema):
    return [[p[f.name] for f in schema] for p in model.predict(states, schema)]


def _max_diff(a, b):
    import numpy as np
    return max(float(np.max(np.abs(np.array(x) - np.array(y)))) for ra, rb in zip(a, b) for x, y in zip(ra, rb))


@pytest.fixture(scope="session")
def both(tiny_export):
    import coxlm
    t = coxlm.load(tiny_export, backend="torch", device="cpu", dtype="fp32")
    m = coxlm.load(tiny_export, backend="mlx", dtype="fp32")
    return t, m


@pytest.mark.parametrize("pass_cost", ["100000", "0"])  # one pass for short requests / always forks
def test_matches_torch(both, monkeypatch, pass_cost):
    monkeypatch.setenv("COXLM_PASS_COST_TOKENS", pass_cost)
    t, m = both
    for schema in (_schema(), _short_schema()):
        for states in (STATES[:1], STATES):  # one state; a batch of states of different lengths
            assert _max_diff(_probs(t, states, schema), _probs(m, states, schema)) < 2e-4


def test_modes_agree(both, monkeypatch):
    """One pass, forks, small fork chunks and the flat (non-nested) many-option path give the same answers."""
    _, m = both
    schema = _schema()
    monkeypatch.setenv("COXLM_PASS_COST_TOKENS", "100000")
    ref = _probs(m, STATES[:2], schema)
    short = _short_schema()
    one_pass = _probs(m, STATES[:2], short)
    monkeypatch.setenv("COXLM_PASS_COST_TOKENS", "0")
    assert _max_diff(one_pass, _probs(m, STATES[:2], short)) < 1e-4  # one pass against forks
    monkeypatch.setenv("COXLM_PASS_COST_TOKENS", "100000")
    for env in ({"COXLM_PASS_COST_TOKENS": "0"}, {"COXLM_PASS_COST_TOKENS": "0", "COXLM_FORK_TOKENS": "40"},
                {"COXLM_NESTED_OPTIONS": "1000"}):
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        assert _max_diff(ref, _probs(m, STATES[:2], schema)) < 1e-4
        for k in env:
            monkeypatch.delenv(k)
        monkeypatch.setenv("COXLM_PASS_COST_TOKENS", "100000")


def test_decide_and_order(both):
    from coxlm import order_items
    t, m = both
    ans = m.decide(STATES[0], _schema())
    assert set(ans.team.probabilities) == {"billing", "support", "sales"}
    assert abs(sum(ans.number.probabilities.values()) - 1) < 1e-5
    steps = ["open the app", "open settings", "crash"]
    a, b = order_items(t, steps, mode="pairwise"), order_items(m, steps, mode="pairwise")
    assert a["order"] == b["order"]
    assert max(abs(x - y) for ra, rb in zip(a["pairs"], b["pairs"]) for x, y in zip(ra, rb) if x is not None) < 1e-3


def test_head_math_matches_torch():
    """The head's maths in MLX against the PyTorch model's formulas on random tensors."""
    torch = pytest.importorskip("torch")
    import mlx.core as mx
    import numpy as np
    import torch.nn.functional as F

    from coxlm.mlx_backend import head_view, pointer_logits, slot_logits

    rng = np.random.default_rng(0)
    B, k, d, dp = 3, 5, 32, 8
    h = rng.normal(size=(B, k, d)).astype(np.float32) * 4
    mu, sd = rng.normal(size=d).astype(np.float32), (rng.random(d) + 0.5).astype(np.float32)
    slot, opts = rng.normal(size=(B, d)).astype(np.float32), rng.normal(size=(B, k, d)).astype(np.float32)
    wq, bq = rng.normal(size=(dp, d)).astype(np.float32), rng.normal(size=dp).astype(np.float32)
    wk, bk = rng.normal(size=(dp, d)).astype(np.float32), rng.normal(size=dp).astype(np.float32)
    proj, log_temp = rng.normal(size=(d, d)).astype(np.float32), np.float32(-1.7)
    T = torch.from_numpy

    got = np.array(head_view(mx.array(h), mx.array(mu), mx.array(sd)))
    assert np.allclose(got, ((T(h) - T(mu)) / T(sd)).numpy(), atol=1e-5)

    got = np.array(pointer_logits(mx.array(slot), mx.array(opts), mx.array(wq), mx.array(bq), mx.array(wk), mx.array(bk)))
    q, kk = F.linear(T(slot), T(wq), T(bq)), F.linear(T(opts), T(wk), T(bk))
    assert np.allclose(got, (torch.einsum("bd,bkd->bk", q, kk) * dp ** -0.5).numpy(), rtol=1e-4, atol=1e-4)

    o2 = opts[0]
    got = np.array(slot_logits(mx.array(slot), mx.array(o2), mx.array(proj), mx.array(log_temp)))
    s, o = F.normalize(T(slot) @ T(proj).T, dim=-1), F.normalize(T(o2) @ T(proj).T, dim=-1)
    want = (s @ o.T) / (F.softplus(torch.tensor(log_temp)) + 1e-4)
    assert np.allclose(got, want.numpy(), rtol=1e-4, atol=1e-4)


def test_mlx_path_needs_no_torch(tiny_export):
    """The MLX backend loads and answers with torch made unimportable."""
    code = ("import sys; sys.modules['torch'] = None\n"
            "import coxlm\n"
            f"m = coxlm.load({tiny_export!r}, backend='mlx')\n"
            "a = m.decide('my card was charged twice', coxlm.questions(t=coxlm.choice(['billing', 'support'])))\n"
            "assert abs(sum(a.t.probabilities.values()) - 1) < 1e-4\n"
            "print('ok')\n")
    src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
    env = dict(os.environ, PYTHONPATH=src + os.pathsep + os.environ.get("PYTHONPATH", ""))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=300)
    assert out.returncode == 0 and "ok" in out.stdout, out.stderr[-2000:]


def test_bf16_matches_torch_bf16(tiny_export):
    """The bf16 path (what Apple silicon runs by default) against PyTorch in bf16: equal within bf16 rounding."""
    import coxlm
    t = coxlm.load(tiny_export, backend="torch", device="cpu", dtype="bf16")
    m = coxlm.load(tiny_export, backend="mlx", dtype="bf16")
    schema = _schema()
    assert _max_diff(_probs(t, STATES, schema), _probs(m, STATES, schema)) < 0.03


def test_quantized_loads(tiny_export):
    """quantize=8 runs and stays close to the unquantized model."""
    import coxlm
    m = coxlm.load(tiny_export, backend="mlx", dtype="fp32")
    q = coxlm.load(tiny_export, backend="mlx", dtype="fp32", quantize=8)
    schema = _schema()
    assert _max_diff(_probs(m, STATES[:2], schema), _probs(q, STATES[:2], schema)) < 0.05


def test_device_without_backend_picks_torch(tiny_export, monkeypatch):
    """encoder= / device= are PyTorch options: passing one without a backend loads PyTorch, even where MLX is the
    default (a demo edited to pass device="mps" on a Mac used to fail instead)."""
    import coxlm
    monkeypatch.delenv("COXLM_BACKEND", raising=False)
    monkeypatch.setattr(coxlm, "_default_backend", lambda path, revision: "mlx")
    m = coxlm.load(tiny_export, device="cpu", dtype="fp32")
    assert not type(m).__module__.endswith("mlx_backend")
