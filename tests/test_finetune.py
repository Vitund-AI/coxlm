"""coxlm.finetune and adapters, on the tiny random export (CPU only, fp32, no download)."""
from __future__ import annotations

import json
import math
import os
import random
import shutil

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

import coxlm  # noqa: E402
from coxlm import Questions, choice, yesno  # noqa: E402

from tiny_export import WORDS  # noqa: E402


class Task(Questions):
    team = choice(["billing", "support", "sales"], instructions="Which team handles this?")
    refund = yesno("Is the customer asking for a refund?")


FILLER = [w for w in WORDS if w not in ("billing", "support", "sales", "refund", "crash", "order", "money")]
CUE = {"billing": "money", "support": "crash", "sales": "order"}


def _examples(n=120, seed=0):
    """A learnable synthetic task: the team is cued by one word, refund by another."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        team = rng.choice(list(CUE))
        refund = rng.random() < 0.5
        words = rng.sample(FILLER, 6) + [CUE[team]] + (["refund"] if refund else [])
        rng.shuffle(words)
        out.append((" ".join(words), {"team": team, "refund": "yes" if refund else "no"}))
    return out


def _load(path, **kw):
    return coxlm.load(path, backend="torch", device="cpu", dtype="fp32", **kw)


@pytest.fixture(scope="module")
def trained(tiny_export, tmp_path_factory):
    out = str(tmp_path_factory.mktemp("adapter"))
    model = _load(tiny_export)
    path, report = coxlm.finetune(model, _examples(), Task, out, steps=40, lr=3e-3, batch_size=8, micro_batch=4,
                                  lora_r=4, holdout=0.25, seed=0, general=True, verbose=False)
    return model, path, report


def _probs(model, states, schema):
    return [[p[f.name] for f in schema] for p in model.predict(states, schema)]


def _max_diff(a, b):
    return max(abs(x - y) for ra, rb in zip(a, b) for va, vb in zip(ra, rb) for x, y in zip(va, vb))


STATES = [t for t, _ in _examples(6, seed=9)]


def test_training_path_reads_what_inference_reads(tiny_export):
    """full_rows (the differentiable training path) gives the inference path's logits, including a many-option
    question (nested forks at inference) and a yes/no question (slot readout)."""
    model = _load(tiny_export)
    many = coxlm.questions(n=choice(["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]),
                           team=Task.team, refund=Task.refund)
    with torch.no_grad():
        ref = [x.clone() for x in model.forward(STATES[:3], many)]
        model.full_rows = True
        got = model.forward(STATES[:3], many)
        model.full_rows = False
    for a, b in zip(ref, got):
        assert torch.allclose(a, b, atol=1e-4)


def test_finetune_lowers_holdout_loss(trained):
    _, path, report = trained
    assert report["data"]["holdout"] == 30 and report["data"]["train"] == 90
    assert report["after"]["overall"]["nll"] < report["before"]["overall"]["nll"] - 0.05
    assert report["after"]["overall"]["accuracy"] >= report["before"]["overall"]["accuracy"]
    for name in ("team", "refund"):
        q = report["after"]["questions"][name]
        assert {"accuracy", "skill", "ece", "automatable_at_95", "majority_accuracy"} <= set(q)
    assert report["general_check"]["before"]["n"] >= 40
    saved = json.load(open(os.path.join(path, "report.json")))
    assert saved["temperature"] == report["temperature"]


def test_adapter_round_trip_and_temperature(trained, tiny_export):
    model, path, report = trained
    cfg = json.load(open(os.path.join(path, "adapter_config.json")))
    assert cfg["peft_type"] == "LORA" and cfg["r"] == 4 and cfg["cox"]["temperature"] == report["temperature"]
    assert cfg["cox"]["base"]["fingerprint"].startswith("sha256:")
    assert set(cfg["cox"]["head_parts"]) == {"pointer_q", "pointer_k", "proj"}  # the temperature is fitted on the held-out part, not trained
    from safetensors.torch import load_file
    keys = list(load_file(os.path.join(path, "adapter_model.safetensors")))
    assert any(k.startswith("base_model.model.layers.0.") and k.endswith("lora_A.weight") for k in keys)  # PEFT names
    assert "cox_head.pointer_q.weight" in keys

    loaded = _load(tiny_export, adapter=path)
    assert loaded.logit_temperature == pytest.approx(report["temperature"])
    schema = Task.__schema__
    assert _max_diff(_probs(model, STATES, schema), _probs(loaded, STATES, schema)) < 1e-5
    # the temperature is applied at load: probabilities are softmax(logits / T)
    with torch.no_grad():
        logits = loaded.forward(STATES[:2], schema)
    want = torch.softmax(logits[0] / report["temperature"], dim=-1)
    got = torch.tensor([p["team"] for p in loaded.predict(STATES[:2], schema)])
    assert torch.allclose(want, got, atol=1e-5)
    ans = loaded.decide(STATES[0], Task)
    assert ans.team.choice in ("billing", "support", "sales")
    assert report["temperature"] != pytest.approx(1.0)  # a real fit, not the identity


def test_merge_equals_unmerged(trained, tiny_export):
    _, path, _ = trained
    plain = _load(tiny_export, adapter=path)
    merged = _load(tiny_export, adapter=path, merge=True)
    assert not hasattr(merged.encoder, "peft_config")
    assert _max_diff(_probs(plain, STATES, Task.__schema__), _probs(merged, STATES, Task.__schema__)) < 1e-4


def test_base_mismatch_is_refused(trained, tmp_path):
    _, path, _ = trained
    from tiny_export import make_tiny_export
    other = make_tiny_export(str(tmp_path / "other"), seed=5)
    with pytest.raises(ValueError, match="trained on"):
        _load(other, adapter=path)


def test_changed_head_is_refused(trained, tiny_export, tmp_path):
    """Same backbone, different head file: still another base."""
    _, path, _ = trained
    copy = str(tmp_path / "copy")
    shutil.copytree(tiny_export, copy)
    from safetensors.torch import load_file, save_file
    head = load_file(os.path.join(copy, "cox_head.safetensors"))
    head["pointer_q.bias"] = head["pointer_q.bias"] + 0.01
    save_file(head, os.path.join(copy, "cox_head.safetensors"))
    with pytest.raises(ValueError, match="trained on"):
        _load(copy, adapter=path)


def test_adapter_on_mlx_not_yet(trained, tiny_export):
    _, path, _ = trained
    with pytest.raises(NotImplementedError, match="MLX"):
        coxlm.load(tiny_export, backend="mlx", adapter=path)


def test_soft_targets_from_jsonl(tiny_export, tmp_path):
    """JSONL rows with soft targets and a per-row schema (the Doom data's format) train and lower held-out loss."""
    rows = []
    for text, ans in _examples(80, seed=3):
        team = ans["team"]
        soft = [0.8 if o == team else 0.1 for o in ("billing", "support", "sales")]
        rows.append({"state": text, "truth": {"team": team}, "targets": {"team": soft},
                     "schema": coxlm.questions(team=Task.team).spec()})
    path = tmp_path / "rows.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    from coxlm.finetuning import read_examples
    ex = read_examples(str(path))
    assert ex[0].targets["team"] == pytest.approx([0.8 if o == rows[0]["truth"]["team"] else 0.1
                                                   for o in ("billing", "support", "sales")])
    _, report = coxlm.finetune(_load(tiny_export), str(path), None, str(tmp_path / "ad"), steps=30, lr=3e-3,
                               batch_size=8, micro_batch=8, lora_r=4, holdout=0.25, general=False, verbose=False,
                               train_head=False)
    assert report["after"]["overall"]["nll"] < report["before"]["overall"]["nll"]
    assert report["config"]["head_parts"] == []


def test_answers_as_dicts_and_lists():
    from coxlm.finetuning import read_examples
    ex = read_examples([("a", {"team": {"billing": 3, "sales": 1}, "refund": True}),
                        ("b", {"team": [0.2, 0.3, 0.5]})], Task)
    assert ex[0].targets["team"] == pytest.approx([0.75, 0.0, 0.25]) and ex[0].labels["refund"] == 1
    assert ex[1].labels == {"team": 2} and "refund" not in ex[1].targets
    with pytest.raises(ValueError, match="not one of its options"):
        read_examples([("a", {"team": "legal"})], Task)
    with pytest.raises(ValueError, match="does not have"):
        read_examples([("a", {"colour": "red"})], Task)


def test_holdout_is_stratified():
    from coxlm.finetuning import read_examples, split_holdout
    rows = [("x", {"team": "billing"})] * 80 + [("y", {"team": "sales"})] * 20
    ex = read_examples(rows, coxlm.questions(team=Task.team))
    train, held = split_holdout(ex, 0.2, seed=1)
    assert len(held) == 20 and not set(train) & set(held)
    assert sum(ex[i].labels["team"] == 2 for i in held) == 4


def test_temperature_fit_recovers_a_scale():
    from coxlm.finetuning import Example, fit_temperature
    rng = random.Random(0)
    schema = Task.__schema__
    logits, exs = [], []
    for _ in range(3000):
        z = [rng.gauss(0, 1.5) for _ in range(3)]
        p = [math.exp(v) for v in z]
        s = sum(p)
        r, acc, lab = rng.random() * s, 0.0, 2
        for i, v in enumerate(p):
            acc += v
            if r <= acc:
                lab = i
                break
        logits.append({"team": [3 * v for v in z]})  # overconfident by a factor of 3
        exs.append(Example("s", schema, {"team": [float(i == lab) for i in range(3)]}, {"team": lab}))
    assert fit_temperature(logits, exs) == pytest.approx(3.0, rel=0.15)


def test_soft_target_calibration():
    """With soft targets, ECE compares confidence with the target probability of the chosen option: a prediction
    equal to the target distribution is perfectly calibrated even though its top option wins more often."""
    from coxlm.finetuning import _ece, _is_soft
    assert _is_soft([0.02, 0.25, 0.73]) and not _is_soft([0.0, 1.0, 0.0])
    conf = [0.73] * 100
    assert _ece(conf, [0.73] * 100) < 1e-9          # against the soft target: calibrated
    assert abs(_ece(conf, [1.0] * 90 + [0.0] * 10) - 0.17) < 1e-9  # against the top answer: looks under-confident


def test_unselected_head_parts_stay_frozen(tiny_export):
    """Only the selected head parts train: train_head=False trains the LoRA alone, and the default leaves the
    temperature (log_temp) to the held-out fit."""
    import torch
    from coxlm.finetuning import _trainable_head
    m = _load(tiny_export)
    assert _trainable_head(m, False) == []
    assert not any(p.requires_grad for part in ("pointer_q", "pointer_k", "proj") for p in getattr(m, part).parameters())
    assert not m.log_temp.requires_grad
    assert "log_temp" not in _trainable_head(m, True) and not m.log_temp.requires_grad
