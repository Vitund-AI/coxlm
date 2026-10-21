"""``coxlm.finetune``: adapt a model export to your own labelled examples with a small LoRA.

What it does, in order:

1. reads the examples (``(text, answers)`` pairs, row dicts, or a JSONL file of rows) and sets a stratified share
   aside as a held-out set (``holdout``);
2. measures the base model on the held-out set, and on a small built-in set of general questions
   (coxlm/data/general_check.json);
3. trains a LoRA on every linear layer of the backbone (and, by default, the head's readout weights) with plain
   cross-entropy on the model's own readout logits (hard labels or soft target distributions), AdamW, warmup then a
   cosine decay, gradient accumulation, bf16 autocast on CUDA and gradient checkpointing. Each example is read
   through the same prompt layout the model uses to answer, so training and inference see the same tokens;
4. fits one temperature on the held-out set (the scalar that minimises its negative log-likelihood), which the
   adapter carries and ``coxlm.load(..., adapter=...)`` applies;
5. measures again and writes the adapter folder plus report.json: held-out accuracy, skill over always answering
   the most common label, calibration error (ECE) and the share of decisions that can be automated at 95%
   accuracy, before and after; and warns if the general questions got worse.

Example rows (JSONL, one object per line):

    {"state": "text, or a JSON object", "truth": {"question": "option", ...}}
    {"state": "...", "targets": {"question": [p_option_1, p_option_2, ...]}, "schema": {...}}

``truth`` names the correct option per question; ``targets`` gives a probability per option (soft targets, in the
order the options are listed); with both, training uses the targets and accuracy is scored against the truth.
``schema`` (optional, per row) is a question set as ``Schema.spec()`` writes it; rows without one use the
``questions`` argument. A row need not answer every question: unanswered ones are left out of its loss.
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
import time
import warnings
from dataclasses import dataclass
from typing import Any, Iterable

from .decide import as_schema, render_state
from .schema import Schema, schema_from_spec

AUTOMATION_ACCURACY = 0.95


@dataclass
class Example:
    state: str
    schema: Schema
    targets: dict  # field name -> probability list over the field's options (sums to 1)
    labels: dict  # field name -> index of the correct option (truth, else the target's top option)


# -- reading examples ----------------------------------------------------------------------------------------------

def _target(field, value) -> tuple[list[float], int | None]:
    """One answer -> (probabilities over the field's options, the hard label's index or None)."""
    opts = list(field.all_options)
    if isinstance(value, bool) and field.kind == "yesno":
        value = "yes" if value else "no"
    if isinstance(value, str):
        if value not in opts:
            raise ValueError(f"question {field.name!r}: {value!r} is not one of its options {opts}")
        i = opts.index(value)
        return [float(j == i) for j in range(len(opts))], i
    if isinstance(value, dict):
        bad = [k for k in value if k not in opts]
        if bad:
            raise ValueError(f"question {field.name!r}: target options {bad} are not among {opts}")
        probs = [float(value.get(o, 0.0)) for o in opts]
    elif isinstance(value, (list, tuple)):
        if len(value) != len(opts):
            raise ValueError(f"question {field.name!r}: {len(value)} target probabilities for {len(opts)} options")
        probs = [float(p) for p in value]
    else:
        raise TypeError(f"question {field.name!r}: an answer is an option name, a {{option: probability}} dict or a "
                        f"list of probabilities, not {type(value).__name__}")
    total = sum(probs)
    if total <= 0 or any(p < 0 for p in probs):
        raise ValueError(f"question {field.name!r}: target probabilities must be non-negative and sum above 0")
    return [p / total for p in probs], None


def _example(state: Any, schema: Schema, truth: dict | None, targets: dict | None) -> Example:
    truth, targets = dict(truth or {}), dict(targets or {})
    unknown = sorted((set(truth) | set(targets)) - set(schema.names))
    if unknown:
        raise ValueError(f"answers for questions the schema does not have: {unknown} (questions: {list(schema.names)})")
    tg, lb = {}, {}
    for f in schema:
        if f.name in targets:
            tg[f.name], i = _target(f, targets[f.name])
            if f.name in truth:
                _, i = _target(f, truth[f.name])
        elif f.name in truth:
            tg[f.name], i = _target(f, truth[f.name])
        else:
            continue
        lb[f.name] = i if i is not None else max(range(len(tg[f.name])), key=tg[f.name].__getitem__)
    if not tg:
        raise ValueError("an example answers none of its questions")
    text = state if isinstance(state, str) else render_state(state)
    return Example(text, schema, tg, lb)


def read_examples(examples, questions=None) -> list[Example]:
    """Examples from a JSONL path, or an iterable of row dicts / (text, answers) pairs. See the module docstring."""
    default = as_schema(questions) if questions is not None else None
    if isinstance(examples, (str, os.PathLike)):
        path = os.fspath(examples)
        with open(path) as f:
            rows = [json.loads(line) for line in f if line.strip()]
    else:
        rows = list(examples)
    out, specs = [], {}
    for n, row in enumerate(rows):
        try:
            if isinstance(row, dict):
                spec = row.get("schema")
                if spec is not None:
                    key = json.dumps(spec, sort_keys=True)
                    if key not in specs:
                        specs[key] = schema_from_spec(spec)
                    schema = specs[key]
                elif default is not None:
                    schema = default
                else:
                    raise ValueError("the row has no schema and no questions= were given")
                if "state" not in row:
                    raise ValueError("the row has no state")
                out.append(_example(row["state"], schema, row.get("truth"), row.get("targets")))
            else:
                text, answers = row
                if default is None:
                    raise ValueError("(text, answers) examples need questions=")
                answers = dict(answers)
                truth = {k: v for k, v in answers.items() if isinstance(v, (str, bool))}
                soft = {k: v for k, v in answers.items() if k not in truth}
                out.append(_example(text, default, truth, soft))
        except (ValueError, TypeError, KeyError) as e:
            raise type(e)(f"example {n}: {e}") from None
    if not out:
        raise ValueError("no examples")
    return out


def split_holdout(examples: list[Example], holdout: float, seed: int) -> tuple[list[int], list[int]]:
    """(train indices, held-out indices): ``holdout`` of the examples, stratified on their labels (each combination
    of correct options is a stratum; combinations seen once are pooled)."""
    if not 0 <= holdout < 1:
        raise ValueError("holdout is a share in [0, 1)")
    rng = random.Random(seed)
    strata: dict = {}
    for i, ex in enumerate(examples):
        key = (ex.schema.names, tuple(sorted(ex.labels.items())))
        strata.setdefault(key, []).append(i)
    pooled, groups = [], []
    for idx in strata.values():
        (groups.append(idx) if len(idx) > 1 else pooled.extend(idx))
    if pooled:
        groups.append(pooled)
    train, held = [], []
    for idx in groups:
        idx = list(idx)
        rng.shuffle(idx)
        k = int(round(holdout * len(idx)))
        held += idx[:k]
        train += idx[k:]
    if holdout > 0 and not held and len(train) > 1:
        held.append(train.pop())
    rng.shuffle(train)
    return train, sorted(held)


# -- measuring -----------------------------------------------------------------------------------------------------

def _logits(model, examples: list[Example], batch: int = 8) -> list[dict]:
    """Per example {field name: logits list}, through the inference path (eval mode, no gradients, temperature 1)."""
    import torch

    model.eval()
    model.full_rows = False
    model.clear_cache()
    out: list[dict] = [dict() for _ in examples]
    groups: dict = {}
    for i, ex in enumerate(examples):
        groups.setdefault(ex.schema, []).append(i)
    with torch.no_grad():
        for schema, idx in groups.items():
            for s in range(0, len(idx), batch):
                part = idx[s:s + batch]
                logits = model.forward([examples[i].state for i in part], schema)
                for fi, f in enumerate(schema):
                    rows = logits[fi].float().cpu().tolist()
                    for j, i in enumerate(part):
                        out[i][f.name] = rows[j]
    model.clear_cache()
    return out


def _softmax(xs: list[float], t: float = 1.0) -> list[float]:
    m = max(xs)
    e = [math.exp((x - m) / t) for x in xs]
    z = sum(e)
    return [v / z for v in e]


def _nll(logits: list[dict], examples: list[Example], t: float) -> float:
    total, n = 0.0, 0
    for lg, ex in zip(logits, examples):
        for name, tgt in ex.targets.items():
            p = _softmax(lg[name], t)
            total -= sum(q * math.log(max(pi, 1e-12)) for q, pi in zip(tgt, p) if q > 0)
            n += 1
    return total / max(n, 1)


T_RANGE = (0.25, 10.0)  # the temperature fit stays inside this range (see fit_temperature)


def fit_temperature(logits: list[dict], examples: list[Example]) -> float:
    """The one temperature T (logits / T) that minimises the held-out negative log-likelihood: a grid over
    log T, then a golden-section refinement around the best point. T is kept within T_RANGE: when every held-out
    answer is right the likelihood keeps improving as T shrinks, and an unbounded fit would make the model claim
    near-certainty from a small sample."""
    lo_t, hi_t = math.log(T_RANGE[0]), math.log(T_RANGE[1])
    grid = [lo_t + (hi_t - lo_t) * i / 120 for i in range(121)]
    vals = [_nll(logits, examples, math.exp(g)) for g in grid]
    b = min(range(len(grid)), key=vals.__getitem__)
    lo, hi = grid[max(b - 1, 0)], grid[min(b + 1, len(grid) - 1)]
    g = (math.sqrt(5) - 1) / 2
    for _ in range(40):
        c, d = hi - g * (hi - lo), lo + g * (hi - lo)
        if _nll(logits, examples, math.exp(c)) < _nll(logits, examples, math.exp(d)):
            hi = d
        else:
            lo = c
    return math.exp((lo + hi) / 2)


def _ece(conf: list[float], correct: list[bool], bins: int = 10) -> float:
    n = len(conf)
    if not n:
        return float("nan")
    total = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(conf) if (lo < c <= hi) or (b == 0 and c == 0)]
        if idx:
            total += abs(sum(conf[i] for i in idx) / len(idx) - sum(correct[i] for i in idx) / len(idx)) * len(idx)
    return total / n


def _automatable(conf: list[float], correct: list[bool], target: float = AUTOMATION_ACCURACY) -> float:
    """The largest share of decisions that can be acted on automatically (the most confident ones first) while
    those acted on are right at least ``target`` of the time."""
    order = sorted(range(len(conf)), key=lambda i: -conf[i])
    best, hits = 0, 0
    for k, i in enumerate(order, 1):
        hits += correct[i]
        if hits / k >= target:
            best = k
    return best / len(conf) if conf else float("nan")


def _metrics(conf, correct, nll_sum, majority) -> dict:
    n = len(conf)
    acc = sum(correct) / n
    out = {"n": n, "accuracy": acc, "nll": nll_sum / n, "ece": _ece(conf, correct),
           f"automatable_at_{int(AUTOMATION_ACCURACY * 100)}": _automatable(conf, correct)}
    if majority is not None:
        out["majority_accuracy"] = majority
        out["skill"] = (acc - majority) / (1 - majority) if majority < 1 else None
    return out


def evaluate(logits: list[dict], examples: list[Example], t: float, majority_labels: dict) -> dict:
    """Per question and overall: accuracy, skill over always answering the training set's most common label
    ((acc - majority acc) / (1 - majority acc)), negative log-likelihood, ECE (10 bins, top-option confidence) and
    the share of decisions automatable at 95% accuracy."""
    per: dict = {}
    for lg, ex in zip(logits, examples):
        for name, tgt in ex.targets.items():
            p = _softmax(lg[name], t)
            top = max(range(len(p)), key=p.__getitem__)
            d = per.setdefault(name, {"conf": [], "correct": [], "nll": 0.0, "label": []})
            d["conf"].append(p[top])
            d["correct"].append(top == ex.labels[name])
            d["label"].append(ex.labels[name])
            d["nll"] -= sum(q * math.log(max(pi, 1e-12)) for q, pi in zip(tgt, p) if q > 0)
    fields, conf, correct, nll, skill_w, skill_n = {}, [], [], 0.0, 0.0, 0
    for name, d in per.items():
        maj = None
        if name in majority_labels:
            maj = sum(lab == majority_labels[name] for lab in d["label"]) / len(d["label"])
        fields[name] = _metrics(d["conf"], d["correct"], d["nll"], maj)
        conf += d["conf"]
        correct += d["correct"]
        nll += d["nll"]
        if fields[name].get("skill") is not None:
            skill_w += fields[name]["skill"] * len(d["conf"])
            skill_n += len(d["conf"])
    overall = _metrics(conf, correct, nll, None) if conf else {}
    if skill_n:
        overall["skill"] = skill_w / skill_n  # decision-weighted mean of the per-question skills
    return {"overall": overall, "questions": fields}


def _general_items():
    from importlib import resources

    from .schema import Field

    data = json.loads(resources.files("coxlm").joinpath("data/general_check.json").read_text())
    qs = data["questions"]
    items = []
    for it in data["items"]:
        q = qs[it["q"]]
        if q["kind"] == "yesno":
            f = Field("answer", ("no", "yes"), instructions=it["instructions"], kind="yesno")
        else:
            f = Field(it["q"], tuple(q["options"]), instructions=q["instructions"], kind=q["kind"])
        items.append(Example(it["text"], Schema((f,)), {}, {f.name: f.all_options.index(it["answer"])}))
    return items


def general_check(model, t: float = 1.0) -> dict:
    """Accuracy on the built-in general questions (coxlm/data/general_check.json), and the list of items missed."""
    items = _general_items()
    logits = _logits(model, items)
    wrong = []
    for k, (lg, ex) in enumerate(zip(logits, items)):
        (name, label), = ex.labels.items()
        p = _softmax(lg[name], t)
        if max(range(len(p)), key=p.__getitem__) != label:
            wrong.append(k)
    return {"n": len(items), "accuracy": 1 - len(wrong) / len(items), "missed": wrong}


# -- training ------------------------------------------------------------------------------------------------------

def _trainable_head(model, train_head) -> list[str]:
    import torch

    if not train_head:
        return []
    from .adapter import HEAD_PARTS
    parts = HEAD_PARTS if train_head is True else tuple(train_head)
    out = []
    for part in parts:
        mod = getattr(model, part, None)
        if mod is None:
            if train_head is not True:
                raise ValueError(f"the model has no head part {part!r} (parts: {HEAD_PARTS})")
            continue
        params = [mod] if isinstance(mod, torch.nn.Parameter) else list(mod.parameters())
        for p in params:
            p.requires_grad_(True)
        out.append(part)
    return out


def _set_checkpointing(model, on: bool) -> bool:
    enc = model.encoder
    try:
        if on:
            enc.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        elif getattr(enc, "is_gradient_checkpointing", False):
            enc.gradient_checkpointing_disable()
        return True
    except (AttributeError, ValueError, NotImplementedError):
        return False


def _fmt(x, pct=False):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return f"{100 * x:.1f}%" if pct else f"{x:.3f}"


def format_report(report: dict) -> str:
    auto = f"automatable_at_{int(AUTOMATION_ACCURACY * 100)}"
    lines = [f"held-out set: {report['data']['holdout']} examples ({report['data']['train']} trained on); "
             f"temperature {report['temperature']:.3f}",
             f"{'question':<32} {'':<7} {'accuracy':>9} {'skill':>7} {'ECE':>7} {'auto@95':>8} {'NLL':>7}"]
    names = ["overall"] + sorted(report["after"]["questions"])
    for name in names:
        for when in ("before", "after"):
            m = report[when]["overall"] if name == "overall" else report[when]["questions"].get(name, {})
            lines.append(f"{(name if when == 'before' else ''):<32} {when:<7} {_fmt(m.get('accuracy'), True):>9} "
                         f"{_fmt(m.get('skill')):>7} {_fmt(m.get('ece')):>7} {_fmt(m.get(auto), True):>8} "
                         f"{_fmt(m.get('nll')):>7}")
    g = report.get("general_check")
    if g:
        lines.append(f"general questions: {_fmt(g['before']['accuracy'], True)} before, "
                     f"{_fmt(g['after']['accuracy'], True)} after ({g['before']['n']} items)")
    for w in report.get("warnings", []):
        lines.append("WARNING: " + w)
    return "\n".join(lines)


def finetune(model, examples, questions=None, out_dir: str | os.PathLike = "cox-adapter", *, steps: int | None = None,
             lr: float = 1e-4, batch_size: int = 8, micro_batch: int = 1, lora_r: int = 8, lora_alpha: int | None = None,
             train_head: bool | Iterable[str] = True, holdout: float = 0.15, seed: int = 0, device: str | None = None,
             dtype: str | None = None, warmup: float = 0.05, weight_decay: float = 0.0, max_grad_norm: float = 1.0,
             gradient_checkpointing: bool = True, general: bool = True, general_tolerance: float = 0.05,
             eval_batch: int = 8, log_every: int = 10, revision: str | None = None, verbose: bool = True):
    """Fine-tune a model export on labelled examples; returns (adapter folder, report dict).

    ``model``: a model export (a folder, or a Hub repo id such as "Vitund/cox-1-3b", pinned with ``revision``) or a
    model already loaded from one with ``coxlm.load`` (PyTorch backend; it is changed in place and ends up carrying
    the adapter). ``examples``: a JSONL path, or an iterable of row dicts or ``(text, answers)`` pairs, where
    ``answers`` maps a question name to its correct option, or to a {option: probability} dict / a probability list
    (soft targets). ``questions``: a ``Questions`` class or ``questions(...)`` schema, for examples without their
    own ``schema``.

    ``steps`` (default: about two passes over the training split, at least 50) optimizer steps of ``batch_size``
    examples each, read ``micro_batch`` at a time (gradient accumulation); ``lr`` peaks after ``warmup`` (a share
    of the steps) and decays to zero on a cosine. ``lora_r`` / ``lora_alpha`` (default 2 * r): the LoRA on every
    linear layer of the backbone. ``train_head``: also train the head's readout weights (True: every part the model
    has; or a list of parts from coxlm.adapter.HEAD_PARTS; False: the LoRA alone). ``holdout``: the share set aside,
    stratified on the labels, for the temperature fit and the report. ``general``: run the built-in general
    questions before and after, warning if accuracy drops by more than ``general_tolerance``.
    """
    import torch

    from .adapter import attach_lora, base_identity, lora_config, save_adapter

    t_start = time.time()
    log = (lambda *a: print("coxlm.finetune:", *a, file=sys.stderr, flush=True)) if verbose else (lambda *a: None)
    if isinstance(model, (str, os.PathLike)):
        from .local import load
        log(f"loading {model}")
        model = load(model, device=device, dtype=dtype, revision=revision)
    if not hasattr(model, "encoder") or not getattr(model, "export_path", None):
        raise ValueError("coxlm.finetune works on model exports loaded with the PyTorch backend "
                         "(e.g. coxlm.load('Vitund/cox-1-3b', backend='torch'))")
    if getattr(model, "adapter_info", None) is not None:
        raise ValueError("this model already carries an adapter; fine-tune from the base export")
    base_identity(model)  # fingerprint the base now (and fail early on a non-export)
    dev = model.device
    out_dir = os.fspath(out_dir)

    data = read_examples(examples, questions)
    train_idx, held_idx = split_holdout(data, holdout, seed)
    train, held = [data[i] for i in train_idx], [data[i] for i in held_idx]
    warn: list[str] = []
    if len(train) < 200:
        warn.append(f"only {len(train)} training examples; a few hundred or more give a steadier adapter")
    if held and len(held) < 50:
        warn.append(f"the held-out set has only {len(held)} examples; its numbers are rough")
    if not held:
        warn.append("no held-out set (holdout=0): no temperature fit and no before/after report")
    for w in warn:
        log("note:", w)
    n_notes = len(warn)

    counts: dict = {}
    for ex in train:
        for name, lab in ex.labels.items():
            counts.setdefault(name, {}).setdefault(lab, 0)
            counts[name][lab] += 1
    majority = {name: max(c, key=c.get) for name, c in counts.items()}

    report: dict = {"data": {"examples": len(data), "train": len(train), "holdout": len(held)}}
    if held:
        log(f"measuring the base model on {len(held)} held-out examples")
        before_logits = _logits(model, held, eval_batch)
        report["before"] = evaluate(before_logits, held, 1.0, majority)
    gen_before = general_check(model) if general else None

    # -- train
    torch.manual_seed(seed)
    attach_lora(model, lora_config(lora_r, lora_alpha))
    head_parts = _trainable_head(model, train_head)
    params = [p for p in model.parameters() if p.requires_grad]
    n_steps = steps if steps is not None else max(50, math.ceil(2 * len(train) / batch_size))
    n_warm = max(1, int(round(warmup * n_steps))) if warmup > 0 else 0
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    def lr_at(step):
        if step < n_warm:
            return (step + 1) / n_warm
        prog = (step - n_warm) / max(1, n_steps - n_warm)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    ckpt = gradient_checkpointing and _set_checkpointing(model, True)
    autocast = dev.type == "cuda"
    log(f"training: {len(train)} examples, {n_steps} steps of {batch_size}, lr {lr}, LoRA r={lora_r}, "
        f"head {head_parts or 'frozen'}, {sum(p.numel() for p in params):,} trainable weights on {dev}")
    rng = random.Random(seed + 1)
    order: list[int] = []
    curve, running, run_n = [], 0.0, 0
    model.train()
    model.full_rows = True
    model.clear_cache()
    try:
        for step in range(n_steps):
            batch = []
            while len(batch) < min(batch_size, len(train)):
                if not order:
                    order = list(range(len(train)))
                    rng.shuffle(order)
                batch.append(order.pop())
            pairs = sum(len(train[i].targets) for i in batch)
            groups: dict = {}
            for i in batch:
                groups.setdefault(train[i].schema, []).append(i)
            step_loss = 0.0
            for schema, idx in groups.items():
                for s in range(0, len(idx), micro_batch):
                    part = [train[i] for i in idx[s:s + micro_batch]]
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
                        logits = model.forward([ex.state for ex in part], schema)
                    loss = 0.0
                    for fi, f in enumerate(schema):
                        rows = [j for j, ex in enumerate(part) if f.name in ex.targets]
                        if not rows:
                            continue
                        tgt = torch.tensor([part[j].targets[f.name] for j in rows], dtype=torch.float32, device=dev)
                        logp = torch.log_softmax(logits[fi][rows].float(), dim=-1)
                        loss = loss - (tgt * logp).sum()
                    if torch.is_tensor(loss):
                        (loss / pairs).backward()
                        step_loss += float(loss.detach()) / pairs
            if max_grad_norm:
                torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            running += step_loss
            run_n += 1
            if (step + 1) % log_every == 0 or step + 1 == n_steps:
                curve.append({"step": step + 1, "train_loss": running / run_n})
                log(f"step {step + 1}/{n_steps}  train loss {running / run_n:.4f}  lr {sched.get_last_lr()[0]:.2e}")
                running, run_n = 0.0, 0
    finally:
        model.full_rows = False
        if ckpt:
            _set_checkpointing(model, False)
        for p in params:
            p.grad = None
        model.eval()
        model.clear_cache()

    # -- temperature and the after report
    temperature = 1.0
    if held:
        after_logits = _logits(model, held, eval_batch)
        temperature = fit_temperature(after_logits, held)
        if min(abs(temperature - b) / b for b in T_RANGE) < 0.01:
            warn.append(f"the temperature fit reached the end of its range ({temperature:.2f}); the held-out set is "
                        "probably too small or too easy to calibrate on")
        report["after_t1"] = evaluate(after_logits, held, 1.0, majority)
        report["after"] = evaluate(after_logits, held, temperature, majority)
    model.logit_temperature = temperature
    report["temperature"] = temperature
    if general:
        gen_after = general_check(model, temperature)
        report["general_check"] = {"before": gen_before, "after": gen_after}
        drop = gen_before["accuracy"] - gen_after["accuracy"]
        if drop > general_tolerance:
            warn.append(f"accuracy on coxlm's general questions fell from {gen_before['accuracy']:.0%} to "
                        f"{gen_after['accuracy']:.0%}: the adapter changed more than your task. Fewer steps, a "
                        "lower lr or more varied examples usually help.")
    if held and report["after"]["overall"]["nll"] > report["before"]["overall"]["nll"]:
        warn.append("the held-out loss is higher after fine-tuning than before; the adapter does not help on "
                    "held-out examples")
    report["warnings"] = warn
    report["config"] = {"steps": n_steps, "lr": lr, "batch_size": batch_size, "micro_batch": micro_batch,
                        "lora_r": lora_r, "lora_alpha": lora_alpha or 2 * lora_r, "head_parts": head_parts,
                        "holdout": holdout, "seed": seed, "warmup": warmup, "weight_decay": weight_decay,
                        "gradient_checkpointing": bool(ckpt), "device": str(dev)}
    report["train_curve"] = curve
    report["seconds"] = round(time.time() - t_start, 1)
    summary = {"rows": len(data), "train_rows": len(train), "holdout_rows": len(held), "steps": n_steps}
    if held:
        summary["holdout_before"] = report["before"]["overall"]
        summary["holdout_after"] = report["after"]["overall"]
    save_adapter(model, out_dir, head_parts=head_parts, temperature=temperature, training=summary)
    from . import __version__
    report["coxlm_version"] = __version__
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    model.adapter_info = {"path": out_dir, "temperature": temperature, "head_parts": head_parts}
    if verbose:
        if held:
            print(format_report(report), flush=True)
        print(f"adapter written to {out_dir}", flush=True)
    for w in warn[n_notes:]:  # the data notes were logged at the start; these are about the result
        warnings.warn(w, stacklevel=2)
    return out_dir, report
