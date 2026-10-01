"""Turning per-field distributions into typed answers, plus state rendering.

The model produces one probability vector per field. This module reads those
vectors out as the three primitives (choice, score, yesno) with a confidence
statistic, and renders structured state (dict / list) into the text the encoder
reads. It has no model dependency so it can be used on stored predictions too.

``confidence`` is the probability of the top option for choice and yes/no
fields. It is the quantity the calibration eval is defined on, so "act only
above 0.9" means "answers that were right about 90% of the time in eval".

A score field reports an expected level, so top-option probability is the wrong
shape for it: mass split evenly over levels 2 and 3 gives a well-supported
score of 2.5 and a top probability of 0.5, and confidence-gated routing would
discard an answer the model is sure of. For score fields ``confidence`` is the
mass on the two levels the score falls between (on the one level, when the
score is exactly an integer): the probability that the true level is one the
reported score sits on or between. ``top_probability`` is always available.
The full distribution is always returned, so you are never locked in.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

from .schema import Field, Schema


@dataclass
class Answer:
    """One field's answer. Which of choice / score / yesno is meaningful depends
    on ``kind``; ``probabilities`` and ``confidence`` are always populated."""

    kind: str
    probabilities: dict[str, float]
    confidence: float
    top_probability: float = 0.0
    choice: str | None = None  # choice + score: the top option
    score: float | None = None  # score: expected level, 1-based, may fall between levels
    legend: dict[float, str] = field(default_factory=dict)  # score: level value -> option (value is 1..k unless set)
    p_yes: float | None = None  # yesno: P(yes)
    none: float | None = None  # probability that nothing offered fits (fields offering the none option)

    @property
    def entropy(self) -> float:
        """Shannon entropy in nats; an alternative spread statistic."""
        return -sum(p * math.log(p) for p in self.probabilities.values() if p > 0)


def _bracket_indices(values: list[float], s: float) -> set[int]:
    """Indices of the level(s) the score sits on or between, on an arbitrary increasing scale.
    Exactly on a level -> that one; between two -> both; past an end -> the nearest."""
    for i, v in enumerate(values):
        if abs(s - v) < 1e-9:
            return {i}
    below = [i for i, v in enumerate(values) if v < s]
    above = [i for i, v in enumerate(values) if v > s]
    if not below:
        return {above[0]}
    if not above:
        return {below[-1]}
    return {below[-1], above[0]}


def answer_for(f: Field, probs: list[float]) -> Answer:
    dist = {o: float(p) for o, p in zip(f.all_options, probs)}
    top = max(range(len(probs)), key=lambda i: probs[i])
    ans = Answer(kind=f.kind, probabilities=dist, confidence=float(probs[top]), top_probability=float(probs[top]), choice=f.all_options[top])
    if f.none_option:
        ans.none = float(probs[-1])
        probs = list(probs[:-1])  # the level arithmetic below is over the real options only
        z = sum(probs) or 1.0
        probs = [q / z for q in probs]
    if f.kind == "score":
        vals = list(f.values) if f.values is not None else [i + 1 for i in range(len(probs))]
        ans.score = float(sum(v * p for v, p in zip(vals, probs)))
        ans.legend = {v: o for v, o in zip(vals, f.options)}
        bracket = float(sum(probs[i] for i in _bracket_indices(vals, ans.score)))
        if f.none_option:
            # a probability again, not a share of the real levels; and if "outside the
            # offered range" is the top answer, that is what the confidence is about
            ans.confidence = ans.none if ans.choice == f.all_options[-1] else bracket * (1.0 - ans.none)
        else:
            ans.confidence = bracket
    elif f.kind == "yesno":
        ans.p_yes = float(probs[f.index_of("yes")])
    return ans


def answers_for(schema: Schema, pred: dict[str, list[float]]) -> dict[str, Answer]:
    """Read out every field of one example from a {field: [probs]} prediction."""
    return {f.name: answer_for(f, pred[f.name]) for f in schema}


# --- state rendering --------------------------------------------------------


def render_state(state: Any) -> str:
    """Render state as the text the encoder reads.

    str    -> as is
    dict   -> one "key: value" line per entry; nested values as compact JSON
    list   -> one numbered line per item (a message sequence, records)
    other  -> JSON
    """
    if isinstance(state, str):
        return state
    if isinstance(state, dict):
        lines = []
        for k, v in state.items():
            lines.append(f"{k}: {v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)}")
        return "\n".join(lines)
    if isinstance(state, (list, tuple)):
        return "\n".join(
            f"{i + 1}. {v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)}" for i, v in enumerate(state)
        )
    return json.dumps(state, ensure_ascii=False)
