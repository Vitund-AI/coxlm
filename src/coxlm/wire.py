"""The native JSON wire format (POST /v1/decide), shared by the client and the server.

Request::

    {
      "states": [<str | object | array>, ...],
      "questions": {
        "<name>": {"type": "choice", "instructions": "...", "options": ["a", "b"] | {"a": "desc", ...}},
        "<name>": {"type": "score",  "instructions": "...", "options": ["low", "high"], "values": [0, 10]?},
        "<name>": {"type": "yesno",  "instructions": "...", "criteria": {"no": "...", "yes": "..."}?}
      }
    }

``criteria`` is accepted as a synonym of ``options`` (and ``kind`` of ``type``).

Response::

    {"model": "...", "answers": [{"<name>": {"kind", "choice", "p_yes", "score", "confidence",
                                             "top_probability", "probabilities", "legend", "none"}}, ...],
     "infer_ms": ...}

One answers dict per state, in order. Pure Python: no model, no torch.
"""
from __future__ import annotations

from typing import Any, Mapping

from .decide import Answer
from .schema import KIND_ALIASES, Field, Schema


def field_to_json(f: Field) -> dict:
    out: dict[str, Any] = {"type": f.kind}
    if f.instructions:
        out["instructions"] = f.instructions
    if f.kind == "yesno":
        if f.descriptions is not None and any(f.descriptions):
            out["criteria"] = dict(zip(f.options, f.descriptions))
    elif f.descriptions is not None:
        out["options"] = dict(zip(f.options, f.descriptions))
    else:
        out["options"] = list(f.options)
    if f.values is not None:
        out["values"] = list(f.values)
    if f.none_option:
        out["none_option"] = True
    return out


def schema_to_json(schema: Schema) -> dict:
    return {f.name: field_to_json(f) for f in schema}


def field_from_json(name: str, spec: Mapping[str, Any]) -> Field:
    if not isinstance(spec, Mapping):
        raise ValueError(f"question {name!r}: expected an object")
    kind = spec.get("type", spec.get("kind", "choice"))
    kind = KIND_ALIASES.get(kind, kind)
    instructions = spec.get("instructions") or ""
    if not isinstance(instructions, str):
        raise ValueError(f"question {name!r}: instructions must be a string")
    raw = spec.get("options", spec.get("criteria"))
    if kind == "yesno":
        desc = None
        if isinstance(raw, Mapping) and raw:
            desc = (str(raw.get("no", "") or ""), str(raw.get("yes", "") or ""))
        return Field(name, ("no", "yes"), instructions=instructions, descriptions=desc, kind="yesno")
    if isinstance(raw, Mapping):
        options = tuple(str(k) for k in raw.keys())
        descs = tuple("" if v is None else str(v) for v in raw.values())
        descriptions = descs if any(descs) else None
    elif isinstance(raw, (list, tuple)):
        options, descriptions = tuple(str(o) for o in raw), None
    else:
        raise ValueError(f"question {name!r}: {kind} needs 'options' (a list, or an object option -> description)")
    values = spec.get("values")
    vals = tuple(float(v) for v in values) if values is not None else None
    return Field(name, options, instructions=instructions, descriptions=descriptions, kind=kind,
                 none_option=bool(spec.get("none_option", False)), values=vals)


def schema_from_json(questions: Mapping[str, Any]) -> Schema:
    if not isinstance(questions, Mapping) or not questions:
        raise ValueError("'questions' must be a non-empty object keyed by question name")
    return Schema(tuple(field_from_json(str(n), s) for n, s in questions.items()))


def answer_to_json(a: Answer) -> dict:
    return {
        "kind": a.kind,
        "choice": a.choice,
        "p_yes": a.p_yes,
        "score": a.score,
        "confidence": a.confidence,
        "top_probability": a.top_probability,
        "probabilities": a.probabilities,
        # JSON object keys are strings; a list of [value, option] pairs keeps the numeric level
        "legend": [[v, o] for v, o in a.legend.items()],
        "none": a.none,
    }


def answer_from_json(d: Mapping[str, Any]) -> Answer:
    legend = d.get("legend") or []
    if isinstance(legend, Mapping):
        legend = list(legend.items())
    return Answer(
        kind=d["kind"],
        probabilities={str(k): float(v) for k, v in (d.get("probabilities") or {}).items()},
        confidence=float(d["confidence"]),
        top_probability=float(d.get("top_probability") or 0.0),
        choice=d.get("choice"),
        score=None if d.get("score") is None else float(d["score"]),
        legend={float(v): str(o) for v, o in legend},
        p_yes=None if d.get("p_yes") is None else float(d["p_yes"]),
        none=None if d.get("none") is None else float(d["none"]),
    )


def decide_request(states: list, schema: Schema) -> dict:
    return {"states": list(states), "questions": schema_to_json(schema)}


def decide_response(answers: list[dict[str, Answer]], model_name: str, infer_ms: float | None = None) -> dict:
    out: dict[str, Any] = {"model": model_name,
                           "answers": [{n: answer_to_json(a) for n, a in row.items()} for row in answers]}
    if infer_ms is not None:
        out["infer_ms"] = round(infer_ms, 1)
    return out
