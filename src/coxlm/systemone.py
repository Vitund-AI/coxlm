"""System One API compatibility: translate a ``/v1/systemone`` request into the
server's matrix-infer input and translate the result back.

The "System One" contract (a public API shape used by several typed-decision
servers) is:

    POST /v1/systemone
    {
      "state": <str|obj|arr>,
      "model": "<model id>",
      "questions": { "<id>": { "type": "noul"|"choice"|"score",
                               "instructions": <str?>,
                               "criteria": <see below> } }
    }
      noul   criteria: {"true": desc?, "false": desc?}      -> answer {"noul": p_yes}
      choice criteria: {option: desc|null, ...} (1-255)      -> {"choice", "probabilities", "confidence"}
      score  criteria: [level_desc, ...] low->high (1-255)   -> {"score", "legend", "probabilities", "confidence"}

    response: { "model", "answers": {<id>: {...}}, "usage": {...}, "latency_ms" }

These functions are PURE dict transforms (no model), so they unit-test on their
own. The server (coxlm.serve) wires them around its ``infer_matrix``. coxlm's
own extension types (multiscore, features, order) are not part of this contract;
they ride the native API. A noul's true/false descriptions are not yet threaded
(coxlm yesno uses its trained no/yes descriptions); everything else round-trips.
"""

from __future__ import annotations

import json
from typing import Any

BASE_TYPES = ("noul", "choice", "score")


def _as_text(x: Any) -> str:
    """state / instructions may be str | object | array -- render to text."""
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    return json.dumps(x, ensure_ascii=False, indent=2)


def _choice_options_text(criteria: dict[str, Any]) -> str:
    lines = []
    for name, desc in criteria.items():
        lines.append(str(name) if desc in (None, "") else f"{name}: {desc}")
    return "\n".join(lines)


def _score_options_text(criteria: list[Any]) -> str:
    # the server's score parser reads "value: label" lines (parse_score); System One levels are 0-indexed
    return "\n".join(f"{i}: {_as_text(d)}" for i, d in enumerate(criteria))


def to_matrix_request(req: dict) -> tuple[list[str], dict]:
    """System One request -> (ordered question ids, infer_matrix input).

    The matrix request is ``{"questions": [...], "states": [{"state": ...}]}``
    with one question per System One question, in the same order. Returns the id
    list so the response can be re-keyed.
    """
    questions = req.get("questions") or {}
    if not isinstance(questions, dict):
        raise ValueError("'questions' must be an object keyed by caller-chosen id")
    ids: list[str] = []
    cox_qs: list[dict] = []
    for qid, spec in questions.items():
        typ = spec.get("type")
        instr = _as_text(spec.get("instructions"))
        criteria = spec.get("criteria")
        if typ == "noul":
            cox_qs.append({"type": "yesno", "question": instr or "answer"})
        elif typ == "choice":
            if not isinstance(criteria, dict) or len(criteria) < 1:
                raise ValueError(f"question {qid!r}: choice needs a criteria object of options")
            cox_qs.append({"type": "choice", "question": instr, "options": _choice_options_text(criteria)})
        elif typ == "score":
            if not isinstance(criteria, (list, tuple)) or len(criteria) < 2:
                raise ValueError(f"question {qid!r}: score needs a criteria array of >=2 levels")
            cox_qs.append({"type": "score", "question": instr, "options": _score_options_text(criteria)})
        else:
            raise ValueError(f"question {qid!r}: unsupported type {typ!r} (System One base types: {BASE_TYPES})")
        ids.append(qid)
    matrix = {"questions": cox_qs, "states": [{"state": _as_text(req.get("state"))}]}
    return ids, matrix


def _usage(matrix_result: dict) -> dict:
    res = (matrix_result.get("results") or [{}])[0]
    # matrix results don't carry token counts; report what we have honestly
    return {"input_tokens": res.get("state_tokens"), "output_tokens": len(res.get("cells", []))}


def from_matrix_result(matrix_result: dict, ids: list[str], req: dict,
                       model_name: str = "cox") -> dict:
    """infer_matrix result (single state) -> System One response."""
    results = matrix_result.get("results") or []
    if not results:
        return {"model": model_name, "answers": {}, "usage": _usage(matrix_result),
                "latency_ms": matrix_result.get("infer_ms")}
    cells = results[0].get("cells", [])
    questions = req.get("questions") or {}
    answers: dict[str, dict] = {}
    for qid, cell in zip(ids, cells):
        kind = cell.get("kind")
        if kind == "yesno":
            answers[qid] = {"type": "noul", "noul": cell.get("p_yes")}
        elif kind == "choice":
            answers[qid] = {"type": "choice", "choice": cell.get("choice"),
                            "confidence": cell.get("confidence"),
                            "probabilities": cell.get("probabilities")}
        elif kind == "score":
            criteria = (questions.get(qid) or {}).get("criteria") or []
            legend = {str(i): _as_text(d) for i, d in enumerate(criteria)}
            answers[qid] = {"type": "score", "score": cell.get("score"),
                            "confidence": cell.get("confidence"),
                            "legend": legend, "probabilities": cell.get("probabilities")}
        else:  # an extension type reached here; pass it through rather than lose it
            answers[qid] = {"type": kind, **{k: v for k, v in cell.items() if k != "kind"}}
    return {"model": model_name, "answers": answers, "usage": _usage(matrix_result),
            "latency_ms": matrix_result.get("infer_ms")}


def answer_systemone(req: dict, infer_matrix, model_name: str = "cox") -> dict:
    """Full round-trip: translate, run ``infer_matrix``, translate back.

    ``infer_matrix`` is the server's matrix-infer callable (state x questions -> cells)."""
    ids, matrix = to_matrix_request(req)
    result = infer_matrix(matrix)
    return from_matrix_result(result, ids, req, model_name=model_name)
