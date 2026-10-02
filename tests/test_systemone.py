"""System One <-> cox translation round-trips without a model."""

from cox.systemone import answer_systemone, from_matrix_result, to_matrix_request

REQ = {
    "state": "The tenant has not paid rent for three months.",
    "model": "cox-latest",
    "questions": {
        "q_action": {"type": "choice", "instructions": "What should we do?",
                     "criteria": {"evict": "begin eviction", "warn": None, "wait": "do nothing"}},
        "q_urgent": {"type": "noul", "instructions": "Is this urgent?",
                     "criteria": {"true": "needs action now", "false": "can wait"}},
        "q_sev": {"type": "score", "instructions": "How severe?",
                  "criteria": ["minor", "moderate", "serious", "critical"]},
    },
}


def test_to_matrix_request_shapes():
    ids, matrix = to_matrix_request(REQ)
    assert ids == ["q_action", "q_urgent", "q_sev"]  # order preserved
    assert matrix["states"] == [{"state": REQ["state"]}]
    qs = matrix["questions"]
    assert qs[0]["type"] == "choice"
    # choice criteria -> "name" or "name: desc" lines
    assert qs[0]["options"].splitlines() == ["evict: begin eviction", "warn", "wait: do nothing"]
    assert qs[1]["type"] == "yesno" and qs[1]["question"] == "Is this urgent?"
    assert qs[2]["type"] == "score"
    # score levels -> 0-indexed "value: label" lines
    assert qs[2]["options"].splitlines() == ["0: minor", "1: moderate", "2: serious", "3: critical"]


def _fake_infer_matrix(matrix):
    # returns one cell per question, in order: choice, yesno, score
    return {
        "infer_ms": 12.3,
        "results": [{"id": 0, "state_tokens": 9, "cells": [
            {"kind": "choice", "choice": "evict", "confidence": 0.42,
             "probabilities": {"evict": 0.6, "warn": 0.25, "wait": 0.15}, "p_yes": None, "score": None},
            {"kind": "yesno", "p_yes": 0.88, "choice": None, "score": None,
             "confidence": 0.76, "probabilities": {"no": 0.12, "yes": 0.88}},
            {"kind": "score", "score": 2.1, "confidence": 0.5,
             "probabilities": {"0": 0.1, "1": 0.2, "2": 0.3, "3": 0.4}, "choice": None, "p_yes": None},
        ]}],
    }


def test_from_matrix_result_maps_answers():
    ids, matrix = to_matrix_request(REQ)
    resp = from_matrix_result(_fake_infer_matrix(matrix), ids, REQ, model_name="cox-1.7b")
    assert resp["model"] == "cox-1.7b"
    a = resp["answers"]
    assert set(a) == {"q_action", "q_urgent", "q_sev"}
    assert a["q_action"] == {"type": "choice", "choice": "evict", "confidence": 0.42,
                             "probabilities": {"evict": 0.6, "warn": 0.25, "wait": 0.15}}
    assert a["q_urgent"] == {"type": "noul", "noul": 0.88}
    assert a["q_sev"]["type"] == "score" and a["q_sev"]["score"] == 2.1
    # score legend reconstructed from the request's criteria (0-indexed)
    assert a["q_sev"]["legend"] == {"0": "minor", "1": "moderate", "2": "serious", "3": "critical"}
    assert resp["usage"] == {"input_tokens": 9, "output_tokens": 3}
    assert resp["latency_ms"] == 12.3


def test_answer_systemone_roundtrip():
    resp = answer_systemone(REQ, _fake_infer_matrix, model_name="cox")
    assert resp["answers"]["q_urgent"]["noul"] == 0.88


def test_rejects_unknown_type():
    bad = {"state": "x", "questions": {"q": {"type": "multilabel", "criteria": {}}}}
    try:
        to_matrix_request(bad)
        assert False, "expected ValueError"
    except ValueError as e:
        assert "unsupported type" in str(e)


def test_score_probabilities_keyed_by_level_index():
    # the real infer_matrix keys score probabilities by level label; the contract keys them "0".."n-1"
    ids, matrix = to_matrix_request(REQ)
    res = _fake_infer_matrix(matrix)
    res["results"][0]["cells"][2]["probabilities"] = {"minor": 0.1, "moderate": 0.2, "serious": 0.3, "critical": 0.4}
    a = from_matrix_result(res, ids, REQ)["answers"]["q_sev"]
    assert a["probabilities"] == {"0": 0.1, "1": 0.2, "2": 0.3, "3": 0.4}
    assert set(a["probabilities"]) == set(a["legend"])


def test_score_probabilities_keyed_by_level_index():
    from coxlm.systemone import _score_probs_by_index
    legend = {"0": "low", "1": "medium", "2": "high"}
    cell = {"probabilities": {"low": 0.2, "medium": 0.5, "high": 0.3}}
    assert _score_probs_by_index(cell, legend) == {"0": 0.2, "1": 0.5, "2": 0.3}
    assert _score_probs_by_index({"probabilities": {"0": 1.0, "1": 0.0, "2": 0.0}}, legend) == {"0": 1.0, "1": 0.0, "2": 0.0}
