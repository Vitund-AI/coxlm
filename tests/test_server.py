import json
import subprocess
import sys
import urllib.request

import pytest

import coxlm
from coxlm import choice, questions, score, yesno


def _get(url):
    with urllib.request.urlopen(url) as r:
        return r.status, r.read().decode()


def _post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_quickstart_through_connect(server):
    url, fake = server
    model = coxlm.connect(url)
    schema = questions(
        team=choice(["billing", "support", "sales"], instructions="Which team handles this?"),
        urgency=score(["low", "medium", "high"], instructions="How urgent is it?"),
        refund=yesno("Is the customer asking for a refund?"))
    ans = model.decide("My card was charged twice!!", schema)
    # the remote answers equal what the model computed in-process
    local = fake.decide("My card was charged twice!!", schema)
    assert ans == local
    assert fake.calls[0][1] == schema  # the server rebuilt exactly the client's schema
    assert ans["team"].choice in ("billing", "support", "sales")
    assert 0.0 <= ans["refund"].p_yes <= 1.0
    assert 1.0 <= ans["urgency"].score <= 3.0
    assert set(ans["team"].probabilities) == {"billing", "support", "sales"}
    assert ans["urgency"].legend == {1.0: "low", 2.0: "medium", 3.0: "high"}


def test_structured_states_and_many_states(server):
    url, fake = server
    model = coxlm.connect(url)
    schema = questions(spam=yesno("Is this spam?"), lang=choice(["en", "de"], instructions="Language?"))
    states = ["hello", {"subject": "WIN", "body": "click"}, ["msg one", "msg two"]] * 15  # 45 > one batch of 32
    out = model.decide_batch(states, schema)
    assert len(out) == 45 and out[1] == fake.decide({"subject": "WIN", "body": "click"}, schema)
    # a list passed to decide() is ONE state (numbered lines), not a batch
    assert model.decide(["msg one", "msg two"], schema) == out[2]


def test_decide_wire_format(server):
    url, _ = server
    code, body = _post(url + "/v1/decide", {
        "states": ["My card was charged twice!!"],
        "questions": {"team": {"type": "choice", "instructions": "Which team?", "options": ["billing", "support"]},
                      "refund": {"type": "yesno", "instructions": "Refund?"},
                      "urgency": {"type": "score", "options": {"low": "can wait", "high": "now"}}}})
    assert code == 200
    assert body["model"] == "coxlm-fake-1b"
    [row] = body["answers"]
    assert set(row) == {"team", "refund", "urgency"}
    assert set(row["team"]) >= {"choice", "confidence", "probabilities"}
    assert row["refund"]["p_yes"] is not None and row["urgency"]["score"] is not None


def test_decide_rejects_bad_requests(server):
    url, _ = server
    code, body = _post(url + "/v1/decide", {"states": "not a list", "questions": {"a": {"type": "yesno"}}})
    assert code == 422 and "states" in body["error"]["message"]
    code, body = _post(url + "/v1/decide", {"states": ["x"], "questions": {"a": {"type": "choice", "options": ["one"]}}})
    assert code == 422
    with pytest.raises(coxlm.CoxlmError, match="questions"):
        coxlm.connect(url)._request("POST", "/v1/decide", {"states": ["x"], "questions": {}})


def test_other_endpoints(server):
    url, _ = server
    code, health = _get(url + "/health")
    assert code == 200 and json.loads(health)["status"] == "ok"
    code, models = _get(url + "/v1/models")
    assert json.loads(models)["data"][0]["id"].startswith("coxlm")
    code, demos = _get(url + "/demos")
    assert isinstance(json.loads(demos), list) and json.loads(demos)
    code, page = _get(url + "/")
    assert "coxlm" in page and "Example/Fake-1B-Base" in page
    code, s1 = _post(url + "/v1/systemone", {"state": "hi", "questions": {
        "q": {"type": "choice", "criteria": {"a": None, "b": "bee"}}, "n": {"type": "noul", "instructions": "ok?"}}})
    assert code == 200 and s1["answers"]["q"]["type"] == "choice" and s1["answers"]["n"]["type"] == "noul"
    code, inf = _post(url + "/infer", {"questions": [{"type": "choice", "question": "Which?", "options": "a\nb"},
                                                     {"type": "order", "question": "Order", "options": "x\ny\nz"}],
                                       "states": [{"state": "text"}]})
    assert code == 200 and "error" not in inf and inf["results"][0]["cells"][1]["kind"] == "order"


def test_unreachable_server_raises():
    with pytest.raises(coxlm.CoxlmError):
        coxlm.connect("http://127.0.0.1:9", timeout=2).decide("x", questions(a=yesno("ok?")))


def test_import_does_not_import_torch():
    code = "import sys, coxlm, coxlm.serve, coxlm.wire, coxlm.systemone; assert 'torch' not in sys.modules, 'torch imported'"
    subprocess.run([sys.executable, "-c", code], check=True)
