import pytest

import coxlm
from coxlm import choice, questions, score, yesno
from coxlm.decide import answers_for
from coxlm.wire import answer_from_json, answer_to_json, schema_from_json, schema_to_json


def deck_schema():
    return questions(
        team=choice(["billing", "support", "sales"], instructions="Which team handles this?"),
        urgency=score(["low", "medium", "high"], instructions="How urgent is it?"),
        refund=yesno("Is the customer asking for a refund?"))


def test_questions_names_fields_by_keyword():
    s = deck_schema()
    assert s.names == ("team", "urgency", "refund")
    assert [f.kind for f in s] == ["choice", "score", "yesno"]
    assert s.field("refund").options == ("no", "yes")
    assert s.field("urgency").options == ("low", "medium", "high")


def test_field_validation():
    with pytest.raises(ValueError):
        choice(["only one"])
    with pytest.raises(ValueError):
        choice(["a", "a"])
    with pytest.raises(ValueError):
        score(["a", "b"], values=[2, 1])
    with pytest.raises(ValueError):
        questions()


def test_descriptions_and_values():
    s = questions(
        intent=choice({"refund": "wants money back", "cancel": "wants to stop the plan"}, instructions="What do they want?"),
        sentiment=score(["neg", "neutral", "pos"], values=[-1, 0, 1]),
        human=yesno("Should a person handle this?", criteria={"no": "a bot can answer", "yes": "needs judgement"}))
    assert s.field("intent").descriptions == ("wants money back", "wants to stop the plan")
    assert s.field("sentiment").values == (-1.0, 0.0, 1.0)
    assert s.field("human").descriptions == ("a bot can answer", "needs judgement")


def test_schema_wire_round_trip():
    s = questions(
        intent=choice({"refund": "wants money back", "cancel": ""}, instructions="What do they want?"),
        sentiment=score(["neg", "neutral", "pos"], values=[-1, 0, 1], instructions="Tone?"),
        human=yesno("Should a person handle this?", criteria={"no": "bot", "yes": "person"}),
        plain=yesno("Is it spam?"),
        team=choice(["a", "b"]))
    assert schema_from_json(schema_to_json(s)) == s


def test_wire_accepts_criteria_and_aliases():
    s = schema_from_json({
        "x": {"type": "noul", "instructions": "Is it?"},
        "y": {"kind": "choice", "criteria": {"a": None, "b": "bee"}},
        "z": {"type": "score", "criteria": ["lo", "hi"]},
    })
    assert [f.kind for f in s] == ["yesno", "choice", "score"]
    assert s.field("y").descriptions == ("", "bee")
    with pytest.raises(ValueError):
        schema_from_json({"x": {"type": "choice"}})
    with pytest.raises(ValueError):
        schema_from_json({})


def test_answer_readout_and_wire_round_trip():
    s = deck_schema()
    ans = answers_for(s, {"team": [0.7, 0.2, 0.1], "urgency": [0.1, 0.4, 0.5], "refund": [0.1, 0.9]})
    assert ans["team"].choice == "billing" and ans["team"].confidence == pytest.approx(0.7)
    assert ans["urgency"].score == pytest.approx(0.1 * 1 + 0.4 * 2 + 0.5 * 3)
    assert ans["refund"].p_yes == pytest.approx(0.9)
    for a in ans.values():
        assert answer_from_json(answer_to_json(a)) == a


def test_public_api_surface():
    for name in ("connect", "load", "questions", "choice", "score", "yesno", "Answer", "Schema", "__version__"):
        assert hasattr(coxlm, name)
    assert coxlm.__version__ == "0.1.0"
