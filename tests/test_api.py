"""The 0.2 API: decide (one state) / decide_batch / decide_iter, answers by attribute and by name, class-based
questions, and the helpers for program flow (pick, ranked, structural pattern matching)."""
from __future__ import annotations

import pytest

import coxlm
from coxlm import Answer, Answers, AnswerSet, Questions, choice, questions, score, yesno


class Ticket(Questions):
    team = choice(["billing", "support", "sales"], instructions="Which team handles this?")
    urgency = score(["low", "medium", "high"], instructions="How urgent is it?")
    refund = yesno("Is the customer asking for a refund?")


class EscalatedTicket(Ticket):
    angry = yesno("Is the customer angry?")


TEXT = "My card was charged twice!!"


def test_class_defines_the_same_questions_as_questions():
    plain = questions(team=Ticket.team, urgency=Ticket.urgency, refund=Ticket.refund)
    assert Ticket.__schema__ == plain
    assert Ticket.__schema__.names == ("team", "urgency", "refund")  # definition order
    assert isinstance(Ticket.team, coxlm.Field) and Ticket.team.name == "team"
    assert EscalatedTicket.__schema__.names == ("team", "urgency", "refund", "angry")  # inherited first


def test_class_rejects_reserved_and_schema_attributes():
    with pytest.raises(ValueError):
        type("Bad", (Questions,), {"items": choice(["a", "b"])})
    with pytest.raises(TypeError):
        type("Bad", (Questions,), {"tags": coxlm.multilabel(["x", "y"])})
    with pytest.raises(TypeError):
        coxlm.connect("http://127.0.0.1:9").decide("x", type("Empty", (Questions,), {}))


def test_decide_returns_the_class(fake_model):
    ans = fake_model.decide(TEXT, Ticket)
    assert isinstance(ans, Ticket) and isinstance(ans, AnswerSet) and not isinstance(ans, Answers)
    assert isinstance(ans.team, Answer) and ans.team is ans["team"]
    assert ans.team.choice in ("billing", "support", "sales")
    assert list(ans) == ["team", "urgency", "refund"] and len(ans) == 3
    assert dict(ans.items())["refund"] is ans.refund
    plain = fake_model.decide(TEXT, Ticket.__schema__)
    assert type(plain) is Answers and plain == ans  # same answers either way
    assert plain.team is plain["team"]


def test_answers_are_read_only_and_helpful(fake_model):
    ans = fake_model.decide(TEXT, Ticket)
    with pytest.raises(AttributeError):
        ans.team = None
    with pytest.raises(AttributeError, match="no question named 'tem'"):
        fake_model.decide(TEXT, Ticket.__schema__).tem
    assert repr(ans).startswith("Ticket(team=")


def test_reserved_question_name_warns():
    with pytest.warns(UserWarning, match="answers\\['items'\\]"):
        questions(items=choice(["a", "b"]))


def test_one_state_and_batches_through_a_server(server):
    url, fake = server
    model = coxlm.connect(url)
    one = model.decide(TEXT, Ticket)
    assert isinstance(one, Ticket) and one == fake.decide(TEXT, Ticket)
    many = model.decide_batch([TEXT, "Quote for 50 seats?"], Ticket)
    assert [type(a) for a in many] == [Ticket, Ticket] and many[0] == one
    assert model.decide_batch((s for s in [TEXT]), Ticket) == [one]  # any iterable
    with pytest.raises(TypeError):
        model.decide_batch(TEXT, Ticket)  # a bare string is one state, not a batch


def test_decide_iter_streams_in_batches(server):
    url, fake = server
    model = coxlm.connect(url)
    pulled = []

    def states():
        for i in range(7):
            pulled.append(i)
            yield f"ticket {i}"

    it = model.decide_iter(states(), Ticket, batch_size=3)
    first = next(it)
    assert isinstance(first, Ticket) and pulled == [0, 1, 2]  # only the first batch was read
    rest = list(it)
    assert len(rest) == 6 and pulled == list(range(7))
    assert [a.team.choice for a in [first, *rest]] == [fake.decide(f"ticket {i}", Ticket).team.choice for i in range(7)]


def _answer(kind, probs, **kw):
    top = max(probs, key=probs.get)
    return Answer(kind=kind, probabilities=probs, confidence=probs[top], top_probability=probs[top], choice=top, **kw)


def test_pick_and_ranked():
    sure = _answer("choice", {"billing": 0.8, "support": 0.15, "sales": 0.05})
    unsure = _answer("choice", {"billing": 0.45, "support": 0.4, "sales": 0.15})
    assert sure.pick(0.7) == "billing" and unsure.pick(0.7) is None and unsure.pick(0.7, default="?") == "?"
    assert unsure.pick() == "billing"
    assert [o for o, _ in unsure.ranked()] == ["billing", "support", "sales"]
    yes = _answer("yesno", {"no": 0.1, "yes": 0.9}, p_yes=0.9)
    assert yes.pick(0.8) == "yes"


def route(ans: Ticket) -> str:
    match ans.team.pick(0.7):
        case "billing":
            return "billing queue"
        case "support" | "sales" as team:
            return f"{team} queue"
        case None:
            return "a person"


def test_match_on_pick_and_patterns():
    sure = Ticket({"team": _answer("choice", {"billing": 0.9, "support": 0.05, "sales": 0.05}),
                   "urgency": _answer("score", {"low": 0.1, "medium": 0.2, "high": 0.7}, score=2.6),
                   "refund": _answer("yesno", {"no": 0.2, "yes": 0.8}, p_yes=0.8)})
    unsure = Ticket({**sure, "team": _answer("choice", {"billing": 0.4, "support": 0.35, "sales": 0.25})})
    assert route(sure) == "billing queue" and route(unsure) == "a person"
    # class patterns on one answer, mapping patterns on the whole set
    match sure.team:
        case Answer(choice="billing", confidence=c) if c > 0.85:
            hit = "confident billing"
        case _:
            hit = "other"
    assert hit == "confident billing"
    match sure:
        case {"refund": Answer(p_yes=p)} if p >= 0.8:
            hit = "refund"
        case _:
            hit = "other"
    assert hit == "refund"
