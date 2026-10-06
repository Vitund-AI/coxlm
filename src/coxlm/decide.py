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
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, ClassVar, Iterable, Iterator, Mapping, TypeVar

from .schema import RESERVED_NAMES, Field, Schema


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

    def pick(self, min_confidence: float = 0.0, default: Any = None) -> Any:
        """The top option ("yes" / "no" for a yes/no question) if ``confidence`` is at least ``min_confidence``,
        otherwise ``default``. Made for branching, where being unsure is its own case:

            match ans.team.pick(0.7):
                case "billing": ...
                case "sales": ...
                case None: ask_a_person()
        """
        return self.choice if self.confidence >= min_confidence else default

    def ranked(self) -> list[tuple[str, float]]:
        """Every option with its probability, most likely first."""
        return sorted(self.probabilities.items(), key=lambda kv: -kv[1])


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


class AnswerSet(Mapping[str, Answer]):
    """The answers to one state: read-only, by attribute (``ans.team``) or by name (``ans["team"]``), and iterable
    like a dict (``for name, a in ans.items()``). The base of ``Answers`` and ``Questions``."""

    __slots__ = ("_answers",)

    def __init__(self, answers: Mapping[str, Answer]) -> None:
        object.__setattr__(self, "_answers", dict(answers))

    def __getitem__(self, name: str) -> Answer:
        return self._answers[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._answers)

    def __len__(self) -> int:
        return len(self._answers)

    if not TYPE_CHECKING:  # hidden from type checkers, so a typo in a Questions subclass's attribute is an error
        def __getattr__(self, name: str) -> Answer:
            if name.startswith("_"):
                raise AttributeError(name)
            try:
                return self._answers[name]
            except KeyError:
                raise AttributeError(f"no question named {name!r} (questions: {', '.join(self._answers)})") from None

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("answers are read-only")

    def __repr__(self) -> str:
        def short(a: Answer) -> str:
            if a.kind == "yesno":
                return f"p_yes={a.p_yes:.2f}"
            if a.kind == "score":
                return f"score={a.score:.2f}"
            return f"{a.choice!r} ({a.confidence:.2f})"
        return f"{type(self).__name__}(" + ", ".join(f"{k}={short(v)}" for k, v in self._answers.items()) + ")"


class Answers(AnswerSet):
    """The answers to a ``questions(...)`` schema: names are only known at run time, so any attribute reads as an
    answer (``ans.team``); ``ans["team"]`` works too."""

    __slots__ = ()

    if TYPE_CHECKING:
        def __getattr__(self, name: str) -> Answer: ...


class Questions(AnswerSet):
    """Define questions as a class; an instance of the class is then the answers to one state.

        class Ticket(coxlm.Questions):
            team = choice(["billing", "support", "sales"], instructions="Which team handles this?")
            refund = yesno("Is the customer asking for a refund?")

        ans = model.decide("My card was charged twice!!", Ticket)   # -> a Ticket
        ans.team.choice, ans.refund.p_yes                         # Ticket.team is the question itself

    Each class attribute built with choice / score / yesno is a question named after the attribute. Subclasses
    inherit their parents' questions. ``Ticket.__schema__`` is the equivalent ``questions(...)`` schema.
    """

    __slots__ = ()
    __schema__: ClassVar[Schema | None] = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        fields: dict[str, Field] = {}
        for base in reversed(cls.__mro__[1:]):
            inherited = base.__dict__.get("__schema__")
            if isinstance(inherited, Schema):
                fields.update({f.name: f for f in inherited})
        for name, value in list(cls.__dict__.items()):
            if isinstance(value, Schema):
                raise TypeError(f"{cls.__name__}.{name}: a whole schema cannot be one question; "
                                "write its questions as attributes, or pass the schema to decide() directly")
            if not isinstance(value, Field):
                continue
            if name in RESERVED_NAMES or name.startswith("_"):
                raise ValueError(f"{cls.__name__}.{name}: question names cannot start with '_' or be one of "
                                 f"{sorted(RESERVED_NAMES)} (the answers' mapping methods)")
            named = replace(value, name=name)
            setattr(cls, name, named)
            fields[name] = named
        cls.__schema__ = Schema(tuple(fields.values())) if fields else None


Q = TypeVar("Q", bound=Questions)


def iter_batches(states: Iterable[Any], batch_size: int) -> Iterator[list]:
    """Consume any iterable of states in lists of at most ``batch_size``."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    batch: list = []
    for st in states:
        batch.append(st)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def as_schema(questions: Schema | type[Questions]) -> Schema:
    """The Schema behind a questions(...) schema or a Questions subclass."""
    if isinstance(questions, Schema):
        return questions
    if isinstance(questions, type) and issubclass(questions, Questions):
        if questions.__schema__ is None:
            raise TypeError(f"{questions.__name__} defines no questions")
        return questions.__schema__
    raise TypeError("questions must be a coxlm schema (coxlm.questions(...)) or a coxlm.Questions subclass, "
                    f"not {type(questions).__name__}")


def answers_class(questions: Schema | type[Questions]) -> type[AnswerSet]:
    """The type one state's answers come back as: the Questions subclass itself, or Answers for a plain schema."""
    return questions if isinstance(questions, type) and issubclass(questions, Questions) else Answers


def answers_for(questions: Schema | type[Questions], pred: dict[str, list[float]]) -> AnswerSet:
    """Read out every field of one example from a {field: [probs]} prediction."""
    schema = as_schema(questions)
    return answers_class(questions)({f.name: answer_for(f, pred[f.name]) for f in schema})


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
