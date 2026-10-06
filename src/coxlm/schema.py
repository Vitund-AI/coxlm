"""Schema definitions for the amortized decision model.

A Schema is a fixed, predefined output space: a list of Fields, each of which is
a categorical choice over an enumerated set of options. Numeric fields are
supported by quantizing a range into labeled buckets, since a bounded numeric
answer is the same softmax head as a categorical one. That collapse is why the
whole output space stays type-safe and cheap: one readout head covers the entire
type system.

Type safety falls out for free here. The model can only ever place probability
mass on options that exist in the schema, so it physically cannot emit an
off-schema value. That guarantees well-formedness, not correctness -- a
confidently wrong option is still a valid option, which is exactly why the
calibration work later matters more than the type guarantee.

Three question kinds sit on the one head:

    choice   pick one of several unordered options            -> choice, probabilities, confidence
    score    rate along ordered levels                         -> score (may fall between levels), probabilities, confidence
    yesno     is this true? a yes/no probability               -> p_yes (P(yes)), probabilities, confidence

They differ only in how the distribution is read out (see coxlm/decide.py), not
in how it is produced. A field may carry ``instructions`` (the question in
words) and a description per option; both are encoded as text by the same
encoder that reads the state, which is what makes a question specifiable rather
than merely named.
"""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable, Mapping, overload

if TYPE_CHECKING:
    from .decide import Answer

MAX_CARDINALITY = 255  # an 8-bit output head; higher cardinality needs a two-stage score-then-choose

KINDS = ("choice", "score", "yesno")
# The reserved abstention option. Offering it is what makes a field a "describe"
# question (the answer must fit the state, and may be that nothing offered does);
# a field without it is a "choose among these" question (pick the best of what
# exists). The model can see which it is, because the option is in what it reads.
NONE_OPTION = "none_of_these"
NONE_DESCRIPTIONS = {"choice": "none of the other options fits", "score": "the true level is outside the offered range"}
# accepted on input only (older spec files and payloads); never emitted
KIND_ALIASES = {"noul": "yesno", "boolean": "yesno", "bool": "yesno"}
# names an answer set already uses (its mapping methods): a question with one of these names is still readable as
# answers["name"], but not as answers.name
RESERVED_NAMES = frozenset({"keys", "items", "values", "get"})


@dataclass(frozen=True)
class Field:
    """One decision: a name and the enumerated options it may take.

    ``instructions`` is the question asked of the state, in words. Empty means
    "the name is the question". ``descriptions`` are per-option clarifications,
    aligned with ``options``; None means the option strings speak for themselves.
    ``kind`` decides how the answer is read out; it never changes the head.
    """

    name: str
    options: tuple[str, ...]
    instructions: str = ""
    descriptions: tuple[str, ...] | None = None
    kind: str = "choice"
    none_option: bool = False  # offer NONE_OPTION as a final, reserved option (never on yes/no fields)
    # score only: explicit numeric level values, aligned with options and strictly
    # increasing. The ordinal head uses only the ORDER of the levels, so these change
    # nothing the model computes -- they set the units the expected level is read out
    # in (e.g. -2..+2 sentiment, or a non-uniform severity scale). None -> levels 1..k.
    values: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if NONE_OPTION in self.options:
            raise ValueError(f"field {self.name!r}: {NONE_OPTION!r} is reserved; set none_option=True instead")
        if self.none_option and self.kind == "yesno":
            raise ValueError(f"yes/no field {self.name!r} is exhaustive and cannot offer {NONE_OPTION!r}")
        if len(self.options) < 2:
            raise ValueError(f"field {self.name!r} needs at least 2 options")
        if len(self.options) > MAX_CARDINALITY:
            raise ValueError(
                f"field {self.name!r} has {len(self.options)} options; max is "
                f"{MAX_CARDINALITY}. Use a two-stage retrieve-then-rank for high cardinality."
            )
        if len(set(self.options)) != len(self.options):
            raise ValueError(f"field {self.name!r} has duplicate options")
        if self.descriptions is not None and len(self.descriptions) != len(self.options):
            raise ValueError(f"field {self.name!r}: descriptions must align with options")
        if self.kind not in KINDS:
            raise ValueError(f"field {self.name!r}: kind must be one of {KINDS}")
        if self.kind == "yesno" and len(self.options) != 2:
            raise ValueError(f"yesno field {self.name!r} must have exactly two options (no, yes)")
        if self.values is not None:
            if self.kind != "score":
                raise ValueError(f"field {self.name!r}: values are only meaningful for score fields")
            if len(self.values) != len(self.options):
                raise ValueError(f"field {self.name!r}: values must align with options ({len(self.options)})")
            if any(b <= a for a, b in zip(self.values, self.values[1:])):
                raise ValueError(f"field {self.name!r}: values must be strictly increasing (levels run low -> high)")

    # A Field written as a class attribute of a coxlm.Questions subclass doubles as the accessor for its answer:
    # on the class it is the question (Ticket.team), on an answer instance it is the answer (ans.team).
    @overload
    def __get__(self, obj: None, owner: Any = None) -> "Field": ...
    @overload
    def __get__(self, obj: object, owner: Any = None) -> "Answer": ...

    def __get__(self, obj, owner=None):
        if obj is None:
            return self
        return obj[self.name]

    @property
    def cardinality(self) -> int:
        return len(self.options)

    def index_of(self, option: str) -> int:
        return self.all_options.index(option)

    @property
    def all_options(self) -> tuple[str, ...]:
        """The options the model scores: the author's, plus the reserved none option if offered."""
        return self.options + ((NONE_OPTION,) if self.none_option else ())

    @property
    def n_outputs(self) -> int:
        return len(self.all_options)

    # -- what the encoder reads ------------------------------------------

    def query_text(self) -> str:
        """The text encoded into this field's query vector."""
        if self.instructions:
            return f"{self.name}: {self.instructions}"
        return self.name

    def option_texts(self) -> tuple[str, ...]:
        """The text encoded for each option, aligned with ``options``."""
        if self.descriptions is None:
            texts = self.options
        else:
            texts = tuple(f"{o}: {d}" if d else o for o, d in zip(self.options, self.descriptions))
        if self.none_option:
            texts = texts + (f"{NONE_OPTION}: {NONE_DESCRIPTIONS.get(self.kind, NONE_DESCRIPTIONS['choice'])}",)
        return texts

    def spec(self) -> dict:
        """A JSON-able description of the question, for docs and the wire format."""
        out: dict = {"kind": self.kind, "options": list(self.options)}
        if self.none_option:
            out["none_option"] = True
        if self.instructions:
            out["instructions"] = self.instructions
        if self.descriptions is not None:
            out["descriptions"] = dict(zip(self.options, self.descriptions))
        if self.values is not None:
            out["values"] = list(self.values)
        return out


def _criteria(criteria: Mapping[str, str] | Iterable[str]) -> tuple[tuple[str, ...], tuple[str, ...] | None]:
    if isinstance(criteria, Mapping):
        return tuple(criteria.keys()), tuple(criteria.values())
    return tuple(criteria), None


def choice(criteria: Mapping[str, str] | Iterable[str], instructions: str = "", name: str = "") -> Field:
    """Pick one of several unordered options.

    ``criteria`` is either a list of option names or a mapping option -> description.
    ``name`` may be left empty when the field is passed to ``questions()``.
    """
    options, descriptions = _criteria(criteria)
    return Field(name, options, instructions=instructions, descriptions=descriptions, kind="choice")


def score(levels: Mapping[str, str] | Iterable[str], instructions: str = "", name: str = "",
          values: Iterable[float] | None = None) -> Field:
    """Rate the state along ordered levels; the answer may fall between two levels.

    ``levels`` is ordered low -> high, either level names or a mapping level -> description.
    ``values`` optionally sets each level's numeric value (aligned with ``levels``,
    strictly increasing); the expected level is then read out in those units instead
    of 1..k. This makes a score self-documenting and lets scores from different fields
    be combined into a single compound number (e.g. a weighted priority).
    """
    options, descriptions = _criteria(levels)
    vals = tuple(float(v) for v in values) if values is not None else None
    return Field(name, options, instructions=instructions, descriptions=descriptions, kind="score", values=vals)


def yesno(instructions: str, criteria: Mapping[str, str] | None = None, name: str = "") -> Field:
    """A yes/no judgment, answered as P(yes). ``criteria`` may clarify what "no"
    and "yes" mean, as {"no": ..., "yes": ...}."""
    descriptions = None
    if criteria:
        descriptions = (criteria.get("no", ""), criteria.get("yes", ""))
    return Field(name, ("no", "yes"), instructions=instructions, descriptions=descriptions, kind="yesno")


def numeric_field(name: str, low: float, high: float, n_buckets: int, instructions: str = "") -> Field:
    """Quantize [low, high] into ``n_buckets`` contiguous buckets -> an ordered score Field.

    A numeric answer becomes a distribution over ordered buckets, which is the
    identical machinery to a categorical field. This is the unification that lets
    one readout head cover both numeric and categorical outputs.
    """
    if not (2 <= n_buckets <= MAX_CARDINALITY):
        raise ValueError("n_buckets must be in [2, 255]")
    edges = [low + (high - low) * i / n_buckets for i in range(n_buckets + 1)]
    opts = tuple(f"[{edges[i]:.3g},{edges[i + 1]:.3g})" for i in range(n_buckets))
    return Field(name=name, options=opts, instructions=instructions, kind="score")


@dataclass(frozen=True)
class Schema:
    """A fixed set of fields the model answers together in one pass."""

    fields: tuple[Field, ...]

    def __post_init__(self) -> None:
        names = [f.name for f in self.fields]
        if len(set(names)) != len(names):
            raise ValueError("duplicate field names in schema")
        if not self.fields:
            raise ValueError("schema needs at least one field")

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(f.name for f in self.fields)

    def field(self, name: str) -> Field:
        for f in self.fields:
            if f.name == name:
                return f
        raise KeyError(name)

    def spec(self) -> dict:
        return {f.name: f.spec() for f in self.fields}

    def __iter__(self):
        return iter(self.fields)

    def __len__(self) -> int:
        return len(self.fields)


def schema_from_spec(spec: dict) -> Schema:
    """Inverse of ``Schema.spec()``: {name: {kind, options, instructions?, descriptions?}}."""
    fields = []
    for name, f in spec.items():
        options = tuple(f["options"])
        desc = None
        if f.get("descriptions"):
            desc = tuple(f["descriptions"].get(o, "") for o in options)
        kind = f.get("kind", "choice")
        kind = KIND_ALIASES.get(kind, kind)
        vals = tuple(float(v) for v in f["values"]) if f.get("values") else None
        fields.append(Field(name, options, instructions=f.get("instructions", ""), descriptions=desc, kind=kind,
                            none_option=bool(f.get("none_option", False)), values=vals))
    return Schema(tuple(fields))


def with_none(schema: "Schema") -> "Schema":
    """The same schema with the none option offered on every field that can carry it."""
    from dataclasses import replace

    return Schema(tuple(f if f.kind == "yesno" else replace(f, none_option=True) for f in schema))


def questions(**fields: Field) -> Schema:
    """Build a Schema from keyword fields, renaming each to its keyword.

        schema = questions(
            action=choice({"refund": "...", "cancel": "..."}, instructions="What does the user want?"),
            frustration=score(["calm", "annoyed", "angry"], instructions="How frustrated is the user?"),
            needs_human=yesno("Should a person handle this?"),
        )
    """
    from dataclasses import replace

    for name in fields:
        if name in RESERVED_NAMES:
            warnings.warn(f"question name {name!r} is also an answer-set method: read it as answers[{name!r}], "
                          f"not answers.{name}", stacklevel=2)
    return Schema(tuple(replace(f, name=name) for name, f in fields.items()))


def multilabel(features, instructions: str = "", question: str = "Is this feature present in the text?") -> Schema:
    """Judge each feature independently as a yes/no: a multi-label / feature-tagging
    schema, where several features can be true at once. This is exactly one yesno
    field per feature (they compete in a `choice`, but are independent here), and
    the model answers them all in one forward pass. ``features`` is a list of
    feature strings, or a mapping feature -> a clarifying description.

        schema = multilabel(["first-person narrator", "past tense", "direct dialogue"])
        # -> P(present) for each, independently
    """
    from dataclasses import replace as _replace

    items = features.items() if hasattr(features, "items") else ((f, "") for f in features)
    fields = []
    seen = set()
    for feat, desc in items:
        slug = re.sub(r"[^a-z0-9]+", "_", str(feat).lower()).strip("_")[:40] or "feature"
        base = slug
        k = 2
        while slug in seen:
            slug = f"{base}_{k}"; k += 1
        seen.add(slug)
        instr = f"{instructions} {question} Feature: {feat}".strip()
        fields.append(_replace(yesno(instr, criteria=({"yes": desc} if desc else None)), name=slug))
    return Schema(tuple(fields))


def multiscore(aspects, levels, instructions: str = "", question: str = "Rate this aspect on the scale.",
               values: Iterable[float] | None = None) -> Schema:
    """Rate each aspect independently on one shared ordered scale: the graded analog
    of ``multilabel``. Exactly one ``score`` field per aspect, all sharing ``levels``
    (and ``values`` if given), answered together in one forward pass. ``aspects`` is a
    list of aspect strings, or a mapping aspect -> a clarifying description.

        schema = multiscore(["food", "service", "ambiance"], ["poor", "ok", "great"])
        # -> an expected level per aspect, independently, on the same scale
    """
    from dataclasses import replace as _replace

    options, descriptions = _criteria(levels)
    lv = options if descriptions is None else dict(zip(options, descriptions))
    items = aspects.items() if hasattr(aspects, "items") else ((a, "") for a in aspects)
    fields, seen = [], set()
    for asp, desc in items:
        slug = re.sub(r"[^a-z0-9]+", "_", str(asp).lower()).strip("_")[:40] or "aspect"
        base, k = slug, 2
        while slug in seen:
            slug = f"{base}_{k}"; k += 1
        seen.add(slug)
        instr = f"{instructions} {question} Aspect: {asp}".strip()
        fields.append(_replace(score(lv, instructions=instr, values=values), name=slug))
    return Schema(tuple(fields))
