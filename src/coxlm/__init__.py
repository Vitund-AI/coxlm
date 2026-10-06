"""coxlm: typed questions in, calibrated probabilities out, in one forward pass.

    import coxlm
    from coxlm import Questions, choice, yesno

    class Ticket(Questions):
        team = choice(["billing", "support"], instructions="Which team handles this?")
        refund = yesno("Is the customer asking for a refund?")

    # a coxlm server (no torch needed); or coxlm.load("model.pt") on a local GPU
    model = coxlm.connect("http://localhost:8000")

    ans = model.decide("My card was charged twice!!", Ticket)        # one state -> a Ticket
    ans.team.choice, ans.team.confidence, ans.refund.p_yes

    results = model.decide_batch([text_1, text_2], Ticket)          # several states -> a list of Tickets

``questions(team=choice(...), ...)`` builds the same questions without a class; its answers are read the same way
(``ans.team`` or ``ans["team"]``).

Importing coxlm does not import torch; only ``load`` does.
"""
from __future__ import annotations

from .client import CoxlmError, RemoteModel, connect
from .decide import Answer, Answers, AnswerSet, Questions, render_state
from .order import order_items
from .schema import Field, Schema, choice, multilabel, multiscore, numeric_field, questions, score, yesno

__version__ = "0.2.1"


def load(path, encoder: str | None = None, device: str | None = None, dtype: str | None = None,
         max_length: int = 2048):
    """Load a checkpoint for local GPU inference (needs ``pip install "coxlm[local]"``).

    Returns a model with the same ``decide(state, questions)`` / ``decide_batch(states, questions)`` contract as
    ``connect()``.
    """
    try:
        from .local import load as _load
        import torch  # noqa: F401
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise ImportError('coxlm.load needs the local extra: pip install "coxlm[local]"') from e
    return _load(path, encoder=encoder, device=device, dtype=dtype, max_length=max_length)


__all__ = [
    "Answer", "AnswerSet", "Answers", "CoxlmError", "Field", "Questions", "RemoteModel", "Schema", "choice", "connect", "load", "multilabel",
    "multiscore", "numeric_field", "order_items", "questions", "render_state", "score", "yesno", "__version__",
]
