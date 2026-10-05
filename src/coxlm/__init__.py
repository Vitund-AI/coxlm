"""coxlm: typed questions in, calibrated probabilities out, in one forward pass.

    import coxlm
    from coxlm import questions, choice

    # a coxlm server (no torch needed); or coxlm.load("model.pt") on a local GPU
    model = coxlm.connect("http://localhost:8000")

    schema = questions(team=choice(["billing", "support"], instructions="Which team handles this?"))

    # one result per text
    answers = model.decide(["My card was charged twice!!"], schema)
    ans = answers[0]
    ans["team"].choice, ans["team"].confidence, ans["team"].probabilities

Importing coxlm does not import torch; only ``load`` does.
"""
from __future__ import annotations

from .client import CoxlmError, RemoteModel, connect
from .decide import Answer, render_state
from .order import order_items
from .schema import Field, Schema, choice, multilabel, multiscore, numeric_field, questions, score, yesno

__version__ = "0.1.0"


def load(path, encoder: str | None = None, device: str | None = None, dtype: str | None = None,
         max_length: int = 2048):
    """Load a checkpoint for local GPU inference (needs ``pip install "coxlm[local]"``).

    Returns a model with the same ``decide(states, schema)`` contract as ``connect()``.
    """
    try:
        from .local import load as _load
        import torch  # noqa: F401
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise ImportError('coxlm.load needs the local extra: pip install "coxlm[local]"') from e
    return _load(path, encoder=encoder, device=device, dtype=dtype, max_length=max_length)


__all__ = [
    "Answer", "CoxlmError", "Field", "RemoteModel", "Schema", "choice", "connect", "load", "multilabel",
    "multiscore", "numeric_field", "order_items", "questions", "render_state", "score", "yesno", "__version__",
]
