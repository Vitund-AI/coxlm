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

import os

from .client import CoxlmError, RemoteModel, connect
from .decide import Answer, Answers, AnswerSet, Questions, render_state
from .order import order_items
from .schema import Field, Schema, choice, multilabel, multiscore, numeric_field, questions, score, yesno

__version__ = "0.5.1"


def load(path, encoder: str | None = None, device: str | None = None, dtype: str | None = None,
         max_length: int = 2048, revision: str | None = None, backend: str | None = None, quantize: int | None = None):
    """Load a checkpoint for local inference.

    Returns a model with the same ``decide(state, questions)`` / ``decide_batch(states, questions)`` contract as
    ``connect()``. ``backend``: "torch" (``pip install "coxlm[local]"``; CUDA, Apple MPS or CPU) or "mlx"
    (``pip install "coxlm[mlx]"``; Apple silicon, model exports such as Vitund/cox-1-3b, no torch). The default
    (or the ``COXLM_BACKEND`` environment variable) is MLX on Apple silicon when mlx is installed and the checkpoint
    is a model export, else torch; passing ``encoder=`` or ``device=`` without a backend picks torch. ``quantize`` (MLX only): 4 or 8 bits for the backbone's linear layers.
    """
    explicit = backend or os.environ.get("COXLM_BACKEND")
    if explicit:
        backend = explicit
    elif encoder is not None or device is not None:
        backend = "torch"  # encoder= / device= are PyTorch options: asking for them picks PyTorch
    else:
        backend = _default_backend(path, revision)
    if backend == "mlx":
        from .mlx_backend import load as _load_mlx, mlx_available
        if not mlx_available():
            raise ImportError('the MLX backend needs the mlx extra on Apple silicon: pip install "coxlm[mlx]"')
        if encoder is not None or device is not None:
            raise ValueError("encoder= and device= are PyTorch options; the MLX backend reads the export folder")
        return _load_mlx(path, max_length=max_length, revision=revision, quantize=quantize, dtype=dtype)
    if backend != "torch":
        raise ValueError(f"unknown backend {backend!r} (torch or mlx)")
    if quantize:
        raise ValueError("quantize= is an MLX option (backend='mlx')")
    try:
        from .local import load as _load
        import torch  # noqa: F401
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise ImportError('coxlm.load needs the local extra: pip install "coxlm[local]"') from e
    return _load(path, encoder=encoder, device=device, dtype=dtype, max_length=max_length, revision=revision)


def _default_backend(path, revision) -> str:
    """MLX on Apple silicon when mlx is installed and `path` is a model export; torch everywhere else."""
    import platform
    if platform.system() != "Darwin" or platform.machine() not in ("arm64", "aarch64"):
        return "torch"
    from .mlx_backend import mlx_available
    if not mlx_available():
        return "torch"
    from .checkpoint import is_export, resolve
    try:
        return "mlx" if is_export(resolve(path, revision=revision)) else "torch"
    except (FileNotFoundError, ImportError):
        return "torch"


__all__ = [
    "Answer", "AnswerSet", "Answers", "CoxlmError", "Field", "Questions", "RemoteModel", "Schema", "choice", "connect", "load", "multilabel",
    "multiscore", "numeric_field", "order_items", "questions", "render_state", "score", "yesno", "__version__",
]
