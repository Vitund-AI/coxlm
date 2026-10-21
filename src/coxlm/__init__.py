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
         max_length: int = 2048, revision: str | None = None, backend: str | None = None, quantize: int | None = None,
         adapter: str | os.PathLike | None = None, adapter_revision: str | None = None, merge: bool = False):
    """Load a checkpoint for local inference.

    Returns a model with the same ``decide(state, questions)`` / ``decide_batch(states, questions)`` contract as
    ``connect()``. ``backend``: "torch" (``pip install "coxlm[local]"``; CUDA, Apple MPS or CPU) or "mlx"
    (``pip install "coxlm[mlx]"``; Apple silicon, model exports such as Vitund/cox-1-3b, no torch). The default
    (or the ``COXLM_BACKEND`` environment variable) is MLX on Apple silicon when mlx is installed and the checkpoint
    is a model export, else torch; passing ``encoder=`` or ``device=`` without a backend picks torch. ``quantize`` (MLX only): 4 or 8 bits for the backbone's linear layers.

    ``adapter``: a fine-tune adapter made by ``coxlm.finetune`` (a folder or a Hub repo id, pinned with
    ``adapter_revision``), applied on top of a model export (PyTorch backend; passing one picks it). It is refused if
    it was trained on another base. The base weights stay as they are unless ``merge=True`` folds the adapter in.
    """
    explicit = backend or os.environ.get("COXLM_BACKEND")
    if explicit:
        backend = explicit
    elif encoder is not None or device is not None or adapter is not None:
        backend = "torch"  # encoder= / device= / adapter= are PyTorch options: asking for them picks PyTorch
    else:
        backend = _default_backend(path, revision)
    if backend == "mlx":
        if adapter is not None:
            raise NotImplementedError("adapters on MLX come later; load the adapter with backend='torch' (CUDA, Apple "
                                      "MPS or CPU)")
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
    return _load(path, encoder=encoder, device=device, dtype=dtype, max_length=max_length, revision=revision,
                 adapter=adapter, adapter_revision=adapter_revision, merge=merge)


def finetune(model, examples, questions=None, out_dir="cox-adapter", **kwargs):
    """Adapt a model export to your own labelled examples with a small LoRA; returns (adapter folder, report).
    See ``coxlm.finetuning.finetune`` for the arguments (needs the local extra)."""
    try:
        from .finetuning import finetune as _finetune
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise ImportError('coxlm.finetune needs the local extra: pip install "coxlm[local]"') from e
    return _finetune(model, examples, questions, out_dir, **kwargs)


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
    "Answer", "AnswerSet", "Answers", "CoxlmError", "Field", "Questions", "RemoteModel", "Schema", "choice", "connect", "finetune", "load", "multilabel",
    "multiscore", "numeric_field", "order_items", "questions", "render_state", "score", "yesno", "__version__",
]
