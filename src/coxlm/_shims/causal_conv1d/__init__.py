"""Shim: expose the Hugging Face hub build of causal-conv1d (kernels-community/causal-conv1d, version 1) under the
package name transformers looks for, so Qwen3.5's linear-attention layers use the fused kernel instead of the slow
PyTorch reference path. No compilation needed. Enabled by coxlm.model when COX_HUB_CONV1D != 0 and `kernels` imports.
"""
import os

from kernels import get_kernel as _get

# Pinned to the commit that version 1 resolves to: with HF_HUB_OFFLINE=1 (as our run scripts set) `version=1` cannot be
# resolved without the network, the import raised, and -- since flash-linear-attention imports causal_conv1d -- BOTH the
# conv and the gated-delta-rule kernels silently fell back to the slow PyTorch reference (2026-10-03).
# COX_CONV1D_REVISION overrides; a host without the cached build needs one online run first.
_REV = os.environ.get("COX_CONV1D_REVISION", "19d5632c27565e020a0aa0169bc7ad44a2c7080b")


def _load():
    try:
        return _get("kernels-community/causal-conv1d", revision=_REV)
    except Exception:
        # offline, kernels' own lookup can miss a snapshot that is in the cache: load the cached build directly
        from pathlib import Path

        from kernels import get_local_kernel
        hub = Path(os.environ.get("HF_HUB_CACHE") or Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub")
        snap = hub / "kernels--kernels-community--causal-conv1d" / "snapshots" / _REV
        if not snap.is_dir():
            raise
        return get_local_kernel(snap)


_k = _load()
causal_conv1d_fn = _k.causal_conv1d_fn
causal_conv1d_update = _k.causal_conv1d_update
__all__ = ["causal_conv1d_fn", "causal_conv1d_update"]
