"""Shim: expose the Hugging Face hub build of causal-conv1d (kernels-community/causal-conv1d, version 1) under the
package name transformers looks for, so Qwen3.5's linear-attention layers use the fused kernel instead of the slow
PyTorch reference path. No compilation needed. Enabled by cox.model when COX_HUB_CONV1D != 0 and `kernels` imports.
"""
from kernels import get_kernel as _get

_k = _get("kernels-community/causal-conv1d", version=1)
causal_conv1d_fn = _k.causal_conv1d_fn
causal_conv1d_update = _k.causal_conv1d_update
__all__ = ["causal_conv1d_fn", "causal_conv1d_update"]
