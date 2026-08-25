# SPDX-License-Identifier: Apache-2.0
"""Fused MoE router via the out-of-tree `router_select` custom TPC op (EXPERIMENTAL).

Enabled with VLLM_HPU_FUSED_ROUTER=1. Loads the custom op built in
experiments/fused_router:
  VLLM_HPU_FUSED_ROUTER_LIB  -> path to hpu_custom_router_select*.so

The kernel lib (librouter_select_fwd_gaudi2_kernels.so) is a COMBINED lib that
aggregates libtpc_kernels.so + the router kernel; GC_KERNEL_PATH must name it as
a SINGLE regular file (vllm's _lazy_init rejects colon lists), set before habana
init (see bench_tier2.py).

The op lib is loaded ONCE at import (not in the hot path) so that
apply_monolithic's call to router_select is a pure op call that torch.compile
captures into the graph (per-call _load() forces a dynamo graph break, which made
the op run eagerly -> 38k kernel instantiations -> slower).
"""
import torch

from vllm_gaudi import envs

_lib_loaded = False


def _load():
    global _lib_loaded
    if _lib_loaded:
        return
    lib = envs.VLLM_HPU_FUSED_ROUTER_LIB
    if not lib:
        raise RuntimeError("VLLM_HPU_FUSED_ROUTER_LIB must point at the router_select op .so")
    torch.ops.load_library(lib)
    _lib_loaded = True


def router_select(router_logits: torch.Tensor, top_k: int = 8):
    """router_logits [T,E] bf16 -> (topk_ids int32 [T,K], topk_weights bf16 [T,K])."""
    ids, weights = torch.ops.custom_op.router_select(router_logits.contiguous(), top_k)
    return ids, weights


def _router_select_op(router_logits: torch.Tensor, top_k: int = 8):
    """Thin, graph-capturable call used inside the traced apply_monolithic.

    No env reads / imports / lib-loading here (all done at module import), so
    torch.compile captures this as a single custom-op node.
    """
    ids, weights = torch.ops.custom_op.router_select(router_logits.contiguous(), top_k)
    return ids, weights


# Load once at import so the hot path is a pure op call (capturable).
_load()
