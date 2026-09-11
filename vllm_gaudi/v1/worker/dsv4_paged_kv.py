# SPDX-License-Identifier: Apache-2.0
"""DeepSeek V4 paged-KV helpers (R1 of the paged-KV reintegration).

DeepSeek V4 registers several KV caches per decoder layer (the compressed MLA
attention cache, the sliding-window cache, the compressor running state, and the
indexer cache). vLLM's ``bind_kv_cache`` groups caches by decoder-layer index and
refuses (``NotImplementedError``) when a single layer index owns more than one
cache on a non-CUDA/XPU/CPU platform. HPU hits that case, so it needs its own
binding that keeps every ``layer_name`` distinct.

Gated by ``VLLM_DSV4_PAGED_KV`` so the manual-buffer fallback stays the default
until the paged path passes its gates (R2-R8); the flag and the manual path are
removed together in R9.
"""

import os

from vllm.v1.worker.utils import extract_layer_index


def dsv4_paged_kv_enabled() -> bool:
    return os.environ.get("VLLM_DSV4_PAGED_KV", "0") == "1"


def hpu_bind_kv_cache(
    kv_caches: dict,
    forward_context: dict,
    runner_kv_caches: list,
    num_attn_module: int = 1,
) -> None:
    """``bind_kv_cache`` without the per-layer-index uniqueness assumption.

    Order ``runner_kv_caches`` by decoder-layer index (stable within a layer by
    name) to match the CUDA path's ordering, but bind every ``layer_name``
    independently so multiple caches per decoder layer are allowed.
    """
    assert len(runner_kv_caches) == 0

    def _sort_key(layer_name: str) -> tuple[int, str]:
        try:
            return (extract_layer_index(layer_name, num_attn_module), layer_name)
        except Exception:
            return (1 << 30, layer_name)

    for layer_name in sorted(kv_caches, key=_sort_key):
        runner_kv_caches.append(kv_caches[layer_name])

    for layer_name, kv_cache in kv_caches.items():
        forward_context[layer_name].bind_kv_cache(kv_cache)
