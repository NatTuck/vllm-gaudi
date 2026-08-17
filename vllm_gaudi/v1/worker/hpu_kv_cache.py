# SPDX-License-Identifier: Apache-2.0

"""HPU-aware KV cache binding.

Upstream :func:`vllm.v1.worker.utils.bind_kv_cache` raises
``NotImplementedError`` whenever two attention layers share a single
``extract_layer_index`` on a non-CUDA/XPU/CPU platform.  This is a
false-positive for HPU when speculative decoding is enabled: the MTP draft
model registers ``mtp.layers.N.self_attn`` layers into the same shared
``static_forward_context`` as the target model's ``model.layers.N.*`` layers,
so both resolve to layer index ``N``.

On HPU the flat ``runner_kv_caches`` list is *not* consumed by the model
forward (the runner pops ``kv_caches`` out and each attention layer reads the
cache bound to ``forward_context[layer_name]``), so a layer-index collision in
that list is harmless.  This module replicates the upstream two-loop binding
but tolerates collisions, which is the correct behavior for HPU.
"""

from collections import defaultdict
from typing import Any

from vllm.model_executor.models.utils import extract_layer_index


def bind_kv_cache(
    kv_caches: dict[str, Any],
    forward_context: dict[str, Any],
    runner_kv_caches: list[Any],
    num_attn_module: int = 1,
) -> None:
    """Bind KV caches to the ModelRunner and the forward context (HPU).

    Mirrors upstream ``bind_kv_cache`` but, instead of raising when multiple
    attention layers share a layer index, appends every colliding cache to the
    flat ``runner_kv_caches`` list (sorted by layer index).  The per-layer
    forward-context binding is unchanged and is what the HPU model actually
    consumes.
    """
    assert len(runner_kv_caches) == 0

    index2name = defaultdict(list)
    for layer_name in kv_caches:
        index2name[extract_layer_index(layer_name, num_attn_module)].append(layer_name)

    for layer_index in sorted(index2name.keys()):
        for layer_name in index2name[layer_index]:
            runner_kv_caches.append(kv_caches[layer_name])

    for layer_name, kv_cache in kv_caches.items():
        forward_context[layer_name].bind_kv_cache(kv_cache)
