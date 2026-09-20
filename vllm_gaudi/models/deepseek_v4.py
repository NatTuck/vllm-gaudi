# SPDX-License-Identifier: Apache-2.0
"""DeepSeek V4 on Gaudi (HPU).

The upstream deepseek_v4 package dispatches to ``vllm_gaudi.models.deepseek_v4``
on HPU (see the dispatch patch in upstream ``__init__.py``). This module reuses
the platform-agnostic top-level classes and MoE from the NVIDIA implementation
and provides HPU-specific attention + decoder/model that construct the exact
param tree the checkpoint expects (so ``load_weights`` works).

NOTE: forward is a work-in-progress. The dense-MLA attention / hyper-connection
(``hc``) math is still being ported; the parameter tree is complete so weights
load and can be snapshotted for fast iteration.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn

from vllm.models.deepseek_v4.attention import DeepseekV4Attention
from vllm.v1.attention.backend import MultipleOf as _MultipleOf
from vllm.v1.attention.backends.mla.indexer import DeepseekV4IndexerBackend as _NvIndexerBackend
from vllm.models.deepseek_v4.nvidia.model import (
    DeepseekV4DecoderLayer as _NvDeepseekV4DecoderLayer,
)
from vllm.models.deepseek_v4.nvidia.model import (
    DeepseekV4ForCausalLM as _NvDeepseekV4ForCausalLM,
)
from vllm.models.deepseek_v4.nvidia.model import (
    DeepseekV4MixtureOfExperts,
    DeepseekV4MoE,
    DeepseekV4Model as _NvDeepseekV4Model,
    _make_deepseek_v4_weights_mapper,
    _use_sequence_parallel,
)
from vllm.sequence import IntermediateTensors



_DIAG_DONE = {"v": True}  # diag disabled: the per-layer device-sync prints throttle the first eager forward

# Debug flags hoisted to module scope: reading them inside a compiled region with
# ``__import__("os").environ`` is a dynamo graph break (the debug bodies also host-sync),
# so they must be constants at trace time.
_DSV4_R6_DEBUG = os.environ.get("DSV4_R6_DEBUG") == "1"
_DSV4_R4B_NOSWA = os.environ.get("DSV4_R4B_NOSWA") == "1"
_HPU_DUMP_ON = os.environ.get("HPU_DUMP") == "1"

# D5d: number of queries per sparse compressed-attention chunk. Each query gets
# its OWN window+top-k keys, so the attention is optimal in FLOPs regardless of
# the chunk size; the chunk only trades kernel-call count against per-chunk
# memory (keys are [chunk, W + k, D]). 64 keeps the unrolled graph moderate.
_DSV4_SPARSE_CHUNK = int(os.environ.get("DSV4_SPARSE_CHUNK", "256"))

# D5d-b: indexer scoring element budget per query chunk. The full score tensor
# is [T, nhead, cap] fp32 (17 GB at 64k); the chunk size is `budget / (nhead*cap)`
# so each chunk stays ~256 MB fp32.
_DSV4_INDEXER_CHUNK_ELEMS = int(os.environ.get("DSV4_INDEXER_CHUNK_ELEMS", str(1 << 26)))

# The compressor is isolated from the layer graph by a *controlled graph break*
# (`torch._dynamo.graph_break()`), NOT by `torch.compiler.disable`. A disable runs
# the wrapper eagerly (HPU compute outside a graph, forbidden); a graph break
# partitions the layer region so the compressor body is traced into its own
# compiled segment (own workspace), with no tensor work at the boundary. The break
# is required because inlining the compressor into the layer graph trips
# aot_autograd's non-differentiable view-mutation assertion and grows the combined
# workspace ~8 GiB (OOM at the production KV util 0.8).

# 2026-09-19 (L0c): ``torch._dynamo.graph_break()`` is a hard error under
# ``fullgraph=True`` (it routes to ``unimplemented`` -> ``Unsupported``, and
# ``should_compile_partial_graph()`` is False when ``one_graph``), and inlining
# the compressor instead OOMs the layer region's arena (PT_DEVMEM OOM during the
# profiling forward, ~8 GiB). So the partition stays **on** and the engine runs
# with ``fullgraph=False`` until the partition has a fullgraph-compatible form.
# ``DSV4_COMPRESSOR_BREAK=0`` disables the break (inline; needs fullgraph=False
# and more HBM than is available today).
_COMPRESSOR_BREAK = os.environ.get("DSV4_COMPRESSOR_BREAK", "1") != "0"

# Temporary diagnostic (DSV4_SYNC_PROBE=1): count host-side synchronize calls and
# print the first few stacks, to find who blocks the host between graph launches.
if os.environ.get("DSV4_SYNC_PROBE") == "1":
    import atexit as _atexit
    import traceback as _tb

    _SP = {"sync": 0, "evsync": 0, "shown": 0}

    def _mk_probe(orig, tag):
        def _probe(*a, **k):
            _SP[tag] += 1
            if _SP["shown"] < 4:
                _SP["shown"] += 1
                print(f"[syncprobe] {tag} call #{_SP[tag]}\n" + "".join(_tb.format_stack()[-7:]),
                      flush=True)
            return orig(*a, **k)
        return _probe

    try:
        torch.hpu.synchronize = _mk_probe(torch.hpu.synchronize, "sync")
    except Exception as _e:  # noqa: BLE001
        print(f"[syncprobe] cannot patch torch.hpu.synchronize: {_e}", flush=True)
    try:
        torch.hpu.Event.synchronize = _mk_probe(torch.hpu.Event.synchronize, "evsync")
    except Exception as _e:  # noqa: BLE001
        print(f"[syncprobe] cannot patch Event.synchronize: {_e}", flush=True)

    def _report():
        print(f"[syncprobe] totals: torch.hpu.synchronize={_SP['sync']} "
              f"Event.synchronize={_SP['evsync']}", flush=True)

    _atexit.register(_report)


def _diag(name: str, t: torch.Tensor, *, layer: int | None = None) -> None:
    """Print once (first forward) a NaN/abs-mean/variance diagnostic for t."""
    if _DIAG_DONE["v"]:
        return
    tag = f"layer{layer}" if layer is not None else name
    numel = t.numel()
    if numel == 0:
        print(f"[diag] {name} {tag}: EMPTY", flush=True)
        return
    if not t.dtype.is_floating_point:
        tf = t.to(torch.float32)
    else:
        tf = t.float()
    tn = torch.isnan(tf)
    ti = torch.isinf(tf)
    n_nan = int(tn.sum())
    n_inf = int(ti.sum())
    a = tf[~(tn | ti)]
    am = float(a.abs().mean()) if a.numel() else float("nan")
    var = float(a.square().mean()) if a.numel() else float("nan")
    fin = torch.isfinite(tf)
    if fin.any():
        amax = float(tf[fin].abs().max())
    else:
        amax = float("nan")
    print(f"[diag] {name} {tag}: dtype={t.dtype} shape={tuple(t.shape)} nan={n_nan} inf={n_inf} "
          f"abs_mean={am:.5g} sq_mean={var:.5g} abs_max={amax:.5g}", flush=True)


def _rmsnorm(x: torch.Tensor, eps: float, weight: torch.Tensor | None = None) -> torch.Tensor:
    """Weighted/weight-free RMSNorm over the last dim.

    Matches the reference rounding order (vLLM ``ir.ops.rms_norm`` and
    transformers ``DeepseekV4RMSNorm``): normalize in fp32, round to the
    weight's dtype, *then* multiply by the weight. The previous form multiplied
    in fp32 and rounded once, which is a different (1-ulp-scale) result.
    """
    orig_dtype = x.dtype
    xf = x.float()
    out = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
    if weight is not None:
        out = out.to(weight.dtype) * weight
    return out.to(orig_dtype)


def _apply_rope(
    x: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    rope_head_dim: int,
) -> torch.Tensor:
    """GPT-J (interleaved-pair) RoPE applied to the LAST rope_head_dim dims.

    Rotates in fp32 with the fp32 cos/sin cache (index_select, not a bf16
    cast of the whole cache) so results match the reference
    ``apply_rotary_pos_emb`` bit-exactly and are graph-stable.
    """
    cache = cos_sin_cache.index_select(0, positions)  # [T, 2*half]
    half = rope_head_dim // 2
    cos, sin = cache.chunk(2, dim=-1)  # [T, half]
    x_pass = x[..., :-rope_head_dim]
    xr = x[..., -rope_head_dim:]
    shape = xr.shape
    xr2 = xr.reshape(*shape[:-1], half, 2)
    x0, x1 = xr2[..., 0], xr2[..., 1]
    dims = x.dim()
    c = cos.unsqueeze(1) if dims == 3 else cos
    s = sin.unsqueeze(1) if dims == 3 else sin
    r0 = x0.float() * c - x1.float() * s
    r1 = x0.float() * s + x1.float() * c
    rot = torch.stack([r0, r1], dim=-1).reshape(shape)
    return torch.cat([x_pass, rot.to(x.dtype)], dim=-1)


def _attn_with_sink(
    q: torch.Tensor,
    k: torch.Tensor,
    mask: torch.Tensor | None,
    sink: torch.Tensor,
    scaling: float,
) -> torch.Tensor:
    """Attention matching the reference: append a per-head sink logit column,
    softmax over [keys + 1 sink], drop the sink column, then mix the values.

    q: [T, H, D]; k (= v): [K, H, D]; mask: [T, K] bool (True = allow), or None.
    sink: [H] per-head logit.

    Reference/eager math (used to validate the FusedSDPA compiled path). einsum
    is fine here; the graph-compiled path uses _attn_with_sink_fsdpa.
    """
    sc = torch.einsum("thd,khd->thk", q.float(), k.float()) * scaling  # [T,H,K]
    if mask is not None:
        sc = sc.masked_fill(~mask.unsqueeze(1), float("-inf"))
    sink_v = sink.float().reshape(1, -1, 1).expand(q.shape[0], -1, 1)  # [T,H,1]
    combined = torch.cat([sc, sink_v], dim=-1)
    combined = combined - combined.max(dim=-1, keepdim=True).values
    probs = torch.softmax(combined, dim=-1)
    scores = probs[..., :-1]  # [T,H,K]
    out = torch.einsum("thk,khd->thd", scores, k.float())  # [T,H,D]
    return out.to(q.dtype)


def _attn_with_sink_fsdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    mask: torch.Tensor | None,
    sink: torch.Tensor,
    scaling: float,
) -> torch.Tensor:
    """Graph-compiled V4 attention via the HPU FusedSDPA kernel.

    The per-head attention sink is folded into the additive ``attn_mask`` as a
    zero-valued extra key column whose bias is the per-head sink logit. This
    computes ``softmax([q·k^T | sink]) · [v | 0]`` == ``_attn_with_sink``
    (validated cos 0.99998) but is graph-compilable on ``hpu_backend``
    (``torch.einsum`` is not). ``softmax_mode='fp32'`` is the only mode that
    compiles here.

    q [T,H,D]; k (= v) [K,H,D]; mask [T,K] bool (True=allow) or None (decode).
    Returns [T,H,D].
    """
    K = k.shape[0]
    zeros = torch.zeros(1, k.shape[1], k.shape[2], dtype=k.dtype, device=k.device)
    k_ext = torch.cat([k, zeros], dim=0)  # [K+1,H,D]
    v_ext = torch.cat([k, zeros], dim=0)
    T, H, D = q.shape
    bias = torch.zeros((1, H, T, K), dtype=torch.float32, device=q.device)
    if mask is not None:
        bias = torch.where(
            mask.unsqueeze(0).unsqueeze(0), bias, torch.full((), float("-inf"), device=q.device)
        )
    sinkcol = sink.float().reshape(1, H, 1, 1).expand(1, H, T, 1)
    bias = torch.cat([bias, sinkcol], dim=-1)  # [1,H,T,K+1]
    qh = q.unsqueeze(0).transpose(1, 2).contiguous()     # [1,H,T,D]
    kh = k_ext.unsqueeze(0).transpose(1, 2).contiguous()  # [1,H,K+1,D]
    vh = v_ext.unsqueeze(0).transpose(1, 2).contiguous()
    out = torch.ops.hpu.sdpa_recomp_fwd(
        qh, kh, vh, bias, 0.0, scaling, False, False, "fp32", None, "right", (-1, -1), None
    )
    out = out[0] if isinstance(out, tuple) else out
    return out.transpose(1, 2).squeeze(0).to(q.dtype)  # [T,H,D]


def _gather_paged_entries(
    page: torch.Tensor,
    bl: torch.Tensor,
    ci: torch.Tensor,
    sbs: int,
    hd: int,
) -> torch.Tensor:
    """Gather cache rows by compressed-entry index from a paged cache view.

    Entry ``ci`` lives in logical block ``ci // sbs`` at offset ``ci % sbs``;
    ``bl`` is the request's full logical->physical block table. ``page`` is the
    strided page view ``[nphys, sbs, hd]`` (``hd`` is the head dim for the
    compressed cache, or the token byte width for the fp8 indexer cache).
    Returns ``[N, hd]`` bf16.

    This is the D5a sparse-gather primitive: the union-dense compressed
    attention reads only the ``window + k`` selected entries per query instead
    of the whole ``cap`` cache. Reads via dim-0 ``index_select`` + dim-1
    ``gather`` (both HPU-supported) rather than advanced indexing on the
    strided page view, which would make the compiler materialise a full slab
    copy inside the model's large graph. The cache is fp8, so the selected rows
    (small) are cast to bf16 before the gather (HPU's gather has no fp8 kernel).
    ``ci`` is clamped into the table's range so masked/invalid entries read a
    real row; callers mask them via the attention bias.
    """
    ci = ci.clamp(min=0, max=bl.numel() * sbs - 1)
    logical = torch.div(ci, sbs, rounding_mode="floor")
    offs = ci % sbs
    phys = bl[logical].clamp(min=0)
    # Index the flat page ``[nphys*sbs, hd]`` at ``phys*sbs + off`` so only the
    # ``[N, hd]`` result is materialised. (Indexing the 3-D view and gathering
    # the offset would materialise ``[N, sbs, hd]`` — ~8.6 GB for N = T*k at a
    # 4k prefill — which OOMs the device.)
    flat = page.reshape(-1, hd)
    idx = phys * sbs + offs
    return flat.index_select(0, idx).to(torch.bfloat16)  # [N, hd]


def _dsv4_page(layer, sbs: int, hd: int) -> torch.Tensor:
    """Build the typed page view ``[nphys, sbs, hd]`` for a paged DSv4 cache.

    P0.2: the packed KV slab is one int8 tensor aliased by caches of several
    dtypes (bf16 SWA, fp8 compressed). Passing those typed views as graph inputs
    makes ``aot_autograd`` trip "non-differentiable view input mutations"
    (multiple view inputs of the same base with different dtypes). Each layer
    therefore stashes only the int8 base plus its byte offset / page size / dtype,
    and the typed view is built **inside** the graph here.
    """
    base = getattr(layer, "_dsv4_slab_i8", None)
    if base is None:
        # Legacy typed view (isolated gates build `_dsv4_slab2d` directly).
        slab2d = layer._dsv4_slab2d
        return slab2d[:, layer._dsv4_page_off:layer._dsv4_page_off +
                      layer._dsv4_page_elems].view(-1, sbs, hd)
    # Order matters: view the *contiguous* int8 base to the target dtype FIRST
    # (free — block_stride is a multiple of elem), then slice the page out of the
    # typed view. Slicing the int8 base first gives a strided int8 tensor whose
    # `.view(dtype)` forces the compiler to materialise the whole page column
    # (~8 GiB of workspace → OOM at the production util). The typed slice is
    # `[nblocks, pe]` with a contiguous last dim, so the final `.view(-1, sbs, hd)`
    # only splits that dim and is also free.
    dtype = layer._dsv4_page_dtype
    elem = torch.tensor([], dtype=dtype).element_size()
    pe = layer._dsv4_page_bytes // elem
    off = layer._dsv4_page_off_bytes // elem
    return base.view(dtype)[:, off:off + pe].view(-1, sbs, hd)


def _attn_with_sink_matmul(
    q: torch.Tensor,
    k: torch.Tensor,
    mask: torch.Tensor | None,
    sink: torch.Tensor,
    scaling: float,
) -> torch.Tensor:
    """FALLBACK (unused): fp32 ``torch.bmm`` V4 attention, no FusedSDPA.

    The forward now uses ``_attn_with_sink_fsdpa`` (fp32 QK^T + fp32 softmax,
    bf16 probs -> bf16 AV), which matches the reference kernel precision much
    better (Step 8 gate: 1.95% >2 bf16 ulp vs 6.28% for this fp32 path) and
    was verified to compile and run at T=1..512, K up to window+compressed cap
    with no segfault. Keep this as the strictly-more-precise fallback in case
    the FusedSDPA kernel regresses; it is intentionally more precise and a
    little slower, which does not contradict the current goals.

    Matches the FusedSDPA contract: the sink is folded into the last key column
    (zero-valued key with per-head sink logit bias).

    q [T,H,D]; k (= v) [K,H,D]; mask [T,K] bool (True=allow) or None.
    Returns [T,H,D].
    """
    K = k.shape[0]
    T, H, D = q.shape
    zeros = torch.zeros(1, H, D, dtype=k.dtype, device=k.device)
    k_ext = torch.cat([k, zeros], dim=0)  # [K+1, H, D]
    qf = q.float(); kf = k_ext.float()
    q_hd = qf.permute(1, 0, 2).contiguous()    # [H, T, D]
    k_hd = kf.permute(1, 0, 2).contiguous()    # [H, K+1, D]
    sc = torch.bmm(q_hd, k_hd.transpose(-1, -2)).permute(1, 0, 2)  # [T, H, K+1]
    sc = sc * scaling
    bias = torch.zeros(T, H, K + 1, dtype=torch.float32, device=q.device)
    if mask is not None:
        bias[:, :, :K].masked_fill_(~mask.unsqueeze(1), float("-inf"))
    bias[:, :, -1] = sink.float().reshape(1, -1)
    combined = sc + bias
    combined = combined - combined.max(dim=-1, keepdim=True).values
    probs = torch.softmax(combined, dim=-1)
    sp = probs.permute(1, 0, 2).contiguous()  # [H, T, K+1]
    kf_p = kf.permute(1, 0, 2).contiguous()   # [H, K+1, D]
    out = torch.bmm(sp, kf_p).permute(1, 0, 2)  # [T, H, D]
    return out.to(q.dtype)



def _apply_inv_rope(
    x: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    rope_head_dim: int,
) -> torch.Tensor:
    cache = cos_sin_cache.index_select(0, positions)
    half = rope_head_dim // 2
    cos, sin = cache.chunk(2, dim=-1)
    x_pass = x[..., :-rope_head_dim]
    xr = x[..., -rope_head_dim:]
    shape = xr.shape
    xr2 = xr.reshape(*shape[:-1], half, 2)
    x0, x1 = xr2[..., 0], xr2[..., 1]
    dims = x.dim()
    c = cos.unsqueeze(1) if dims == 3 else cos
    s = sin.unsqueeze(1) if dims == 3 else sin
    r0 = x0.float() * c + x1.float() * s
    r1 = -x0.float() * s + x1.float() * c
    rot = torch.stack([r0, r1], dim=-1).reshape(shape)
    return torch.cat([x_pass, rot.to(x.dtype)], dim=-1)


def _mhc_pre_broadcast(
    x: torch.Tensor,
    fn_broadcast: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult: float,
    sinkhorn_iters: int,
    hc_mult: int,
    norm_weight: torch.Tensor | None,
    norm_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    T, H = x.shape
    x_float = x.float()
    mixes = x_float @ fn_broadcast.t()  # [T, hc_mult3]
    sqrsum = x_float.square().sum(-1, keepdim=True)
    mixes = mixes * torch.rsqrt(sqrsum / H + rms_eps)
    pre_logits = mixes[:, :hc_mult] * hc_scale[0] + hc_base[:hc_mult]
    pre_mix = torch.sigmoid(pre_logits) + hc_pre_eps  # [T, hc_mult]
    post_logits = mixes[:, hc_mult : 2 * hc_mult] * hc_scale[1] + hc_base[hc_mult : 2 * hc_mult]
    post_mix = torch.sigmoid(post_logits) * hc_post_mult  # [T, hc_mult]
    comb_logits = (
        mixes[:, 2 * hc_mult :].view(T, hc_mult, hc_mult) * hc_scale[2]
        + hc_base[2 * hc_mult :].view(1, hc_mult, hc_mult)
    )
    comb_mix = torch.softmax(comb_logits, dim=-1) + hc_sinkhorn_eps
    comb_mix = comb_mix / (comb_mix.sum(-2, keepdim=True) + hc_sinkhorn_eps)
    for _ in range(sinkhorn_iters - 1):
        comb_mix = comb_mix / (comb_mix.sum(-1, keepdim=True) + hc_sinkhorn_eps)
        comb_mix = comb_mix / (comb_mix.sum(-2, keepdim=True) + hc_sinkhorn_eps)
    residual_out = (pre_mix.unsqueeze(-1) * x.unsqueeze(1)).to(x.dtype)  # [T, hc_mult, H]
    collapsed_pre = residual_out.sum(1)  # [T, H] before the input RMSNorm
    layer_input = _rmsnorm(collapsed_pre, norm_eps, norm_weight)  # [T, H]
    import os as _hcc
    if _hcc.environ.get("HPU_DUMP") == "1" and _hcc.environ.get("HPU_DUMP_MHC") == "1":
        _HPU_CAP.setdefault("mhc_pre", {})[0] = pre_mix.detach().float().clone()
        _HPU_CAP.setdefault("mhc_collapsed", {})[0] = collapsed_pre.detach().float().clone()
    return residual_out, post_mix.unsqueeze(-1), comb_mix, layer_input


def _mhc_pre_broadcast_compilable(
    x: torch.Tensor,
    fn_broadcast: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult: float,
    sinkhorn_iters: int,
    hc_mult: int,
    norm_weight: torch.Tensor | None,
    norm_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``_mhc_pre_broadcast`` without HPU_DUMP — safe for ``torch.compile``."""
    T, H = x.shape
    x_float = x.float()
    mixes = x_float @ fn_broadcast.t()
    sqrsum = x_float.square().sum(-1, keepdim=True)
    mixes = mixes * torch.rsqrt(sqrsum / H + rms_eps)
    pre_logits = mixes[:, :hc_mult] * hc_scale[0] + hc_base[:hc_mult]
    pre_mix = torch.sigmoid(pre_logits) + hc_pre_eps
    post_logits = mixes[:, hc_mult : 2 * hc_mult] * hc_scale[1] + hc_base[hc_mult : 2 * hc_mult]
    post_mix = torch.sigmoid(post_logits) * hc_post_mult
    comb_logits = (
        mixes[:, 2 * hc_mult :].view(T, hc_mult, hc_mult) * hc_scale[2]
        + hc_base[2 * hc_mult :].view(1, hc_mult, hc_mult)
    )
    comb_mix = torch.softmax(comb_logits, dim=-1) + hc_sinkhorn_eps
    comb_mix = comb_mix / (comb_mix.sum(-2, keepdim=True) + hc_sinkhorn_eps)
    for _ in range(sinkhorn_iters - 1):
        comb_mix = comb_mix / (comb_mix.sum(-1, keepdim=True) + hc_sinkhorn_eps)
        comb_mix = comb_mix / (comb_mix.sum(-2, keepdim=True) + hc_sinkhorn_eps)
    residual_out = (pre_mix.unsqueeze(-1) * x.unsqueeze(1)).to(x.dtype)
    collapsed_pre = residual_out.sum(1)
    layer_input = _rmsnorm(collapsed_pre, norm_eps, norm_weight)
    return residual_out, post_mix.unsqueeze(-1), comb_mix, layer_input


def _mhc_fused_post_pre(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult: float,
    sinkhorn_iters: int,
    hc_mult: int,
    norm_weight: torch.Tensor | None,
    norm_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mixed_residual = torch.einsum("...ij,...ih->...jh", comb_res_mix.float(), residual.float())
    post_term = post_layer_mix.float() * x.unsqueeze(-2).float()
    residual_cur = (mixed_residual + post_term).to(residual.dtype)  # [..., hc_mult, H]
    outer = residual_cur.shape[:-2]
    r_flat = residual_cur.reshape(-1, hc_mult, residual_cur.shape[-1])
    T = r_flat.shape[0]
    H = r_flat.shape[-1]
    xf = r_flat.view(T, hc_mult * H).float()
    mixes = xf @ fn.t()
    sqrsum = xf.square().sum(-1, keepdim=True)
    mixes = mixes * torch.rsqrt(sqrsum / (hc_mult * H) + rms_eps)
    pre_logits = mixes[:, :hc_mult] * hc_scale[0] + hc_base[:hc_mult]
    pre_mix = torch.sigmoid(pre_logits) + hc_pre_eps
    post_logits = mixes[:, hc_mult : 2 * hc_mult] * hc_scale[1] + hc_base[hc_mult : 2 * hc_mult]
    post_mix_cur = torch.sigmoid(post_logits) * hc_post_mult
    comb_logits = (
        mixes[:, 2 * hc_mult :].view(T, hc_mult, hc_mult) * hc_scale[2]
        + hc_base[2 * hc_mult :].view(1, hc_mult, hc_mult)
    )
    comb_mix_cur = torch.softmax(comb_logits, dim=-1) + hc_sinkhorn_eps
    comb_mix_cur = comb_mix_cur / (comb_mix_cur.sum(-2, keepdim=True) + hc_sinkhorn_eps)
    for _ in range(sinkhorn_iters - 1):
        comb_mix_cur = comb_mix_cur / (comb_mix_cur.sum(-1, keepdim=True) + hc_sinkhorn_eps)
        comb_mix_cur = comb_mix_cur / (comb_mix_cur.sum(-2, keepdim=True) + hc_sinkhorn_eps)
    layer_input_cur = torch.sum(pre_mix.unsqueeze(-1) * r_flat.float(), dim=1).to(x.dtype)
    if norm_weight is not None:
        layer_input_cur = _rmsnorm(layer_input_cur, norm_eps, norm_weight)
    return (
        residual_cur,
        post_mix_cur.view(*outer, hc_mult, 1),
        comb_mix_cur.view(*outer, hc_mult, hc_mult),
        layer_input_cur.view(*outer, H),
    )


def _mhc_fused_post_pre_compilable(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult: float,
    sinkhorn_iters: int,
    hc_mult: int,
    norm_weight: torch.Tensor | None,
    norm_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``_mhc_fused_post_pre`` without HPU_DUMP — safe for ``torch.compile``."""
    mixed_residual = torch.matmul(
        comb_res_mix.float().transpose(-1, -2), residual.float()
    )
    post_term = post_layer_mix.float() * x.unsqueeze(-2).float()
    residual_cur = (mixed_residual + post_term).to(residual.dtype)
    outer = residual_cur.shape[:-2]
    r_flat = residual_cur.reshape(-1, hc_mult, residual_cur.shape[-1])
    T = r_flat.shape[0]
    H = r_flat.shape[-1]
    xf = r_flat.view(T, hc_mult * H).float()
    mixes = xf @ fn.t()
    sqrsum = xf.square().sum(-1, keepdim=True)
    mixes = mixes * torch.rsqrt(sqrsum / (hc_mult * H) + rms_eps)
    pre_logits = mixes[:, :hc_mult] * hc_scale[0] + hc_base[:hc_mult]
    pre_mix = torch.sigmoid(pre_logits) + hc_pre_eps
    post_logits = mixes[:, hc_mult : 2 * hc_mult] * hc_scale[1] + hc_base[hc_mult : 2 * hc_mult]
    post_mix_cur = torch.sigmoid(post_logits) * hc_post_mult
    comb_logits = (
        mixes[:, 2 * hc_mult :].view(T, hc_mult, hc_mult) * hc_scale[2]
        + hc_base[2 * hc_mult :].view(1, hc_mult, hc_mult)
    )
    comb_mix_cur = torch.softmax(comb_logits, dim=-1) + hc_sinkhorn_eps
    comb_mix_cur = comb_mix_cur / (comb_mix_cur.sum(-2, keepdim=True) + hc_sinkhorn_eps)
    for _ in range(sinkhorn_iters - 1):
        comb_mix_cur = comb_mix_cur / (comb_mix_cur.sum(-1, keepdim=True) + hc_sinkhorn_eps)
        comb_mix_cur = comb_mix_cur / (comb_mix_cur.sum(-2, keepdim=True) + hc_sinkhorn_eps)
    layer_input_cur = torch.sum(pre_mix.unsqueeze(-1) * r_flat.float(), dim=1).to(x.dtype)
    if norm_weight is not None:
        layer_input_cur = _rmsnorm(layer_input_cur, norm_eps, norm_weight)
    return (
        residual_cur,
        post_mix_cur.view(*outer, hc_mult, 1),
        comb_mix_cur.view(*outer, hc_mult, hc_mult),
        layer_input_cur.view(*outer, H),
    )


def _mhc_post_compilable(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    """mHC post block, einsum-free.

    Equivalent to ``mhc_post_torch`` but uses ``torch.matmul`` (the upstream
    ``torch.einsum`` is not graph-compilable on ``hpu_backend``).
    """
    mixed_residual = torch.matmul(
        comb_res_mix.float().transpose(-1, -2), residual.float()
    )
    post_term = post_layer_mix.float() * x.unsqueeze(-2).float()
    return (mixed_residual + post_term).to(residual.dtype)


def _hc_head(
    hs_flat: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
) -> torch.Tensor:
    T, hc_mult, H = hs_flat.shape
    x_flat = hs_flat.reshape(T, hc_mult * H)
    x_normed = _rmsnorm(x_flat, rms_eps)
    mixes = x_normed.float() @ fn.t()  # [T, hc_mult]
    pre = torch.sigmoid(mixes * hc_scale + hc_base) + hc_eps  # [T, hc_mult]
    out = torch.matmul(pre.unsqueeze(-2), hs_flat.float()).squeeze(-2).to(hs_flat.dtype)
    return out


def _model_head(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    post_mix: torch.Tensor,
    res_mix: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    norm_weight: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
) -> torch.Tensor:
    """Post-layer head: mHC post + hc_head + final RMSNorm.

    Factored out so it can be `torch.compile`d as one graph (the model forward
    itself stays eager glue for the dynamic layer loop).
    """
    hidden_states = _mhc_post_compilable(hidden_states, residual, post_mix, res_mix)
    hidden_states = _hc_head(hidden_states, fn, scale, base, rms_eps, hc_eps)
    return _rmsnorm(hidden_states, rms_eps, norm_weight)




class HPUDeepseekV4IndexerBackend(_NvIndexerBackend):
    """HPU indexer-cache backend.

    The HPU indexer runs as compiled Python and is block-size agnostic, so
    accept any kernel block size. The upstream CUDA backend pins 256, which
    conflicts with the HPU MLA kernel's 128 when the indexer cache and the
    compressed MLA cache are merged into one packed KV cache group
    (``select_common_block_size`` then finds no common size).
    """

    @staticmethod
    def get_supported_kernel_block_sizes():
        return [_MultipleOf(1)]


class DeepseekV4HPUAttention(DeepseekV4Attention):
    """HPU DeepSeek V4 attention.

    The shared NVIDIA base provides the (bf16/fp8) q/kv/out projections and the
    compressor/indexer parameter tree (so checkpoint weights load). This HPU
    subclass re-implements the attention forward to faithfully replicate the
    reference sparse attention: it uses the *public* transformers compressor /
    indexer modules (fed with the loaded weights), ports the private
    buffer/compressed/overlap cache state, and runs the core with
    ``_attn_with_sink`` (a verified port of ``eager_attention_forward``).
    """

    from vllm_gaudi.attention.backends.hpu_attn import HPUMLAAttentionBackend

    backend_cls = HPUMLAAttentionBackend
    use_fp8_ds_mla_layout = False

    def __init__(self, *args, **kwargs) -> None:
        # The shared base and DeepseekV4Indexer allocate torch.cuda.Event in
        # __init__; HPU has no cuda streams, so redirect to torch.hpu.Event.
        _orig_event = torch.cuda.Event
        torch.cuda.Event = torch.hpu.Event  # type: ignore[assignment, misc]

        # The base __init__ calls _resolve_dsv4_kv_cache_dtype; with
        # use_fp8_ds_mla_layout=False it takes the bf16 branch and returns
        # ("auto", bf16) — no hack needed.
        super().__init__(*args, **kwargs)
        torch.cuda.Event = _orig_event  # type: ignore[assignment, misc]
        # The checkpoint stores a full-head ``attn_sink`` ([n_heads]) that is
        # replicated across TP ranks (CUDA pads n_local_heads up to the global
        # head count so the sink param matches). Match that here so weight
        # loading needs no TP head-sharding. The dense fallback also pads Q to
        # the global head count.
        self.padded_heads = self.n_heads
        self.attn_sink = nn.Parameter(
            torch.full((self.n_heads,), -float("inf"), dtype=torch.float32),
            requires_grad=False,
        )
        _ws = getattr(self, "window_size", 128)
        print(f"[HPU_ATTN] window_size from base={_ws}, forcing to 128", flush=True)
        self.window_size = 128

        # Compressor kernel state — flat tensor attributes, no dict
        vllm_config = kwargs.get("vllm_config", None) or (args[0] if args else None)
        self._hf_config = getattr(vllm_config, "model_config", None).hf_config
        self._compress_cache = None
        self._comp_coff = 2 if self.compress_ratio == 4 else 1
        self._cap = 0
        # D5d attention kind (static per layer): 0 window-only, 1 CSA (per-query
        # top-k gather), 2 HCA (shared dense compressed entries).
        if getattr(self, "compressor", None) is None or self.compress_ratio <= 1:
            self._attn_kind = 0
        elif self.compress_ratio == 4:
            self._attn_kind = 1
        else:
            self._attn_kind = 2
        # D3/D5b: size the compressed-entry space and the state ring from the
        # real config instead of a hardcoded 4096. `cap` is the number of
        # compressed entries the layer can hold (entries are produced every
        # `compress_ratio` tokens), so it must cover `max_model_len`; the state
        # ring must hold a whole prefill chunk plus the compressor window.
        self._max_model_len = int(
            getattr(getattr(vllm_config, "model_config", None), "max_model_len", 0) or 0)
        self._max_num_batched_tokens = int(
            getattr(getattr(vllm_config, "scheduler_config", None),
                    "max_num_batched_tokens", 0) or 0)
        # Completed-window entry counters (R9: the paged caches hold the data).
        self._c_n = torch.tensor(0, dtype=torch.int64)
        self._i_n = torch.tensor(0, dtype=torch.int64)

        # The indexer k_cache must advertise an HPU-friendly supported kernel
        # block size (see HPUDeepseekV4IndexerBackend).
        if getattr(self, "indexer", None) is not None:
            self.indexer.k_cache.get_attn_backend = lambda: HPUDeepseekV4IndexerBackend

    @classmethod
    def get_padded_num_q_heads(cls, num_heads: int) -> int:
        return num_heads

    def get_kv_cache_spec(self, vllm_config):
        # R1: upstream-faithful spec — the compressed MLA cache for
        # compress_ratio > 1, None for the SWA-only layers (their cache is the
        # separately registered ``swa_cache`` module).
        return super().get_kv_cache_spec(vllm_config)

        # The dense-MLA fallback computes attention over the full context in
        # torch and does not read the paged KV cache, but the HPU worker still
        # needs a normal, small, framework-compatible cache. Return a plain
        # bf16 FullAttentionSpec (num_kv_heads=1, head_dim) instead of the
        # memory-heavy fp8_ds_mla MLA spec, so allocation succeeds and the KV
        # coordinator / bind_kv_cache work.
        from vllm.v1.kv_cache_interface import FullAttentionSpec

        return FullAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=torch.bfloat16,
        )

    def _fused_qnorm_rope_kv_insert(self, q, kv, positions, attn_metadata):
        return q

    @staticmethod
    def _dequant_fp8_weight(lin) -> torch.Tensor:
        """Dequantize an fp8 ReplicatedLinear/ColumnParallelLinear weight to
        bf16, matching how the linear's own forward dequantizes it. Handles
        block-scaled fp8 (scale is [out/blk, in/blk]), per-output-row scale,
        and per-tensor scale."""
        w = lin.weight.detach()
        if w.dtype.is_floating_point:
            return w
        wf = w.float()
        s = getattr(lin, "weight_scale", None)
        if s is None or s.numel() == 0:
            s = getattr(lin, "weight_scale_inv", None)
        s = s.float()
        if s.dim() == 2 and s.shape[0] != wf.shape[0] and s.shape[1] != wf.shape[1]:
            bs = getattr(getattr(lin, "quant_config", None), "weight_block_size", None)
            if bs is None:
                bs = (128, 128)
            from vllm_gaudi.extension.ops import dequant_block_fp8_weight_naive
            return dequant_block_fp8_weight_naive(
                lin.weight.detach(), s.detach(), bs, torch.float32
            ).to(torch.bfloat16)
        if s.numel() == wf.shape[0] or (s.dim() == 2 and s.shape[0] == wf.shape[0]):
            return (wf * s.reshape(-1, 1)).to(torch.bfloat16)
        if s.numel() == 1:
            return (wf * s).to(torch.bfloat16)
        return (wf * s.view(1, -1)).to(torch.bfloat16)

    def _copy_fused_compressor(self, src, dst, head_dim: int) -> None:
        """Copy the vLLM DeepseekCompressor (fused wkv|wgate + ape + norm) into a
        transformers public compressor (kv_proj / gate_proj / position_bias /
        kv_norm)."""
        fused = src.fused_wkv_wgate.weight.data
        kv_out = dst.kv_proj.out_features
        dst.kv_proj.weight.data.copy_(fused[:kv_out])
        dst.gate_proj.weight.data.copy_(fused[kv_out:])
        dst.position_bias.data.copy_(src.ape.data)
        dst.kv_norm.weight.data.copy_(src.norm.weight.data)



    @staticmethod
    def _build_compress_rope_cache(hf_config, max_model_len, rate, dev):
        """Build the yarn compress RoPE cos/sin cache (shared across layers).

        Constructs a transformers ``AutoConfig``/``DeepseekV4CSACompressor`` to
        read the compress rope parameters. Must be called OUTSIDE any compiled
        region; the result is identical for every compressor layer (same
        ``compress_rope_theta`` / partial-rotary factor), so it is built once
        and shared instead of duplicated per layer (at 1M that is ~268 MB each).
        """
        from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
            DeepseekV4CSACompressor,
        )
        from transformers import AutoConfig
        ckpt = _v4_resolve_ckpt_dir(
            getattr(hf_config, "_name_or_path", "") or ""
        )
        pub_cfg = AutoConfig.from_pretrained(ckpt)
        pub = DeepseekV4CSACompressor(pub_cfg)
        inv = pub.rotary_emb.compress_inv_freq.float().to(dev)
        scale = pub.rotary_emb.compress_attention_scaling
        # Entries are RoPE'd at position ``ci * rate``; ci < cap = max_model_len // rate.
        p = torch.arange(max_model_len + rate + 1, dtype=torch.float32, device=dev)
        freqs = p.unsqueeze(1) * inv.unsqueeze(0)
        cos = freqs.cos() * scale
        sin = freqs.sin() * scale
        return torch.cat([cos, sin], dim=-1).to(torch.bfloat16)

    def _make_compress_cache(self, shared_rope_cache=None):
        """Build the per-layer compressor buffers (D3/D5b).

        Must be called OUTSIDE any compiled region. Invoked once per layer from
        ``DeepseekV4HPUModel.load_weights`` after the weights are loaded.

        ``cap`` (the number of compressed entries) is ``max_model_len //
        compress_ratio`` — the hardcoded 4096 truncated every context beyond
        16k (CSA) / 512k (HCA). The state ring is sized to hold a whole prefill
        chunk (``max_num_batched_tokens``) plus the compressor window, since
        the vectorized state path writes the chunk then reads a window back.
        The compress RoPE cache is built once and passed in (``shared_rope_cache``)
        so it is not duplicated across the 43 layers.
        """
        if self.compressor is None or self.compress_ratio <= 1:
            return
        if self._compress_cache is not None:
            return
        rate = self.compress_ratio
        D = self.head_dim
        dev = self.compressor.ape.device
        max_model_len = self._max_model_len or 4096
        cap = max(1, max_model_len // rate)
        self._cap = cap
        if shared_rope_cache is not None:
            self._compress_cache = shared_rope_cache
        else:
            self._compress_cache = self._build_compress_rope_cache(
                self._hf_config, max_model_len, rate, dev)

        # State ring: the vectorized path writes the whole chunk (T positions)
        # and then reads a `win`-sized window per token, so the ring must hold
        # T + win - 1 slots. Address by absolute position modulo `ring`; one
        # extra row (index `ring`) is a scratch slot for padding rows.
        coff = self._comp_coff
        win = 2 * rate if coff == 2 else rate
        tokens = self._max_num_batched_tokens or max(max_model_len, 4096)
        ring = tokens + win
        state_dim = 2 * coff * D
        self._dsv4_comp_state = torch.zeros(ring + 1, state_dim, dtype=torch.bfloat16,
                                            device=dev)
        # Dedicated fp32 scale buffer for the indexer k_cache. HPU cannot mutate
        # two dtype-views (fp8 keys + fp32 scales) of the same cache in one
        # graph, so the scales live here, indexed by compressed-entry index.
        # One extra row (index `cap`) is a scratch slot for invalid/padding
        # entries so they never overwrite a live entry's scale.
        if self.indexer is not None:
            self._dsv4_indexer_scales = torch.zeros(cap + 1, dtype=torch.float32,
                                                    device=dev)
            # The indexer has its own (narrower) state width; give it a separate
            # state buffer (absolute-position addressed, +1 scratch row) so it
            # never shares the main compressor's buffer.
            idx_state_dim = 2 * coff * self.indexer.head_dim
            self._dsv4_indexer_state = torch.zeros(ring + 1, idx_state_dim,
                                                   dtype=torch.bfloat16, device=dev)

    def _comp_rmsnorm(self, x, w):
        # Match the reference DeepseekV4RMSNorm: normalize in fp32, round to the
        # input dtype, then apply the weight (do not fuse the weight multiply in
        # fp32 — the reference rounds before it).
        xf = x.float()
        out = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + self.eps)
        return w * out.to(x.dtype)

    @staticmethod
    def _comp_weight(lin, in_features: int) -> torch.Tensor:
        """Linear weight as ``[out_features, in_features]``.

        The HPU compile stack can lazily store a linear's weight transposed
        (observed mid-first-forward on the compressor linears); normalize to the
        ``[out, in]`` layout the forward math (``x @ W.t()``) assumes.
        """
        w = lin.weight.detach()
        if w.dim() == 2 and w.shape[-1] != in_features:
            w = w.t()
        return w

    def _dsv4_aux_entry(self, layer):
        """Fetch the paged-KV metadata entry for an aux cache layer.

        Prefers the layer prefix (the standard machinery key: metadata keyed by
        layer name), falling back to the KV-cache group id.
        """
        from vllm.forward_context import get_forward_context

        md = get_forward_context().attn_metadata
        aux = getattr(md, "dsv4_aux_meta", None)
        if not aux:
            return None
        prefix = getattr(layer, "prefix", None)
        if prefix is not None and prefix in aux:
            return aux[prefix]
        gid = getattr(layer, "_kv_cache_gid", None)
        if gid is not None and gid in aux:
            return aux[gid]
        return None

    def _comp_state_paged_vectorized(self, kv, gate, positions, comp_module=None,
                                      head_dim=None, state_buf=None):
        """R4 vectorized: write per-token state to a bf16 ring buffer and
        recompute compressed entries (no per-token .item() loops, no strided
        fp32 page views — HPU-graph-safe).

        Each token writes its ``(kv, gate)`` state to
        ``state_buf[position % win]`` (ring buffer, size ``win``). Then for
        each token a window of the last ``win`` state rows is read back and
        passed through the compression math.

        ``comp_module`` selects main/indexer compressor; ``head_dim`` overrides
        the projected head width (indexer: 128). ``state_buf`` defaults to
        a per-layer buffer allocated in ``_make_compress_cache``.

        Returns ``(comp [nw, D], nw, sel_indices)``.
        """
        rate = self.compress_ratio
        comp_module = comp_module if comp_module is not None else self.compressor
        D = head_dim if head_dim is not None else self.head_dim
        coff = self._comp_coff
        dev = kv.device
        state_dim = 2 * coff * D
        win = 2 * rate if coff == 2 else rate

        # State buffer allocated in _make_compress_cache (ring + 1 rows).
        buf = state_buf if state_buf is not None else self._dsv4_comp_state
        # Address by ABSOLUTE position modulo `ring` (= cap), not modulo `win`:
        # each position maps to a distinct slot for any T <= cap, so the window
        # read is correct (the old `% win` form overwrote states within a
        # prefill chunk longer than `win`). `ring` is also the scratch slot.
        ring = buf.shape[0] - 1

        # Write to the state buffer indexed by absolute position. Padding rows
        # (position < 0) go to the scratch slot so they never clobber a real
        # token's state.
        state = torch.cat([kv, gate], dim=-1).to(torch.bfloat16)  # [T, state_dim] bf16
        pos = positions.reshape(-1).to(torch.int64)
        valid_pos = pos >= 0
        ridx = torch.where(valid_pos, pos % ring,
                           torch.full_like(pos, ring))  # scratch slot for padding
        buf[ridx] = state

        # Read window: for each token, gather the last 'win' positions
        T = pos.numel()
        if T == 0:
            return kv.new_zeros((0, D)), torch.tensor(0, dtype=torch.int64, device=dev), None
        rel = torch.arange(-win + 1, 1, device=dev, dtype=torch.int64)
        win_pos = pos.unsqueeze(1) + rel.unsqueeze(0)  # [T, win]
        valid = win_pos >= 0
        safe_pos = win_pos.clamp(min=0)
        ridx_win = safe_pos % ring
        states = buf[ridx_win]  # [T, win, state_dim] bf16

        # APE by each window slot's absolute position-within-rate (the reference
        # adds position_bias[j] to the j-th token of the window, not one bias
        # broadcast to the whole window).
        ape = comp_module.ape  # [rate, coff*D] fp32
        ape_idx = (safe_pos % rate).reshape(-1)          # [T*win]
        ape_win = ape[ape_idx].view(T, win, coff * D)    # [T, win, coff*D]

        if coff == 2:
            # CSA: stream 0 in [0:rate], stream 1 in [rate:2*rate]
            kv_s0 = states[:, :rate, :D].float()
            gate_s0 = states[:, :rate, 2*D:3*D].float() + ape_win[:, :rate, :D]
            kv_s1 = states[:, rate:, D:2*D].float()
            gate_s1 = states[:, rate:, 3*D:4*D].float() + ape_win[:, rate:, D:2*D]
            nk = torch.cat([kv_s0, kv_s1], dim=1)
            ng = torch.cat([gate_s0, gate_s1], dim=1)
        else:
            # HCA: single stream
            nk = states[:, :, :D].float()
            ng = states[:, :, D:2*D].float() + ape_win[:, :, :D]
        ng = ng.masked_fill(~valid.unsqueeze(-1), float("-inf"))
        # Padding rows have no valid window -> all -inf -> softmax NaN. Zero them
        # (their outputs are discarded) so the graph stays NaN-free.
        all_invalid = ~valid.any(dim=1)  # [T]
        ng = torch.where(all_invalid.view(T, 1, 1), torch.zeros_like(ng), ng)
        soft = torch.softmax(ng.float(), dim=1)
        # Match the reference's bf16 aggregation: round the products to bf16,
        # accumulate in fp32, then round the result (torch.sum on bf16 uses fp32
        # accumulation). Keeping the accumulation in fp32 keeps the compiled
        # graph shape-independent (a plain bf16 .sum is not).
        comp_all = (nk.to(torch.bfloat16) * soft.to(torch.bfloat16)).float().sum(dim=1).to(
            torch.bfloat16)  # [T, D]
        comp_all = self._comp_rmsnorm(comp_all, comp_module.norm.weight)
        idx = torch.div(pos + 1, rate, rounding_mode="floor") - 1
        comp_all = _apply_rope(comp_all, (idx * rate).clamp(min=0),
                               self._compress_cache, self.rope_head_dim)

        # Select window-end-aligned entries by absolute position. Real positions
        # are contiguous and leading, padding rows (pos < 0) are trailing, so
        # nreal = #(pos >= 0) and the first real position is p0 = max - nreal + 1.
        # The aligned rows are s, s+rate, ... where s = (-(p0+1)) % rate is a ROW
        # index. A FIXED-size selection (max_sel rows) keeps every shape static
        # (no bool-mask indexing / data-dependent shapes); invalid rows (>= nreal)
        # are clamped to a real row only for the gather, while the returned `sel`
        # stays unclamped so the writers redirect them to pad-block slots.
        nreal = valid_pos.sum()
        p0 = pos.max() - nreal + 1
        s = (-(p0 + 1)) % rate
        max_sel = (T + rate - 1) // rate
        sel = s + torch.arange(max_sel, device=dev, dtype=torch.int64) * rate
        nw = (sel < nreal).sum().to(torch.int64)
        sel_clamped = sel.clamp(max=(nreal - 1).clamp(min=0))
        comp = comp_all.index_select(0, sel_clamped)  # [max_sel, D], static
        return comp, nw, sel

    def _write_paged_compressed(self, comp, sel, positions):
        """R5: write the compressor's compressed entries into the paged
        compressed-MLA cache (the attention layer's own cache, fp8).

        Uses ``page[blocks, offs] = comp`` (strided fp8 advanced-index scatter,
        which compiles fine on HPU — same pattern as ``_write_paged_swa_cache``).
        ``sel`` indexes the rows that completed a window.
        """
        if comp.shape[0] == 0:
            return
        from vllm.forward_context import get_forward_context
        ctx = get_forward_context()
        if ctx is None:
            return
        md = ctx.attn_metadata
        aux = getattr(md, "dsv4_aux_meta", None)
        if not aux:
            return
        entry = aux.get(self.prefix)
        if entry is None:
            return
        slab2d = getattr(self, "_dsv4_slab2d", None)
        if slab2d is None:
            return
        slots = entry["slot_mapping"]
        if sel is not None:
            # `sel` is the fixed-size aligned-row selection from
            # `_comp_state_paged_vectorized`. Entries with row >= nreal are
            # invalid (no window completed); clamp them to the last row of the
            # padded slot_mapping, which is a pad-block slot, so their writes
            # land harmlessly outside the request's real blocks.
            sel_slot = sel.clamp(max=slots.shape[0] - 1)
            slots = slots[sel_slot]
        sbs = entry["storage_block_size"]
        hd = self.head_dim
        page = _dsv4_page(self, sbs, hd)
        blocks = torch.div(slots, sbs, rounding_mode="floor")
        offs = slots % sbs
        # Strided fp8 advanced-index scatter compiles fine (SWA cache gate).
        page[blocks, offs] = comp.to(page.dtype)
        if _DSV4_R6_DEBUG and comp.shape[0] > 0:
            _off_elems = self._dsv4_page_off
            _b0 = int(blocks[0].item())
            _o0 = int(offs[0].item())
            _probe = slab2d[_b0, _off_elems + _o0 * hd:_off_elems + _o0 * hd + 3].to(
                torch.bfloat16).float().tolist()
            print(f"[R5-write] layer={getattr(self, '_layer_idx', '?')} n={comp.shape[0]} "
                  f"block={_b0} off={_o0} "
                  f"val0={comp[0, :3].float().tolist()} probe={_probe} "
                  f"page_off={self._dsv4_page_off} "
                  f"page_elems={self._dsv4_page_elems} slab_cols={slab2d.shape[1]}", flush=True)

    def _read_paged_compressed(self, cap):
        """R6: gather the compressed entries ``[cap, head_dim]`` from the paged
        compressed-MLA cache (written by R5) by global block/slot id.

        Entry ``ci`` lives at physical slot
        ``bt0[ci // storage_block_size] * sbs + ci % storage_block_size``, so
        block ``b`` holds entries ``[b*sbs, (b+1)*sbs)`` in order: selecting the
        request's physical blocks in logical order and reshaping gives the
        entries in ``ci`` order. Entries beyond the live count are masked out by
        the caller. The cache is fp8, so selected blocks are cast to bf16 before
        the read (HPU has no fp8 gather kernel). Returns None when the paged path
        is unavailable.
        """
        from vllm.forward_context import get_forward_context

        md = get_forward_context().attn_metadata
        aux = getattr(md, "dsv4_aux_meta", None)
        _dbg = _DSV4_R6_DEBUG
        if _dbg:
            print(f"[R6-enter] prefix={self.prefix!r} has_aux={bool(aux)} "
                  f"slab={getattr(self, '_dsv4_slab2d', None) is not None}", flush=True)
        if not aux:
            return None
        entry = aux.get(self.prefix)
        if entry is None:
            if _dbg:
                print(f"[R6] no aux entry for prefix={self.prefix!r}; "
                      f"keys={list(aux.keys())[:6]}", flush=True)
            return None
        slab2d = getattr(self, "_dsv4_slab2d", None)
        if slab2d is None:
            return None
        bl = entry.get("block_list_full")
        if bl is None or bl.dim() != 1:
            if _dbg:
                print(f"[R6] block_list_full missing for prefix={self.prefix!r}",
                      flush=True)
            return None
        sbs = entry["storage_block_size"]
        hd = self.head_dim
        page = _dsv4_page(self, sbs, hd)
        nblocks = (cap + sbs - 1) // sbs
        logical = torch.arange(nblocks, device=bl.device, dtype=torch.int64)
        logical = logical.clamp(max=bl.numel() - 1)
        phys = bl[logical].clamp(min=0)
        pages = page.index_select(0, phys).to(torch.bfloat16)  # [nblocks, sbs, hd]
        comp = pages.reshape(-1, hd)[:cap]
        if _DSV4_R6_DEBUG:
            live = int(self._c_n)
            if live > 0:
                print(f"[R6] layer={getattr(self, '_layer_idx', '?')} live={live} "
                      f"cache_dtype={page.dtype} bl0={int(bl[0])} phys0={int(phys[0])} "
                      f"got0={comp[0, :3].float().tolist()} "
                      f"page_off={self._dsv4_page_off} "
                      f"page_elems={self._dsv4_page_elems} slab_cols={slab2d.shape[1]}", flush=True)
        return comp

    def _indexer_kcache_aux(self):
        """Return ``(layer, info)`` for the indexer k_cache, or ``(None, None)``."""
        layer = getattr(self.indexer, "k_cache", None) if self.indexer is not None else None
        if layer is None:
            return None, None
        info = self._dsv4_aux_entry(layer)
        if info is None or getattr(layer, "_dsv4_slab2d", None) is None:
            return None, None
        return layer, info

    def _quant_indexer_key(self, k):
        """Quantize indexer keys to the paged k_cache format: fp8 + a per-token
        fp32 scale, using the standard HPU fp8 path.

        ``k`` is ``[n, 128]`` bf16. Returns ``(x_fp8 [n,128], scale [n] fp32)``
        where ``dequant = cast_from_fp8(x_fp8, scale)``. ``FP8_MAX`` is the HPU
        native max (240 = e4m3fnuz on Gaudi2, 448 on Gaudi3), so the quantized
        values stay in the hardware's representable range.
        """
        from vllm_gaudi.extension.ops import FP8_MAX
        kf = k.to(torch.bfloat16)
        amax = kf.float().abs().amax(dim=-1).clamp(min=1e-4)
        scale = (amax / FP8_MAX).to(torch.float32)  # dequant multiplier
        inv = (1.0 / scale).reshape(-1, 1)
        x = torch.ops.hpu.cast_to_fp8_v2(kf, inv, False, False,
                                         torch.float8_e4m3fn)[0]
        return x, scale

    def _write_paged_indexer(self, icomp, sel, positions, nw=None):
        """R7: write the indexer compressed keys into the paged indexer k_cache
        as fp8 + a per-token fp32 scale.

        The upstream k_cache layout is per block of ``sbs`` tokens:
        ``[0, sbs*128)`` fp8 bytes, ``[sbs*128, sbs*128+sbs*4)`` fp32 scales.
        Per-entry contiguous 1-D slice copies (eager fp8 advanced-index scatter
        is a no-op on HPU).
        """
        layer, info = self._indexer_kcache_aux()
        if layer is None or icomp.shape[0] == 0:
            return
        slab2d = layer._dsv4_slab2d
        slots = info["slot_mapping"]
        if sel is not None:
            # Fixed-size aligned-row selection; invalid rows (>= nreal) clamp to
            # the last (pad-block) row of the padded slot_mapping so their writes
            # land harmlessly outside the request's real blocks.
            sel_slot = sel.clamp(max=slots.shape[0] - 1)
            slots = slots[sel_slot]
        sbs = info["storage_block_size"]
        tok = self.indexer.head_dim  # 128 fp8 bytes/token
        off = layer._dsv4_page_off
        fp8_elems = sbs * tok
        blocks = torch.div(slots, sbs, rounding_mode="floor")
        offs = slots % sbs
        x, scale = self._quant_indexer_key(icomp)
        # Keys (fp8) go into the paged k_cache: one strided-view mutation.
        fp8_page = slab2d[:, off:off + fp8_elems].view(-1, sbs, tok)
        fp8_page[blocks, offs] = x.to(slab2d.dtype)
        # Scales go into the dedicated fp32 buffer, indexed by the compressed
        # entry index ci = (pos+1)//rate - 1 (HPU cannot mutate two dtype-views
        # of the same cache in one graph — see notes/hpu-clean.md §4).
        rate = self.compress_ratio
        pos_all = positions.reshape(-1)
        if sel is not None:
            if nw is not None:
                # `nw` is the number of valid (window-end-aligned, real) rows;
                # rows [nw, max_sel) are invalid and must not touch a live entry.
                valid = (torch.arange(sel.shape[0], device=sel.device) < nw)
                sel_ci = torch.where(valid, sel, torch.zeros_like(sel))
                ci = ((pos_all[sel_ci] + 1) // rate - 1).clamp(min=0, max=self._cap - 1)
                ci = torch.where(valid, ci, torch.full_like(ci, self._cap))
            else:
                sel_ci = sel.clamp(max=pos_all.numel() - 1)
                ci = ((pos_all[sel_ci] + 1) // rate - 1).clamp(min=0, max=self._cap - 1)
        else:
            ci = torch.arange(icomp.shape[0], device=pos_all.device).clamp(max=self._cap - 1)
        self._dsv4_indexer_scales[ci] = scale.to(torch.float32)

    def _read_paged_indexer(self, cap):
        """R7: gather indexer compressed keys ``[cap, 128]`` from the paged
        indexer k_cache and dequantize (fp8 * per-token scale) to bf16.

        Keys come from the paged k_cache; scales come from the dedicated fp32
        buffer (``_dsv4_indexer_scales``). Entries beyond the live count stay
        zero and are masked by the caller.
        """
        layer, info = self._indexer_kcache_aux()
        if layer is None:
            return None
        bl = info.get("block_list_full")
        if bl is None or bl.dim() != 1:
            return None
        slab2d = layer._dsv4_slab2d
        sbs = info["storage_block_size"]
        tok = self.indexer.head_dim
        off = layer._dsv4_page_off
        fp8_elems = sbs * tok
        nblocks = (cap + sbs - 1) // sbs
        logical = torch.arange(nblocks, device=bl.device, dtype=torch.int64)
        logical = logical.clamp(max=bl.numel() - 1)
        phys = bl[logical].clamp(min=0)
        fp8_page = slab2d[:, off:off + fp8_elems].view(-1, sbs, tok)
        x = fp8_page.index_select(0, phys).view(torch.float8_e4m3fn)
        x = x.reshape(-1, tok)[:cap]
        sc = self._dsv4_indexer_scales[:cap]
        out = torch.ops.hpu.cast_from_fp8(x, sc.reshape(-1, 1), torch.bfloat16)
        if _DSV4_R6_DEBUG:
            _live = min(int(self._c_n), cap)
            if _live > 0:
                print(f"[R7] layer={getattr(self, '_layer_idx', '?')} live={_live} "
                      f"tok={tok} off={off} pe={layer._dsv4_page_elems} "
                      f"bl0={int(bl[0])} phys0={int(phys[0])} "
                      f"got0={out[0, :3].float().tolist()} "
                      f"c_n={int(self._c_n)} i_n={int(self._i_n)}", flush=True)
        return out

    def _compressor_compilable(self, hidden, qr, positions, is_prompt):
        """Graph-compilable compressor forward.

        Computes KV/gate projections, writes per-token state via
        ``_comp_state_paged_vectorized`` (bf16 ring buffer, no ``.item()``
        loops), writes completed compressed entries to the paged compressed-MLA
        cache (``page[blocks, offs] = comp``, fp8 strided scatter), and runs
        the scorer / top-k / block_bias logic with pure tensor ops (no
        data-dependent Python branches, static ``cap`` shape).

        ``is_prompt`` is a scalar tensor (1 for prefill, 0 for decode). This
        function runs over the FULL (possibly padded) batch; padding rows
        (position < 0) are handled inside by masks (ring scratch slot +
        all-invalid-row zeroing + ``nreal``-based aligned-row selection), so no
        bool-mask indexing / dynamic shapes ever reach the graph.
        """
        if self.compressor is None or self.compress_ratio <= 1:
            return None, None, None, None
        rate = self.compress_ratio
        D = self.head_dim
        dev = hidden.device
        cache = self._compress_cache
        T = hidden.shape[0]

        # Live compressed-entry count from absolute positions (stateless).
        _pos = positions.reshape(-1)
        _last = _pos.max() if _pos.numel() > 0 else torch.tensor(-1, dtype=torch.int64, device=dev)
        live = ((_last + 1) // rate).to(torch.int64)

        c = self.compressor
        fused = self._comp_weight(c.fused_wkv_wgate, hidden.shape[-1])
        coff = self._comp_coff
        kv_out = coff * D
        kv_w = fused[:kv_out]
        gate_w = fused[kv_out:]
        kv = hidden @ kv_w.t()
        gate = hidden @ gate_w.t()

        # R4 vectorized state write + window compression
        comp, _nw_p, sel_p = self._comp_state_paged_vectorized(kv, gate, positions)
        if comp.shape[0] > 0:
            self._write_paged_compressed(comp, sel_p, positions)
        # ---- indexer (only CSA, ratio 4) ----
        has_idx = rate == 4 and self.indexer is not None
        if has_idx:
            # CSA gathers its top-k entries in the attention (D5d), so the
            # dense compressed read ([cap, D]; ~268 MB at 1M) is not needed.
            compressed_kv_full = kv.new_zeros((0, D))
        else:
            compressed_kv_full = self._read_paged_compressed(self._cap)
            if compressed_kv_full is None:
                compressed_kv_full = kv.new_zeros((self._cap, D))

        if has_idx:
            idx = self.indexer
            idx_h = idx.head_dim
            ifused = self._comp_weight(idx.compressor.fused_wkv_wgate, hidden.shape[-1])
            ikv_w = ifused[: 2 * idx_h]
            igate_w = ifused[2 * idx_h:]
            ikv = hidden @ ikv_w.t()
            igate = hidden @ igate_w.t()
            icomp, _nw_i, _sel_i = self._comp_state_paged_vectorized(
                ikv, igate, positions, comp_module=idx.compressor, head_dim=idx_h,
                state_buf=getattr(self, "_dsv4_indexer_state", None))
            if icomp.shape[0] > 0:
                self._write_paged_indexer(icomp, _sel_i, positions, _nw_i)
            idx_compressed = self._read_paged_indexer(self._cap)
            if idx_compressed is None:
                idx_compressed = kv.new_zeros((self._cap, idx_h))
        else:
            idx_h = D
            idx_compressed = torch.zeros(0, D, dtype=torch.bfloat16, device=dev)

        # ---- scorer / topk / block_bias (tensor-only, no Python branches) ----
        # Only the static `has_idx` (CSA vs HCA) selects a path; all
        # data-dependent bounds (live-entry count) are expressed as tensor masks
        # (the causal threshold subsumes the live count), so nothing here
        # graph-breaks.
        cap = self._cap
        clen = live.to(torch.int64)
        # CSA uses `topk` (per-query gather); HCA uses `block_bias` (dense causal
        # mask). Each returns an empty tensor for the field the other uses.
        block_bias = compressed_kv_full.new_zeros((T, 0))
        topk = torch.empty(T, 0, dtype=torch.long, device=dev)

        if has_idx:
            # CSA: indexer top-k over the compressed entries.
            nhead = self.indexer.n_head
            qb_w = self.indexer.wq_b.weight.detach()
            if qb_w.dtype == torch.float8_e4m3fn:
                from vllm_gaudi.extension.ops import dequant_block_fp8_weight_naive
                wq_s = getattr(self.indexer.wq_b, "weight_scale", None)
                if wq_s is None:
                    wq_s = getattr(self.indexer.wq_b, "weight_scale_inv", None)
                bs = getattr(getattr(self.indexer.wq_b, "quant_config", None),
                             "weight_block_size", (128, 128))
                if wq_s is not None and wq_s.dim() == 2:
                    qb = dequant_block_fp8_weight_naive(qb_w, wq_s, bs, torch.bfloat16)
                else:
                    s = wq_s if wq_s is not None else torch.ones(
                        qb_w.shape[0], device=qb_w.device, dtype=torch.float32)
                    qb = (qb_w.float() * s.reshape(-1, 1)).to(torch.bfloat16)
            else:
                qb = qb_w.to(torch.bfloat16)
            wp = self.indexer.weights_proj.weight
            ss = self.indexer.head_dim ** -0.5
            ws = self.indexer.n_head ** -0.5
            q = (qr @ qb.t()).view(T, nhead, idx_h).to(torch.bfloat16)
            q = _apply_rope(q, positions, cache, self.rope_head_dim)
            w = (hidden @ wp.t()).float() * ws  # [T, nhead]
            k_static = min(self.indexer.topk_tokens, cap)
            entry = torch.arange(cap, device=dev)
            # Query-chunk the scoring: the full score tensor is
            # [T, nhead, cap] fp32 (17 GB at 64k, 68 GB at 256k). The chunk size
            # is derived from `cap` so the per-chunk tensor stays within the
            # element budget; it is a static Python int per compiled shape.
            c_idx = max(1, min(T, _DSV4_INDEXER_CHUNK_ELEMS // max(1, nhead * cap)))
            Tpad = ((T + c_idx - 1) // c_idx) * c_idx
            pad = Tpad - T
            if pad:
                q = torch.cat([q, q.new_zeros((pad, nhead, idx_h))], dim=0)
                w = torch.cat([w, w.new_zeros((pad, nhead))], dim=0)
                positions_i = torch.cat(
                    [positions, positions.new_full((pad,), -1)], dim=0)
            else:
                positions_i = positions
            outs = []
            for j in range(Tpad // c_idx):
                sl = slice(j * c_idx, (j + 1) * c_idx)
                # Score over ALL cap entries; the causal mask zeroes the
                # not-yet-live ones (entry >= causal_threshold), so no dynamic slice.
                sc = torch.matmul(q[sl].float(), idx_compressed.float().t())
                sc = torch.relu(sc) * ss
                sc = (sc * w[sl].unsqueeze(-1)).sum(dim=1)  # [c_idx, cap]
                causal_threshold = (positions_i[sl] + 1) // rate
                future = entry.unsqueeze(0) >= causal_threshold.unsqueeze(-1)
                sc = sc.masked_fill(future, float("-inf"))
                tk = sc.topk(k_static, dim=-1).indices
                valid_topk = tk < causal_threshold.unsqueeze(-1)
                outs.append(torch.where(valid_topk, tk, torch.full_like(tk, -1)))
            topk = torch.cat(outs, dim=0)[:T]
        else:
            # HCA: dense causal mask over the live compressed entries.
            block_bias = compressed_kv_full.new_full((T, cap), float("-inf"))
            causal = (positions + 1) // rate
            entry = torch.arange(cap, device=dev)
            block_bias.masked_fill_(
                (entry.unsqueeze(0) < causal.unsqueeze(-1)) & (entry.unsqueeze(0) < clen),
                0.0)

        return compressed_kv_full, live, block_bias, topk

    def _run_compressor(self, hidden_states, q_residual, positions, is_prompt):
        """CSA compressor/indexer for prefill, isolated by a controlled graph break.

        The compressor must be its own compiled graph: inlining it into the layer
        graph trips ``aot_autograd``'s non-differentiable view-mutation assertion
        and grows the combined workspace ~8 GiB (OOM at util 0.8). The
        ``torch._dynamo.graph_break()`` calls partition the layer region so the
        compressor body is traced into its own compiled segment with its own
        workspace. The boundary carries no tensor work.

        This is a real break, **not** ``torch.compiler.disable``: the body stays
        compiled. ``compiler.disable`` would run the wrapper eagerly, i.e. HPU
        compute outside a graph, which is forbidden.

        Default (``_COMPRESSOR_BREAK`` False): inlined, no break — required under
        the engine's global ``fullgraph=True``.
        """
        if _COMPRESSOR_BREAK:
            torch._dynamo.graph_break()
        out = self._run_compressor_impl(
            hidden_states, q_residual, positions, is_prompt)
        if _COMPRESSOR_BREAK:
            torch._dynamo.graph_break()
        return out

    def _run_compressor_decode(self, hidden_states, q_residual, positions, is_prompt):
        """Decode compressor; see ``_run_compressor`` for the controlled break."""
        if _COMPRESSOR_BREAK:
            torch._dynamo.graph_break()
        out = self._run_compressor_impl(
            hidden_states, q_residual, positions, is_prompt)
        if _COMPRESSOR_BREAK:
            torch._dynamo.graph_break()
        return out

    def _run_compressor_impl(self, hidden_states, q_residual, positions, is_prompt):
        if self.compressor is None or self.compress_ratio <= 1:
            D = self.head_dim
            dev = hidden_states.device
            return (torch.zeros(0, D, dtype=torch.bfloat16, device=dev),
                    torch.tensor(0, dtype=torch.int64, device=dev), None, None)
        ckv_full, c_n, bb, topk = self._compressor_compilable(
            hidden_states, q_residual, positions, is_prompt)
        if ckv_full is None:
            cap = max(self._cap, 1)
            dev = hidden_states.device
            D = self.head_dim
            return (torch.zeros(cap, D, dtype=torch.bfloat16, device=dev),
                    torch.tensor(0, dtype=torch.int64, device=dev), None, None)
        return ckv_full, c_n, bb, topk

    def _o_proj(self, o: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # o: [T, n_local_heads, head_dim] -- inverse RoPE + wo_a (group bmm) + wo_b
        _diag("op_invrope_in", o)
        o = _apply_inv_rope(
            o, positions, self.rotary_emb.cos_sin_cache, self.rope_head_dim
        )
        _diag("op_after_invrope", o)
        if _HPU_DUMP_ON:
            _HPU_CAP.setdefault("op_after_invrope", {})[getattr(self, "_layer_idx", -1)] = o.detach().float().clone()
        num_tokens = o.shape[0]
        heads_per_group = self.n_local_heads // self.n_local_groups
        # [T, n_local_groups, heads_per_group, head_dim]
        o_g = o.view(num_tokens, self.n_local_groups, heads_per_group, self.head_dim)
        # wo_a is a TP-sharded ColumnParallelLinear; its weight is laid out as
        # [n_local_groups * o_lora_rank, heads_per_group * head_dim]. Reshape to
        # [n_local_groups, o_lora_rank, heads_per_group * head_dim] for a group bmm.
        # Dequant block-fp8 weight on-the-fly (no CPU round-trip, no cache).
        from vllm_gaudi.extension.ops import _dequant_fp8_weight, apply_block_fp8_linear_hpu_gemm
        wo_a_bf16 = _dequant_fp8_weight(self.wo_a.weight, self.wo_a.weight_scale_inv)
        w = wo_a_bf16.view(
            self.n_local_groups, self.o_lora_rank, heads_per_group * self.head_dim
        )
        o_flat = o_g.reshape(num_tokens, self.n_local_groups, heads_per_group * self.head_dim)
        _diag("wo_a_weight", self.wo_a.weight)
        z = torch.einsum("tgk,gdk->tgd", o_flat.float(), w).to(o.dtype)  # [T, n_groups, o_lora_rank]
        _diag("op_after_einsum", z)
        if _HPU_DUMP_ON:
            _HPU_CAP.setdefault("z_einsum", {})[getattr(self, "_layer_idx", -1)] = z.detach().float().clone()
        if _HPU_DUMP_ON:
            _HPU_CAP.setdefault("wo_a_dq", {})[getattr(self, "_layer_idx", -1)] = \
                wo_a_bf16.detach().float().clone()
            _HPU_CAP.setdefault("wo_a_raw", {})[getattr(self, "_layer_idx", -1)] = \
                self.wo_a.weight.detach().float().clone()
            si = getattr(self.wo_a, "weight_scale_inv", None)
            _HPU_CAP.setdefault("wo_a_scale", {})[getattr(self, "_layer_idx", -1)] = \
                (si.detach().float().clone() if si is not None else None)
            _HPU_CAP.setdefault("wo_b_w", {})[getattr(self, "_layer_idx", -1)] = \
                self.wo_b.weight.detach().float().clone()
            bs = getattr(self.wo_b, "weight_scale_inv", None)
            if bs is None:
                bs = getattr(self.wo_b, "weight_scale", None)
            _HPU_CAP.setdefault("wo_b_scale", {})[getattr(self, "_layer_idx", -1)] = \
                (bs.detach().float().clone() if bs is not None else None)
        z = z.reshape(num_tokens, self.n_local_groups * self.o_lora_rank)
        out = apply_block_fp8_linear_hpu_gemm(
            z, self.wo_b.weight, self.wo_b.weight_scale_inv,
            self.wo_b.quant_config.weight_block_size,
        )
        out = out[0] if isinstance(out, tuple) else out
        from vllm.distributed import tensor_model_parallel_all_reduce

        out = tensor_model_parallel_all_reduce(out)
        _diag("op_after_wo_b", out)
        return out

    def _o_proj_compilable(self, o: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """O projection without HPU_DUMP / _diag — safe for torch.compile.

        Applies fp32 composition through the full projection chain (same pattern
        as the MoE routed/shared experts): keep all intermediates in fp32, cast
        to bf16 only at the very end. This prevents the ulp-stacking tail from
        multiple sequential bf16 truncations.
        """
        from vllm_gaudi.extension.ops import _dequant_fp8_weight
        o = _apply_inv_rope(
            o, positions, self.rotary_emb.cos_sin_cache, self.rope_head_dim
        )
        num_tokens = o.shape[0]
        heads_per_group = self.n_local_heads // self.n_local_groups
        o_g = o.view(num_tokens, self.n_local_groups, heads_per_group, self.head_dim)
        wo_a_bf16 = _dequant_fp8_weight(self.wo_a.weight, self.wo_a.weight_scale_inv)
        w = wo_a_bf16.view(
            self.n_local_groups, self.o_lora_rank, heads_per_group * self.head_dim
        )
        o_flat = o_g.reshape(num_tokens, self.n_local_groups, heads_per_group * self.head_dim)
        # fp32 group matmul — keep in fp32 (no cast to bf16)
        z = torch.matmul(
            o_flat.float().permute(1, 0, 2), w.float().transpose(-1, -2)
        ).permute(1, 0, 2)
        z = z.reshape(num_tokens, self.n_local_groups * self.o_lora_rank)
        # fp32 wo_b matmul — dequant to bf16 then cast to fp32 for the product
        wo_b_bf16 = _dequant_fp8_weight(self.wo_b.weight, self.wo_b.weight_scale_inv)
        out = (z.float() @ wo_b_bf16.float().t()).to(torch.bfloat16)
        from vllm.distributed import tensor_model_parallel_all_reduce
        out = tensor_model_parallel_all_reduce(out)
        return out

    def forward_mqa(self, q, kv, positions, output) -> None:
        # Dense attention over the full context via torch SDPA.
        T = q.shape[0]
        k = kv.unsqueeze(1).expand(T, self.n_local_heads, self.head_dim)
        v = k
        o = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=True
        )
        output.copy_(o)

    def _sparse_comp_attn(
        self,
        q: torch.Tensor,
        positions: torch.Tensor,
        win_kv: torch.Tensor,
        start: torch.Tensor,
        span: int,
        comp_full: torch.Tensor | None,
        comp_bb: torch.Tensor | None,
        topk: torch.Tensor | None,
        kind: int,
    ) -> torch.Tensor:
        """D5c/D5d: sparse compressed attention over per-query keys.

        The old dense path attended ``window_union + cap`` keys per query, where
        the window union was the shared ``W + T - 1`` read and ``cap =
        max_model_len // rate`` (262144 for CSA at 1M). This gathers, per query,
        its own ``W`` window slots and its ``index_topk`` compressed entries, so
        the attention is ``O(T * (W + k))`` and the per-chunk memory is bounded.

        ``kind`` is a static per-layer int (resolved at trace time):
          * 0 — window only (SWA layers);
          * 1 — CSA: per-query top-k gather, batched over the query chunk
                (``sdpa_recomp_fwd`` supports B>1 with MQA, D5d.0);
          * 2 — HCA: shared dense compressed entries, one query chunk at a time
                (per-query gather would be ``T * cap`` traffic).

        The window is gathered per query from the D2 union read by position, so
        no union blow-up: keys are ``W + k`` (CSA) instead of ``W + T - 1 + cap``.
        """
        W = self.window_size
        H = self.n_local_heads
        D = self.head_dim
        scaling = self.scale
        sink = self.attn_sink[:H]
        T = q.shape[0]
        dev = q.device
        # Never pad a short (decode) batch up to the prefill chunk size: T=1 with
        # chunk=256 would process 256 rows (255 padding) per layer per step.
        chunk = min(_DSV4_SPARSE_CHUNK, max(T, 1))
        Tpad = ((T + chunk - 1) // chunk) * chunk
        pad = Tpad - T
        if pad:
            q = torch.cat([q, q.new_zeros((pad, H, D))], dim=0)
            positions = torch.cat([positions, positions.new_full((pad,), -1)], dim=0)
            if topk is not None:
                topk = torch.cat([topk, topk.new_full((pad, topk.shape[1]), -1)], dim=0)
            if comp_bb is not None:
                comp_bb = torch.cat(
                    [comp_bb, comp_bb.new_full((pad, comp_bb.shape[1]), float("-inf"))],
                    dim=0)

        if kind == 1:
            info = self._dsv4_aux_entry(self)
            slab2d = getattr(self, "_dsv4_slab2d", None)
            bl = None if info is None else info.get("block_list_full")
            if info is None or slab2d is None or bl is None:
                raise RuntimeError(
                    f"DSv4 sparse compressed attention needs the paged cache for "
                    f"prefix={getattr(self, 'prefix', None)!r}: info={info is not None}, "
                    f"slab={slab2d is not None}, block_list_full={bl is not None}")
            sbs = info["storage_block_size"]
            page = _dsv4_page(self, sbs, D)
            k_static = topk.shape[1]

        outs = []
        nchunk = Tpad // chunk
        for j in range(nchunk):
            sl = slice(j * chunk, (j + 1) * chunk)
            p = positions[sl]  # [chunk]
            if kind == 2:
                # HCA: the shared D2 union window (position-masked per query) +
                # the shared dense compressed entries, one query chunk at a time
                # so the [1,H,chunk,span+cap] bias stays bounded (the full-T bias
                # would be ~3 GB at 1M). The union window is validated by the D2
                # gate; slicing a per-chunk union is avoided (it was buggy).
                win_ids = torch.arange(span, device=dev)
                win_pos = start + win_ids  # [span]
                win_mask = (
                    (win_pos.unsqueeze(0) >= 0)
                    & (win_pos.unsqueeze(0) <= p.unsqueeze(1))
                    & (win_pos.unsqueeze(0) > p.unsqueeze(1) - W)
                )  # [chunk, span]
                keys = torch.cat([win_kv, comp_full], dim=0)  # [span+cap, D]
                comp_mask = torch.isfinite(comp_bb[sl])
                mask = torch.cat([win_mask, comp_mask], dim=1)  # [chunk, span+cap]
                # The sink is folded in as an extra zero-valued key row (the
                # bias carries its per-head logit); k must have K+1 rows.
                keys_ext = torch.cat([keys, keys.new_zeros(1, D)], dim=0)
                qh = q[sl].permute(1, 0, 2).unsqueeze(0)  # [1,H,chunk,D]
                kh = keys_ext.unsqueeze(0).unsqueeze(0)   # [1,1,span+cap+1,D]
                bias = torch.zeros((1, H, chunk, keys.shape[0]),
                                   dtype=torch.float32, device=dev)
                bias = torch.where(mask.reshape(1, 1, chunk, -1), bias,
                                   torch.full((), float("-inf"), device=dev))
                bias = torch.cat(
                    [bias, sink.float().reshape(1, H, 1, 1).expand(1, H, chunk, 1)],
                    dim=-1)
                o = torch.ops.hpu.sdpa_recomp_fwd(
                    qh, kh, kh, bias, 0.0, scaling, False, False, "fp32", None,
                    "right", (-1, -1), None)
                o = o[0] if isinstance(o, tuple) else o
                outs.append(o.squeeze(0).permute(1, 0, 2))  # [chunk,H,D]
                continue

            # kind 0/1: per-query window gather (batch = chunk).
            wp = p.unsqueeze(1) - W + 1 + torch.arange(W, device=dev)  # [chunk,W]
            widx = (wp - start).clamp(min=0, max=span - 1)
            win_rows = win_kv.index_select(0, widx.reshape(-1)).view(chunk, W, D)
            win_mask = wp >= 0  # [chunk,W]
            if kind == 1:
                tk = topk[sl]  # [chunk,k]
                comp_rows = _gather_paged_entries(
                    page, bl, tk.reshape(-1), sbs, D).view(chunk, k_static, D)
                comp_mask = tk >= 0
                keys = torch.cat([win_rows, comp_rows], dim=1)  # [chunk,W+k,D]
                mask = torch.cat([win_mask, comp_mask], dim=1)  # [chunk,W+k]
            else:
                keys = win_rows
                mask = win_mask
            bias = torch.zeros((chunk, H, 1, keys.shape[1]),
                               dtype=torch.float32, device=dev)
            bias = torch.where(mask.reshape(chunk, 1, 1, -1), bias,
                               torch.full((), float("-inf"), device=dev))
            bias = torch.cat(
                [bias, sink.float().reshape(1, H, 1, 1).expand(chunk, H, 1, 1)],
                dim=-1)
            # Sink as an extra zero key row (bias has W+k+1 columns).
            keys_ext = torch.cat([keys, keys.new_zeros(chunk, 1, D)], dim=1)
            qh = q[sl].unsqueeze(2)   # [chunk,H,1,D]
            kh = keys_ext.unsqueeze(1)  # [chunk,1,W+k+1,D]
            o = torch.ops.hpu.sdpa_recomp_fwd(
                qh, kh, kh, bias, 0.0, scaling, False, False, "fp32", None,
                "right", (-1, -1), None)
            o = o[0] if isinstance(o, tuple) else o
            outs.append(o.squeeze(2))  # [chunk,H,D]
        return torch.cat(outs, dim=0)[:T]

    def _forward_compilable(
        self,
        qr: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        win_kv: torch.Tensor,
        start: torch.Tensor,
        span: int,
        comp_full: torch.Tensor | None,
        comp_bb: torch.Tensor | None,
        topk: torch.Tensor | None,
        kind: int,
    ) -> torch.Tensor:
        """From normed qr+kv through wq_b -> RoPE -> sparse attention -> O proj.

        ``qr`` and ``kv`` are already normed by the eager outer (fused_wqa_wkv
        stays in eager because the compressor needs intermediate ``qr``). No
        Python control flow on tensor data, no env var checks — safe for
        ``torch.compile``. ``kind`` is a static per-layer int (see
        ``_sparse_comp_attn``).

        Args:
            qr: ``[T, q_lora_rank]`` normed Q residual.
            kv: ``[T, head_dim]`` normed KV.
            positions: ``[T]`` absolute positions.
            win_kv: ``[span, head_dim]`` D2 union window read.
            start: scalar first position of ``win_kv``.
            span: ``W + T - 1`` (union window length).
            comp_full: ``[cap, D]`` compressed entries (HCA) or None.
            comp_bb: ``[T, cap]`` additive compressed bias (HCA) or None.
            topk: ``[T, k]`` selected compressed indices (CSA) or None.
            kind: 0 window-only, 1 CSA, 2 HCA.

        Returns:
            ``[T, D]`` attention output.
        """
        num_tokens = qr.shape[0]

        from vllm_gaudi.extension.ops import apply_block_fp8_linear_hpu_gemm
        q = apply_block_fp8_linear_hpu_gemm(
            qr,
            self.wq_b.weight,
            self.wq_b.weight_scale_inv,
            self.wq_b.quant_config.weight_block_size,
        ).view(num_tokens, self.n_local_heads, self.head_dim)
        q = _rmsnorm(q, self.eps)
        q = _apply_rope(q, positions, self.rotary_emb.cos_sin_cache, self.rope_head_dim)
        kv = _apply_rope(kv, positions, self.rotary_emb.cos_sin_cache, self.rope_head_dim)

        o = self._sparse_comp_attn(
            q, positions, win_kv, start, span, comp_full, comp_bb, topk, kind)

        o = self._o_proj_compilable(o, positions)
        return o



    def _fused_qkv(self, hidden_states):
        """Shared preamble: fused_wqa_wkv → split → rmsnorm → rope."""
        qr_kv = self.fused_wqa_wkv(hidden_states)
        qr_kv = qr_kv[0] if isinstance(qr_kv, tuple) else qr_kv
        qr, kv = qr_kv.split([self.q_lora_rank, self.head_dim], dim=-1)
        qr = _rmsnorm(qr, self.eps, self.q_norm.weight)
        kv = _rmsnorm(kv, self.eps, self.kv_norm.weight)
        return qr, kv

    def _write_paged_swa_cache(self, kv_roped):
        """Write roped KV into the paged SWA cache (VLLM_DSV4_PAGED_KV).

        Uses a 2-D view of the packed slab (stashed by the model runner) so the
        scatter indexes dim 0 (blocks) rather than the whole flat slab: HPU index
        ops fail when the indexed dim >= 2**31 and the slab is ~1.9e10 elements,
        but the 2-D dim-0 size is only ``num_blocks+1``. The page is viewed 3-D
        ``(num_blocks+1, block_size, head_dim)`` and slots are written with
        advanced indexing on ``(block, off)``; this compiles as a single graph
        (see ``src/stepP1_probe4.py``).

        Scatter by (block, off) in place using the group's slot_mapping from the
        HPU attention metadata.
        """
        if _DSV4_R4B_NOSWA:
            return
        info = self._dsv4_aux_entry(self.swa_cache_layer)
        if info is None:
            return
        slab2d = getattr(self.swa_cache_layer, "_dsv4_slab2d", None)
        if slab2d is None:
            return
        slots = info["slot_mapping"]
        bs = info["block_size"]
        head_dim = self.head_dim
        kv = kv_roped.reshape(-1, head_dim)
        # 3-D page view: (num_blocks+1, block_size, head_dim); slot-major within a page.
        page = _dsv4_page(self.swa_cache_layer, bs, head_dim)
        blocks = torch.div(slots, bs, rounding_mode="floor")
        offs = slots % bs
        page[blocks, offs] = kv

    def _read_paged_swa_window(self, last_pos, window, extra=0):
        """Read the sliding-window union from the paged SWA cache (R3).

        Reads ``window + extra`` positions ending at ``last_pos``. A multi-token
        prefill chunk needs the union of its per-query windows
        ``[min_p-W+1, max_p]`` (length ``W + T - 1``), so the caller passes
        ``extra=T-1`` and then masks each query back to its own window. Decode
        passes ``extra=0`` (the read is already exactly the last window).

        ``last_pos`` is the last real position in the batch (scalar tensor).
        Returns ``(win_kv [W+extra, head_dim], win_n, win_start, valid)``.

        The paged path is the only path: a missing aux entry / slab / full block
        table is a hard error (silently dropping context is worse than failing).

        Graph-safe: positions -> logical block -> physical block via the
        request's full block table, then ``page[phys, off]`` advanced indexing.
        """
        layer = self.swa_cache_layer
        info = self._dsv4_aux_entry(layer)
        slab2d = getattr(layer, "_dsv4_slab2d", None)
        bl = None if info is None else info.get("block_list_full")
        if info is None or slab2d is None or bl is None or bl.dim() != 1:
            raise RuntimeError(
                f"DSv4 paged SWA window read unavailable for prefix="
                f"{getattr(self, 'prefix', None)!r}: info={info is not None}, "
                f"slab={slab2d is not None}, "
                f"block_list_full={None if bl is None else tuple(bl.shape)}. The "
                f"paged path is the only path; a missing aux entry/block table "
                f"would silently drop context.")
        bs = info["block_size"]
        head_dim = self.head_dim
        span = window + extra
        page = _dsv4_page(layer, bs, head_dim)
        end = last_pos + 1
        start = torch.clamp(end - span, min=0)
        win_ids = torch.arange(span, device=end.device, dtype=torch.int64)
        wpos = start + win_ids
        valid = wpos < end
        wpos_c = torch.clamp(wpos, max=end - 1)
        logical = torch.div(wpos_c, bs, rounding_mode="floor")
        woffs = wpos_c % bs
        phys = bl[logical]
        # Read via dim-0 index_select + dim-1 gather (both HPU-supported) rather
        # than advanced indexing on the strided page view, which makes the
        # compiler materialize a full slab copy inside the model's large graph.
        # The page is fp8 (kv_cache_dtype=fp8_inc); HPU's gather has no fp8
        # kernel, so cast the selected rows (small) to bf16 first.
        pages = page.index_select(0, phys).to(torch.bfloat16)  # [span, bs, head_dim]
        gidx = woffs.view(-1, 1, 1).expand(-1, 1, head_dim)
        win_kv = torch.gather(pages, 1, gidx).squeeze(1)  # [span, head_dim]
        win_n = torch.clamp(end - start, max=span)
        return win_kv, win_n, start, valid

    def _swa_window_mask(self, start, win_n, span, pos_flat, real, window):
        """Per-query sliding-window mask over the union window read.

        ``start``/``win_n``/``span`` come from ``_read_paged_swa_window``; slot
        ``i`` holds absolute position ``start + i``. Real query rows attend only
        their own causal window ``[p-window+1, p]`` (and only present slots);
        padding rows (``real`` false) attend everything present so their
        discarded output stays finite.
        """
        win_ids = torch.arange(span, device=pos_flat.device)
        win_present = win_ids < win_n
        win_pos = start + win_ids
        win_mask = (
            (win_pos[None, :] <= pos_flat[:, None])
            & (win_pos[None, :] > pos_flat[:, None] - window)
            & win_present[None, :]
        )
        return win_mask | (~real)[:, None]

    def forward_prefill(self, positions, hidden_states):
        """Prefill across a (possibly padded) prefill batch.

        vllm pads the prefill to a fixed length; only rows with position >= 0
        are real tokens and they are the leading ``[0:nreal)`` rows. The window
        cache keeps the last ``min(nreal, window)`` real kv rows. The whole
        (padded) batch flows through the compressor; padding rows are handled by
        masks inside ``_compressor_compilable``. The query side
        (qr/kv/positions) stays the full padded batch so the returned hidden has
        the shape vllm expects; padding query rows are masked to attend every
        slot so their (discarded) output stays finite.

        Graph-safe: no ``int(tensor)`` host sync, no data-dependent Python
        control flow, no bool-mask indexing. The real token count is a tensor
        (``nreal``), used only in tensor masks and static-size selection.
        """
        qr, kv = self._fused_qkv(hidden_states)
        kv_roped = _apply_rope(kv, positions, self.rotary_emb.cos_sin_cache, self.rope_head_dim)
        self._write_paged_swa_cache(kv_roped)
        T = kv_roped.shape[0]
        pos_flat = positions.reshape(-1)
        dev = kv_roped.device
        W = self.window_size

        # D2 union window read: this chunk's queries span [max-T+1, max], so read
        # W + T - 1 positions; ``_sparse_comp_attn`` gathers each query's own
        # [p-W+1, p] window from it by position.
        win_kv, win_n, start, _valid = self._read_paged_swa_window(
            pos_flat.max(), W, extra=T - 1)

        # `is_prompt` is a trace-time constant (the compressor body does not
        # read it); a device scalar here is an unnecessary HPU allocation.
        comp_full, comp_n, comp_bb, topk = self._run_compressor(
            hidden_states, qr, pos_flat, 1)

        span = W + T - 1
        kind = self._attn_kind
        if kind == 1:
            return self._forward_compilable(qr, kv, positions, win_kv, start, span,
                                            None, None, topk, kind)
        if kind == 2:
            return self._forward_compilable(qr, kv, positions, win_kv, start, span,
                                            comp_full, comp_bb, None, kind)
        return self._forward_compilable(qr, kv, positions, win_kv, start, span,
                                        None, None, None, kind)

    def forward_decode(self, positions, hidden_states):
        """Decode: append 1 token to ring buffer, run compressor (no reset), full mask."""
        # Decode may pass a 1-D [H] hidden; the compressor/rope assume [T, H].
        if hidden_states.dim() == 1:
            hidden_states = hidden_states.unsqueeze(0)
        qr, kv = self._fused_qkv(hidden_states)
        kv_roped = _apply_rope(kv, positions, self.rotary_emb.cos_sin_cache, self.rope_head_dim)
        self._write_paged_swa_cache(kv_roped)
        pos_flat = positions.reshape(-1)
        W = self.window_size
        # R3/R8/R9: window straight from the paged SWA cache (no ring buffer).
        # No assert: it would be a dynamo graph break inside the compiled region.
        win_kv, win_n, start, _valid = self._read_paged_swa_window(pos_flat[-1], W)
        comp_full, comp_n, comp_bb, topk = self._run_compressor_decode(
            hidden_states, qr, pos_flat, 0)

        kind = self._attn_kind
        if kind == 1:
            return self._forward_compilable(qr, kv, positions, win_kv, start, W,
                                            None, None, topk, kind)
        if kind == 2:
            return self._forward_compilable(qr, kv, positions, win_kv, start, W,
                                            comp_full, comp_bb, None, kind)
        return self._forward_compilable(qr, kv, positions, win_kv, start, W,
                                        None, None, None, kind)

    def forward(self, positions, hidden_states, llama_4_scaling=None):
        return self.forward_decode(positions, hidden_states)


_HPU_CAP: dict = {"attn_in": {}, "attn_out": {}}
_DUMP_ON = _HPU_DUMP_ON
_MHC_PHASE = "attn"
_MHC_LAYER = -1


class DeepseekV4HPUDecoderLayer(_NvDeepseekV4DecoderLayer):
    """HPU decoder layer. Subclasses the NVIDIA decoder so the shared
    ``set_moe_parameters`` isinstance checks pass; __init__/forward are
    overridden to use the HPU attention and to skip CUDA streams/events."""

    def __init__(
        self,
        vllm_config,
        prefix,
        topk_indices_buffer: torch.Tensor | None = None,
        aux_stream_list: list | None = None,
        eager_scratch_pool=None,
    ):
        nn.Module.__init__(self)
        import re as _re
        m = _re.search(r"layers\.(\d+)$", prefix or "")
        self._layer_idx = int(m.group(1)) if m else -1

        from vllm.model_executor.layers.layernorm import RMSNorm

        config = vllm_config.model_config.hf_config
        self.hidden_size = config.hidden_size
        self.use_sequence_parallel = _use_sequence_parallel(vllm_config)

        self.rms_norm_eps = config.rms_norm_eps
        self.attn = DeepseekV4HPUAttention(
            vllm_config,
            prefix=f"{prefix}.attn",
            topk_indices_buffer=topk_indices_buffer,
            aux_stream_list=None,
            eager_scratch_pool=None,
        )
        self.attn._layer_idx = self._layer_idx
        if self.use_sequence_parallel:
            self.attn.wo_b.reduce_results = False
        self.ffn = DeepseekV4MoE(
            vllm_config,
            prefix=f"{prefix}.ffn",
            use_sequence_parallel=self.use_sequence_parallel,
        )
        # Shared expert uses native block-fp8 GEMM via Fp8LinearMethod.apply()
        # (gate_up_proj and down_proj are ReplicatedLinear loaded as block-fp8).
        # No monkeypatch needed -- the default forward already dispatches through
        # the HPU block-fp8 path.
        self.ffn._v4_ckpt_path = _v4_resolve_ckpt_dir(getattr(config, "_name_or_path", None) or "")
        self.ffn._v4_layer_idx = getattr(self, "_layer_idx", -1)
        # Register empty native-fp8 routed buffers at construction so state_dict
        # carries them for the snapshot pipeline (fresh load fills them in
        # load_weights; snapshot restore fills them directly). This also frees
        # the dead FusedMoE routed weights first so the fp8 buffers and the
        # fp4 FusedMoE copies never coexist in HBM.
        _register_v4_fp8_placeholders(self.ffn, config)

        self.attn_norm = RMSNorm(self.hidden_size, self.rms_norm_eps)
        self.ffn_norm = RMSNorm(self.hidden_size, self.rms_norm_eps)
        self.hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.hc_post_alpha = 2.0
        mix_hc = (2 + self.hc_mult) * self.hc_mult
        hc_dim = self.hc_mult * self.hidden_size
        self.hc_attn_fn = nn.Parameter(
            torch.empty((mix_hc, hc_dim), dtype=torch.float32), requires_grad=False
        )
        self.hc_attn_fn_broadcast: torch.Tensor | None = None
        self.hc_ffn_fn = nn.Parameter(
            torch.empty((mix_hc, hc_dim), dtype=torch.float32), requires_grad=False
        )
        self.hc_attn_base = nn.Parameter(
            torch.empty(mix_hc, dtype=torch.float32), requires_grad=False
        )
        self.hc_ffn_base = nn.Parameter(
            torch.empty(mix_hc, dtype=torch.float32), requires_grad=False
        )
        self.hc_attn_scale = nn.Parameter(
            torch.empty(3, dtype=torch.float32), requires_grad=False
        )
        self.hc_ffn_scale = nn.Parameter(
            torch.empty(3, dtype=torch.float32), requires_grad=False
        )

    def _mhc(self, x, residual, post_mix, res_mix):
        """mHC pre/fused — shared by prefill and decode."""
        if residual is None:
            if x.dim() == 2:
                return _mhc_pre_broadcast_compilable(
                    x, self.hc_attn_fn_broadcast, self.hc_attn_scale,
                    self.hc_attn_base, self.rms_norm_eps, self.hc_eps,
                    self.hc_eps, self.hc_post_alpha, self.hc_sinkhorn_iters,
                    self.hc_mult, self.attn_norm.weight, self.rms_norm_eps,
                )
            from vllm.model_executor.kernels.mhc import mhc_pre_torch
            return (x, *mhc_pre_torch(
                x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base,
                self.rms_norm_eps, self.hc_eps, self.hc_eps,
                self.hc_post_alpha, self.hc_sinkhorn_iters,
            ))
        return _mhc_fused_post_pre_compilable(
            x, residual, post_mix, res_mix,
            self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base,
            self.rms_norm_eps, self.hc_eps, self.hc_eps,
            self.hc_post_alpha, self.hc_sinkhorn_iters, self.hc_mult,
            self.attn_norm.weight, self.rms_norm_eps,
        )

    def _mhc_post_ffn(self, x, residual, post_mix, res_mix):
        return _mhc_fused_post_pre_compilable(
            x, residual, post_mix, res_mix,
            self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base,
            self.rms_norm_eps, self.hc_eps, self.hc_eps,
            self.hc_post_alpha, self.hc_sinkhorn_iters, self.hc_mult,
            self.ffn_norm.weight, self.rms_norm_eps,
        )

    def _forward_inner_prefill(
        self, x, positions, input_ids, post_mix=None, res_mix=None, residual=None,
    ):
        residual, post_mix, res_mix, x = self._mhc(x, residual, post_mix, res_mix)
        if _DUMP_ON:
            # attn_in (post input-norm / mHC) for the prefill step; compare with
            # ref_cache.pt["attn_in"][0][layer]. Module-const guard => no graph break.
            _HPU_CAP["attn_in"][self._layer_idx] = x.detach().float().clone()
        x = self.attn.forward_prefill(positions, x)
        if _DUMP_ON:
            _HPU_CAP.setdefault("attn_out", {})[self._layer_idx] = x.detach().float().clone()
        residual, post_mix, res_mix, x = self._mhc_post_ffn(x, residual, post_mix, res_mix)
        if _DUMP_ON:
            # MoE input (post ffn-norm / mHC collapse) for the prefill step.
            _HPU_CAP.setdefault("moe_in", {})[self._layer_idx] = x.detach().float().clone()
        x = self.ffn(x, input_ids)
        if _DUMP_ON:
            _HPU_CAP.setdefault("moe_out", {})[self._layer_idx] = x.detach().float().clone()
        return x, residual, post_mix, res_mix

    def _forward_inner_decode(
        self, x, positions, input_ids, post_mix=None, res_mix=None, residual=None,
    ):
        residual, post_mix, res_mix, x = self._mhc(x, residual, post_mix, res_mix)
        x = self.attn.forward_decode(positions, x)
        residual, post_mix, res_mix, x = self._mhc_post_ffn(x, residual, post_mix, res_mix)
        x = self.ffn(x, input_ids)
        return x, residual, post_mix, res_mix

    def forward(self, x, positions, input_ids, post_mix=None, res_mix=None, residual=None):
        # Dispatch on the (bucketed, static) token count so the compiled layer
        # region is actually invoked: prefill has T > 1, decode has T == 1.
        if positions.numel() > 1:
            x, residual, post_mix, res_mix = self._forward_inner_prefill(
                x, positions, input_ids, post_mix, res_mix, residual
            )
        else:
            x, residual, post_mix, res_mix = self._forward_inner_decode(
                x, positions, input_ids, post_mix, res_mix, residual
            )
        return x, residual, post_mix, res_mix


class DeepseekV4HPUModel(_NvDeepseekV4Model):
    """HPU DeepSeek V4 model.

    Subclasses the NVIDIA model so ``load_weights`` / ``get_expert_mapping`` /
    ``finalize_*`` (which handle the checkpoint's stacked ``wq_a``/``wkv``
    params, expert weights, and the mHC broadcast) are inherited. Only
    construction (no CUDA streams/events, HPU attention) and forward are
    overridden."""

    def __init__(self, *, vllm_config, prefix: str = ""):
        nn.Module.__init__(self)

        from vllm.distributed import get_pp_group
        from vllm.model_executor.layers.layernorm import RMSNorm
        from vllm.model_executor.layers.vocab_parallel_embedding import (
            VocabParallelEmbedding,
        )
        from vllm.model_executor.models.utils import PPMissingLayer, make_layers

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        self.use_sequence_parallel = _use_sequence_parallel(vllm_config)
        self.use_mega_moe = (
            vllm_config.kernel_config.moe_backend == "deep_gemm_mega_moe"
        )
        self.vocab_size = config.vocab_size
        self.hc_eps = config.hc_eps
        self.hc_mult = config.hc_mult
        self.hc_dim = self.hc_mult * config.hidden_size
        self.rms_norm_eps = config.rms_norm_eps

        self.topk_indices_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            config.index_topk,
            dtype=torch.int32,
        )

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: DeepseekV4HPUDecoderLayer(vllm_config, prefix=prefix),
            prefix=f"{prefix}.layers",
        )

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, self.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.hc_head_fn = nn.Parameter(
            torch.empty(self.hc_mult, self.hc_dim, dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_head_base = nn.Parameter(
            torch.empty(self.hc_mult, dtype=torch.float32), requires_grad=False
        )
        self.hc_head_scale = nn.Parameter(
            torch.empty(1, dtype=torch.float32), requires_grad=False
        )

        self._mtp_hidden_buffer = None

    def forward(self, input_ids, positions, intermediate_tensors, inputs_embeds=None):
        # Graph break sources (this method stays eager):
        #   1. Layer loop ``for idx, layer in enumerate(islice(...))`` — Python
        #      dynamic loop; torch.compile cannot unroll a variable-length layer
        #      sequence.
        #   2. ``V4_DEBUG`` block — ``try/except`` + ``os.environ`` + ``print``.
        #   3. ``HPU_DUMP`` save block — ``try/except`` + ``os.environ`` +
        #      ``torch.save``.
        #   4. ``_diag()`` calls — Python-level diagnostic.
        # Individual layer forwards ARE compiled via
        # ``DeepseekV4HPUDecoderLayer._forward_inner()`` (``@torch.compile``).
        from itertools import islice

        from vllm.distributed import get_pp_group

        import os as _mdbg
        if _mdbg.environ.get("V4_DEBUG") == "1":
            n = getattr(self, "_model_dbg_n", 0) + 1
            self._model_dbg_n = n
            if n <= 12:
                try:
                    from vllm.forward_context import get_forward_context
                    amm = get_forward_context().attn_metadata
                    ispm = bool(getattr(amm, "is_prompt", None)) if amm is not None else None
                except Exception as e:
                    ispm = f"err:{e}"
                posm = positions.reshape(-1)
                print(f"[v4m] fwd#{n} iid={tuple(input_ids.shape)} posmin={int(posm.min()):d} "
                      f"posmax={int(posm.max()):d} is_prompt={ispm} is_first={get_pp_group().is_first_rank}", flush=True)

        if get_pp_group().is_first_rank:
            hidden_states = self.embed_input_ids(input_ids) if inputs_embeds is None else inputs_embeds
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]

        # On HPU input_ids/positions are 2D [batch, seq], so embed returns 3D
        # [batch, seq, hidden]. Flatten to the token dimension for the decoder
        # (which expects [T, hidden]); restore the batch dim before returning.
        hidden_shape = hidden_states.shape
        if hidden_states.dim() == 3:
            hidden_states = hidden_states.view(-1, self.config.hidden_size)
            positions = positions.reshape(-1)

        residual, post_mix, res_mix = None, None, None
        import os as _os
        _dbg = _os.environ.get("V4_DEBUG") == "1"
        for idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer),
            start=self.start_layer,
        ):
            if _dbg:
                print(f"[v4model] layer {idx} in", flush=True)
            hidden_states, residual, post_mix, res_mix = layer(
                hidden_states, positions, input_ids, post_mix, res_mix, residual
            )
            if _dbg:
                print(f"[v4model] layer {idx} out", flush=True)
            _diag(f"after_layer", hidden_states, layer=idx)
        _DIAG_DONE["v"] = True

        import os as _hdump
        if _hdump.environ.get("HPU_DUMP") == "1" and not getattr(self, "_hpu_dumped", False):
            self._hpu_dumped = True
            rank = get_pp_group().rank_in_group
            try:
                from vllm.distributed import get_tensor_model_parallel_rank as _gtmr
                rank = _gtmr()
                _dump_dir = _hdump.environ.get("HPU_DUMP_DIR", ".")
                os.makedirs(_dump_dir, exist_ok=True)
                torch.save(_HPU_CAP, os.path.join(_dump_dir, f"hpu_cap_tp{rank}.pt"))
                print(f"[hpu_dump] tp{rank} saved {len(_HPU_CAP['attn_in'])} layers", flush=True)
            except Exception as _e:
                print(f"[hpu_dump] rank{rank} save err: {_e}", flush=True)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})

        # Reconstruct the multi-stream [T, hc_mult, H] hidden state from the 2D
        # ffn output + the mHC residual/mixes, then collapse it via hc_head and
        # the final RMSNorm. `_model_head` is one compiled graph (the model
        # forward stays eager glue for the dynamic layer loop); the V4_DEBUG path
        # keeps the eager step-by-step prints.
        #
        # Manual fp32 RMSNorm instead of the vLLM RMSNorm module:
        # HPURMSNorm.forward_oot -> torch.ops.hpu.rms_norm FAILS to compile with
        # force_static_compile on this stack (synStatus 26), and the
        # rms_norm_fast path is inaccurate. The manual decomposition compiles.
        if _dbg:
            hidden_states = _mhc_post_compilable(hidden_states, residual, post_mix, res_mix)
            print(f"[v4model] after mhc_post shape={tuple(hidden_states.shape)}", flush=True)
            hidden_states = _hc_head(
                hidden_states,
                self.hc_head_fn,
                self.hc_head_scale,
                self.hc_head_base,
                self.rms_norm_eps,
                self.hc_eps,
            )
            print(f"[v4model] after hc_head shape={tuple(hidden_states.shape)}", flush=True)
            hidden_states = _rmsnorm(hidden_states, self.rms_norm_eps, self.norm.weight)
            print(f"[v4model] after norm shape={tuple(hidden_states.shape)}", flush=True)
        else:
            hidden_states = self._get_compiled_head()(
                hidden_states, residual, post_mix, res_mix,
                self.hc_head_fn, self.hc_head_scale, self.hc_head_base,
                self.norm.weight, self.rms_norm_eps, self.hc_eps,
            )
        if hidden_shape is not None and len(hidden_shape) == 3:
            hidden_states = hidden_states.view(*hidden_shape[:2], self.config.hidden_size)
        return hidden_states

    def _get_compiled_head(self):
        """Return a torch.compile'd `_model_head` (cached).

        The model head (mHC post + hc_head + final RMSNorm) must be a single
        compiled graph; run eager it is a stray HPU compute step outside any
        compiled region (V8).
        """
        fn = getattr(self, "_compiled_head_fn", None)
        if fn is None:
            from vllm_gaudi.utils import HPUCompileConfig
            fn = torch.compile(_model_head,
                               **HPUCompileConfig(fullgraph=True).get_compile_args())
            self._compiled_head_fn = fn
        return fn

    def _prep_v4_routed_all(self) -> None:
        """Pre-convert every layer's routed experts fp4 -> native fp8 (load time).

        Called from ``load_weights`` (fresh load) so the native-fp8 buffers are
        filled before snapshot capture; the forward then reads them directly
        (no per-token fp4->fp8 requant). Idempotent.
        """
        import time as _t
        t0 = _t.time()
        n = 0
        for layer in self.layers[self.start_layer:self.end_layer]:
            ffn = getattr(layer, "ffn", None)
            if ffn is not None and hasattr(ffn, "packed_w1_weight") \
                    and not getattr(ffn, "_v4_fp8_filled", False):
                _register_v4_routed_packed(ffn, "cpu")
                n += 1
        print(f"[v4] prepped {n} layers' routed fp4 buffers in {_t.time() - t0:.1f}s", flush=True)

    def _prep_compress_caches(self) -> None:
        """Build every layer's compressor buffers at load time.

        ``_make_compress_cache`` constructs a transformers config/module, so it
        is graph-unsafe and must run outside the compiled forward (Step 4.5).
        The compress RoPE cache is built once (D5b) and shared across layers:
        it is identical for every compressor layer and ~268 MB at 1M context.
        """
        n = 0
        shared = None
        for layer in self.layers[self.start_layer:self.end_layer]:
            attn = getattr(layer, "attn", None)
            if attn is not None and hasattr(attn, "_make_compress_cache"):
                attn._make_compress_cache(shared_rope_cache=shared)
                if shared is None:
                    shared = getattr(attn, "_compress_cache", None)
                if getattr(attn, "_compress_cache", None) is not None:
                    n += 1
        print(f"[v4] built compress caches for {n} layers (rope cache shared: "
              f"{shared is not None})", flush=True)
        self._check_v4_routed_all()

    def _check_v4_routed_all(self) -> None:
        """Load/restore-time invariant: every routed expert's packed fp4 buffers
        must be present. Run here (called by both fresh load and snapshot restore)
        so the ``.item()`` host sync never appears in the compiled forward."""
        for layer in self.layers[self.start_layer:self.end_layer]:
            ffn = getattr(layer, "ffn", None)
            if ffn is not None and hasattr(ffn, "_v4_fp8_ready"):
                _check_v4_routed_ready(ffn)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Routed-expert weights (.experts.) are bypassed: the FusedMoE params
        # are freed at construction and the native-fp8 path reads the packed
        # fp4 from the checkpoint directly. Filter them out so the (freed)
        # FusedMoE params are not repopulated, then build the fp8 buffers.
        weights = ((n, w) for (n, w) in weights if ".experts." not in n)
        loaded = super().load_weights(weights)
        self._prep_v4_routed_all()
        self._prep_compress_caches()
        return loaded


class DeepseekV4ForCausalLM(_NvDeepseekV4ForCausalLM):
    """HPU DeepSeek V4. Reuses the NVIDIA top-level (embed/norm/logits/MoE) but
    with the HPU model (attention + decoder)."""

    model_cls = DeepseekV4HPUModel

    def compute_logits(self, hidden_states):
        return self.logits_processor(self.lm_head, hidden_states)

    def warmup(self, device: torch.device) -> None:
        """Run one forward with expected prefill and decode shapes to trigger
        HPU graph compilation (``torch.compile`` compiles lazily on first call
        for each shape).

        Call once after model load, before the first inference request.
        """
        self.eval()
        with torch.no_grad():
            prefill_ids = torch.zeros((1, 128), dtype=torch.long, device=device)
            prefill_pos = torch.arange(128, dtype=torch.long, device=device).unsqueeze(0)
            _ = self.model(
                prefill_ids, prefill_pos, None,
            )
            decode_ids = torch.zeros((1, 1), dtype=torch.long, device=device)
            decode_pos = torch.tensor([[128]], dtype=torch.long, device=device)
            _ = self.model(
                decode_ids, decode_pos, None,
            )


def _hpu_v4_shared_mlp_forward(self, x: torch.Tensor) -> torch.Tensor:
    """Shared expert forward using native block-fp8 GEMM + fp32 composition.

    ``self`` is the shared_experts module with ``gate_up_proj`` (fused
    gate+up, ReplicatedLinear) and ``down_proj`` (ReplicatedLinear), both
    loaded as block-fp8 by Fp8LinearMethod. Dequants on-the-fly on HPU.
    The gate/up matmuls run bf16 (fast); the silu/clamp/product run in fp32
    to avoid compounding bf16 rounding into a multi-ulp tail (same as the
    routed experts). Returns fp32; the MoE forward casts to bf16 at the end.
    """
    from vllm_gaudi.extension.ops import apply_block_fp8_linear_hpu_gemm
    gu = self.gate_up_proj
    dp = self.down_proj
    gu_out = apply_block_fp8_linear_hpu_gemm(
        x, gu.weight, gu.weight_scale_inv, gu.quant_config.weight_block_size,
    )
    gate, up = gu_out.chunk(2, dim=-1)
    limit = getattr(self, "swiglu_limit", getattr(self, "limit", 10.0))
    gate = torch.clamp(gate.float(), max=limit)
    up = torch.clamp(up.float(), min=-limit, max=limit)
    h = torch.nn.functional.silu(gate) * up  # fp32
    out = apply_block_fp8_linear_hpu_gemm(
        h.to(x.dtype), dp.weight, dp.weight_scale_inv, dp.quant_config.weight_block_size,
    )
    return out.float()


def _v4_resolve_ckpt_dir(name_or_path: str) -> str:
    """Resolve a local dir holding the checkpoint safetensors.

    For a local path returns it as-is; for a hub id resolves the HF cache
    snapshot dir.
    """
    import glob as _g
    import os
    p = name_or_path or ""
    if p and os.path.isdir(p) and _g.glob(os.path.join(p, "model-*.safetensors")):
        return p
    name = p.replace("/", "--")
    root = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    mdir = os.path.join(root, "hub", f"models--{name}")
    for snap in sorted(_g.glob(os.path.join(mdir, "snapshots", "*"))):
        if _g.glob(os.path.join(snap, "model-*.safetensors")):
            return snap
    return p


def _fp4_nibble_bf16(n: torch.Tensor) -> torch.Tensor:
    """Decode packed e2m1 nibbles (uint8, 0..15) to bf16 bit patterns.

    A per-nibble LUT gather is the obvious decode but is pathologically slow on
    HPU (measured ~17x slower on the real expert shapes). The nibble math is
    kept in **uint8** (int32 temporaries peak ~6 GiB higher and OOM the engine)
    and converted to int16 only for the final 16-bit pattern. Mapping: idx 0 ->
    0; idx 1 -> 0.5 (exp field 126, mantissa 0); idx >= 2 -> exp field
    ``126 + (idx >> 1)``, mantissa ``idx & 1``; sign is bit 3.
    """
    idx = n & 7
    sign = (n >> 3) & 1
    zero = torch.zeros((), dtype=torch.uint8, device=n.device)
    expf = torch.where(idx == 0, zero, 126 + (idx >> 1))
    mantf = torch.where(idx >= 2, idx & 1, zero)
    return ((expf.to(torch.int16) << 7) | (mantf.to(torch.int16) << 6)
            | (sign.to(torch.int16) << 15)).view(torch.bfloat16)


def _load_fp4_custom_op():
    """Load the custom fp4->fp8 TPC-kernel op, or return None if unavailable.

    The op extension (``hpu_custom_fp4_dequant*.so``) is loaded via
    ``torch.ops.load_library``; the TPC kernel library
    (``libfp4_dequant_kernels.so``) is loaded by the graph compiler via
    ``GC_KERNEL_PATH``, which must point at it *before* process start. Override
    the extension path with ``DSV4_FP4_OP_LIB``.
    """
    import glob
    pat = os.environ.get("DSV4_FP4_OP_LIB")
    if not pat:
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
        pat = os.path.join(root, "src", "custom_ops", "fp4_dequant",
                           "hpu_custom_fp4_dequant*.so")
    hits = glob.glob(pat)
    if not hits:
        return None
    try:
        torch.ops.load_library(hits[0])
        return torch.ops.custom_op.fp4_dequant
    except Exception:  # noqa: BLE001
        return None


_FP4_CUSTOM_OP = (_load_fp4_custom_op()
                   if os.environ.get("DSV4_FP4_CUSTOM", "0") == "1" else None)


_FP4_BLK = 64  # fp8 bytes per interleave chunk (32 packed fp4 bytes -> 64 fp8)


def _fp4_blk_perm(n_blk: int, device) -> torch.Tensor:
    """Block permutation for the kernel's native [o0 || o1] fp8 layout.

    The kernel emits o0 = even 64-byte chunks, o1 = odd chunks (avoiding a
    stack/DmaTranspose). A consumer that needs the target element order can
    un-permute the large weight, or permute the much smaller activation with
    this same permutation -- the GEMM sum is unchanged because both contraction
    operands are permuted identically.
    """
    p = torch.cat([torch.arange(0, n_blk, 2), torch.arange(1, n_blk, 2)])
    return p.to(device)


def _fp4_perm_blocks(x: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    """Apply the 64-element block permutation along the last dimension."""
    d = x.shape[-1]
    return x.reshape(*x.shape[:-1], d // _FP4_BLK, _FP4_BLK)[..., perm, :].reshape(*x.shape[:-1], d)


def _fp8_expert_gemm(x: torch.Tensor, wq: torch.Tensor) -> torch.Tensor:
    """Batched expert GEMM with scale-free fp8 weights.

    ``x`` [G, T, K] bf16, ``wq`` [G, N, K] fp8 (the fp4 kernel's scale-folded
    output, so no weight scale is needed). Returns [G, T, N] fp32. The activation
    is dynamically quantised per expert. ``fp8_gemm_v2`` is 2D-only, so loop over
    the (small) expert count.
    """
    outs = []
    for g in range(x.shape[0]):
        amax = x[g].abs().amax().clamp(min=1e-12)
        scale = (amax / 448.0).to(torch.float32)
        xq = torch.ops.hpu.cast_to_fp8_v2(
            x[g], (1.0 / scale).to(torch.float32), False, False, torch.float8_e4m3fn)[0]
        outs.append(torch.ops.hpu.fp8_gemm_v2(
            A=xq, trans_A=False, B=wq[g], trans_B=True, D=None,
            out_dtype=torch.bfloat16, A_scale_inv=scale, B_scale_inv=None,
            bias=None, accumulate=False))
    return torch.stack(outs, 0).float()


def _dequant_v4_fp4(blocks: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Dequantize MXFP4 packed expert weight to bf16.

    ``blocks`` [R, C] uint8: each byte holds two e2m1 nibbles (lo = even col,
    hi = odd col). ``scales`` [R, G], one per 32 columns: either raw uint8 e8m0
    exponent bytes (real scale = 2^(b-127)) or float8_e8m0fnu (the value is
    already the scale). Returns [R, 2*C] bf16, matching the reference
    (transformers dequants fp4 -> bf16 and computes in bf16).

    When the custom TPC kernel is available it decodes fp4 -> fp8 in one fused
    pass (exact; ~4x faster than the pure-Python decode), then this function
    casts to bf16 and applies the per-32 e8m0 scale.
    """
    G = scales.shape[-1]
    R, C = blocks.shape
    B = C // G  # bytes per group -> 32 values per group (B == 16)
    if _FP4_CUSTOM_OP is not None:
        # Fold the per-32 e8m0 scale (an exact power of two, 2^k) into the TPC
        # kernel: it adds k to the e4m3fn exponent field, with the subnormal
        # path exact for the checkpoint's scale range. Removes the separate
        # bf16 scale multiply + broadcast.
        if scales.dtype == torch.uint8:
            kexp = scales.to(torch.int16) - 127
        else:
            kexp = ((scales.view(torch.int32) >> 23) & 0xFF) - 127
        kexp = kexp.to(torch.int8).repeat_interleave(B, dim=1)  # [R, C]
        o0, o1 = _FP4_CUSTOM_OP(blocks, kexp)
        # The kernel emits the interleave in its native [o0 || o1] block order
        # (o0 = even 64-element chunks, o1 = odd) to avoid a stack/DmaTranspose;
        # the consumer permutes the much smaller activation to match. Stay in
        # fp8: the per-32 e8m0 scale is already folded into the exponent, so the
        # fp8 weights are scale-free and can feed the fp8 GEMM directly (no
        # fp8->bf16 cast, half the weight bytes).
        return torch.cat([o0, o1], dim=-1)
    if scales.dtype == torch.uint8:
        s = torch.pow(2.0, scales.float() - 127.0).to(torch.bfloat16).reshape(R, G, 1)
    else:
        s = scales.to(torch.bfloat16).reshape(R, G, 1)
    # Pure-Python fallback: build the bf16 bit pattern arithmetically. The
    # checkpoint stores packed fp4 as int8; view as uint8 so the nibble shifts
    # are LOGICAL (0..15), not arithmetic (which sign-extends the MSB).
    ni = blocks.view(torch.uint8).reshape(R, G, B)
    lo = _fp4_nibble_bf16(ni & 0x0F) * s
    hi = _fp4_nibble_bf16(ni >> 4) * s
    tgt = torch.stack([lo, hi], dim=-1).reshape(R, C * 2).contiguous()
    # Match the custom kernel's native [o0 || o1] block order.
    n_blk = (C * 2) // _FP4_BLK
    perm = _fp4_blk_perm(n_blk, tgt.device)
    return tgt.reshape(R, n_blk, _FP4_BLK)[:, perm, :].reshape(R, C * 2)


def _free_v4_fused_moe_routed(moe) -> None:
    """Free the bypassed FusedMoE's routed-expert weights to reclaim HBM.

    The routed experts are computed by the custom fp8 path, so the FusedMoE's
    packed fp4 ``w13_weight``/``w2_weight`` (and scales) are dead weight. This
    must happen BEFORE the (much larger) native-fp8 buffers are allocated so
    the two never coexist (fp8 ~2x fp4; together they exceed per-card HBM).
    """
    try:
        re_ = moe.experts.routed_experts
        for attr in ("w13_weight", "w2_weight", "w13_weight_scale",
                     "w2_weight_scale"):
            p = getattr(re_, attr, None)
            if p is not None and p.numel() > 0:
                setattr(re_, attr, torch.nn.Parameter(torch.empty(0), requires_grad=False))
    except Exception:
        pass


def _register_v4_fp8_placeholders(moe, config) -> None:
    """Register EMPTY native-fp8 routed-expert buffers at construction.

    Shapes come from the model config (``hidden_size``, ``moe_intermediate_size``)
    and the per-rank expert count, so they are known before any weights are
    read. Registering at construction means ``state_dict`` always carries them,
    so the snapshot pipeline (capture + fast restore) captures/restores them
    without re-reading the checkpoint. ``_register_v4_routed_packed`` fills them
    on fresh load; the dead FusedMoE routed weights are freed here first so the
    packed buffers and the fp4 FusedMoE copies never coexist in HBM.

    We store the checkpoint's native **packed fp4** (34 GB/rank) — NOT
    pre-converted fp8 (69 GB/rank, which does not fit alongside the rest of the
    model on 4x96 GB). The forward dequants only the active experts fp4 -> bf16
    with the sign-correct LUT (``_dequant_v4_fp4``).
    """
    if hasattr(moe, "packed_w1_weight"):
        return
    try:
        n_local = moe.experts_end_idx - moe.experts_start_idx
        hidden = config.hidden_size
        inter = config.moe_intermediate_size
    except Exception:
        return
    # Free the dead FusedMoE routed weights BEFORE allocating the packed buffers.
    _free_v4_fused_moe_routed(moe)
    moe.register_buffer("packed_w1_weight",
                        torch.empty(n_local, inter, hidden // 2, dtype=torch.uint8))
    moe.register_buffer("packed_w1_scale",
                        torch.empty(n_local, inter, hidden // 32, dtype=torch.float32))
    moe.register_buffer("packed_w3_weight",
                        torch.empty(n_local, inter, hidden // 2, dtype=torch.uint8))
    moe.register_buffer("packed_w3_scale",
                        torch.empty(n_local, inter, hidden // 32, dtype=torch.float32))
    moe.register_buffer("packed_w2_weight",
                        torch.empty(n_local, hidden, inter // 2, dtype=torch.uint8))
    moe.register_buffer("packed_w2_scale",
                        torch.empty(n_local, hidden, inter // 32, dtype=torch.float32))
    # Persisted ready flag: 0 until the packed buffers are filled (fresh load via
    # _register_v4_routed_packed, OR snapshot restore). Captured/restored by the
    # snapshot so restore does not rebuild the buffers.
    moe.register_buffer("_v4_fp8_ready", torch.zeros((), dtype=torch.int8))


def _register_v4_routed_packed(moe, dev) -> None:
    """Load packed fp4 routed experts into registered buffers at load time.

    Reads the checkpoint safetensors for all local experts and stacks the
    packed fp4 weights + e8m0 scales into registered buffers (34 GB/rank).
    Because this runs at load time (not lazily on first forward),
    ``moe.state_dict()`` carries them for the snapshot pipeline. The forward
    dequants only the active experts fp4 -> bf16 (``_dequant_v4_fp4``, which
    is sign-correct for the int8-packed bytes).

    Also frees the bypassed FusedMoE's routed weights to reclaim their HBM.
    """
    if hasattr(moe, "packed_w1_weight") and (
            getattr(moe, "_v4_fp8_filled", False)
            or bool(getattr(moe, "_v4_fp8_ready", torch.tensor(0)).item())):
        return

    import glob as _g
    from safetensors import safe_open
    import vllm_gaudi.models.deepseek_v4 as _MOD

    # Ensure the dead FusedMoE routed weights are freed before building.
    _free_v4_fused_moe_routed(moe)

    path = getattr(moe, "_v4_ckpt_path", None)
    lidx = getattr(moe, "_v4_layer_idx", -1)
    if getattr(_MOD, "_SHARD_MAP", None) is None:
        _MOD._SHARD_MAP = {}
        for f in _g.glob(path + "/model-*.safetensors"):
            with safe_open(f, framework="pt") as sf:
                for k in sf.keys():
                    _MOD._SHARD_MAP[k] = f

    def load(name):
        f = _MOD._SHARD_MAP.get(name)
        if f is None:
            raise KeyError(name)
        with safe_open(f, framework="pt") as sf:
            return sf.get_tensor(name)

    # e8m0 scales decode to their real float32 value (float8_e8m0fnu's .float()
    # IS the scale; raw uint8 bytes decode as 2^(b-127)).
    def _scale_float(t):
        if t.dtype == torch.float8_e8m0fnu:
            return t.float()
        if t.dtype == torch.uint8:
            return torch.pow(2.0, t.float() - 127.0)
        return t.float()

    start, end = moe.experts_start_idx, moe.experts_end_idx

    w1_list, s1_list, w3_list, s3_list, w2_list, s2_list = [], [], [], [], [], []
    for g in range(start, end):
        P = f"layers.{lidx}.ffn.experts.{g}."
        # Keep packed weights as uint8 bytes (bit-preserving int8 view).
        w1_list.append(load(P + "w1.weight").view(torch.uint8))
        s1_list.append(_scale_float(load(P + "w1.scale")))
        w3_list.append(load(P + "w3.weight").view(torch.uint8))
        s3_list.append(_scale_float(load(P + "w3.scale")))
        w2_list.append(load(P + "w2.weight").view(torch.uint8))
        s2_list.append(_scale_float(load(P + "w2.scale")))

    moe.register_buffer("packed_w1_weight", torch.stack(w1_list, dim=0))
    moe.register_buffer("packed_w1_scale", torch.stack(s1_list, dim=0))
    moe.register_buffer("packed_w3_weight", torch.stack(w3_list, dim=0))
    moe.register_buffer("packed_w3_scale", torch.stack(s3_list, dim=0))
    moe.register_buffer("packed_w2_weight", torch.stack(w2_list, dim=0))
    moe.register_buffer("packed_w2_scale", torch.stack(s2_list, dim=0))
    moe._v4_fp8_filled = True
    if hasattr(moe, "_v4_fp8_ready"):
        moe._v4_fp8_ready.fill_(1)


def _dequant_v4_fp4_to_fp8(
    blocks: torch.Tensor,
    scales: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dequant MXFP4 packed weights to bf16, then dynamic-quant to fp8.

    Handles both 2D [R, C] and 3D [B, R, C] inputs (batched experts).

    Args:
        blocks: uint8 tensor of packed fp4 nibbles, shape [R, C] or [B, R, C].
        scales: uint8 e8m0 exponent bytes, shape [R, G] or [B, R, G].

    Returns:
        fp8_weight: uint8 view of float8_e4m3fn, same shape as blocks but with
            the column dim doubled (since fp4 packing halves it).
        per_32_scales: float32 per-32-group scales (amax/448, then *0.5 for
            e4m3fnuz range), shape [R, G] or [B, R, G].
    """
    is_3d = blocks.dim() == 3
    if is_3d:
        B, R, C = blocks.shape
        G = scales.shape[-1]
        # view as uint8: logical nibble shifts (see _dequant_v4_fp4 docstring)
        blk = blocks.view(torch.uint8).reshape(B, R, G, C // G)
        s_raw = scales
    else:
        R, C = blocks.shape
        G = scales.shape[-1]
        blk = blocks.view(torch.uint8).reshape(R, G, C // G)
        s_raw = scales

    # --- Step 1: dequant fp4 -> bf16 (arithmetic bit-pattern decode) ---
    sub = torch.stack([_fp4_nibble_bf16(blk & 0x0F),
                       _fp4_nibble_bf16(blk >> 4)], dim=-1).reshape(*blk.shape[:-1], (C // G) * 2)

    # Decode e8m0 scales: real_scale = 2^(byte - 127)
    if s_raw.dtype == torch.uint8:
        s_val = torch.pow(2.0, s_raw.float() - 127.0).to(torch.bfloat16)
    else:
        s_val = s_raw.to(torch.bfloat16)

    if is_3d:
        sub = sub * s_val.unsqueeze(-1)
        bf16_weight = sub.reshape(B, R, C * 2).contiguous()
    else:
        sub = sub * s_val.unsqueeze(-1)
        bf16_weight = sub.reshape(R, C * 2).contiguous()

    # --- Step 2: dynamic quant bf16 -> fp8 with per-32-group scales ---
    # Reshape to [*, G, 32] for per-group amax and quantization
    if is_3d:
        bf16_view = bf16_weight.reshape(B, R, G, 32)
    else:
        bf16_view = bf16_weight.reshape(R, G, 32)

    # amax per group of 32 columns
    amax = bf16_view.abs().max(dim=-1, keepdim=True).values  # [*, G, 1]
    # Scale = amax / 448 (fp8 max), then *0.5 to fit e4m3fnuz (max 240)
    per_32_scales = (amax / 448.0) * 0.5  # [*, G, 1]
    # Avoid division by zero
    per_32_scales = per_32_scales.clamp(min=1e-12)

    # Quantize: bf16 / scale -> fp8 (reshape to [*, G, 32] so scale [*, G, 1] broadcasts)
    inv_scale = 1.0 / per_32_scales.to(torch.float32)
    fp8_view = torch.ops.hpu.cast_to_fp8_v2(
        bf16_view, inv_scale, False, False, torch.float8_e4m3fn
    )[0]

    # Return fp8 as float8_e4m3fn (HPU can store/decode it correctly for these
    # small values) + per-32 scales as float32 (squeezed last dim).
    return fp8_view.reshape(*bf16_weight.shape), per_32_scales.squeeze(-1).to(torch.float32)


def _make_block_scales(
    per_32_scales: torch.Tensor,
) -> torch.Tensor:
    """Convert per-32-column scales to [128, 128] block scales via max-pool.

    Groups 4 adjacent per-32 scales along the column dimension into 128-wide
    blocks. The row dimension is left as-is (each row is its own block).

    Args:
        per_32_scales: float32 [R, G] where G = num_32col_groups.

    Returns:
        block_scales: float32 [R_blocks, N_blocks] where R_blocks = R,
            N_blocks = G // 4 (each block covers 128 columns = 4 * 32).
    """
    R, G = per_32_scales.shape
    assert G % 4 == 0, f"G must be divisible by 4, got G={G}"
    # Reshape to [R, G//4, 4] and max over the last dim
    grouped = per_32_scales.reshape(R, G // 4, 4)
    block_scales = grouped.max(dim=-1).values  # [R, G//4]
    return block_scales


def _check_v4_routed_ready(moe) -> None:
    """Load-time invariant check; never runs in the compiled forward."""
    if not bool(getattr(moe, "_v4_fp8_ready", torch.tensor(0)).item()):
        raise RuntimeError(
            "DeepSeek V4 routed fp4 buffers not ready (expected pre-registration "
            "at load or snapshot restore); refusing to load from checkpoint at runtime."
        )


def _compute_v4_routed(moe, flat: torch.Tensor, topk_ids, topk_weights) -> torch.Tensor:
    """Compiled active-expert fp4 routed experts (pure torch, no built-in op).

    The built-in ``torch.ops.hpu.mixture_of_experts.mxfp4_fused_weights`` cannot
    express DSv4's activation ``silu(clamp(gate, max=limit)) * clamp(up, +-limit)``
    (the non-bias path supports silu/gelu only; the gpt-oss bias path hardcodes
    beta=1), so the MXFP4 logic is implemented here in pure compiled torch.

    Only the experts actually routed to are dequantized. Each token selects K
    experts, so at most ``T*K`` distinct local experts are active; we gather
    ``G = min(n_local, T*K)`` experts (a static, graph-safe bound — the
    ``_gather_swigluoai_moe`` pattern) and dequantize just those fp4 -> bf16.
    This replaces the previous all-expert per-forward dequant (n_local experts,
    ~3.2 GB/layer/forward). The expert MLP runs densely over the G gathered
    experts (static shapes; no ``[T*K, 2*inter, H]`` per-pair materialization).

    For each active expert:
      h = silu(clamp(x@w1^T, <=limit)) * clamp(x@w3^T, +-limit)
      out = h@w2^T * weight
    Accumulation is fp32 (cast to bf16 at the end).
    """
    # The packed fp4 routed buffers are prepared ONCE at load time
    # (_register_v4_routed_packed, called from load_weights) or restored from the
    # snapshot (_v4_fp8_ready=1). There is deliberately NO runtime-load fallback
    # here: a missing buffer means a load/restore bug and must fail loudly, never
    # re-read the checkpoint in the forward. The readiness check runs once in the
    # eager model forward (DeepseekV4HPUModel.forward) so its `.item()` host sync
    # never graph-breaks this compiled region.
    if moe.packed_w1_weight.device != flat.device:
        for _name in ("packed_w1_weight", "packed_w1_scale",
                      "packed_w2_weight", "packed_w2_scale",
                      "packed_w3_weight", "packed_w3_scale"):
            setattr(moe, _name, getattr(moe, _name).to(flat.device))

    start, end = moe.experts_start_idx, moe.experts_end_idx
    limit = float(getattr(moe, "swiglu_limit", 10.0) or 10.0)
    H = flat.shape[1]
    inter = moe.packed_w1_weight.shape[1]
    n_local = end - start
    T, K = topk_ids.shape

    # Per-(token, local-expert) combine weight, static shape, sync-free scatter
    # (mirrors _gather_swigluoai_moe). Non-local pairs are zeroed here.
    local_ids = topk_ids - start                       # [T, K] global -> local
    in_range = (local_ids >= 0) & (local_ids < n_local)
    safe_ids = torch.where(in_range, local_ids, torch.zeros_like(local_ids))
    gate_w = flat.new_zeros((T, n_local), dtype=torch.float32)
    gate_w.scatter_add_(
        1, safe_ids,
        torch.where(in_range, topk_weights, torch.zeros_like(topk_weights)).to(torch.float32))
    # Hit count per local expert; <= T*K distinct experts are active.
    hit = flat.new_zeros((n_local,), dtype=torch.float32)
    hit.scatter_add_(0, safe_ids.reshape(-1), in_range.reshape(-1).to(torch.float32))

    # Gather the (at most) T*K most-used local experts. Because at most T*K are
    # active, every routed expert is included; the arbitrary choice among the
    # zero-hit experts contributes nothing (gate_w == 0). Sorted ascending so the
    # accumulation order matches the reference's expert-index order.
    g = min(n_local, T * K)
    gather_ids = torch.topk(hit, g, sorted=False).indices
    gather_ids, _ = torch.sort(gather_ids)

    def _deq_gathered(packed, scale, idx):
        p = packed[idx]
        s = scale[idx]
        n = p.shape[0]
        R, C = p.shape[1], p.shape[-1]
        Gs = s.shape[-1]
        # Keep the natural dtype: fp8 (scale-free) from the custom kernel, bf16
        # from the pure-Python fallback. _compute_v4_routed branches on it.
        return _dequant_v4_fp4(p.reshape(-1, C), s.reshape(-1, Gs)).view(n, R, C * 2)

    # Keep w1/w3 separate: concatenating them along `inter` forces a transposed
    # DMA on HPU (the `ffn/cat` DmaTranspose); two bmms are cheaper than one
    # cat + one bmm.
    w1_g = _deq_gathered(moe.packed_w1_weight, moe.packed_w1_scale, gather_ids)  # [G, inter, H]
    w3_g = _deq_gathered(moe.packed_w3_weight, moe.packed_w3_scale, gather_ids)  # [G, inter, H]
    w2_g = _deq_gathered(moe.packed_w2_weight, moe.packed_w2_scale, gather_ids)  # [G, H, inter]

    # Dense expert MLP over the gathered experts (static shapes). The weights
    # arrive in the fp4 kernel's native [o0 || o1] 64-element block order, so
    # permute the much smaller activation identically (the GEMM sum is
    # unchanged) instead of un-permuting the large weight.
    perm_h = _fp4_blk_perm(H // _FP4_BLK, flat.device)
    xe = _fp4_perm_blocks(flat, perm_h).unsqueeze(0).expand(g, T, H)  # [G, T, H]
    perm_i = _fp4_blk_perm(inter // _FP4_BLK, flat.device)
    if w1_g.dtype == torch.float8_e4m3fn:
        # Scale-free fp8 weights -> fp8 expert GEMMs (no fp8->bf16 cast).
        gate = _fp8_expert_gemm(xe, w1_g)                  # [G, T, inter] fp32
        up = _fp8_expert_gemm(xe, w3_g)                    # [G, T, inter] fp32
        gate = torch.clamp(gate, max=limit)
        up = torch.clamp(up, min=-limit, max=limit)
        h = (torch.nn.functional.silu(gate) * up).to(flat.dtype)  # [G, T, inter]
        y = _fp8_expert_gemm(_fp4_perm_blocks(h, perm_i), w2_g)   # [G, T, H] fp32
    else:
        gate = torch.bmm(xe, w1_g.transpose(1, 2)).float()  # [G, T, inter] bf16 -> fp32
        up = torch.bmm(xe, w3_g.transpose(1, 2)).float()    # [G, T, inter] bf16 -> fp32
        gate = torch.clamp(gate, max=limit)
        up = torch.clamp(up, min=-limit, max=limit)
        h = (torch.nn.functional.silu(gate) * up).to(flat.dtype)  # [G, T, inter]
        y = torch.bmm(_fp4_perm_blocks(h, perm_i), w2_g.transpose(1, 2)).float()  # [G, T, H]
    gate_wg = gate_w.index_select(1, gather_ids).t().unsqueeze(-1)  # [G, T, 1]
    out = (y * gate_wg).sum(0)                             # [T, H] fp32
    return out.to(flat.dtype)


def _make_block_scales_batched(
    per_32_scales: torch.Tensor,
) -> torch.Tensor:
    """Batched version of _make_block_scales for [B, R, G] input."""
    B, R, G = per_32_scales.shape
    assert G % 4 == 0, f"G must be divisible by 4, got G={G}"
    grouped = per_32_scales.reshape(B, R, G // 4, 4)
    return grouped.max(dim=-1).values


def _dequant_block_fp8_naive(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    block_size_m: int = 1,
    block_size_n: int = 128,
) -> torch.Tensor:
    """Dequantize block-fp8 (e4m3fn) weight to bf16.

    Args:
        weight: fp8 (float8_e4m3fn) [M, N]
        weight_scale: float32 [M_blocks, N_blocks] where M_blocks = M // block_size_m,
            N_blocks = N // block_size_n
        block_size_m: row block size (1 for per-row blocks)
        block_size_n: column block size (128 for 128-col blocks)

    Returns:
        bf16 weight [M, N]
    """
    M, N = weight.shape
    M_blocks, N_blocks = weight_scale.shape
    w_view = weight.view(M_blocks, block_size_m, N_blocks, block_size_n)
    s_view = weight_scale.view(M_blocks, 1, N_blocks, 1)
    dequant = w_view.to(torch.bfloat16) * s_view.to(torch.bfloat16)
    return dequant.reshape(M, N).contiguous()


def _hpu_deepseek_v4_moe_forward(
    self: DeepseekV4MoE,
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """HPU DeepSeek V4 MoE forward (clean torch, reference-exact bf16)."""
    from vllm.distributed import tensor_model_parallel_all_reduce

    org_shape = hidden_states.shape
    flat = hidden_states.reshape(-1, self.hidden_size)

    router_logits = self.gate(flat)[0]
    scores = torch.sqrt(torch.nn.functional.softplus(router_logits))
    top_k = self.n_activated_experts
    if self.gate.tid2eid is not None:
        if input_ids is None:
            raise ValueError("DeepSeek V4 hash MoE routing requires input_ids.")
        topk_ids = self.gate.tid2eid[input_ids.reshape(-1)].long()
        topk_weights = scores.gather(1, topk_ids)
    else:
        bias = self.gate.e_score_correction_bias
        sel = scores if bias is None else scores + bias.to(scores.dtype)
        _, topk_ids = torch.topk(sel, top_k, dim=-1, sorted=False)
        topk_weights = scores.gather(1, topk_ids)
    topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
    topk_weights = topk_weights * self.routed_scaling_factor
    if _DUMP_ON:
        _lidx = getattr(self, "_v4_layer_idx", -1)
        _HPU_CAP.setdefault("router_logits", {})[_lidx] = router_logits.detach().float().clone()
        _HPU_CAP.setdefault("topk_ids", {})[_lidx] = topk_ids.detach().clone()
        _HPU_CAP.setdefault("topk_weights", {})[_lidx] = topk_weights.detach().float().clone()

    routed = _compute_v4_routed(self, flat, topk_ids, topk_weights)

    if self.tp_size > 1:
        routed = tensor_model_parallel_all_reduce(routed)
    if _DUMP_ON:
        _HPU_CAP.setdefault("routed_out", {})[getattr(self, "_v4_layer_idx", -1)] = \
            routed.detach().float().clone()
    if self.shared_experts is not None:
        _shared = self.shared_experts(flat)
        # The shared expert's RowParallelLinear down_proj is built with
        # reduce_results=False (upstream reduces routed+shared together inside the
        # FusedMoE). This custom forward bypasses the FusedMoE, so the shared
        # expert's TP partial must be all-reduced here -- otherwise every rank
        # contributes only its own shard (~1/tp_size of the full output).
        _dp = getattr(self.shared_experts, "down_proj", None)
        if self.tp_size > 1 and _dp is not None and not getattr(_dp, "reduce_results", True):
            _shared = tensor_model_parallel_all_reduce(_shared)
        if _DUMP_ON:
            _HPU_CAP.setdefault("shared_out", {})[getattr(self, "_v4_layer_idx", -1)] = \
                _shared.detach().float().clone()
        routed = routed.float() + _shared.float()
    final_hidden_states = routed.to(torch.bfloat16)

    return final_hidden_states.view(org_shape)


# Install the HPU MoE forward on the shared upstream class (module import applies
# the patch once). The top-level/mtp classes remain unregistered for now.
DeepseekV4MoE.forward = _hpu_deepseek_v4_moe_forward  # type: ignore[method-assign]


def _hpu_sel_deepseek_v4_mxfp4_moe_backend(config):
    """HPU: no MXFP4 MoE kernel. The clean-bf16 forward dequants fp4->bf16 and
    computes the routed experts itself (bypassing the FusedMoE), so the FusedMoE
    only needs to construct/load without a kernel -> no-op NONE backend."""
    from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import Mxfp4MoeBackend

    return Mxfp4MoeBackend.NONE, None


# The official DeepSeek-V4-Flash checkpoint has fp4 (MXFP4) routed experts; the
# CUDA/ROCm MXFP4 backend selector has no HPU candidate and would raise. Patch it
# to the no-op backend on HPU (see _hpu_deepseek_v4_moe_forward).
import vllm.model_executor.layers.quantization.mxfp4 as _mx
_mx.select_deepseek_v4_mxfp4_moe_backend = _hpu_sel_deepseek_v4_mxfp4_moe_backend

DeepSeekV4MTP = None
DSparkDeepseekV4ForCausalLM = None
