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
    """Weighted/weight-free RMSNorm over the last dim."""
    xf = x.float()
    out = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
    if weight is not None:
        out = out * weight.float()
    return out.to(x.dtype)


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
        # Window KV cache as ring buffer with tensor counter
        self._win_cache_device = self.fused_wqa_wkv.weight.device
        _ws = getattr(self, "window_size", 128)
        print(f"[HPU_ATTN] window_size from base={_ws}, forcing to 128", flush=True)
        self.window_size = 128
        self._win_cache = torch.zeros(
            self.window_size, self.head_dim,
            dtype=torch.bfloat16, device=self._win_cache_device,
        )
        self._win_n = torch.tensor(0, dtype=torch.int64, device=self._win_cache.device)
        self._decode_pos = torch.tensor(0, dtype=torch.int64, device=self._win_cache.device)

        # Compressor kernel state — flat tensor attributes, no dict
        vllm_config = kwargs.get("vllm_config", None) or (args[0] if args else None)
        self._hf_config = getattr(vllm_config, "model_config", None).hf_config
        self._compress_cache = None
        self._comp_coff = 2 if self.compress_ratio == 4 else 1
        self._cap = 0
        # Window buffers (allocated at first compressor use)
        self._c_win_kv = torch.empty(0)
        self._c_win_gate = torch.empty(0)
        self._c_win_n = torch.tensor(0, dtype=torch.int64)
        self._i_win_kv = torch.empty(0)
        self._i_win_gate = torch.empty(0)
        self._i_win_n = torch.tensor(0, dtype=torch.int64)
        # Compressed KV capacity buffers
        self._c_comp = torch.empty(0)
        self._c_n = torch.tensor(0, dtype=torch.int64)
        self._i_comp = torch.empty(0)
        self._i_n = torch.tensor(0, dtype=torch.int64)
        # Overlap buffers
        self._c_ovl_kv = torch.empty(0)
        self._c_ovl_gate = torch.empty(0)
        self._c_ovl_n = torch.tensor(0, dtype=torch.int64)
        self._i_ovl_kv = torch.empty(0)
        self._i_ovl_gate = torch.empty(0)
        self._i_ovl_n = torch.tensor(0, dtype=torch.int64)

        # Under the paged-KV flag the indexer k_cache must advertise an
        # HPU-friendly supported kernel block size (see HPUDeepseekV4IndexerBackend).
        from vllm_gaudi.v1.worker.dsv4_paged_kv import dsv4_paged_kv_enabled

        if dsv4_paged_kv_enabled() and getattr(self, "indexer", None) is not None:
            self.indexer.k_cache.get_attn_backend = lambda: HPUDeepseekV4IndexerBackend

    @classmethod
    def get_padded_num_q_heads(cls, num_heads: int) -> int:
        return num_heads

    def get_kv_cache_spec(self, vllm_config):
        from vllm_gaudi.v1.worker.dsv4_paged_kv import dsv4_paged_kv_enabled

        if dsv4_paged_kv_enabled():
            # R1: upstream-faithful spec — the compressed MLA cache for
            # compress_ratio > 1, None for the SWA-only layers (their cache is
            # the separately registered ``swa_cache`` module).
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



    def _make_compress_cache(self):
        """Build the yarn compress RoPE cos/sin cache once at init.

        Must be called OUTSIDE any compiled region (it constructs a transformers
        ``AutoConfig``/``DeepseekV4CSACompressor``). It is invoked once per layer
        from ``DeepseekV4HPUModel.load_weights`` after the weights are loaded.
        """
        if self.compressor is None or self.compress_ratio <= 1:
            return
        if self._compress_cache is not None:
            return
        rate = self.compress_ratio
        D = self.head_dim
        cap = 4096
        self._cap = cap
        dev = self.compressor.ape.device
        from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
            DeepseekV4CSACompressor,
        )
        from transformers import AutoConfig
        # vllm's hf_config (vllm.transformers_utils.configs.deepseek_v4) exposes
        # only the legacy compress_ratios and drops the transformers fields the
        # public CSA compressor reads (compress_rates, rope_parameters). Rebuild
        # a real transformers config from the checkpoint dir so its __post_init__
        # derives compress_rates/layer_types/rope_parameters as the CPU reference
        # does.
        ckpt = _v4_resolve_ckpt_dir(
            getattr(self._hf_config, "_name_or_path", "") or ""
        )
        pub_cfg = AutoConfig.from_pretrained(ckpt)
        pub = DeepseekV4CSACompressor(pub_cfg)
        inv = pub.rotary_emb.compress_inv_freq.float().to(dev)
        scale = pub.rotary_emb.compress_attention_scaling
        p = torch.arange(cap * rate + 16, dtype=torch.float32, device=dev)
        freqs = p.unsqueeze(1) * inv.unsqueeze(0)
        cos = freqs.cos() * scale
        sin = freqs.sin() * scale
        self._compress_cache = torch.cat([cos, sin], dim=-1).to(torch.bfloat16)
        idx_h = self.indexer.head_dim if (rate == 4 and self.indexer is not None) else D
        max_tokens = rate * 2 + 128  # prefill up to 128 tokens + leftover
        # The stored per-token kv/gate width is coff*D (CSA rate4 keeps two
        # streams -> 2*D; HCA keeps one -> D). Sizing by coff*D keeps the
        # window buffer width == the projected kv width for both layer types.
        cwid = self._comp_coff * D
        self._c_win_kv = torch.zeros(max_tokens, cwid, dtype=torch.bfloat16, device=dev)
        self._c_win_gate = torch.zeros(max_tokens, cwid, dtype=torch.bfloat16, device=dev)
        self._c_comp = torch.zeros(cap, D, dtype=torch.bfloat16, device=dev)
        self._c_ovl_kv = torch.zeros(rate, D, dtype=torch.bfloat16, device=dev)
        # gates are fp32 to match the reference position_bias precision
        self._c_ovl_gate = torch.zeros(rate, D, dtype=torch.float32, device=dev)
        self._i_win_kv = torch.zeros(max_tokens, 2 * idx_h, dtype=torch.bfloat16, device=dev)
        self._i_win_gate = torch.zeros(max_tokens, 2 * idx_h, dtype=torch.bfloat16, device=dev)
        self._i_comp = torch.zeros(cap, idx_h, dtype=torch.bfloat16, device=dev)
        self._i_ovl_kv = torch.zeros(rate, idx_h, dtype=torch.bfloat16, device=dev)
        self._i_ovl_gate = torch.zeros(rate, idx_h, dtype=torch.float32, device=dev)

    def _comp_rmsnorm(self, x, w):
        xf = x.float()
        out = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + self.eps)
        return (out * w.float()).to(x.dtype)

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

    def _comp_store_shift(self, kv_buf, gate_buf, n, kv_new, gate_new, rate):
        """Append new (kv, gate) tokens, return the window-aligned prefix to
        compress plus the leftover count, and compact the leftover to the front.

        The window MUST be captured BEFORE the leftover is moved to the front
        (otherwise it clobbers the first ``left`` window rows), and the gate
        buffer must be written alongside the kv buffer.
        """
        # FIXME(Step 5.5/6): host-sync the counter to a Python int; HPU eager
        # mis-handles tensor slice bounds (`buf[tensor:tensor]`). The paged-KV
        # rework removes this counter entirely.
        n = int(n) if torch.is_tensor(n) else n
        T = kv_new.shape[0]
        total = n + T
        if __import__("os").environ.get("V4_COMP_DEBUG") == "1":
            print(f"[store] n={n} T={T} kv_new={tuple(kv_new.shape)} "
                  f"buf={tuple(kv_buf.shape)} total={total}", flush=True)
        kv_buf[n:total] = kv_new
        gate_buf[n:total] = gate_new
        usable = (total // rate) * rate
        left = total - usable
        ck = kv_buf[:usable].clone()
        cg = gate_buf[:usable].clone()
        kv_buf[:left] = kv_buf[usable:total].clone()
        gate_buf[:left] = gate_buf[usable:total].clone()
        return usable // rate, left, ck, cg

    def _comp_windows(self, ck, cg, first_pos, Dk, rate, ape, norm_w,
                      cache, ovl_n, ovl_kv, ovl_gate, rope_dim):
        nw = ck.shape[0] // rate
        if nw == 0:
            return ck.new_zeros((0, Dk))
        ckv = ck.view(nw, rate, 2 * Dk)
        cgv = cg.view(nw, rate, 2 * Dk) + ape
        nk = ck.new_zeros(nw, 2 * rate, Dk)
        # Gates stay fp32 (position_bias is fp32 in the reference), so the
        # softmax sees the same values as the reference.
        ng = cgv.new_full((nw, 2 * rate, Dk), float("-inf"))
        nk[:, rate:] = ckv[:, :, Dk:].to(ck.dtype)
        ng[:, rate:] = cgv[:, :, Dk:]
        nk[1:, :rate] = ckv[:-1, :, :Dk].to(ck.dtype)
        ng[1:, :rate] = cgv[:-1, :, :Dk]
        nk[0, :rate] = torch.where(ovl_n > 0, ovl_kv[:rate].to(ck.dtype), nk[0, :rate])
        ng[0, :rate] = torch.where(ovl_n > 0, ovl_gate[:rate].to(ng.dtype), ng[0, :rate])
        # Match the reference: softmax in fp32, cast to the value dtype, then
        # product/sum in the value dtype.
        soft = torch.softmax(ng.float(), dim=1).to(nk.dtype)
        comp = (nk * soft).sum(dim=1)
        comp = self._comp_rmsnorm(comp, norm_w)
        pos = torch.arange(nw, device=ck.device) * rate + first_pos
        comp = _apply_rope(comp, pos, cache, rope_dim)
        return comp

    def _comp_windows_hca(self, ck, cg, first_pos, Dk, rate, ape, norm_w,
                          cache, rope_dim):
        nw = ck.shape[0] // rate
        if nw == 0:
            return ck.new_zeros((0, Dk))
        ckv = ck.view(nw, rate, Dk)
        cgv = cg.view(nw, rate, Dk) + ape
        # Match the reference: softmax fp32 -> value dtype, product/sum in value dtype.
        soft = torch.softmax(cgv.float(), dim=1).to(ckv.dtype)
        comp = (ckv * soft).sum(dim=1)
        comp = self._comp_rmsnorm(comp, norm_w)
        pos = torch.arange(nw, device=ck.device) * rate + first_pos
        comp = _apply_rope(comp, pos, cache, rope_dim)
        return comp

    def _comp_fill_overlap(self, chunk_kv, chunk_gate, Dk, rate, ape, ovl_kv, ovl_gate, ovl_n):
        nw = chunk_kv.shape[0] // rate
        # nw is a Python int; wrap it as a device tensor so torch.where is valid
        # in eager too (dynamo otherwise constant-folds the python bool).
        ovl_n.copy_(torch.where(
            chunk_kv.new_tensor(nw) > 0,
            torch.tensor(rate, dtype=torch.int64, device=ovl_n.device),
            torch.tensor(0, dtype=torch.int64, device=ovl_n.device)))
        # Pad chunk with rate dummy rows to make [-1] safe when usable<rate.
        # Only meaningful when at least one full window closed; when nw==0 the
        # chunk is empty and there is no overlap to carry (ovl_n already 0).
        if nw > 0:
            safe_kv = torch.cat([chunk_kv, chunk_kv.new_zeros(rate, 2 * Dk)], dim=0)
            safe_ga = torch.cat([chunk_gate + ape, chunk_gate.new_zeros(rate, 2 * Dk)], dim=0)
            last = safe_kv.view(-1, rate, 2 * Dk)[-1]
            lastg = safe_ga.view(-1, rate, 2 * Dk)[-1]
            ovl_kv[:rate] = last[:, :Dk]
            ovl_gate[:rate] = lastg[:, :Dk]

    @torch.compiler.disable
    def _compressor_compilable(self, hidden, qr, positions, is_prompt, real=None):
        # FIXME(Step 5.5/6): this method uses data-dependent tensor slice bounds
        # (`buf[n:total]` with tensor n/total), which miscompile inside the
        # compiled layer region (RuntimeError: expanded size 0 vs 1024). Run it
        # eagerly (graph break) until the paged-KV rework makes the compressor
        # fully graph-safe with fixed shapes / slot_mapping.
        T_full = hidden.shape[0]
        if real is not None:
            # Real-token selection (bool-mask indexing) must stay out of the
            # compiled region; it is safe here (eager, graph-disabled).
            hidden = hidden[real]
            qr = qr[real]
            positions = positions[real]
        if self.compressor is None or self.compress_ratio <= 1:
            return None, None, None, None
        rate = self.compress_ratio
        D = self.head_dim
        dev = hidden.device
        # Built once at load time (see _make_compress_cache); never build here.
        cache = self._compress_cache

        # Reset all state on prefill
        zero = torch.tensor(0, dtype=torch.int64, device=dev)
        self._c_win_n = torch.where(is_prompt > 0, zero, self._c_win_n)
        self._c_n = torch.where(is_prompt > 0, zero, self._c_n)
        self._c_ovl_n = torch.where(is_prompt > 0, zero, self._c_ovl_n)
        self._i_win_n = torch.where(is_prompt > 0, zero, self._i_win_n)
        self._i_n = torch.where(is_prompt > 0, zero, self._i_n)
        self._i_ovl_n = torch.where(is_prompt > 0, zero, self._i_ovl_n)

        c = self.compressor
        fused = self._comp_weight(c.fused_wkv_wgate, hidden.shape[-1])
        coff = self._comp_coff
        kv_out = coff * D
        kv_w = fused[:kv_out]
        gate_w = fused[kv_out:]
        # position_bias stays fp32 (matches the reference; bf16 would round the
        # softmax logits differently).
        ape = c.ape
        norm_w = c.norm.weight

        if __import__("os").environ.get("V4_COMP_DEBUG") == "1":
            print(f"[comp] hidden={tuple(hidden.shape)} qr={tuple(qr.shape)} "
                  f"c_win_n={self._c_win_n} rate={rate}", flush=True)
        kv = hidden @ kv_w.t()
        gate = hidden @ gate_w.t()
        _nw_c, left_c, ck, cg = self._comp_store_shift(
            self._c_win_kv, self._c_win_gate, self._c_win_n, kv, gate, rate)
        self._c_win_n = torch.tensor(left_c, dtype=torch.int64, device=dev)
        if rate == 4:
            comp = self._comp_windows(ck, cg, self._c_n * rate, D, rate, ape,
                                      norm_w, cache, self._c_ovl_n,
                                      self._c_ovl_kv, self._c_ovl_gate, self.rope_head_dim)
        else:
            comp = self._comp_windows_hca(ck, cg, self._c_n * rate, D, rate, ape,
                                          norm_w, cache, self.rope_head_dim)
        self._c_comp[self._c_n:self._c_n + comp.shape[0]] = comp
        self._c_n += comp.shape[0]
        if rate == 4:
            self._comp_fill_overlap(ck, cg, D, rate, ape,
                                    self._c_ovl_kv, self._c_ovl_gate, self._c_ovl_n)
        compressed_kv_full = self._c_comp

        # ---- indexer (only CSA, ratio 4) ----
        has_idx = rate == 4 and self.indexer is not None
        if has_idx:
            idx_h = self._i_comp.shape[-1]
            idx = self.indexer
            ifused = self._comp_weight(idx.compressor.fused_wkv_wgate, hidden.shape[-1])
            ikv_w = ifused[: 2 * idx_h]
            igate_w = ifused[2 * idx_h:]
            iape = idx.compressor.ape
            inorm_w = idx.compressor.norm.weight
            ikv = hidden @ ikv_w.t()
            igate = hidden @ igate_w.t()
            _nw_i, left_i, ik, ig = self._comp_store_shift(
                self._i_win_kv, self._i_win_gate, self._i_win_n, ikv, igate, rate)
            self._i_win_n = torch.tensor(left_i, dtype=torch.int64, device=dev)
            icomp = self._comp_windows(ik, ig, self._i_n * rate, idx_h, rate, iape,
                                       inorm_w, cache, self._i_ovl_n,
                                       self._i_ovl_kv, self._i_ovl_gate, self.rope_head_dim)
            self._i_comp[self._i_n:self._i_n + icomp.shape[0]] = icomp
            self._i_n += icomp.shape[0]
            self._comp_fill_overlap(ik, ig, idx_h, rate, iape,
                                    self._i_ovl_kv, self._i_ovl_gate, self._i_ovl_n)
            idx_compressed = self._i_comp
        else:
            idx_h = D
            idx_compressed = torch.zeros(0, D, dtype=torch.bfloat16, device=dev)

        # ---- scorer / topk / block_bias ----
        T = hidden.shape[0]
        # clen is used as a python int (slice bounds, arange, min(), topk k,
        # full_like fill). The underlying counter is a 0-dim tensor for the
        # graph-compilable parts; materialize it here for the data-dependent
        # topk logic, which is correct in eager (== correct everywhere).
        clen = int(self._c_n)
        cap = self._cap
        if has_idx and clen > 0:
            nhead = self.indexer.n_head
            qb_w = self.indexer.wq_b.weight.detach()
            if qb_w.dtype == torch.float8_e4m3fn:
                # indexer.wq_b is block-fp8 that force-channel-fp8 converted to
                # per-row fp8. Dequant correctly on HPU for the scale rank that
                # is actually stored: per-row (1-D) or block (2-D).
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
            scores = torch.matmul(q.float(), idx_compressed[:clen].float().t())
            scores = torch.relu(scores) * ss
            w = (hidden @ wp.t()).float() * ws
            scores = (scores * w.unsqueeze(-1)).sum(dim=1)
            causal_threshold = (positions + 1) // rate
            entry = torch.arange(clen, device=scores.device)
            future = entry.unsqueeze(0) >= causal_threshold.unsqueeze(-1)
            scores = scores.masked_fill(future, float("-inf"))
            k = min(self.indexer.topk_tokens, clen)
            topk = scores.topk(k, dim=-1).indices
            invalid = topk >= causal_threshold.unsqueeze(-1)
            topk = torch.where(invalid, torch.full_like(topk, -1), topk)
            valid = topk >= 0
            safe = torch.where(valid, topk, torch.full_like(topk, clen))
            bb = compressed_kv_full.new_full((1, 1, T, cap + 1), float("-inf"))
            safe_cap = torch.where(valid, topk, torch.full_like(topk, cap))
            bb.scatter_(-1, safe_cap.unsqueeze(0).unsqueeze(0), 0.0)
            block_bias = bb[..., :cap][0, 0]
        elif clen > 0 and T > 1:
            causal = (positions + 1) // rate
            entry = torch.arange(cap, device=dev)
            block_bias = compressed_kv_full.new_full((T, cap), float("-inf"))
            block_bias.masked_fill_(
                (entry.unsqueeze(0) < causal.unsqueeze(-1)) & (entry.unsqueeze(0) < clen),
                0.0)
            topk = torch.empty(T, 0, dtype=torch.long, device=dev)
        else:
            block_bias = None
            topk = torch.empty(T, 0, dtype=torch.long, device=dev)
        if real is not None and block_bias is not None and T_full != T:
            # The compressor ran on the real tokens only ([nreal, cap]); scatter
            # the per-query block_bias back to the padded batch so the caller
            # (compiled forward_prefill) gets a static [T, cap] mask without
            # data-dependent slicing.
            bb_full = block_bias.new_full((T_full, block_bias.shape[-1]), float("-inf"))
            bb_full[real] = block_bias
            block_bias = bb_full
        return compressed_kv_full, self._c_n, block_bias, topk

    def _run_compressor(self, hidden_states, q_residual, positions, is_prompt, real=None):
        """Run the graph-compilable CSA compressor/indexer over real tokens.
        Returns (compressed_kv_full [CAP,D], compressed_n, block_bias [T,CAP] or None, _).
        ``real`` is the optional real-token mask (prefill); it is applied inside
        the compiler-disabled compressor so no bool indexing hits the graph."""
        if self.compressor is None or self.compress_ratio <= 1:
            D = self.head_dim
            dev = hidden_states.device
            return (torch.zeros(0, D, dtype=torch.bfloat16, device=dev),
                    torch.tensor(0, dtype=torch.int64, device=dev), None, None)
        ckv_full, c_n, bb, _ = self._compressor_compilable(
            hidden_states, q_residual, positions, is_prompt, real)
        if ckv_full is None:
            cap = max(self._cap, 1)
            dev = hidden_states.device
            D = self.head_dim
            return (torch.zeros(cap, D, dtype=torch.bfloat16, device=dev),
                    torch.tensor(0, dtype=torch.int64, device=dev), None, None)
        return ckv_full, c_n, bb, None

    def _o_proj(self, o: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # o: [T, n_local_heads, head_dim] -- inverse RoPE + wo_a (group bmm) + wo_b
        _diag("op_invrope_in", o)
        o = _apply_inv_rope(
            o, positions, self.rotary_emb.cos_sin_cache, self.rope_head_dim
        )
        _diag("op_after_invrope", o)
        if __import__("os").environ.get("HPU_DUMP") == "1":
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
        if __import__("os").environ.get("HPU_DUMP") == "1":
            _HPU_CAP.setdefault("z_einsum", {})[getattr(self, "_layer_idx", -1)] = z.detach().float().clone()
        if __import__("os").environ.get("HPU_DUMP") == "1":
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

    def _forward_compilable(
        self,
        qr: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        past_kv: torch.Tensor | None,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """From normed qr+kv through wq_b -> RoPE -> attention -> O projection.

        ``qr`` and ``kv`` are already normed by the eager outer (fused_wqa_wkv
        stays in eager because the compressor needs intermediate ``qr``). No
        Python control flow, no env var checks — safe for ``torch.compile``.

        Args:
            qr: ``[T, q_lora_rank]`` normed Q residual.
            kv: ``[T, head_dim]`` normed KV.
            positions: ``[T]`` absolute positions.
            past_kv: ``[K, n_heads, D]`` past KV (window + compressed).
            mask: ``[T, K]`` bool mask or ``None`` for decode.

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

        # past_kv already includes ALL keys (window + compressed from the eager
        # outer); do NOT append cur_kv here (decode should not attend to self).
        # FusedSDPA path: fp32 QK^T + fp32 softmax, bf16 probs -> bf16 AV,
        # matching the reference kernel precision (verified at T=1..512,
        # K up to window+compressed cap).
        o = _attn_with_sink_fsdpa(
            q, past_kv, mask, self.attn_sink[: self.n_local_heads], self.scale
        )

        o = self._o_proj_compilable(o, positions)
        return o



    def _fused_qkv(self, hidden_states):
        """Shared preamble: fused_wqa_wkv → split → rmsnorm → rope."""
        qr_kv = self.fused_wqa_wkv(hidden_states)
        qr_kv = qr_kv[0] if isinstance(qr_kv, tuple) else qr_kv
        qr, kv = qr_kv.split([self.q_lora_rank, self.head_dim], dim=-1)
        qr = _rmsnorm(qr, self.eps, self.q_norm.weight.data)
        kv = _rmsnorm(kv, self.eps, self.kv_norm.weight.data)
        return qr, kv

    def _write_paged_swa_cache(self, kv_roped):
        """Write roped KV into the paged SWA cache (VLLM_DSV4_PAGED_KV).

        Uses a contiguous flat view of the packed slab (stashed by the model
        runner) so the scatter avoids strided views, which HPU's graph compiler
        rejects.

        The SWA cache is a strided view into the packed slab, so scatter by
        (block, offset) in place using the group's slot_mapping from the HPU
        attention metadata.
        """
        from vllm.forward_context import get_forward_context

        md = get_forward_context().attn_metadata
        aux = getattr(md, "dsv4_aux_meta", None)
        if not aux:
            return
        gid = getattr(self.swa_cache_layer, "_kv_cache_gid", None)
        if gid is None or gid not in aux:
            return
        slab = getattr(self.swa_cache_layer, "_dsv4_flat_slab", None)
        if slab is None:
            return
        info = aux[gid]
        slots = info["slot_mapping"]
        bs = info["block_size"]
        head_dim = self.head_dim
        kv = kv_roped.reshape(-1, head_dim)
        T = kv.shape[0]
        blocks = torch.div(slots, bs, rounding_mode="floor")
        offs = slots % bs
        # Flat scatter into the contiguous slab: element index =
        # layer_offset + block*block_stride_elems + off*head_dim + d.
        base = (self.swa_cache_layer._dsv4_flat_offset + blocks * self.swa_cache_layer._dsv4_flat_stride +
                offs * head_dim)
        idx = base.unsqueeze(1) + torch.arange(head_dim, device=kv.device, dtype=slots.dtype).unsqueeze(0)
        slab[idx] = kv

    def forward_prefill(self, positions, hidden_states):
        """Prefill across a (possibly padded) prefill batch.

        vllm pads the prefill to a fixed length; only rows with position >= 0
        are real tokens and they are the leading ``[0:nreal)`` rows. The window
        cache keeps the last ``min(nreal, window)`` real kv rows; the compressor
        sees only real tokens. The query side (qr/kv/positions) stays the full
        padded batch so the returned hidden has the shape vllm expects; padding
        query rows are masked to attend every slot so their (discarded) output
        stays finite.

        Graph-safe: no ``int(tensor)`` host sync, no data-dependent Python
        control flow, no bool-mask indexing in the compiled region. The real
        token count is a tensor and the real-token selection happens inside the
        compiler-disabled compressor.
        """
        qr, kv = self._fused_qkv(hidden_states)
        kv_roped = _apply_rope(kv, positions, self.rotary_emb.cos_sin_cache, self.rope_head_dim)
        self._write_paged_swa_cache(kv_roped)
        T = kv_roped.shape[0]
        pos_flat = positions.reshape(-1)
        real = pos_flat >= 0
        dev = kv_roped.device
        W = self.window_size
        D = self.head_dim

        # Window cache = the last min(nreal, W) real kv rows. Static-shape
        # gather (fixed [W, D] output, dynamic index values): slot j holds the
        # token at position ``start + j`` where start = max(0, nreal - W).
        nreal = real.long().sum()
        start = torch.clamp(nreal - W, min=0)
        win_idx = (start + torch.arange(W, device=dev)).clamp(max=T - 1)
        self._win_cache.copy_(kv_roped.gather(0, win_idx[:, None].expand(-1, D)))
        self._win_n.copy_(torch.clamp(nreal, max=W))
        # decode appends at the next free ring slot = number of real tokens seen.
        self._decode_pos.copy_(nreal)

        is_prompt = torch.tensor(1, dtype=torch.int64, device=dev)
        comp_full, comp_n, comp_bb, _topk = self._run_compressor(
            hidden_states, qr, pos_flat, is_prompt, real)

        cap = self._cap
        past_kv = torch.cat([self._win_cache, comp_full], dim=0)
        past_kv = past_kv.unsqueeze(1).expand(-1, self.n_local_heads, self.head_dim)

        # Window mask: real query rows attend present, causal window slots;
        # padding query rows attend every slot (finite, discarded).
        win_ids = torch.arange(W, device=dev)
        win_present = win_ids < self._win_n
        win_pos = start + win_ids
        win_mask = (win_pos[None, :] <= pos_flat[:, None]) & win_present[None, :]
        win_mask = win_mask | (~real)[:, None]

        # Compressed mask: real rows attend present/finite compressed entries;
        # padding rows attend every entry.
        comp_ids = torch.arange(cap, device=dev)
        comp_present = comp_ids[None, :] < comp_n.reshape(-1)[None, :]  # [1, CAP]
        if comp_bb is not None:
            comp_valid = comp_present & torch.isfinite(comp_bb[:, :cap])
        else:
            comp_valid = comp_present.expand(T, -1)
        comp_mask = comp_valid | (~real)[:, None]
        mask = torch.cat([win_mask, comp_mask], dim=-1)

        return self._forward_compilable(qr, kv, positions, past_kv, mask)

    def forward_decode(self, positions, hidden_states):
        """Decode: append 1 token to ring buffer, run compressor (no reset), full mask."""
        # Decode may pass a 1-D [H] hidden; the compressor/rope assume [T, H].
        if hidden_states.dim() == 1:
            hidden_states = hidden_states.unsqueeze(0)
        qr, kv = self._fused_qkv(hidden_states)
        kv_roped = _apply_rope(kv, positions, self.rotary_emb.cos_sin_cache, self.rope_head_dim)
        self._write_paged_swa_cache(kv_roped)
        # Ring-buffer write via index_copy_ with an explicit [1] slot index:
        # graph-safe (fixed shapes), unlike `buf[tensor] = value`.
        slot = (self._decode_pos % self.window_size).reshape(1)
        self._win_cache.index_copy_(0, slot, kv_roped.reshape(1, -1))
        self._decode_pos.add_(1)
        self._win_n.add_(1).clamp_(max=self.window_size)

        pos_flat = positions.reshape(-1)
        is_prompt = torch.tensor(0, dtype=torch.int64, device=pos_flat.device)
        comp_full, comp_n, comp_bb, _topk = self._run_compressor(hidden_states, qr, pos_flat, is_prompt)

        cap = self._cap
        past_kv = torch.cat([self._win_cache, comp_full], dim=0)
        past_kv = past_kv.unsqueeze(1).expand(-1, self.n_local_heads, self.head_dim)

        # Decode attends its real past context in the window: the first _win_n
        # slots (ring not yet full) or all slots once the ring is full.
        win_ids = torch.arange(self.window_size, device=pos_flat.device)
        win_present = (win_ids < self._win_n).unsqueeze(0)  # [1, WIN]
        comp_ids = torch.arange(cap, device=pos_flat.device)
        comp_present = comp_ids.unsqueeze(0) < comp_n.unsqueeze(-1)
        mask = torch.cat([win_present, comp_present], dim=-1)

        return self._forward_compilable(qr, kv, positions, past_kv, mask)

    def forward(self, positions, hidden_states, llama_4_scaling=None):
        return self.forward_decode(positions, hidden_states)


_HPU_CAP: dict = {"attn_in": {}, "attn_out": {}}
_DUMP_ON = __import__("os").environ.get("HPU_DUMP") == "1"
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
                    self.hc_mult, self.attn_norm.weight.data, self.rms_norm_eps,
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
            self.attn_norm.weight.data, self.rms_norm_eps,
        )

    def _mhc_post_ffn(self, x, residual, post_mix, res_mix):
        return _mhc_fused_post_pre_compilable(
            x, residual, post_mix, res_mix,
            self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base,
            self.rms_norm_eps, self.hc_eps, self.hc_eps,
            self.hc_post_alpha, self.hc_sinkhorn_iters, self.hc_mult,
            self.ffn_norm.weight.data, self.rms_norm_eps,
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

        # Reconstruct the multi-stream [T, hc_mult, H] hidden state from the
        # 2D ffn output + the mHC residual/mixes, then collapse it via hc_head.
        # einsum-free HPU version (upstream mhc_post_torch uses torch.einsum,
        # which is not graph-compilable on hpu_backend).
        hidden_states = _mhc_post_compilable(hidden_states, residual, post_mix, res_mix)
        if _dbg:
            print(f"[v4model] after mhc_post shape={tuple(hidden_states.shape)}", flush=True)
        hidden_states = _hc_head(
            hidden_states,
            self.hc_head_fn,
            self.hc_head_scale,
            self.hc_head_base,
            self.rms_norm_eps,
            self.hc_eps,
        )
        if _dbg:
            print(f"[v4model] after hc_head shape={tuple(hidden_states.shape)}", flush=True)
        # Use the manual fp32 RMSNorm instead of the vLLM RMSNorm module.
        # HPURMSNorm.forward_oot calls habana FusedRMSNorm -> torch.ops.hpu.rms_norm,
        # which FAILS to compile with force_static_compile on this stack
        # (synStatus 26 "Can not compile graph", reproducible standalone; the
        # rms_norm_fast path compiles but is inaccurate). The manual
        # decomposition is what every layer norm already uses and compiles.
        hidden_states = _rmsnorm(hidden_states, self.rms_norm_eps, self.norm.weight)
        if _dbg:
            print(f"[v4model] after norm shape={tuple(hidden_states.shape)}", flush=True)
        if hidden_shape is not None and len(hidden_shape) == 3:
            hidden_states = hidden_states.view(*hidden_shape[:2], self.config.hidden_size)
        return hidden_states

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
        """Build every layer's compressor RoPE cache at load time.

        ``_make_compress_cache`` constructs a transformers config/module, so it
        is graph-unsafe and must run outside the compiled forward (Step 4.5).
        """
        n = 0
        for layer in self.layers[self.start_layer:self.end_layer]:
            attn = getattr(layer, "attn", None)
            if attn is not None and hasattr(attn, "_make_compress_cache"):
                attn._make_compress_cache()
                if getattr(attn, "_compress_cache", None) is not None:
                    n += 1
        print(f"[v4] built compress caches for {n} layers", flush=True)

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


_FP4_VALUES = (
    +0.0, +0.5, +1.0, +1.5, +2.0, +3.0, +4.0, +6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
)


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


def _dequant_v4_fp4(blocks: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Dequantize MXFP4 packed expert weight to bf16.

    ``blocks`` [R, C] uint8: each byte holds two e2m1 nibbles (lo = even col,
    hi = odd col). ``scales`` [R, G], one per 32 columns: either raw uint8 e8m0
    exponent bytes (real scale = 2^(b-127)) or float8_e8m0fnu (the value is
    already the scale). Returns [R, 2*C] bf16, matching the reference
    (transformers dequants fp4 -> bf16 and computes in bf16).
    """
    G = scales.shape[-1]
    R, C = blocks.shape
    B = C // G  # bytes per group -> 32 values per group (B == 16)
    # The checkpoint stores packed fp4 as int8; view as uint8 so the nibble
    # shifts below are LOGICAL (0..15), not arithmetic (which sign-extends the
    # MSB and maps ~50% of bytes to negative LUT indices, destroying the
    # negative fp4 values).
    blk = blocks.view(torch.uint8).reshape(R, G, B)
    lut = torch.tensor(_FP4_VALUES, dtype=torch.bfloat16, device=blocks.device)
    lo = (blk & 0x0F).to(torch.long)
    hi = (blk >> 4).to(torch.long)
    sub = torch.empty(R, G, B * 2, dtype=torch.bfloat16, device=blocks.device)
    sub[:, :, 0::2] = lut[lo]
    sub[:, :, 1::2] = lut[hi]
    if scales.dtype == torch.uint8:
        s = torch.pow(2.0, scales.float() - 127.0).to(torch.bfloat16).reshape(R, G, 1)
    else:
        s = scales.to(torch.bfloat16).reshape(R, G, 1)
    sub = sub * s
    return sub.reshape(R, C * 2).contiguous()


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

    # --- Step 1: dequant fp4 -> bf16 ---
    lut = torch.tensor(_FP4_VALUES, dtype=torch.bfloat16, device=blocks.device)
    lo = (blk & 0x0F).to(torch.long)
    hi = (blk >> 4).to(torch.long)
    if is_3d:
        sub = torch.empty(B, R, G, (C // G) * 2, dtype=torch.bfloat16, device=blocks.device)
        sub[:, :, :, 0::2] = lut[lo]
        sub[:, :, :, 1::2] = lut[hi]
    else:
        sub = torch.empty(R, G, (C // G) * 2, dtype=torch.bfloat16, device=blocks.device)
        sub[:, :, 0::2] = lut[lo]
        sub[:, :, 1::2] = lut[hi]

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


@torch.compiler.disable
def _check_v4_routed_ready(moe) -> None:
    """Load-time invariant check, kept out of the compiled MoE graph."""
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
    # re-read the checkpoint in the forward. The check is compiler-disabled so
    # its `.item()` host sync never graph-breaks the compiled MoE region.
    _check_v4_routed_ready(moe)
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
        return _dequant_v4_fp4(p.reshape(-1, C), s.reshape(-1, Gs)).view(n, R, C * 2).to(flat.dtype)

    w13_g = torch.cat([
        _deq_gathered(moe.packed_w1_weight, moe.packed_w1_scale, gather_ids),
        _deq_gathered(moe.packed_w3_weight, moe.packed_w3_scale, gather_ids),
    ], dim=1)  # [G, 2*inter, H]
    w2_g = _deq_gathered(moe.packed_w2_weight, moe.packed_w2_scale, gather_ids)  # [G, H, inter]

    # Dense expert MLP over the gathered experts (static shapes).
    xe = flat.unsqueeze(0).expand(g, T, H)                 # [G, T, H] view
    gu = torch.bmm(xe, w13_g.transpose(1, 2))              # [G, T, 2*inter] bf16
    gate, up = gu[..., :inter].float(), gu[..., inter:].float()
    gate = torch.clamp(gate, max=limit)
    up = torch.clamp(up, min=-limit, max=limit)
    h = (torch.nn.functional.silu(gate) * up).to(flat.dtype)  # [G, T, inter]
    y = torch.bmm(h, w2_g.transpose(1, 2)).float()         # [G, T, H] fp32
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
