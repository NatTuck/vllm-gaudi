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

from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4CSACompressor,
    DeepseekV4HCACompressor,
)

_DIAG_DONE = {"v": False}


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
    """GPT-J (interleaved-pair) RoPE applied to the LAST rope_head_dim dims."""
    cache = cos_sin_cache.to(x.dtype).index_select(0, positions)  # [T, 2*half]
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
    r0 = x0 * c - x1 * s
    r1 = x0 * s + x1 * c
    rot = torch.stack([r0, r1], dim=-1).reshape(shape)
    return torch.cat([x_pass, rot], dim=-1)


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



def _apply_inv_rope(
    x: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    rope_head_dim: int,
) -> torch.Tensor:
    cache = cos_sin_cache.to(x.dtype).index_select(0, positions)
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
    r0 = x0 * c + x1 * s
    r1 = -x0 * s + x1 * c
    rot = torch.stack([r0, r1], dim=-1).reshape(shape)
    return torch.cat([x_pass, rot], dim=-1)


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
    if __import__("os").environ.get("HPU_DUMP") == "1" and _MHC_PHASE == "ffn":
        _HPU_CAP.setdefault("mhc_ffn_prenorm", {})[_MHC_LAYER] = layer_input_cur.detach().float().clone()
    if norm_weight is not None:
        layer_input_cur = _rmsnorm(layer_input_cur, norm_eps, norm_weight)
    return (
        residual_cur,
        post_mix_cur.view(*outer, hc_mult, 1),
        comb_mix_cur.view(*outer, hc_mult, hc_mult),
        layer_input_cur.view(*outer, H),
    )


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
    out = torch.einsum("tm,tmh->th", pre, hs_flat.float()).to(hs_flat.dtype)
    return out


class _DSV4CompressionState:
    """Port of the private ``DeepseekV4HCACache`` / ``DeepseekV4CSACache`` state
    (the internal buffer / compressed / overlap bookkeeping the public
    compressor & indexer modules drive via their ``past_key_values`` arg).

    State is keyed by entry name ("compressor" / "indexer"): each holds the
    pending source tokens between windows (``buffer_*``), the running list of
    compressed KV entries emitted so far (``compressed_kv``), and how many
    windows have closed (``entry_count``). CSA additionally carries per-name
    overlap state for the two-series (Ca/Cb) window scheme. This replaces the
    private methods ``store_compression_weights`` / ``update_compressor_states``
    / ``update_overlap_state``.
    """

    def __init__(self, config, compress_rate: int):
        self.config = config
        self.compress_rate = compress_rate
        self.buffer_kv: dict[str, torch.Tensor | None] = {"compressor": None, "indexer": None}
        self.buffer_gate: dict[str, torch.Tensor | None] = {"compressor": None, "indexer": None}
        self.compressed_kv: dict[str, torch.Tensor | None] = {"compressor": None, "indexer": None}
        self.entry_count: dict[str, int] = {"compressor": 0, "indexer": 0}
        self.overlap_kv: dict[str, torch.Tensor | None] = {"compressor": None, "indexer": None}
        self.overlap_gate: dict[str, torch.Tensor | None] = {"compressor": None, "indexer": None}

    def store_compression_weights(self, name, kv, gate):
        first_window_position = self.entry_count[name] * self.compress_rate
        buffered_kv, buffered_gate = self.buffer_kv[name], self.buffer_gate[name]
        if buffered_kv is not None and buffered_kv.shape[1]:
            kv = torch.cat([buffered_kv, kv], dim=1)
            gate = torch.cat([buffered_gate, gate], dim=1)
        usable = (kv.shape[1] // self.compress_rate) * self.compress_rate
        self.buffer_kv[name], self.buffer_gate[name] = kv[:, usable:], gate[:, usable:]
        return kv[:, :usable], gate[:, :usable], first_window_position

    def update_compressor_states(self, name, compressed):
        if self.compressed_kv[name] is None:
            self.compressed_kv[name] = compressed
        elif compressed.shape[1] > 0:
            self.compressed_kv[name] = torch.cat([self.compressed_kv[name], compressed], dim=1)
        self.entry_count[name] += compressed.shape[1]
        return self.compressed_kv[name]

    def update_overlap_state(self, name, chunk_kv, chunk_gate, head_dim):
        prior_kv, prior_gate = self.overlap_kv[name], self.overlap_gate[name]
        self.overlap_kv[name] = chunk_kv[:, -1, :, :head_dim].clone()
        self.overlap_gate[name] = chunk_gate[:, -1, :, :head_dim].clone()
        return prior_kv, prior_gate

    def reset(self):
        for d in (self.buffer_kv, self.buffer_gate, self.compressed_kv, self.overlap_kv, self.overlap_gate):
            for k in d:
                d[k] = None
        for k in self.entry_count:
            self.entry_count[k] = 0


class _DSV4PkvShim:
    """Minimal stand-in for ``past_key_values`` so the public compressor/indexer
    modules can reach a single layer's ported state via
    ``past_key_values.layers[layer_idx]``."""

    def __init__(self, layer_idx: int, state: _DSV4CompressionState):
        self._layer_idx = layer_idx
        self._state = state
        self.layers = [None] * (layer_idx + 1)
        self.layers[layer_idx] = state


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
    use_flashmla_fp8_layout = True

    def __init__(self, *args, **kwargs) -> None:
        # The shared base and DeepseekV4Indexer allocate torch.cuda.Event in
        # __init__; HPU has no cuda streams, so redirect to torch.hpu.Event.
        _orig_event = torch.cuda.Event
        torch.cuda.Event = torch.hpu.Event  # type: ignore[assignment, misc]
        try:
            super().__init__(*args, **kwargs)
        finally:
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
        # Manual full-context KV cache (roped kv per layer) so DECODE steps can
        # attend to the prompt + previously generated tokens. Without this, each
        # generated token attends only to itself (num_tokens=1) -> content-free
        # output. Reset on prefill, append on decode. (Not a real paged cache.)
        self._kv_cache: list[torch.Tensor] | None = None
        # Compressor state (DeepSeek V4 sparse attention): per-token kv/score
        # states + the compressed-KV cache written at boundary positions.
        self._comp_kv_states: torch.Tensor | None = None
        self._comp_score_states: torch.Tensor | None = None
        self._comp_kv_cache: list[torch.Tensor] = []
        self._comp_kv_positions: list[int] = []
        self._comp_coff = 2 if self.compress_ratio == 4 else 1
        # Last cached token position, to detect a new sequence (reset caches).
        self._last_pos: int | None = None
        # True once the window cache has been seeded with the prompt's roped kv
        # read from the engine's paged cache (for the first decode of a request).
        self._paged_ctx_seeded: bool = False

        # Public sparse-attention modules (built lazily after weights load) and
        # the ported private compression state.
        vllm_config = kwargs.get("vllm_config", None) or (args[0] if args else None)
        self._hf_config = getattr(vllm_config, "model_config", None).hf_config
        self._pub_sparse = None
        self._comp_state = None
        self._pkv_shim = None

    @classmethod
    def get_padded_num_q_heads(cls, num_heads: int) -> int:
        return num_heads

    def get_kv_cache_spec(self, vllm_config):
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
        w = lin.weight.data
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
                lin.weight.data, s.data, bs, torch.float32
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

    def _build_public_sparse(self):
        """Lazily build the public transformers compressor/indexer from the
        checkpoint weights already loaded into the vLLM compressor/indexer.

        A fresh transformers config is loaded (from the checkpoint path) because
        the server's ``hf_config`` carries a flat ``rope_parameters`` dict (for
        the vLLM rope builder), while the public transformers compressor/indexer
        expect the nested per-rope-type form."""
        if self._pub_sparse is not None:
            return self._pub_sparse
        cfg = self._hf_config
        path = getattr(cfg, "_name_or_path", None)
        if path:
            from transformers import AutoConfig
            try:
                cfg = AutoConfig.from_pretrained(path, trust_remote_code=True)
            except Exception:
                cfg = self._hf_config
        if self.compress_ratio == 4:
            pub = DeepseekV4CSACompressor(cfg)
        else:
            pub = DeepseekV4HCACompressor(cfg)
        comp = self.compressor
        self._copy_fused_compressor(comp, pub, self.head_dim)
        if self.compress_ratio == 4:
            idx = self.indexer
            pub_idx = pub.indexer
            self._copy_fused_compressor(idx.compressor, pub_idx, idx.head_dim)
            wq = self._dequant_fp8_weight(idx.wq_b)
            pub_idx.q_b_proj.weight.data.copy_(wq)
            pub_idx.scorer.weights_proj.weight.data.copy_(
                idx.weights_proj.weight.data
            )
        pub = pub.to(torch.bfloat16).to(comp.device)
        self._pub_sparse = pub
        print(f"[diag] built public {type(pub).__name__} ratio={self.compress_ratio} "
              f"layer={getattr(self, '_layer_idx', -1)}", flush=True)
        return pub

    def _compression(self):
        if self._comp_state is None:
            self._comp_state = _DSV4CompressionState(
                self._hf_config, self.compress_ratio
            )
            self._pkv_shim = _DSV4PkvShim(getattr(self, "_layer_idx", 0), self._comp_state)
        return self._comp_state, self._pkv_shim

    def _run_compressor(self, hidden_states, q_residual, positions):
        """Run the public compressor/indexer over real tokens -> (compressed_kv,
        block_bias). Returns squeezed [n_comp, D] and [n_valid, n_comp] mask (or
        None)."""
        if self.compressor is None or self.compress_ratio <= 1:
            return None, None, None
        pub = self._build_public_sparse()
        state, shim = self._compression()
        hs = hidden_states.unsqueeze(0)
        qr = q_residual.unsqueeze(0)
        pos = positions.unsqueeze(0)
        compressed_kv, block_bias = pub(
            hs, qr, pos, shim, getattr(self, "_layer_idx", 0)
        )
        n_comp = compressed_kv.shape[2]
        if n_comp == 0 or block_bias is None:
            return None, None, None
        ckv = compressed_kv[0, 0]  # [n_comp, D]
        bb = block_bias[0, 0]  # [n_valid, n_comp]
        if os.environ.get("HPU_DUMP") == "1" and getattr(self, "_layer_idx", -1) == 2:
            _HPU_CAP.setdefault("comp_input", {})[0] = hs[0].detach().float().clone()
            _HPU_CAP.setdefault("comp_kv", {})[0] = ckv.detach().float().clone()
            _HPU_CAP.setdefault("comp_bb", {})[0] = bb.detach().float().clone()
            _HPU_CAP.setdefault("comp_pos", {})[0] = positions.detach().cpu().clone()
        return ckv, bb, n_comp

    def _dequant_wo_a(self) -> torch.Tensor:
        # wo_a is fp8 with a per-block or per-output-row scale; the einsum needs
        # the REAL weight (fp8 bytes dequantized by that scale). Cache once.
        cached = getattr(self, "_wo_a_weight_dequant", None)
        if cached is None:
            w = self.wo_a.weight.data.float()
            # Prefer the block scale; fall back to whatever scale is present.
            s = getattr(self.wo_a, "weight_scale_inv", None)
            if s is None or s.numel() == 0:
                s = getattr(self.wo_a, "weight_scale", None)
            s = s.float()
            if s.dim() == 2:
                from vllm_gaudi.extension.ops import dequant_block_fp8_weight_naive

                # wo_a is BLOCK-fp8. scale_adjustment halves the fp8 weight bytes
                # (*0.5) but does NOT double this block scale, so the block dequant
                # lands at exactly 0.5x the checkpoint's true value (measured ratio
                # 2.000). Double it so the einsum uses the real checkpoint value.
                cached = (
                    dequant_block_fp8_weight_naive(
                        self.wo_a.weight.data,
                        self.wo_a.weight_scale_inv.data,
                        self.wo_a.quant_config.weight_block_size,
                        torch.float32,
                    )
                    * 2.0
                ).detach()
            elif s.numel() == w.shape[0]:
                # per-output-row scale: dequant[r, c] = w[r, c] * s[r].
                # The fp8 weight was *0.5'd by scale_adjustment to fit e4m3fnuz,
                # but this per-row scale was NOT *2.0'd (it's a 1D scale, unlike
                # the block/uint8 scales handled in gaudi_weight_wrapper). Undo the
                # halving so the dequant matches the checkpoint's real block values.
                cached = (w * s.view(-1, 1) * 2.0).detach()
            else:
                # per-input-col scale: dequant[r, c] = w[r, c] * s[c]
                cached = (w * s.view(1, -1)).detach()
            self._wo_a_weight_dequant = cached
            si = getattr(self.wo_a, "weight_scale_inv", None)
            ss = getattr(self.wo_a, "weight_scale", None)
            print(f"[diag] wo_a dequant: scale_inv={tuple(si.shape) if si is not None else 'NA'} "
                  f"scale={tuple(ss.shape) if ss is not None else 'NA'} bs={getattr(self.wo_a.quant_config,'weight_block_size',None)} "
                  f"w_dq abs_mean={float(cached.abs().mean()):.5g} abs_max={float(cached.abs().max()):.5g}", flush=True)
        return cached

    def _prep_attn_weights(self) -> dict:
        """Read all attention fp8 linears from the checkpoint and dequantize with
        their block scales to the exact bf16 weights (caching once).

        The HPU's fused/per-row scale representation is a lossy conversion of the
        checkpoint block scales (fused_wqa_wkv, wq_b, wo_a, wo_b all end up with
        per-row scales that leave small element residuals). Dequantizing the
        checkpoint's own block-fp8 weights (fp8 * e8m0, block [128,128]) recovers
        the exact reference value. The checkpoint fp8 is e4m3fn (max 448), which
        the HPU's e4m3fnuz decode cannot represent (>240 -> NaN), so this runs on
        CPU as one-time load-time weight prep and caches the bf16 result on HPU.

        Returns dict w/ keys: wq_a, wkv, wq_b, wo_a, wo_b (TP-sliced to this rank).
        """
        cached = getattr(self, "_attn_w", None)
        if cached is not None:
            return cached
        import glob as _glob
        from safetensors import safe_open
        from vllm_gaudi.extension.ops import dequant_block_fp8_weight_naive
        from vllm.distributed import get_tensor_model_parallel_rank

        path = getattr(self._hf_config, "_name_or_path", None)
        lidx = getattr(self, "_layer_idx", -1)
        dev = self.wq_b.weight.device
        rank = get_tensor_model_parallel_rank()
        P = f"layers.{lidx}.attn."

        import vllm_gaudi.models.deepseek_v4 as _MOD
        if getattr(_MOD, "_SHARD_MAP", None) is None:
            _MOD._SHARD_MAP = {}
            for f in _glob.glob(path + "/model-*.safetensors"):
                with safe_open(f, framework="pt") as sf:
                    for k in sf.keys():
                        _MOD._SHARD_MAP[k] = f

        def load(name):
            f = _MOD._SHARD_MAP.get(name)
            if f is None:
                raise KeyError(name)
            with safe_open(f, framework="pt") as sf:
                return sf.get_tensor(name)

        def dq(name):
            w = load(P + name + ".weight")
            s = load(P + name + ".scale").float()
            return dequant_block_fp8_weight_naive(w, s, (128, 128), torch.float32).to(
                torch.bfloat16
            )

        wq_a = dq("wq_a")  # [q_lora_rank, hidden] replicated
        wkv = dq("wkv")  # [head_dim, hidden] replicated
        wq_b_full = dq("wq_b")  # [n_heads*head_dim, q_lora_rank]
        wq_b = wq_b_full[rank * self.n_local_heads * self.head_dim:
                         (rank + 1) * self.n_local_heads * self.head_dim]
        wo_a_full = dq("wo_a")  # [n_groups*o_lora_rank, ...]
        wo_a = wo_a_full[rank * self.n_local_groups * self.o_lora_rank:
                         (rank + 1) * self.n_local_groups * self.o_lora_rank]
        wo_b_full = dq("wo_b")  # [hidden, n_groups*o_lora_rank]
        wo_b = wo_b_full[:, rank * self.n_local_groups * self.o_lora_rank:
                         (rank + 1) * self.n_local_groups * self.o_lora_rank]
        out = dict(wq_a=wq_a.to(dev), wkv=wkv.to(dev), wq_b=wq_b.to(dev),
                   wo_a=wo_a.to(dev), wo_b=wo_b.to(dev))
        self._attn_w = out
        return out

    def _dequant_wq_b(self) -> torch.Tensor:
        """Dequantize the block-fp8 `wq_b` (TP-sharded ColumnParallelLinear) to
        the true bf16 weight, caching once.

        `scale_adjustment` halves the fp8 weight bytes (*0.5) to fit e4m3fnuz but
        does NOT double this block scale (the e8m0 branch of gaudi_weight_wrapper
        returns before scale_adjustment), so the vLLM apply path dequantizes to
        exactly 0.5x. Mirrors `_dequant_wo_a`: dequant(halved_weight, scale) then
        *2.0 recovers the real weight (verified == fp8*e8m0 to ~1e-6). Running the
        query projection as a plain matmul on the real weight also avoids the
        fused block-fp8 apply path's per-element residual."""
        cached = getattr(self, "_wq_b_dequant", None)
        if cached is None:
            w = self.wq_b.weight.data.float()
            s = getattr(self.wq_b, "weight_scale_inv", None)
            if s is None or s.numel() == 0:
                s = getattr(self.wq_b, "weight_scale", None)
            s = s.float()
            if s.dim() == 2:
                from vllm_gaudi.extension.ops import dequant_block_fp8_weight_naive

                bs = getattr(self.wq_b.quant_config, "weight_block_size", None) or (128, 128)
                cached = (
                    dequant_block_fp8_weight_naive(
                        self.wq_b.weight.data, s.data, bs, torch.float32
                    )
                    * 2.0
                )
            elif s.numel() == w.shape[0]:
                cached = w * s.view(-1, 1) * 2.0
            else:
                cached = w * s.view(1, -1)
            self._wq_b_dequant = cached.to(torch.bfloat16).detach()
        return self._wq_b_dequant

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
        w = self._prep_attn_weights()["wo_a"].view(
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
                self._dequant_wo_a().detach().float().clone()
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
        # wo_b is RowParallel (TP-sliced over the input = group-pair dims); each
        # rank computes only ITS group-pair's contribution. Use the exact bf16
        # wo_b from the checkpoint and all-reduce across TP ranks to combine all
        # groups (matching the RowParallel reduce; HPU is not sequence parallel).
        wo_b = self._prep_attn_weights()["wo_b"]
        out = torch.matmul(z, wo_b.t())
        out = out[0] if isinstance(out, tuple) else out
        from vllm.distributed import tensor_model_parallel_all_reduce

        out = tensor_model_parallel_all_reduce(out)
        _diag("op_after_wo_b", out)
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

    def _compress_one(self, p: int, state_idx: int) -> torch.Tensor | None:
        """Compress the last coff*ratio states into a single [head_dim] key.

        Mirrors ``_fused_kv_compress_norm_rope_insert_sparse_attn``: softmax over
        the gathered score, weighted sum of kv, RMSNorm, RoPE at compressed_pos.
        """
        ratio = self.compress_ratio
        coff = self._comp_coff
        head_dim = self.head_dim
        gather = coff * ratio
        dev = self._comp_kv_states.device
        start = p - gather + 1
        g = torch.arange(gather, device=dev)
        pos = start + g
        valid = pos >= 0
        # state buffer index for each gathered position q (q -> state_idx - (p-q))
        si = (state_idx - (p - pos)).clamp(0, self._comp_kv_states.shape[0] - 1)
        ho = (g >= ratio).long() * head_dim
        ar = torch.arange(head_dim, device=dev)
        flat_idx = si[:, None] * (coff * head_dim) + ho[:, None] + ar[None, :]
        kv_flat = self._comp_kv_states.reshape(-1)
        score_flat = self._comp_score_states.reshape(-1)
        kv_v = kv_flat[flat_idx]  # [gather, head_dim]
        score_v = score_flat[flat_idx]
        score_v = torch.where(
            valid[:, None], score_v, torch.full_like(score_v, float("-inf"))
        )
        w = torch.softmax(score_v, dim=0)
        compressed = (kv_v * w).sum(0)  # [head_dim] fp32
        compressed = _rmsnorm(compressed, self.eps, self.compressor.norm.weight.data)
        cpos = (p // ratio) * ratio
        cpos_t = torch.tensor([cpos], dtype=torch.long, device=dev)
        compressed = _apply_rope(
            compressed.unsqueeze(0), cpos_t,
            self.rotary_emb.cos_sin_cache, self.rope_head_dim,
        ).squeeze(0)
        return compressed.to(self._comp_kv_states.dtype)

    def _compress_tokens(self, hidden_states, positions, is_new: bool):
        """Advance the compressor: store per-token states, emit compressed kv.

        Returns ``(values, pos)`` where values is ``[n_compressed, head_dim]``
        and pos is ``[n_compressed]`` (the boundary position each compressed
        token represents), or ``(None, None)`` if none exist yet.
        """
        if self.compressor is None or self.compress_ratio <= 1:
            return None, None
        ratio = self.compress_ratio
        coff = self._comp_coff
        head_dim = self.head_dim
        comp = self.compressor
        num_tokens = hidden_states.shape[0]
        pos_flat = positions.reshape(-1)
        if is_new:
            self._comp_kv_states = None
            self._comp_score_states = None
            self._comp_kv_cache = []
            self._comp_kv_positions = []
        kv_score = comp.fused_wkv_wgate(hidden_states)
        kv_score = kv_score[0] if isinstance(kv_score, tuple) else kv_score
        kv, score = kv_score.split([coff * head_dim, coff * head_dim], dim=-1)
        score = score + comp.ape[pos_flat % ratio]
        kv_f, score_f = kv.float(), score.float()
        if self._comp_kv_states is None:
            self._comp_kv_states = kv_f
            self._comp_score_states = score_f
        else:
            self._comp_kv_states = torch.cat([self._comp_kv_states, kv_f], 0)
            self._comp_score_states = torch.cat([self._comp_score_states, score_f], 0)
        T_total = self._comp_kv_states.shape[0]
        base = T_total - num_tokens
        for i in range(num_tokens):
            p = int(pos_flat[i])
            if p < 0:
                continue
            if (p + 1) % ratio != 0:
                continue
            ck = self._compress_one(p, base + i)
            if ck is not None:
                self._comp_kv_cache.append(ck)
                self._comp_kv_positions.append((p // ratio) * ratio)
        if not self._comp_kv_cache:
            return None, None
        values = torch.stack(self._comp_kv_cache, 0).to(hidden_states.dtype)
        pos = torch.tensor(self._comp_kv_positions, dtype=torch.long,
                           device=hidden_states.device)
        return values, pos

    def _paged_context_kv(self):
        """Read the prompt's roped KV from the engine's paged cache.

        The engine prefills the prompt through its own path (writing roped kv to
        the flat paged cache bound via ``bind_kv_cache``), while this attention
        forward is only invoked for decode steps. For a single short request the
        context tokens occupy the ``context_len`` slots immediately before the
        current token's slot, so the prompt kv is ``key_cache[slot-ctx_len:slot]``.

        Returns ``[ctx_len, 1, head_dim]`` roped kv, or None if unavailable.
        """
        try:
            if getattr(self, "kv_cache", None) is None:
                return None
            from vllm.forward_context import get_forward_context
            am = get_forward_context().attn_metadata
            if am is None:
                return None
            sm = getattr(am, "slot_mapping", None)
            cl = getattr(am, "context_lens_tensor", None)
            if sm is None or cl is None:
                return None
            key_cache = self.kv_cache[0]
            cur_slot = int(sm.reshape(-1)[0])
            ctx_len = int(cl.reshape(-1)[0])
            if ctx_len <= 0 or cur_slot - ctx_len < 0:
                return None
            ctx = key_cache[cur_slot - ctx_len:cur_slot]
            if ctx.numel() == 0:
                return None
            return ctx
        except Exception:
            return None

    def forward(self, positions, hidden_states, llama_4_scaling=None):
        num_tokens = hidden_states.shape[0]
        import os as _fdbg
        if _fdbg.environ.get("V4_DEBUG") == "1" and getattr(self, "_fwd_dbg_n", 0) < 4:
            self._fwd_dbg_n = getattr(self, "_fwd_dbg_n", 0) + 1
            try:
                from vllm.forward_context import get_forward_context
                amf = get_forward_context().attn_metadata
                ispf = bool(getattr(amf, "is_prompt", None)) if amf is not None else None
            except Exception as e:
                ispf = f"err:{e}"
            pos = positions.reshape(-1)
            sm = cl = bl = None
            try:
                sm = getattr(amf, "slot_mapping", None)
                cl = getattr(amf, "context_lens_tensor", None)
                bl = getattr(amf, "block_list", None)
            except Exception:
                pass
            kc_info = "nobind"
            try:
                if getattr(self, "kv_cache", None) is not None:
                    kc = self.kv_cache[0]
                    nnz = int((kc.abs().sum(dim=(-1, -2)) > 0).sum())
                    kc_info = f"slots={kc.shape[0]} nnz_slots={nnz} am={float(kc.float().abs().mean()):.3g}"
            except Exception as e:
                kc_info = f"err:{e}"
            print(f"[dbg] fwd#{self._fwd_dbg_n} T={num_tokens} is_prompt={ispf} "
                  f"posmin={int(pos.min()):d} posmax={int(pos.max()):d} "
                  f"hs={tuple(hidden_states.shape)} cache={0 if self._kv_cache is None else len(self._kv_cache)} "
                  f"lastpos={self._last_pos} sm={sm.reshape(-1).tolist() if sm is not None else None} "
                  f"cl={cl.reshape(-1).tolist() if cl is not None else None} "
                  f"bl={bl.reshape(-1).tolist() if bl is not None else None} kc[{kc_info}]", flush=True)

        def _out(r):
            return r[0] if isinstance(r, tuple) else r

        _diag("attn_input", hidden_states)
        _attn_w = self._prep_attn_weights()
        qr_kv = torch.cat(
            [
                torch.matmul(hidden_states, _attn_w["wq_a"].t()),
                torch.matmul(hidden_states, _attn_w["wkv"].t()),
            ],
            dim=-1,
        )
        _diag("attn_fused_wqa_wkv", qr_kv)
        _diag("attn_fused_wqa_wkv_weight", self.fused_wqa_wkv.weight)
        if not _DIAG_DONE["v"]:
            w = self.fused_wqa_wkv.weight
            wf = w.to(torch.float32)
            nnan = torch.isnan(wf)
            print(f"[diag] fused_w weight per-128-row-block NaN: "
                  f"{[int(nnan[i:i+128].sum()) for i in range(0, w.shape[0], 128)]}", flush=True)
            for attr in ("scale", "weight_scale", "weight_scale_1", "weight_scale_2"):
                if hasattr(self.fused_wqa_wkv, attr):
                    s = getattr(self.fused_wqa_wkv, attr)
                    try:
                        _diag(f"fused_wqa_wkv.{attr}", s)
                    except Exception as e:
                        print(f"[diag] fused_wqa_wkv.{attr} err: {e}", flush=True)
        qr, kv = qr_kv.split([self.q_lora_rank, self.head_dim], dim=-1)
        _diag("attn_qr", qr)
        _diag("attn_kv", kv)
        qr = _rmsnorm(qr, self.eps, self.q_norm.weight.data)
        kv = _rmsnorm(kv, self.eps, self.kv_norm.weight.data)
        _diag("attn_qr_normed", qr)
        _diag("attn_kv_normed", kv)
        if __import__("os").environ.get("HPU_DUMP") == "1" and getattr(self, "_layer_idx", -1) == 0:
            _HPU_CAP.setdefault("attn_qr_normed", {})[0] = qr.detach().float().clone()
        q = torch.matmul(qr, _attn_w["wq_b"].t()).view(
            num_tokens, self.n_local_heads, self.head_dim
        )
        _diag("attn_q", q)
        if __import__("os").environ.get("HPU_DUMP") == "1" and getattr(self, "_layer_idx", -1) == 0:
            _HPU_CAP.setdefault("attn_q_pre_bnorm", {})[0] = q.detach().float().clone()
        # per-head q RMSNorm (no weight) then RoPE on the rope dims
        q = _rmsnorm(q, self.eps)
        _diag("attn_q_normed", q)
        if __import__("os").environ.get("HPU_DUMP") == "1" and getattr(self, "_layer_idx", -1) in (0, 1, 2):
            _HPU_CAP.setdefault("attn_q_normed", {})[getattr(self, "_layer_idx", -1)] = q.detach().float().clone()
            _HPU_CAP.setdefault("attn_kv_normed", {})[getattr(self, "_layer_idx", -1)] = kv.detach().float().clone()
        q = _apply_rope(q, positions, self.rotary_emb.cos_sin_cache, self.rope_head_dim)
        kv = _apply_rope(kv, positions, self.rotary_emb.cos_sin_cache, self.rope_head_dim)
        _diag("attn_q_roped", q)
        _diag("attn_kv_roped", kv)
        if __import__("os").environ.get("HPU_DUMP") == "1" and getattr(self, "_layer_idx", -1) in (0, 1, 2):
            _HPU_CAP.setdefault("attn_q_roped", {})[getattr(self, "_layer_idx", -1)] = q.detach().float().clone()
            _HPU_CAP.setdefault("attn_kv_roped", {})[getattr(self, "_layer_idx", -1)] = kv.detach().float().clone()
        # Reset/append decision uses attn_metadata.is_prompt (reliable: True only
        # on the genuine prefill forward, False on decode). The prompt prefill is
        # a PADDED forward (input_ids [1,128], positions -1..4 for a 5-token
        # prompt), so on prefill keep only the real tokens (position >= 0).
        import os as _odbg
        is_prompt = None
        try:
            from vllm.forward_context import get_forward_context
            am0 = get_forward_context().attn_metadata
            is_prompt = bool(getattr(am0, "is_prompt", None)) if am0 is not None else None
        except Exception as e:
            is_prompt = f"err:{e}"
        pos_flat = positions.reshape(-1)
        if os.environ.get("HPU_DUMP") == "1" and getattr(self, "_layer_idx", -1) == 0:
            _HPU_CAP.setdefault("pos_flat", {})[0] = pos_flat.detach().cpu().clone()
        if is_prompt is True:
            # Prefill: reset caches with the real (non-padding) prompt tokens.
            valid = pos_flat >= 0
            self._comp_kv_states = None
            self._comp_score_states = None
            self._comp_kv_cache = []
            self._comp_kv_positions = []
            state, _shim = self._compression()
            state.reset()
            if bool(valid.any()):
                self._kv_cache = [kv[valid]]
                self._last_pos = int(pos_flat[valid].max())
                window_kv = kv[valid]
                window_pos = pos_flat[valid]
                comp_valid = valid
                real_hs = hidden_states[comp_valid]
                real_pos = pos_flat[comp_valid]
            else:
                self._kv_cache = [kv]
                self._last_pos = int(pos_flat[-1])
                window_kv = kv
                window_pos = pos_flat
                comp_valid = torch.ones_like(pos_flat, dtype=torch.bool)
                real_hs = hidden_states
                real_pos = pos_flat
            # Run the public compressor over the real prompt tokens so the
            # compressed KV + block_bias match the reference sparse path.
            comp_v, comp_bb, n_comp = self._run_compressor(real_hs, qr[comp_valid], real_pos)
            comp_p = None
            if __import__("os").environ.get("HPU_DUMP") == "1" and getattr(self, "_layer_idx", -1) in (0, 1, 2, 3):
                _HPU_CAP.setdefault("comp_v", {})[getattr(self, "_layer_idx", -1)] = (
                    comp_v.detach().float().clone() if comp_v is not None else None
                )
                _HPU_CAP.setdefault("comp_p", {})[getattr(self, "_layer_idx", -1)] = (
                    comp_p
                )
        else:
            # Decode: the cache holds PAST tokens (prompt + prior decode). The
            # current token is appended AFTER its own attention (so it does not
            # attend to itself).
            if self._kv_cache is None:
                self._kv_cache = [kv]
            self._last_pos = (self._last_pos if self._last_pos is not None else -1) + num_tokens
            window_kv = torch.cat(self._kv_cache, dim=0)
            window_pos = None
            comp_v, comp_bb, n_comp = self._run_compressor(hidden_states, qr, pos_flat)
            comp_p = None
            comp_valid = None
        if _odbg.environ.get("V4_DEBUG") == "1":
            print(f"[diag] reset is_prompt={is_prompt} pos0={int(pos_flat[0]):d} "
                  f"T={num_tokens} cache_n={len(self._kv_cache)} "
                  f"cache_tok={self._kv_cache[0].shape[0] if self._kv_cache else 0} "
                  f"last_pos={self._last_pos}", flush=True)
        if window_kv.shape[0] > self.window_size:
            window_kv = window_kv[-self.window_size:]
            if window_pos is not None:
                window_pos = window_pos[-self.window_size:]
        # Build [window ∪ compressed] kv (reference order: sliding-window branch
        # first, compressed entries appended) + the combined mask.
        if comp_v is not None:
            all_kv = torch.cat([window_kv, comp_v], dim=0)
        else:
            all_kv = window_kv
        k = all_kv.unsqueeze(1).expand(
            all_kv.shape[0], self.n_local_heads, self.head_dim
        )
        # Sparse/compressed attention over [window ∪ compressed].
        #  - Prefill: attend over the REAL query tokens with a position-causal
        #    mask for the window keys (query at position q sees keys with
        #    position <= q) and the compressor's block_bias (causality +
        #    indexer validity) for the compressed keys. The padding query rows
        #    are zeroed and dropped downstream.
        #  - Decode: the single (newest) query attends over all past keys (all
        #    positions < current, since the current kv is not yet cached).
        if is_prompt is True:
            q_att = q[comp_valid]  # [n_valid, H, D]
            q_pos = pos_flat[comp_valid]
            if window_pos is None:
                window_pos = torch.arange(window_kv.shape[0], device=q_pos.device)
            win_mask = q_pos[:, None] >= window_pos[None, :]  # [n_valid, n_win]
            if comp_bb is not None:
                mask = torch.cat([win_mask, torch.isfinite(comp_bb)], dim=-1)
            else:
                mask = win_mask
            o_valid = _attn_with_sink(
                q_att, k, mask, self.attn_sink[: self.n_local_heads], self.scale
            )
            o = torch.zeros_like(q)
            o[comp_valid] = o_valid.to(o.dtype)
            # DIAG: also compute a FULL-causal attention over the real prompt
            # tokens only (no compressed keys), to isolate whether the sparse
            # [window ∪ compressed] path is what differs from the reference.
            if __import__("os").environ.get("HPU_DUMP") == "1" and getattr(self, "_layer_idx", -1) in (0, 1):
                import torch.nn.functional as _F
                kf = window_kv.unsqueeze(0).unsqueeze(0).expand(1, self.n_local_heads, -1, self.head_dim)
                _HPU_CAP.setdefault("attn_sparse", {})[0] = o_valid.detach().float().clone()
                o_full = _F.scaled_dot_product_attention(
                    q_att.unsqueeze(0).transpose(1, 2), kf, kf, is_causal=True
                ).transpose(1, 2).squeeze(0)
                _HPU_CAP.setdefault("attn_full", {})[0] = o_full.detach().float().clone()
                _HPU_CAP.setdefault("attn_qreal", {})[0] = q_att.detach().float().clone()
        else:
            o = _attn_with_sink(
                q, k, None, self.attn_sink[: self.n_local_heads], self.scale
            )
            # Append the current decode token's kv to the cache for the NEXT step.
            if self._kv_cache is not None:
                self._kv_cache.append(kv)
        _diag("attn_sdpa_out", o)
        import os as _os2
        if _os2.environ.get("V4_DEBUG") == "1":
            with torch.no_grad():
                # token-to-token scores: [T, n_keys] averaged over heads
                sc = (q.float() @ all_kv.float().transpose(0, 1)) * (self.head_dim**-0.5)
                # report the LAST REAL (non-padding) query row
                if is_prompt is True:
                    pp = positions.reshape(-1)
                    vidx = (pp >= 0).nonzero().reshape(-1)
                    last_row = int(vidx[-1].item()) if vidx.numel() else sc.shape[0] - 1
                else:
                    last_row = sc.shape[0] - 1
                last = sc[last_row]  # last real query row over all cached keys
                sm = torch.softmax(last, dim=-1)
                ent = float(-(sm * torch.log(sm + 1e-12)).sum(-1).mean())
                nkeys = last.shape[-1]
                uni = float(torch.log(torch.tensor(nkeys, dtype=torch.float32)))
                topk_id, topk_v = torch.topk(sm.mean(0), min(3, nkeys))
                comp_txt = "none"
                if comp_v is not None:
                    ckf = comp_v.float()
                    comp_txt = (f"n={comp_v.shape[0]} am={float(ckf.abs().mean()):.3g} "
                                f"mx={float(ckf.abs().max()):.3g} nan={int(torch.isnan(ckf).sum())}")
                winlen = window_kv.shape[0] if window_kv is not None else 0
                print(f"[diag] attn prefill={is_prompt} T={num_tokens} keys={all_kv.shape[0]} "
                      f"(comp={0 if comp_v is None else comp_v.shape[0]}, win={winlen}) "
                      f"entropy={ent:.4g} uniform={uni:.4g} topkey={topk_id.tolist()} topw={[round(x,3) for x in topk_v.tolist()]} "
                      f"comp_kv[{comp_txt}]", flush=True)
        import os as _oob
        if _oob.environ.get("HPU_DUMP") == "1":
            _HPU_CAP.setdefault("attn_pre_o", {})[getattr(self, "_layer_idx", -1)] = o.detach().float().clone()
        return self._o_proj(o, positions)


_HPU_CAP: dict = {"attn_in": {}, "attn_out": {}}
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
        # Clean-bf16 shared expert. The fused MoE dequants the shared expert with
        # per-row scales that are ~hundreds off (shared_out ~30x too small).
        # Dequant the checkpoint's own block-fp8 shared w1/w2/w3 to bf16 and run
        # clean matmuls (same pattern as _prep_attn_weights).
        se = getattr(self.ffn, "shared_experts", None)
        if se is not None:
            se._v4_ckpt_path = getattr(config, "_name_or_path", None)
            se._v4_layer_idx = getattr(self, "_layer_idx", -1)
            _se_li = getattr(self, "_layer_idx", -1)

            def _se_clean_forward(x, _se=se):
                out = _hpu_v4_shared_mlp_forward(_se, x)
                if __import__("os").environ.get("HPU_DUMP") == "1" and _se_li in (0, 1):
                    _HPU_CAP.setdefault("moe_shared", {})[_se_li] = out.detach().float().clone()
                return out

            se.forward = _se_clean_forward

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

    def forward(self, x, positions, input_ids, post_mix=None, res_mix=None, residual=None):
        import os
        if not getattr(self, "_moe_scale_fixed", False):
            self._moe_scale_fixed = True
            # FIXED 2026-08-27: scale_adjustment (VLLM_SCALE_ADJUSTMENT=1) already
            # doubles ALL fp8 block scales (after the uint8 e8m0 decode for the
            # routed experts) in gaudi_weight_wrapper, and *0.5's the fp8 weights.
            # The previous code doubled the routed/shared expert block scales a
            # SECOND time here, so the dequant value was 2x too large -> the MoE
            # output blew up monotonically in the deep layers (layers 41-42 ffn_out
            # 1.3->12) and the model echoed the prompt. Removing the doubling bounds
            # the magnitudes and lets the model produce sensible output. No manual
            # scale doubling is applied (scale_adjustment is the single source of
            # the fp8 conversion).
        if __import__("os").environ.get("HPU_DUMP") == "1" and getattr(self, "_layer_idx", -1) == 0 \
                and not getattr(self, "_routed_w_captured", False):
            self._routed_w_captured = True
            try:
                ex = getattr(self.ffn, "experts", None)
                if ex is not None and hasattr(ex, "routed_experts"):
                    ex = ex.routed_experts
                if ex is not None:
                    for attr in ("w13_weight", "w2_weight", "w13_weight_scale",
                                 "w2_weight_scale", "w13_weight_scale_inv",
                                 "w2_weight_scale_inv", "w13_scale_inv", "w2_scale_inv"):
                        v = getattr(ex, attr, None)
                        if v is not None:
                            _HPU_CAP.setdefault("routed_" + attr, {})[0] = \
                                v.detach().float().clone()
            except Exception as e:
                print(f"[routed] wcap err {e}", flush=True)
        if __import__("os").environ.get("HPU_DUMP") == "1" and getattr(self, "_layer_idx", -1) == 0 \
                and not getattr(self, "_gate_captured", False):
            self._gate_captured = True
            try:
                g = getattr(getattr(self.ffn, "gate", None), "weight", None)
                if g is not None:
                    _HPU_CAP.setdefault("gate_w", {})[0] = g.detach().float().clone()
            except Exception as e:
                print(f"[gate] cap err {e}", flush=True)
        if __import__("os").environ.get("HPU_DUMP") == "1" and getattr(self, "_layer_idx", -1) == 0 \
                and not getattr(self, "_se_w_captured", False):
            self._se_w_captured = True
            try:
                se = getattr(self.ffn, "shared_experts", None)
                _HPU_CAP.setdefault("se_gu_w", {})[0] = se.gate_up_proj.weight.detach().float().clone()
                _HPU_CAP.setdefault("se_dn_w", {})[0] = se.down_proj.weight.detach().float().clone()
                for tag, mod in (("se_gu_s", se.gate_up_proj), ("se_dn_s", se.down_proj)):
                    bs = getattr(mod, "weight_scale_inv", None)
                    if bs is None:
                        bs = getattr(mod, "weight_scale", None)
                    _HPU_CAP.setdefault(tag, {})[0] = bs.detach().float().clone() if bs is not None else None
            except Exception as e:
                print(f"[se] wcap err {e}", flush=True)
        dbg = os.environ.get("V4_DEBUG") == "1"
        if dbg:
            print(f"[v4dec] x.dim={x.dim()} x.shape={tuple(x.shape)} residual_none={residual is None} "
                  f"hc_mult={self.hc_mult} attn_fn0={tuple(self.hc_attn_fn.shape)}")
        _diag("dec_x_in", x)
        # mHC pre (first layer) / fused post+pre (subsequent) for the attn block.
        if residual is None:
            if x.dim() == 2:
                residual, post_mix, res_mix, x = _mhc_pre_broadcast(
                    x,
                    self.hc_attn_fn_broadcast,
                    self.hc_attn_scale,
                    self.hc_attn_base,
                    self.rms_norm_eps,
                    self.hc_eps,
                    self.hc_eps,
                    self.hc_post_alpha,
                    self.hc_sinkhorn_iters,
                    self.hc_mult,
                    self.attn_norm.weight.data,
                    self.rms_norm_eps,
                )
            else:
                from vllm.model_executor.kernels.mhc import mhc_pre_torch

                residual = x
                post_mix, res_mix, x = mhc_pre_torch(
                    x,
                    self.hc_attn_fn,
                    self.hc_attn_scale,
                    self.hc_attn_base,
                    self.rms_norm_eps,
                    self.hc_eps,
                    self.hc_eps,
                    self.hc_post_alpha,
                    self.hc_sinkhorn_iters,
                )
        else:
            residual, post_mix, res_mix, x = _mhc_fused_post_pre(
                x,
                residual,
                post_mix,
                res_mix,
                self.hc_attn_fn,
                self.hc_attn_scale,
                self.hc_attn_base,
                self.rms_norm_eps,
                self.hc_eps,
                self.hc_eps,
                self.hc_post_alpha,
                self.hc_sinkhorn_iters,
                self.hc_mult,
                self.attn_norm.weight.data,
                self.rms_norm_eps,
            )

        _diag("dec_after_attn_mhcpre", x)
        import os as _hdbg
        if _hdbg.environ.get("HPU_DUMP") == "1":
            _HPU_CAP["attn_in"][self._layer_idx] = x.detach().float().clone()
        x = self.attn(positions, x, None)
        if _hdbg.environ.get("HPU_DUMP") == "1":
            _HPU_CAP["attn_out"][self._layer_idx] = x.detach().float().clone()
        _diag("dec_after_attn", x)

        global _MHC_PHASE, _MHC_LAYER
        _MHC_PHASE = "ffn"
        _MHC_LAYER = self._layer_idx
        residual, post_mix, res_mix, x = _mhc_fused_post_pre(
            x,
            residual,
            post_mix,
            res_mix,
            self.hc_ffn_fn,
            self.hc_ffn_scale,
            self.hc_ffn_base,
            self.rms_norm_eps,
            self.hc_eps,
            self.hc_eps,
            self.hc_post_alpha,
            self.hc_sinkhorn_iters,
            self.hc_mult,
            self.ffn_norm.weight.data,
            self.rms_norm_eps,
        )
        _diag("dec_after_ffn_pre", x)
        if __import__("os").environ.get("HPU_DUMP") == "1" and getattr(self, "_layer_idx", -1) in (0, 1):
            _HPU_CAP.setdefault("ffn_norm_w", {})[getattr(self, "_layer_idx", -1)] = \
                self.ffn_norm.weight.detach().float().clone()
            _HPU_CAP.setdefault("ffn_in", {})[getattr(self, "_layer_idx", -1)] = x.detach().float().clone()

        x = self.ffn(x, input_ids)
        _diag("dec_after_ffn", x)
        if __import__("os").environ.get("HPU_DUMP") == "1" and getattr(self, "_layer_idx", -1) in (0, 1):
            _HPU_CAP.setdefault("ffn_out", {})[getattr(self, "_layer_idx", -1)] = x.detach().float().clone()
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

        from vllm.model_executor.kernels.mhc import mhc_post_torch

        # Reconstruct the multi-stream [T, hc_mult, H] hidden state from the
        # 2D ffn output + the mHC residual/mixes, then collapse it via hc_head.
        hidden_states = mhc_post_torch(hidden_states, residual, post_mix, res_mix)
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
        hidden_states = self.norm(hidden_states)
        if _dbg:
            print(f"[v4model] after norm shape={tuple(hidden_states.shape)}", flush=True)
        if hidden_shape is not None and len(hidden_shape) == 3:
            hidden_states = hidden_states.view(*hidden_shape[:2], self.config.hidden_size)
        return hidden_states


class DeepseekV4ForCausalLM(_NvDeepseekV4ForCausalLM):
    """HPU DeepSeek V4. Reuses the NVIDIA top-level (embed/norm/logits/MoE) but
    with the HPU model (attention + decoder)."""

    model_cls = DeepseekV4HPUModel

    def compute_logits(self, hidden_states):
        import os as _os

        if _os.environ.get("V4_DEBUG") == "1":
            print(f"[v4head] compute_logits hidden={tuple(hidden_states.shape)}", flush=True)
            hs = hidden_states.float()
            print(f"[diag] pre_logits_hidden abs_mean={float(hs.abs().mean()):.5g} "
                  f"abs_max={float(hs.abs().max()):.5g} sq_mean={float(hs.square().mean()):.5g}", flush=True)
        _diag("pre_logits_hidden", hidden_states)
        if _os.environ.get("V4_DEBUG") == "1":
            w = getattr(self.lm_head, "weight", None)
            if w is not None:
                wf = w.float()
                print(f"[diag] lm_head weight dtype={w.dtype} shape={tuple(w.shape)} "
                      f"abs_mean={float(wf.abs().mean()):.5g} abs_max={float(wf.abs().max()):.5g}", flush=True)
            s = getattr(self.lm_head, "weight_scale_inv", None) or getattr(self.lm_head, "weight_scale", None)
            if s is not None:
                print(f"[diag] lm_head scale dtype={s.dtype} shape={tuple(s.shape)} "
                      f"mean={float(s.float().mean()):.5g} min={float(s.float().min()):.5g} max={float(s.float().max()):.5g}", flush=True)
        logits = self.logits_processor(self.lm_head, hidden_states)
        if _os.environ.get("V4_DEBUG") == "1":
            print(f"[v4head] logits done {tuple(logits.shape)}", flush=True)
        _diag("logits", logits)
        if _os.environ.get("V4_DEBUG") == "1":
            lf = logits.reshape(-1, logits.shape[-1]).float()[-1]
            fin = torch.isfinite(lf)
            if fin.any():
                safe = torch.where(fin, lf, torch.full_like(lf, float("-inf")))
                top5, top5i = torch.topk(safe, 5)
                print(f"[diag] logits last-token top5_ids={top5i.tolist()} "
                      f"top5_vals={['%.4g' % v for v in top5.tolist()]} "
                      f"all_nan={bool((~fin).all())}", flush=True)
                for probe_tok in (11111,):
                    if probe_tok < safe.numel():
                        rank = int((safe > safe[probe_tok]).sum())
                        print(f"[diag] probe token {probe_tok} rank={rank} val={float(safe[probe_tok]):.4g}", flush=True)
        _DIAG_DONE["v"] = True
        return logits


def _prep_v4_shared_weights(se) -> tuple:
    """Dequantize the checkpoint's block-fp8 shared-expert w1/w2/w3 to exact bf16."""
    import glob as _g
    from safetensors import safe_open
    from vllm_gaudi.extension.ops import dequant_block_fp8_weight_naive
    import vllm_gaudi.models.deepseek_v4 as _MOD

    path = getattr(se, "_v4_ckpt_path", None)
    lidx = getattr(se, "_v4_layer_idx", -1)
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

    def dq(name):
        w = load(name + ".weight")
        s = load(name + ".scale").float()
        return dequant_block_fp8_weight_naive(w, s, (128, 128), torch.float32).to(
            torch.bfloat16
        )

    P = f"layers.{lidx}.ffn.shared_experts."
    return dq(P + "w1"), dq(P + "w2"), dq(P + "w3")


def _hpu_v4_shared_mlp_forward(self, x: torch.Tensor) -> torch.Tensor:
    """Clean-bf16 shared expert: silu(x@w1^T)*(x@w3^T)@w2^T from checkpoint dequant."""
    w = getattr(self, "_v4_clean_shared_w", None)
    if w is None:
        w = _prep_v4_shared_weights(self)
        dev = x.device
        self._v4_clean_shared_w = w = tuple(t.to(dev) for t in w)
    w1, w2, w3 = w
    h = torch.nn.functional.silu(torch.matmul(x, w1.t())) * torch.matmul(x, w3.t())
    return torch.matmul(h, w2.t())


def _hpu_deepseek_v4_moe_forward(
    self: DeepseekV4MoE,
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """HPU DeepSeek V4 MoE forward.

    The upstream FusedMoE path passes ``router_logits=hidden_states`` (routing on
    the raw post-norm input). That routes with the wrong logits (~3x off). Compute
    the actual gate logits here, then run the experts with them. Kept in the plugin
    so upstream vLLM stays pristine.
    """
    org_shape = hidden_states.shape
    _diag("moe_in", hidden_states)
    router_logits = self.gate(hidden_states)[0]
    if os.environ.get("HPU_DUMP") == "1":
        import re as _moe_re
        m = _moe_re.search(r"\.(\d+)\.ffn$", self.prefix or "")
        lidx = int(m.group(1)) if m else 0
        _HPU_CAP.setdefault("router_logits", {})[lidx] = (
            router_logits.detach().float().clone()
        )
        _HPU_CAP.setdefault("moe_in", {})[lidx] = hidden_states.detach().float().clone()
    final_hidden_states = self.experts(
        hidden_states=hidden_states,
        router_logits=router_logits,
        input_ids=input_ids,
    )
    _diag("moe_out", final_hidden_states)
    if os.environ.get("HPU_DUMP") == "1":
        _HPU_CAP.setdefault("moe_out", {})[lidx] = (
            final_hidden_states.detach().float().clone()
        )
    return final_hidden_states.view(org_shape)


# Install the HPU MoE forward on the shared upstream class (module import applies
# the patch once). The top-level/mtp classes remain unregistered for now.
DeepseekV4MoE.forward = _hpu_deepseek_v4_moe_forward  # type: ignore[method-assign]

DeepSeekV4MTP = None
DSparkDeepseekV4ForCausalLM = None
