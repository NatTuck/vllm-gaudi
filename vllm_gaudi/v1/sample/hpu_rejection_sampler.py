# SPDX-License-Identifier: Apache-2.0

import os
import time
import torch
from dataclasses import replace
from typing import Optional

from vllm.v1.sample import rejection_sampler
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import SamplerOutput
from vllm_gaudi.utils import HPUCompileConfig

PLACEHOLDER_TOKEN_ID = rejection_sampler.PLACEHOLDER_TOKEN_ID
GREEDY_TEMPERATURE = rejection_sampler.GREEDY_TEMPERATURE

_COMPILE_ARGS = HPUCompileConfig().get_compile_args()


@torch.compile(**(_COMPILE_ARGS or {}))
def _rejection_sample_tensor(
    draft_token_ids: torch.Tensor,  # [num_tokens] compacted, request-major
    cu_num_draft_tokens: torch.Tensor,  # [batch_size]
    max_spec_len: int,
    draft_probs: Optional[torch.Tensor],  # [num_tokens, vocab_size] or None
    target_probs: torch.Tensor,  # [num_tokens, vocab_size]
    bonus_token_ids: torch.Tensor,  # [batch_size, 1]
    uniform_probs: torch.Tensor,  # [num_tokens]
    inv_q: torch.Tensor,  # [batch_size, vocab_size]
    is_greedy: torch.Tensor,  # [batch_size]
) -> torch.Tensor:
    """Vectorized greedy + random rejection sampling (pure tensor ops).

    Works entirely on a fixed ``[batch_size, max_spec_len]`` grid: the compacted,
    request-major inputs are reshaped to the grid (the MTP scheduler assigns a
    uniform ``num_spec`` draft tokens to every active spec-decode request, so
    ``num_tokens == batch_size * max_spec_len``). No ``repeat_interleave`` /
    data-dependent output shapes and no boolean-mask scatters, so the whole
    kernel is captured in a single compiled graph (zero graph breaks).

    Returns ``[batch_size, max_spec_len + 1]`` int32 output. Greedy requests
    accept draft tokens equal to the target argmax; random requests use the
    stock target_prob/draft_prob ratio vs a uniform draw. Positions after the
    first rejection are filled with PLACEHOLDER; a bonus token is appended when
    the whole sequence was accepted.
    """
    batch_size = cu_num_draft_tokens.shape[0]
    device = target_probs.device
    vocab_size = target_probs.shape[-1]

    starts = torch.zeros_like(cu_num_draft_tokens)
    starts[1:] = cu_num_draft_tokens[:-1]
    counts = cu_num_draft_tokens - starts  # [batch_size]

    # Reshape compacted request-major inputs to the fixed [batch, max_spec_len] grid.
    draft_ids = draft_token_ids.view(batch_size, max_spec_len)
    target_g = target_probs.view(batch_size, max_spec_len, vocab_size)
    uniform_g = uniform_probs.view(batch_size, max_spec_len)
    draft_g = None if draft_probs is None else draft_probs.view(
        batch_size, max_spec_len, vocab_size)

    pos = torch.arange(max_spec_len, device=device)  # [max_spec_len]
    valid = pos.unsqueeze(0) < counts.unsqueeze(1)  # [batch, max_spec_len]

    # ---- Recovered tokens (random path) ----
    if draft_g is None:
        prob = target_g.clone()
        gather_idx = draft_ids.clamp(min=0, max=vocab_size - 1).long()
        prob.scatter_(2, gather_idx.unsqueeze(-1),
                      torch.zeros(batch_size, max_spec_len, 1, device=device))
        prob = torch.where((draft_ids >= 0).unsqueeze(-1), prob, target_g)
    else:
        prob = torch.clamp(target_g - draft_g, min=0.0)
    score = prob * inv_q.unsqueeze(1)  # [batch, max_spec_len, vocab]
    recovered = score.argmax(dim=-1).to(draft_token_ids.dtype)

    # ---- Acceptance ----
    gather_idx = draft_ids.clamp(min=0, max=vocab_size - 1).long()
    if draft_g is None:
        draft_prob = torch.ones(batch_size, max_spec_len, device=device)
    else:
        draft_prob = draft_g.gather(2, gather_idx.unsqueeze(-1)).squeeze(-1)
    target_prob = target_g.gather(2, gather_idx.unsqueeze(-1)).squeeze(-1)

    target_argmax = target_g.argmax(dim=-1)
    greedy_accepted = (draft_ids == target_argmax) & (draft_ids >= 0)

    safe_draft_prob = torch.where(draft_prob > 0, draft_prob,
                                  torch.ones_like(draft_prob))
    random_accepted = (draft_ids >= 0) & (draft_prob > 0) & (
        target_prob / safe_draft_prob >= uniform_g)

    is_greedy_g = is_greedy.unsqueeze(1)  # [batch, 1]
    accepted = torch.where(is_greedy_g, greedy_accepted, random_accepted)

    random_chosen = torch.where(accepted, draft_ids, recovered)
    chosen = torch.where(is_greedy_g, target_argmax, random_chosen).to(torch.int32)

    # ---- Scatter into [batch, max_spec_len] padded layout ----
    chosen_pad = torch.where(valid, chosen, torch.full(
        (batch_size, max_spec_len), PLACEHOLDER_TOKEN_ID, dtype=torch.int32,
        device=device))
    accepted_pad = accepted & valid

    # ---- Sticky-rejection fill ----
    effective_accepted = accepted_pad | (~valid)
    mismatches = ~effective_accepted
    first_reject = torch.where(mismatches.any(dim=1),
                               mismatches.int().argmax(dim=1),
                               torch.full((batch_size,), max_spec_len, device=device))
    keep_mask = (pos.unsqueeze(0) <= first_reject.unsqueeze(1)) & valid

    output = torch.full((batch_size, max_spec_len + 1), PLACEHOLDER_TOKEN_ID,
                        dtype=torch.int32, device=device)
    output[:, :max_spec_len] = torch.where(keep_mask, chosen_pad,
                                           output[:, :max_spec_len])

    # Bonus token where the whole (actual) sequence was accepted. Non-accepted
    # rows write into their own bonus column, which is never read because those
    # rows have already been rejected.
    all_accepted = ~mismatches.any(dim=1)
    bonus_pos = counts.clamp(max=max_spec_len).long()
    bonus_vals = bonus_token_ids[:, 0]
    output[torch.arange(batch_size, device=device), bonus_pos] = torch.where(
        all_accepted, bonus_vals, torch.full_like(bonus_vals, PLACEHOLDER_TOKEN_ID))

    return output


def _generate_uniform_probs(
    num_tokens: int,
    cu_num_draft_tokens: torch.Tensor,
    generators: dict[int, torch.Generator],
    device: torch.device,
) -> torch.Tensor:
    """Device-side uniform draws (no per-request host generator seeding, no D2H)."""
    del cu_num_draft_tokens, generators
    return torch.rand(num_tokens, dtype=torch.float64, device=device)


@torch.compile(**(_COMPILE_ARGS or {}))
def _apply_topk_topp(
    logits: torch.Tensor,  # [num_tokens, vocab_size] float32
    k: torch.Tensor,  # [num_tokens] int64
    p: Optional[torch.Tensor],  # [num_tokens] or None
    max_k: int,
) -> torch.Tensor:
    """Apply top-k (and top-p) masking to logits without a full-vocab sort.

    The stock ``apply_top_k_top_p`` sorts the full vocab and (on HPU) falls into
    an eager, un-compiled path that is slow for a 152k vocab. This instead uses
    ``topk(max_k)`` (cheap) plus a softmax/cumsum over only the top-``max_k``
    values, then scatters the masked top-k values back into the full vocab. It
    matches the stock top-k/top-p semantics: keep the largest tokens whose
    cumulative probability exceeds ``p`` (or all of the top-k when ``p`` is
    None), masking everything else to -inf. A per-row ``~keep.any()`` guard keeps
    at least the largest token even for near-uniform distributions (the stock
    sort path would mask everything and produce NaNs). Captured in a single
    compiled HPU graph (no graph breaks).
    """
    topk_vals, topk_idx = logits.topk(max_k, dim=-1)  # [num_tokens, max_k] desc
    pos = torch.arange(max_k, device=logits.device).unsqueeze(0)  # [1, max_k]
    in_k = pos < k.unsqueeze(-1)  # [num_tokens, max_k]
    neg_inf = torch.tensor(float('-inf'), device=logits.device)
    probs = torch.where(in_k, topk_vals, neg_inf).softmax(dim=-1)
    cum = probs.cumsum(dim=-1)
    if p is not None:
        keep = (cum > (1.0 - p).unsqueeze(-1)) & in_k
    else:
        keep = in_k
    keep = keep | (~keep.any(dim=-1, keepdim=True))
    keep_vals = torch.where(keep, topk_vals, neg_inf)
    full = torch.full_like(logits, float('-inf'))
    full.scatter_(1, topk_idx, keep_vals)
    return full


@torch.compile(**(_COMPILE_ARGS or {}))
def _rejection_sample_fused(
    draft_token_ids: torch.Tensor,  # [num_tokens] compacted, request-major
    cu_num_draft_tokens: torch.Tensor,  # [batch_size]
    max_spec_len: int,
    target_logits: torch.Tensor,  # [num_tokens, vocab_size] RAW logits
    k: torch.Tensor,  # [num_tokens] int64
    p: Optional[torch.Tensor],  # [num_tokens] or None
    max_k: int,
    bonus_token_ids: torch.Tensor,  # [batch_size, 1]
    uniform_probs: torch.Tensor,  # [num_tokens]
    inv_q: torch.Tensor,  # [batch_size, max_k] Gumbel inv_q over the top-k subset
    is_greedy: torch.Tensor,  # [batch_size]
) -> torch.Tensor:
    """Fused top-k/top-p + rejection sampling, operating only on the topk subset.

    Replaces the two-step path (``_apply_topk_topp`` materialising a full 152k
    masked logits tensor, then ``rejection_sample`` softmaxing the full vocab and
    building a ``[batch, vocab]`` Gumbel ``inv_q``) with a single compiled kernel
    that runs ``topk(max_k)`` once and does everything on the compact
    ``[num_tokens, max_k]`` subset: top-p masking, renormalization, greedy/random
    acceptance, and Gumbel-max recovered-token sampling. Avoids every redundant
    full-vocab pass. Returns ``[batch_size, max_spec_len + 1]`` int32.
    """
    batch_size = cu_num_draft_tokens.shape[0]
    device = target_logits.device
    num_tokens = target_logits.shape[0]

    starts = torch.zeros_like(cu_num_draft_tokens)
    starts[1:] = cu_num_draft_tokens[:-1]
    counts = cu_num_draft_tokens - starts  # [batch_size]

    # ---- top-k then top-p, all on the compact subset ----
    vals, idx = target_logits.topk(max_k, dim=-1)  # [num_tokens, max_k] desc
    pos = torch.arange(max_k, device=device).unsqueeze(0)  # [1, max_k]
    in_k = pos < k.unsqueeze(-1)  # [num_tokens, max_k]
    neg_inf = torch.tensor(float('-inf'), device=device)
    probs = torch.where(in_k, vals, neg_inf).softmax(dim=-1)
    if p is not None:
        cum = probs.cumsum(dim=-1)
        keep = (cum > (1.0 - p).unsqueeze(-1)) & in_k
    else:
        keep = in_k
    keep = keep | (~keep.any(dim=-1, keepdim=True))
    probs = torch.where(keep, probs, torch.zeros_like(probs))
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    # ---- reshape to the fixed [batch, max_spec_len] grid ----
    draft_ids = draft_token_ids.view(batch_size, max_spec_len)  # [b, ns]
    tgt_g = probs.view(batch_size, max_spec_len, max_k)  # [b, ns, max_k]
    idx_g = idx.view(batch_size, max_spec_len, max_k)  # [b, ns, max_k]
    uniform_g = uniform_probs.view(batch_size, max_spec_len)  # [b, ns]

    # ---- target probability of the drafted token ----
    draft_idx = draft_ids.clamp(min=0, max=target_logits.shape[-1] - 1)
    match = idx_g == draft_idx.unsqueeze(-1)  # [b, ns, max_k]
    target_prob = (tgt_g * match).sum(dim=-1)  # [b, ns]

    # ---- argmax (greedy) and Gumbel-max (recovered) over the subset ----
    target_argmax_k = tgt_g.argmax(dim=-1)  # [b, ns] position in top-k
    target_argmax = idx_g.gather(-1, target_argmax_k.unsqueeze(-1)).squeeze(-1)
    score = tgt_g * inv_q.unsqueeze(1)  # [b, ns, max_k] (inv_q [b, max_k] shared over positions)
    recovered_k = score.argmax(dim=-1)
    recovered = idx_g.gather(-1, recovered_k.unsqueeze(-1)).squeeze(-1)

    # ---- acceptance ----
    greedy_accepted = (draft_ids == target_argmax) & (draft_ids >= 0)
    random_accepted = (draft_ids >= 0) & (target_prob > 0) & (target_prob >= uniform_g)
    is_greedy_g = is_greedy.unsqueeze(1)  # [b, 1]
    accepted = torch.where(is_greedy_g, greedy_accepted, random_accepted)

    random_chosen = torch.where(accepted, draft_ids, recovered)
    chosen = torch.where(is_greedy_g, target_argmax, random_chosen).to(torch.int32)

    # ---- sticky-rejection fill into the padded output grid ----
    pos_grid = torch.arange(max_spec_len, device=device).unsqueeze(0)  # [1, ns]
    valid = pos_grid < counts.unsqueeze(1)  # [b, ns]
    chosen_pad = torch.where(valid, chosen, torch.full(
        (batch_size, max_spec_len), PLACEHOLDER_TOKEN_ID, dtype=torch.int32, device=device))
    effective_accepted = accepted | (~valid)
    mismatches = ~effective_accepted
    first_reject = torch.where(mismatches.any(dim=1),
                               mismatches.int().argmax(dim=1),
                               torch.full((batch_size,), max_spec_len, device=device))
    keep_mask = (pos_grid <= first_reject.unsqueeze(1)) & valid

    output = torch.full((batch_size, max_spec_len + 1), PLACEHOLDER_TOKEN_ID,
                        dtype=torch.int32, device=device)
    output[:, :max_spec_len] = torch.where(keep_mask, chosen_pad, output[:, :max_spec_len])

    all_accepted = ~mismatches.any(dim=1)
    bonus_pos = counts.clamp(max=max_spec_len).long()
    bonus_vals = bonus_token_ids[:, 0]
    output[torch.arange(batch_size, device=device), bonus_pos] = torch.where(
        all_accepted, bonus_vals, torch.full_like(bonus_vals, PLACEHOLDER_TOKEN_ID))

    return output


def rejection_sample(
    # [num_tokens]
    draft_token_ids: torch.Tensor,
    # [batch_size]
    num_draft_tokens: list[int],
    max_spec_len: int,
    # [batch_size]
    cu_num_draft_tokens: torch.Tensor,
    # [num_tokens, vocab_size]
    draft_probs: Optional[torch.Tensor],
    # [num_tokens, vocab_size]
    target_probs: torch.Tensor,
    # [batch_size, 1]
    bonus_token_ids: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    synthetic_mode: bool = False,
    synthetic_conditional_rates: Optional[torch.Tensor] = None,
    use_fp64_gumbel: bool = False,
) -> torch.Tensor:
    """Pure-tensor, torch-compiled rejection sampling for HPU.

    All device math runs inside the compiled ``_rejection_sample_tensor`` kernel
    (zero graph breaks, zero D2H). Uniform/Gumbel random numbers are drawn on
    device here — a stock boundary — and fed into the compiled kernel; no
    per-request host generator seeding is performed.
    """
    if synthetic_mode:
        raise NotImplementedError("Synthetic rejection sampling is not supported on HPU.")

    _T = os.environ.get("FASTQWEN_STEP_TIMING") == "1"
    _t0 = time.time() if _T else None

    device = target_probs.device
    batch_size = len(num_draft_tokens)
    num_tokens = draft_token_ids.shape[0]

    cu = cu_num_draft_tokens.to(device, non_blocking=True)
    if _T:
        _now = time.time()
        print(f"[step_timing]     rs[enter]={1000*(_now-_t0):.2f}ms", flush=True)
        _t0 = _now

    if sampling_metadata.all_greedy:
        is_greedy = torch.ones(batch_size, dtype=torch.bool, device=device)
        uniform_probs = torch.ones(num_tokens, device=device)
    else:
        temp = sampling_metadata.temperature
        if temp is None:
            is_greedy = torch.ones(batch_size, dtype=torch.bool, device=device)
        else:
            is_greedy = temp[:batch_size] == GREEDY_TEMPERATURE
        uniform_probs = _generate_uniform_probs(
            num_tokens, cu, sampling_metadata.generators, device)
    if _T:
        _now = time.time()
        print(f"[step_timing]     rs[uniform]={1000*(_now-_t0):.2f}ms", flush=True)
        _t0 = _now

    # Gumbel inv_q: [batch_size, vocab_size], device-side.
    # Convert target logits to probabilities (needed for the acceptance ratio and
    # recovered-token sampling). Done here (host, outside the compiled kernel) so
    # the compiled kernel stays small; a softmax inside the compiled graph over the
    # large vocab is slow on HPU.
    target_probs_f = target_probs.float().softmax(dim=-1, dtype=torch.float32)
    vocab_size = target_probs_f.shape[-1]
    if _T:
        _now = time.time()
        print(f"[step_timing]     rs[softmax]={1000*(_now-_t0):.2f}ms", flush=True)
        _t0 = _now
    q = torch.empty((batch_size, vocab_size), dtype=torch.float32, device=device)
    q.exponential_()
    inv_q = q.reciprocal()
    if _T:
        _now = time.time()
        print(f"[step_timing]     rs[invq]={1000*(_now-_t0):.2f}ms", flush=True)
        _t0 = _now

    output = _rejection_sample_tensor(
        draft_token_ids, cu, max_spec_len, draft_probs, target_probs_f,
        bonus_token_ids, uniform_probs, inv_q, is_greedy)
    if _T:
        _now = time.time()
        print(f"[step_timing]     rs[kernel]={1000*(_now-_t0):.2f}ms", flush=True)
    return output.to(torch.int32)


def rejection_sample_fused(
    draft_token_ids: torch.Tensor,  # [num_tokens]
    num_draft_tokens: list[int],
    max_spec_len: int,
    cu_num_draft_tokens: torch.Tensor,  # [batch_size]
    target_logits: torch.Tensor,  # [num_tokens, vocab_size] RAW (temp-scaled) logits
    k: torch.Tensor,  # [num_tokens] int64
    p: Optional[torch.Tensor],  # [num_tokens] or None
    max_k: int,
    bonus_token_ids: torch.Tensor,  # [batch_size, 1]
    sampling_metadata: SamplingMetadata,
) -> torch.Tensor:
    """Host wrapper for the fused top-k/top-p + rejection-sampling kernel."""
    device = target_logits.device
    batch_size = len(num_draft_tokens)
    num_tokens = draft_token_ids.shape[0]

    cu = cu_num_draft_tokens.to(device, non_blocking=True)
    if sampling_metadata.all_greedy:
        is_greedy = torch.ones(batch_size, dtype=torch.bool, device=device)
        uniform_probs = torch.ones(num_tokens, device=device)
    else:
        temp = sampling_metadata.temperature
        if temp is None:
            is_greedy = torch.ones(batch_size, dtype=torch.bool, device=device)
        else:
            is_greedy = temp[:batch_size] == GREEDY_TEMPERATURE
        uniform_probs = _generate_uniform_probs(
            num_tokens, cu, sampling_metadata.generators, device)

    # Gumbel inv_q over the compact top-k subset only (small), host-side.
    q = torch.empty((batch_size, max_k), dtype=torch.float32, device=device)
    q.exponential_()
    inv_q = q.reciprocal()

    output = _rejection_sample_fused(
        draft_token_ids, cu, max_spec_len, target_logits, k, p, max_k,
        bonus_token_ids, uniform_probs, inv_q, is_greedy)
    return output.to(torch.int32)


class HpuRejectionSampler(rejection_sampler.RejectionSampler):
    """HPU RejectionSampler with a fast spec-decode path.

    The stock ``__call__`` applies ``apply_sampling_constraints`` (top-k/top-p
    which sorts over the full vocab, plus a triton ``expand_batch_to_tokens``
    that falls back to a slow Python loop on HPU) on every step. When there are
    no logits processors and no effective top-k/top-p constraint, those steps
    are unnecessary; we apply only the temperature scaling directly and go
    straight to the compiled ``rejection_sample`` kernel. Falls back to the
    stock implementation for every other case (logprobs, constraints, etc.).
    """

    def __call__(
        self,
        metadata,
        draft_probs,
        logits,
        sampling_metadata,
    ) -> SamplerOutput:
        # top_k <= 0 (e.g. -1) and top_p >= 1.0 mean "no constraint".
        top_k = sampling_metadata.top_k
        top_p = sampling_metadata.top_p
        has_topk = top_k is not None and int(top_k.max()) > 0
        has_topp = top_p is not None and float(top_p.min()) < 1.0
        has_processors = (
            not sampling_metadata.no_penalties
            or bool(sampling_metadata.bad_words_token_ids)
            or sampling_metadata.allowed_token_ids_mask is not None
        )
        if os.environ.get("FASTQWEN_STEP_TIMING") == "1":
            print(f"[step_timing]   top_k={top_k} top_p={top_p} topk={has_topk} "
                  f"topp={has_topp} no_penalties={sampling_metadata.no_penalties}",
                  flush=True)
        # Stock fallback: logits processors, logprobs, or top-p-only (no fast
        # top-p kernel without an accompanying top-k).
        if (has_processors or sampling_metadata.max_num_logprobs is not None
                or (has_topp and not has_topk)):
            if os.environ.get("FASTQWEN_STEP_TIMING") == "1":
                print(f"[step_timing]   reject_sampler_FASTPATH=fallback "
                      f"(processors={has_processors} "
                      f"nlogprobs={sampling_metadata.max_num_logprobs})", flush=True)
            return super().__call__(metadata, draft_probs, logits, sampling_metadata)
        if os.environ.get("FASTQWEN_STEP_TIMING") == "1":
            print(f"[step_timing]   reject_sampler_FASTPATH=taken "
                  f"no_penalties={sampling_metadata.no_penalties}", flush=True)

        device = logits.device
        batch_size = len(metadata.num_draft_tokens)
        _T = os.environ.get("FASTQWEN_STEP_TIMING") == "1"
        _t0 = time.time() if _T else None
        if _T:
            _now = time.time()
            print(f"[step_timing]     rs_call[enter]={1000*(_now-_t0):.2f}ms", flush=True)
            _t0 = _now

        # Bonus token via the regular sampler (no logprobs needed).
        _bn_t0 = time.time() if _T else None
        bonus_logits = logits[metadata.bonus_logits_indices]
        bonus_out = self.sampler(
            logits=bonus_logits,
            sampling_metadata=replace(sampling_metadata, max_num_logprobs=-1),
            predict_bonus_token=True,
        )
        bonus_token_ids = bonus_out.sampled_token_ids
        if _T:
            torch.hpu.synchronize()
            _now = time.time()
            print(f"[step_timing]     rs_call[bonus]={1000*(_now-_t0):.2f}ms "
                  f"bonus_synced={1000*(_now-_bn_t0):.2f}ms", flush=True)
            _t0 = _now

        target_logits = logits[metadata.target_logits_indices].float()
        if _T:
            torch.hpu.synchronize()
            _now = time.time()
            print(f"[step_timing]     rs_call[gather]={1000*(_now-_t0):.2f}ms "
                  f"gather_synced={1000*(_now-_bn_t0):.2f}ms", flush=True)
            _tk_t0 = _now
            _t0 = _now

        # Temperature scaling (greedy -> 1), applied per request. The compacted
        # target rows are request-major with max_spec_len==1, so row i is request i.
        fused_args = None
        if not sampling_metadata.all_greedy:
            temp = sampling_metadata.temperature
            if temp is not None:
                temp_f = temp[:batch_size].float().to(device)
                temp_safe = torch.where(temp_f == rejection_sampler.GREEDY_TEMPERATURE,
                                        torch.ones_like(temp_f), temp_f)
                target_logits = target_logits / temp_safe.unsqueeze(-1)

            if has_topk:
                # Fused path: pass raw (temp-scaled) logits + k/p to the fused
                # top-k/top-p rejection kernel; it never materialises the full
                # vocab. Resolve the expand helper at call time so the HPU patch
                # (which replaces the triton kernel with a pure-PyTorch version)
                # is honored.
                num_tokens = target_logits.shape[0]
                cu = metadata.cu_num_draft_tokens
                expand = rejection_sampler.expand_batch_to_tokens
                tk = expand(top_k, cu, num_tokens).to(device, dtype=torch.int64)
                tp = None
                if has_topp:
                    tp = expand(top_p, cu, num_tokens).to(device)
                max_k = int(top_k.max())
                fused_args = (tk, tp, max_k)
                if _T:
                    torch.hpu.synchronize()
                    _now = time.time()
                    print(f"[step_timing]     rs_call[topk_prep]={1000*(_now-_t0):.2f}ms "
                          f"topk_prep_synced={1000*(_now-_tk_t0):.2f}ms", flush=True)
                    _t0 = _now

        if fused_args is not None:
            tk, tp, max_k = fused_args
            _fj_t0 = time.time() if _T else None
            output_token_ids = rejection_sample_fused(
                metadata.draft_token_ids,
                metadata.num_draft_tokens,
                metadata.max_spec_len,
                metadata.cu_num_draft_tokens,
                target_logits,
                tk,
                tp,
                max_k,
                bonus_token_ids,
                sampling_metadata,
            )
            if _T:
                torch.hpu.synchronize()
                _now = time.time()
                print(f"[step_timing]     rs_call[fused_kernel]={1000*(_now-_t0):.2f}ms "
                      f"fused_synced={1000*(_now-_fj_t0):.2f}ms", flush=True)
                _t0 = _now
        else:
            output_token_ids = rejection_sample(
                metadata.draft_token_ids,
                metadata.num_draft_tokens,
                metadata.max_spec_len,
                metadata.cu_num_draft_tokens,
                draft_probs,
                target_logits,
                bonus_token_ids,
                sampling_metadata,
            )
        if _T:
            _now = time.time()
            print(f"[step_timing]     rs_call[rej_sample]={1000*(_now-_t0):.2f}ms", flush=True)
        return SamplerOutput(sampled_token_ids=output_token_ids, logprobs_tensors=None)


rejection_sampler.rejection_sample = rejection_sample
