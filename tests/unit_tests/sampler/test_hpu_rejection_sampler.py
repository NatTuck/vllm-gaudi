# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import habana_frameworks.torch  # noqa: F401

from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.platforms import current_platform
from vllm.v1.sample.logits_processor import LogitsProcessors

from vllm_gaudi.v1.sample.hpu_rejection_sampler import (
    HpuRejectionSampler,
    GREEDY_TEMPERATURE,
    PLACEHOLDER_TOKEN_ID,
)

DEVICE = current_platform.device_type
VOCAB = 16


def _metadata(num_spec: int, draft_ids: list[int]) -> SpecDecodeMetadata:
    batch = len(draft_ids) // num_spec
    cu = [num_spec * (b + 1) for b in range(batch)]
    return SpecDecodeMetadata(
        draft_token_ids=torch.tensor(draft_ids, dtype=torch.int32, device=DEVICE),
        num_draft_tokens=[num_spec] * batch,
        cu_num_draft_tokens=torch.tensor(cu, dtype=torch.int32, device=DEVICE),
        cu_num_sampled_tokens=torch.tensor(cu, dtype=torch.int32, device=DEVICE),
        target_logits_indices=torch.tensor(
            [b * (num_spec + 1) + i for b in range(batch) for i in range(num_spec)],
            device=DEVICE),
        bonus_logits_indices=torch.tensor(
            [b * (num_spec + 1) + num_spec for b in range(batch)], device=DEVICE),
        logits_indices=torch.tensor(
            [i for i in range(batch * (num_spec + 1))], device=DEVICE),
    )


def _logits(num_spec: int, argmax: list[int]) -> torch.Tensor:
    rows = len(argmax)
    logits = torch.full((rows, VOCAB), -10.0, device=DEVICE, dtype=torch.float32)
    for i, a in enumerate(argmax):
        logits[i, a] = 10.0
    return logits


def _sampling_metadata(top_k: int = 8, top_p: float = 1.0,
                       temperature: float = GREEDY_TEMPERATURE) -> SamplingMetadata:
    return SamplingMetadata(
        temperature=torch.tensor([temperature], device=DEVICE),
        all_greedy=temperature == GREEDY_TEMPERATURE,
        all_random=temperature != GREEDY_TEMPERATURE,
        top_p=torch.tensor([top_p], device=DEVICE),
        top_k=torch.tensor([top_k], device=DEVICE),
        generators={},
        max_num_logprobs=None,
        no_penalties=True,
        prompt_token_ids=None,
        frequency_penalties=torch.zeros(1, device=DEVICE),
        presence_penalties=torch.zeros(1, device=DEVICE),
        repetition_penalties=torch.ones(1, device=DEVICE),
        output_token_ids=[[]],
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=LogitsProcessors([]),
    )


def _test_greedy_case(draft_ids: list[int], argmax: list[int], expected: list[int]) -> None:
    num_spec = len(argmax) - 1
    sampler = HpuRejectionSampler(Sampler())
    metadata = _metadata(num_spec, draft_ids)
    sampling_metadata = _sampling_metadata()
    out = sampler(metadata, None, _logits(num_spec, argmax), sampling_metadata)
    got = out.sampled_token_ids[0].tolist()
    assert got == expected, f"draft={draft_ids} argmax={argmax}: got {got}, want {expected}"


def test_fastpath_greedy_batch2() -> None:
    # batch=2, num_spec=2. Exercises the view-based extraction across request
    # boundaries (logits are request-major: b0 rows 0..2, b1 rows 3..5).
    num_spec = 2
    sampler = HpuRejectionSampler(Sampler())
    metadata = _metadata(num_spec, [3, 5, 1, 2])  # b0 drafts [3,5], b1 drafts [1,2]
    # argmax per row (b0: 3,5,7 ; b1: 1,2,4)
    logits = torch.full((6, VOCAB), -10.0, device=DEVICE, dtype=torch.float32)
    for i, a in enumerate([3, 5, 7, 1, 2, 4]):
        logits[i, a] = 10.0
    sampling_metadata = _sampling_metadata()
    out = sampler(metadata, None, logits, sampling_metadata)
    got = out.sampled_token_ids.tolist()
    assert got == [[3, 5, 7], [1, 2, 4]], f"batch2: got {got}"


def test_fastpath_greedy_all_accepted() -> None:
    # num_spec=2, drafts match the per-position argmax => full accept + bonus.
    _test_greedy_case(draft_ids=[3, 5], argmax=[3, 5, 7], expected=[3, 5, 7])


def test_fastpath_greedy_first_reject() -> None:
    # Position 1's draft is rejected => the target's recovered token (argmax 5)
    # is emitted at position 1, then placeholders after (no bonus).
    _test_greedy_case(draft_ids=[3, 1], argmax=[3, 5, 7], expected=[3, 5, PLACEHOLDER_TOKEN_ID])


def test_fastpath_greedy_immediate_reject() -> None:
    # Position 0's draft is rejected => the target's recovered token (argmax 3)
    # is emitted at position 0, then placeholders.
    _test_greedy_case(draft_ids=[0, 5], argmax=[3, 5, 7],
                      expected=[3, PLACEHOLDER_TOKEN_ID, PLACEHOLDER_TOKEN_ID])


def test_fastpath_metadata_cache_reuse() -> None:
    # The sampler caches per-request k/p/temp keyed on the SamplingMetadata object
    # identity. Running the same metadata object twice must be correct (RNG stays
    # fresh, draft ids come from the per-step metadata).
    num_spec = 2
    sampler = HpuRejectionSampler(Sampler())
    sampling_metadata = _sampling_metadata()
    logits = _logits(num_spec, [3, 5, 7])
    for _ in range(3):
        out = sampler(_metadata(num_spec, [3, 5]), None, logits, sampling_metadata)
        assert out.sampled_token_ids[0].tolist() == [3, 5, 7]


def test_fastpath_non_greedy_shape() -> None:
    # Random path (temp > 0): not deterministic, but must stay within [0, VOCAB)
    # or PLACEHOLDER and keep shape [batch, num_spec+1].
    num_spec = 2
    sampler = HpuRejectionSampler(Sampler())
    sampling_metadata = _sampling_metadata(temperature=0.7)
    metadata = _metadata(num_spec, [3, 5])
    out = sampler(metadata, None, _logits(num_spec, [3, 5, 7]), sampling_metadata)
    ids = out.sampled_token_ids
    assert ids.shape == (1, num_spec + 1)
    for x in ids[0].tolist():
        assert x == PLACEHOLDER_TOKEN_ID or (0 <= x < VOCAB), f"out of range id {x}"
