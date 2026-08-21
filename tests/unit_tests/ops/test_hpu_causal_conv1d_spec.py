# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the speculative-decode conv1d path (causal_conv1d_pytorch.py).

These tests exercise numerical correctness on CPU — no Gaudi hardware required.
The spec path (``hpu_causal_conv1d_update`` with ``num_accepted_tokens``) is
compared against a manual sequential reference that mirrors the CUDA
``_causal_conv1d_update_kernel`` IS_SPEC_DECODING semantics.
"""

from __future__ import annotations

import torch

from vllm_gaudi.ops.causal_conv1d_pytorch import hpu_causal_conv1d_update


def _spec_conv_reference(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    conv_state_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    query_start_loc: torch.Tensor,
    width: int,
):
    """Manual sequential reference for the spec conv path.

    For each spec sequence, read ``width - 1`` history values starting at
    ``offset = num_accepted[seq] - 1``, convolve the ``seql`` tokens, then
    roll the state left by 1 (relative to ``offset``) and append the tokens
    (length ``state_len``).
    """
    state_len = conv_state.shape[1]
    out = torch.empty_like(x)
    qsl = query_start_loc.tolist()
    for seq_id in range(len(qsl) - 1):
        start, end = qsl[seq_id], qsl[seq_id + 1]
        seql = end - start
        if seql <= 0:
            continue
        block = int(conv_state_indices[seq_id])
        offset = max(int(num_accepted_tokens[seq_id]) - 1, 0)
        B = conv_state[block]  # (state_len, dim)
        prior = B[offset:offset + width - 1]  # (width-1, dim)
        x_seq = x[start:end]  # (seql, dim)
        seq_input = torch.cat([prior, x_seq], dim=0)  # (w-1+seql, dim)
        # Depthwise conv1d (groups=dim), matching _depthwise_conv1d_tpc.
        conv_in = seq_input.unsqueeze(0).transpose(1, 2)  # (1, dim, w-1+seql)
        dim = weight.size(0)
        conv_out = torch.zeros(1, dim, seql)
        for k in range(width):
            conv_out = conv_out + conv_in[:, :, k:k + seql] * weight[:, k:k + 1].unsqueeze(0)
        if bias is not None:
            conv_out = conv_out + bias.unsqueeze(0).unsqueeze(-1)
        out[start:end] = conv_out.squeeze(0).transpose(0, 1)
        tail_len = state_len - seql
        keep = B[offset + 1:offset + 1 + tail_len]
        new_B = torch.cat([keep, x_seq], dim=0)
        conv_state[block] = new_B
    return out


def _make_conv_inputs(
    num_seqs: int,
    num_spec: int,
    dim: int = 8,
    width: int = 4,
    *,
    seed: int = 0,
):
    torch.manual_seed(seed)
    seq_len = num_spec + 1
    N = num_seqs * seq_len
    x = torch.randn(N, dim)
    weight = torch.randn(dim, width)
    bias = torch.randn(dim)
    state_len = width - 1 + num_spec
    conv_state = torch.randn(num_seqs, state_len, dim)
    conv_state_indices = torch.arange(num_seqs, dtype=torch.long)
    query_start_loc = torch.arange(num_seqs + 1, dtype=torch.long) * seq_len
    num_accepted = torch.randint(1, num_spec + 2, (num_seqs,))
    return x, conv_state, weight, bias, conv_state_indices, num_accepted, query_start_loc, state_len


def test_spec_conv_matches_sequential():
    """Spec conv should equal the manual rolling-window reference."""
    num_seqs, num_spec = 3, 1
    x, conv_state, weight, bias, indices, num_acc, qsl, state_len = _make_conv_inputs(
        num_seqs, num_spec
    )
    ref_state = conv_state.clone()

    out = hpu_causal_conv1d_update(
        x,
        conv_state,
        weight,
        bias,
        activation=None,
        conv_state_indices=indices,
        num_accepted_tokens=num_acc,
        query_start_loc=qsl,
    )
    ref_out = _spec_conv_reference(
        x, ref_state, weight, bias, indices, num_acc, qsl, width=weight.shape[1]
    )
    torch.testing.assert_close(out, ref_out, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(conv_state, ref_state, atol=1e-5, rtol=1e-5)


def test_spec_conv_matches_sequential_activation():
    """Spec conv with silu activation."""
    num_seqs, num_spec = 4, 2
    x, conv_state, weight, bias, indices, num_acc, qsl, state_len = _make_conv_inputs(
        num_seqs, num_spec, seed=3
    )
    ref_state = conv_state.clone()

    out = hpu_causal_conv1d_update(
        x,
        conv_state,
        weight,
        bias,
        activation="silu",
        conv_state_indices=indices,
        num_accepted_tokens=num_acc,
        query_start_loc=qsl,
    )
    ref_out = _spec_conv_reference(
        x, ref_state, weight, bias, indices, num_acc, qsl, width=weight.shape[1]
    )
    ref_out = torch.nn.functional.silu(ref_out)
    torch.testing.assert_close(out, ref_out, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(conv_state, ref_state, atol=1e-5, rtol=1e-5)
