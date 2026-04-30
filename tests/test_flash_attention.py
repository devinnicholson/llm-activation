from __future__ import annotations

import math

import pytest
import torch

from scratch_llm.systems.flash_attention import (
    CAUSAL_MASK_VALUE,
    get_flashattention_autograd_function_pytorch,
)


def reference_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    if is_causal:
        n_queries = scores.shape[-2]
        n_keys = scores.shape[-1]
        causal_mask = torch.arange(n_queries)[..., None] >= torch.arange(n_keys)
        causal_mask = causal_mask.to(scores.device)
        scores = scores.masked_fill(~causal_mask, CAUSAL_MASK_VALUE)
    return torch.softmax(scores, dim=-1) @ v, torch.logsumexp(scores, dim=-1)


def make_inputs():
    torch.manual_seed(0)
    batch_size = 3
    n_queries = 96
    n_keys = 96
    d_model = 32
    q = torch.randn(batch_size, n_queries, d_model, requires_grad=True)
    k = torch.randn(batch_size, n_keys, d_model, requires_grad=True)
    v = torch.randn(batch_size, n_keys, d_model, requires_grad=True)
    grad_out = torch.randn(batch_size, n_queries, d_model)
    return q, k, v, grad_out


@pytest.mark.parametrize("is_causal", [False, True])
def test_flash_attention_forward_matches_reference(is_causal):
    q, k, v, _ = make_inputs()
    out = get_flashattention_autograd_function_pytorch().apply(q, k, v, is_causal)
    expected_out, expected_lse = reference_attention(q, k, v, is_causal)

    saved_lse = [
        tensor for tensor in out.grad_fn.saved_tensors if tensor.shape == expected_lse.shape
    ]
    assert len(saved_lse) == 1
    torch.testing.assert_close(out, expected_out, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(saved_lse[0], expected_lse, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("is_causal", [False, True])
def test_flash_attention_backward_matches_reference(is_causal):
    q_ref, k_ref, v_ref, grad_out = make_inputs()
    reference_attention(q_ref, k_ref, v_ref, is_causal)[0].backward(grad_out)

    q, k, v, grad_out = make_inputs()
    get_flashattention_autograd_function_pytorch().apply(q, k, v, is_causal).backward(grad_out)

    torch.testing.assert_close(q.grad, q_ref.grad, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(k.grad, k_ref.grad, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(v.grad, v_ref.grad, rtol=1e-4, atol=1e-4)
