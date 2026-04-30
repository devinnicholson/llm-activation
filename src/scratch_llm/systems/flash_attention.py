from __future__ import annotations

import math

import torch

DEFAULT_QUERY_BLOCK_SIZE = 64
DEFAULT_KEY_BLOCK_SIZE = 64
CAUSAL_MASK_VALUE = -1e6


def _flatten_attention_tensor(x: torch.Tensor) -> tuple[torch.Tensor, torch.Size]:
    leading_shape = x.shape[:-2]
    return x.reshape(-1, x.shape[-2], x.shape[-1]), leading_shape


def _causal_mask(
    *,
    query_start: int,
    key_start: int,
    n_queries: int,
    n_keys: int,
    device: torch.device,
) -> torch.Tensor:
    query_positions = query_start + torch.arange(n_queries, device=device)
    key_positions = key_start + torch.arange(n_keys, device=device)
    return query_positions[:, None] >= key_positions[None, :]


class FlashAttentionPytorch(torch.autograd.Function):
    """Blockwise FlashAttention-style autograd function using standard PyTorch ops.

    The forward pass never materializes the full attention matrix. It computes each
    query block with online log-sum-exp updates across key/value blocks, saves the
    per-row log-sum-exp tensor, and returns exact attention output up to normal
    floating-point differences.

    The backward pass recomputes attention probabilities blockwise from Q, K, V, O,
    dO, and the saved log-sum-exp values.
    """

    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool):
        q_flat, leading_shape = _flatten_attention_tensor(q)
        k_flat, _ = _flatten_attention_tensor(k)
        v_flat, _ = _flatten_attention_tensor(v)

        batch, n_queries, d_head = q_flat.shape
        n_keys = k_flat.shape[1]
        d_value = v_flat.shape[-1]
        scale = 1.0 / math.sqrt(d_head)

        q_work = q_flat.float()
        k_work = k_flat.float()
        v_work = v_flat.float()
        out = torch.empty(batch, n_queries, d_value, device=q.device, dtype=torch.float32)
        logsumexp = torch.empty(batch, n_queries, device=q.device, dtype=torch.float32)

        for query_start in range(0, n_queries, DEFAULT_QUERY_BLOCK_SIZE):
            query_stop = min(query_start + DEFAULT_QUERY_BLOCK_SIZE, n_queries)
            q_block = q_work[:, query_start:query_stop, :]
            q_block_size = query_stop - query_start

            row_max = torch.full(
                (batch, q_block_size),
                -torch.inf,
                device=q.device,
                dtype=torch.float32,
            )
            row_sum = torch.zeros(batch, q_block_size, device=q.device, dtype=torch.float32)
            acc = torch.zeros(batch, q_block_size, d_value, device=q.device, dtype=torch.float32)

            for key_start in range(0, n_keys, DEFAULT_KEY_BLOCK_SIZE):
                key_stop = min(key_start + DEFAULT_KEY_BLOCK_SIZE, n_keys)
                k_block = k_work[:, key_start:key_stop, :]
                v_block = v_work[:, key_start:key_stop, :]
                scores = q_block @ k_block.transpose(-2, -1) * scale
                if is_causal:
                    mask = _causal_mask(
                        query_start=query_start,
                        key_start=key_start,
                        n_queries=q_block_size,
                        n_keys=key_stop - key_start,
                        device=q.device,
                    )
                    scores = scores.masked_fill(~mask, CAUSAL_MASK_VALUE)

                block_max = scores.max(dim=-1).values
                new_row_max = torch.maximum(row_max, block_max)
                old_scale = torch.exp(row_max - new_row_max)
                exp_scores = torch.exp(scores - new_row_max[..., None])

                acc = acc * old_scale[..., None] + exp_scores @ v_block
                row_sum = row_sum * old_scale + exp_scores.sum(dim=-1)
                row_max = new_row_max

            out[:, query_start:query_stop, :] = acc / row_sum[..., None]
            logsumexp[:, query_start:query_stop] = row_max + torch.log(row_sum)

        out = out.reshape(*leading_shape, n_queries, d_value).to(q.dtype)
        logsumexp = logsumexp.reshape(*leading_shape, n_queries)
        ctx.save_for_backward(q, k, v, out, logsumexp)
        ctx.is_causal = is_causal
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        q, k, v, out, logsumexp = ctx.saved_tensors
        q_flat, leading_shape = _flatten_attention_tensor(q)
        k_flat, _ = _flatten_attention_tensor(k)
        v_flat, _ = _flatten_attention_tensor(v)
        out_flat, _ = _flatten_attention_tensor(out)
        grad_out_flat, _ = _flatten_attention_tensor(grad_out)
        logsumexp_flat = logsumexp.reshape(-1, logsumexp.shape[-1])

        batch, n_queries, d_head = q_flat.shape
        n_keys = k_flat.shape[1]
        d_value = v_flat.shape[-1]
        scale = 1.0 / math.sqrt(d_head)

        q_work = q_flat.float()
        k_work = k_flat.float()
        v_work = v_flat.float()
        out_work = out_flat.float()
        grad_out_work = grad_out_flat.float()

        dq = torch.zeros_like(q_work)
        dk = torch.zeros_like(k_work)
        dv = torch.zeros(batch, n_keys, d_value, device=v.device, dtype=torch.float32)

        row_delta = (grad_out_work * out_work).sum(dim=-1)

        for query_start in range(0, n_queries, DEFAULT_QUERY_BLOCK_SIZE):
            query_stop = min(query_start + DEFAULT_QUERY_BLOCK_SIZE, n_queries)
            q_block = q_work[:, query_start:query_stop, :]
            do_block = grad_out_work[:, query_start:query_stop, :]
            delta_block = row_delta[:, query_start:query_stop]
            lse_block = logsumexp_flat[:, query_start:query_stop]
            q_block_size = query_stop - query_start

            dq_block = torch.zeros_like(q_block)
            for key_start in range(0, n_keys, DEFAULT_KEY_BLOCK_SIZE):
                key_stop = min(key_start + DEFAULT_KEY_BLOCK_SIZE, n_keys)
                k_block = k_work[:, key_start:key_stop, :]
                v_block = v_work[:, key_start:key_stop, :]

                scores = q_block @ k_block.transpose(-2, -1) * scale
                if ctx.is_causal:
                    mask = _causal_mask(
                        query_start=query_start,
                        key_start=key_start,
                        n_queries=q_block_size,
                        n_keys=key_stop - key_start,
                        device=q.device,
                    )
                    scores = scores.masked_fill(~mask, CAUSAL_MASK_VALUE)

                probs = torch.exp(scores - lse_block[..., None])
                dp = do_block @ v_block.transpose(-2, -1)
                ds = probs * (dp - delta_block[..., None])

                dq_block += ds @ k_block * scale
                dk[:, key_start:key_stop, :] += ds.transpose(-2, -1) @ q_block * scale
                dv[:, key_start:key_stop, :] += probs.transpose(-2, -1) @ do_block

            dq[:, query_start:query_stop, :] = dq_block

        dq = dq.reshape(*leading_shape, n_queries, d_head).to(q.dtype)
        dk = dk.reshape(*leading_shape, n_keys, d_head).to(k.dtype)
        dv = dv.reshape(*leading_shape, n_keys, d_value).to(v.dtype)
        return dq, dk, dv, None


def get_flashattention_autograd_function_pytorch() -> type[torch.autograd.Function]:
    return FlashAttentionPytorch


def get_flashattention_autograd_function_triton() -> type[torch.autograd.Function]:
    raise NotImplementedError("Triton FlashAttention is the next Assignment 2 milestone.")
