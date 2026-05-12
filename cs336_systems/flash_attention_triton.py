"""Flash attention implementation.

Used to debug the actual triton implementation. Tested with:
uv run modal run scripts/execute_tests.py --k test_flash_forward_pass_triton
"""

import triton
import triton.language as tl
import math
import torch
from torch import Tensor
from jaxtyping import Float
from einops import rearrange


@triton.jit
def flash_fwd_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    is_causal: tl.constexpr,
    O_ptr,
    L_ptr,
    stride_qs,
    stride_qq,
    stride_qd,
    stride_ks,
    stride_kk,
    stride_kd,
    stride_vs,
    stride_vk,
    stride_vd,
    stride_ob,
    stride_oq,
    stride_od,
    stride_lb,
    stride_lq,
    N_QUERIES,
    N_KEYS,
    softmax_scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
):
    # Program indices: each program handles one query tile for one batch/head
    query_tile_index = tl.program_id(0)
    seq_index = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + seq_index * stride_qs,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + seq_index * stride_ks,
        shape=(D, N_KEYS),
        strides=(stride_kd, stride_kk),
        offsets=(0, 0),
        block_shape=(D, K_TILE_SIZE),
        order=(0, 1),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + seq_index * stride_vs,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    q_tile = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")

    # Initialize accumulators in SRAM (tl.float32 for precision).
    mi = tl.full([Q_TILE_SIZE], float("-inf"), dtype=tl.float32)
    li = tl.zeros([Q_TILE_SIZE], dtype=tl.float32)
    oi = tl.zeros([Q_TILE_SIZE, D], dtype=tl.float32)

    for k_tile_idx in range(0, tl.cdiv(N_KEYS, K_TILE_SIZE)):
        k_tile = tl.load(K_block_ptr, boundary_check=(1, 0), padding_option="zero")
        v_tile = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
        s_tile = tl.dot(q_tile, k_tile) * softmax_scale

        if is_causal:
            # Identify the global row and column indices for the current tiles.
            start_q = query_tile_index * Q_TILE_SIZE
            start_k = k_tile_idx * K_TILE_SIZE

            # Create relative row/col indices for the current block
            rows = start_q + tl.arange(0, Q_TILE_SIZE)
            cols = start_k + tl.arange(0, K_TILE_SIZE)

            # Mask: True where Key index > Query index
            mask = rows[:, None] < cols[None, :]
            s_tile = tl.where(mask, float("-inf"), s_tile)

        mi_new = tl.maximum(mi, tl.max(s_tile, axis=1))
        p_tile = tl.exp(s_tile - mi_new[:, None])
        alpha = tl.exp(mi - mi_new)
        li = alpha * li + tl.sum(p_tile, axis=1)
        oi = oi * alpha[:, None]
        oi = tl.dot(p_tile.to(v_tile.dtype), v_tile, acc=oi)
        mi = mi_new

        # Advance K and V block pointers
        K_block_ptr = tl.advance(K_block_ptr, (0, K_TILE_SIZE))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))

    # Final normalization before write-back.
    oi = oi / li[:, None]
    l_sum_exp = mi + tl.log(li)

    # Write back to HBM (only once per program execution).
    O_block_ptr = tl.make_block_ptr(
        O_ptr + seq_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    tl.store(O_block_ptr, oi.to(O_ptr.dtype.element_ty), boundary_check=(0, 1))

    L_block_ptr = tl.make_block_ptr(
        L_ptr + seq_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )
    tl.store(L_block_ptr, l_sum_exp, boundary_check=(0,))


class TritonFlashAttention(torch.autograd.Function):
    """Flash attention implementation."""

    @staticmethod
    def forward(
        ctx,
        Q: Float[Tensor, "... queries d"],
        K: Float[Tensor, "... keys d"],
        V: Float[Tensor, "... keys d"],
        is_causal: bool,
    ) -> Float[Tensor, "... queries d"]:
        assert Q.shape[:-2] == K.shape[:-2] == V.shape[:-2], "Batch/Head mismatch"
        assert Q.shape[-1] == K.shape[-1] == V.shape[-1], "Embedding D mismatch"
        assert K.shape[-2] == V.shape[-2], "Sequence length Nk mismatch"

        ctx.Q_TILE_SIZE = 64
        ctx.K_TILE_SIZE = 64

        # Reshape the inputs to 3D tensors to simplify the algorithm.
        # contiguous() is needed to make sure the tl.make_block_ptr API to
        # function correctly.
        Q_flat = rearrange(Q, "... q d -> (...) q d").contiguous()
        K_flat = rearrange(K, "... k d -> (...) k d").contiguous()
        V_flat = rearrange(V, "... k d -> (...) k d").contiguous()

        n_sequences, n_queries, d = Q_flat.shape
        n_keys = K_flat.shape[1]
        softmax_scale = 1.0 / math.sqrt(d)

        O_flat = torch.zeros_like(Q_flat)
        L_flat = torch.zeros((n_sequences, n_queries), device=Q.device, dtype=Q.dtype)

        grid = (triton.cdiv(n_queries, ctx.Q_TILE_SIZE), n_sequences)
        flash_fwd_kernel[grid](
            Q_ptr=Q_flat,
            K_ptr=K_flat,
            V_ptr=V_flat,
            is_causal=is_causal,
            O_ptr=O_flat,
            L_ptr=L_flat,
            stride_qs=Q_flat.stride(0),
            stride_qq=Q_flat.stride(1),
            stride_qd=Q_flat.stride(2),
            stride_ks=K_flat.stride(0),
            stride_kk=K_flat.stride(1),
            stride_kd=K_flat.stride(2),
            stride_vs=V_flat.stride(0),
            stride_vk=V_flat.stride(1),
            stride_vd=V_flat.stride(2),
            stride_ob=O_flat.stride(0),
            stride_oq=O_flat.stride(1),
            stride_od=O_flat.stride(2),
            stride_lb=L_flat.stride(0),
            stride_lq=L_flat.stride(1),
            N_QUERIES=n_queries,
            N_KEYS=n_keys,
            softmax_scale=softmax_scale,
            D=d,
            Q_TILE_SIZE=ctx.Q_TILE_SIZE,
            K_TILE_SIZE=ctx.K_TILE_SIZE,
        )

        ctx.save_for_backward(Q_flat, K_flat, V_flat, O_flat, L_flat)
        return O_flat.view(Q.shape)

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("Backward pass not yet implemented.")
