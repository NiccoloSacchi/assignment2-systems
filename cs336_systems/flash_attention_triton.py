"""Flash attention implementation.

Used to debug the actual triton implementation. Tested with:
uv run modal run scripts/execute_tests.py --k "test_flash and triton"
"""

import triton
import triton.language as tl
import math
import torch
from torch import Tensor
from jaxtyping import Float
from einops import rearrange


def check_cuda(tensor, name):
    if not tensor.is_cuda:
        raise ValueError(
            f"Tensor '{name}' must be on a CUDA device, but found {tensor.device}"
        )


fwd_triton_configs = [
    # Large tiles for high-end GPUs (A100/H100)
    triton.Config({"Q_TILE_SIZE": 128, "K_TILE_SIZE": 128}, num_stages=3, num_warps=8),
    triton.Config({"Q_TILE_SIZE": 128, "K_TILE_SIZE": 64}, num_stages=4, num_warps=4),
    # Mid-range tiles
    triton.Config({"Q_TILE_SIZE": 64, "K_TILE_SIZE": 64}, num_stages=4, num_warps=4),
    triton.Config({"Q_TILE_SIZE": 64, "K_TILE_SIZE": 64}, num_stages=2, num_warps=8),
    # Small tiles for high d_model or lower-end GPUs
    triton.Config({"Q_TILE_SIZE": 32, "K_TILE_SIZE": 32}, num_stages=2, num_warps=4),
]


@triton.autotune(
    configs=fwd_triton_configs,
    # When any of these inputs change then Triton will run autotuning.
    key=["D_MODEL", "IS_CAUSAL"],
)
@triton.jit
def flash_fwd_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
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
    NUM_QUERIES: tl.constexpr,
    NUM_KEYS: tl.constexpr,
    softmax_scale,
    IS_CAUSAL: tl.constexpr,
    D_MODEL: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
):
    # Program indices: each program handles one query tile for one batch/head
    query_tile_index = tl.program_id(0)
    seq_index = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + seq_index * stride_qs,
        shape=(NUM_QUERIES, D_MODEL),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D_MODEL),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + seq_index * stride_ks,
        shape=(D_MODEL, NUM_KEYS),
        strides=(stride_kd, stride_kk),
        offsets=(0, 0),
        block_shape=(D_MODEL, K_TILE_SIZE),
        order=(0, 1),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + seq_index * stride_vs,
        shape=(NUM_KEYS, D_MODEL),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D_MODEL),
        order=(1, 0),
    )

    q_tile = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")

    # Initialize accumulators in SRAM (tl.float32 for precision).
    Mi = tl.full([Q_TILE_SIZE], float("-inf"), dtype=tl.float32)
    Li = tl.zeros([Q_TILE_SIZE], dtype=tl.float32)
    Oi = tl.zeros([Q_TILE_SIZE, D_MODEL], dtype=tl.float32)

    for k_tile_idx in range(0, tl.cdiv(NUM_KEYS, K_TILE_SIZE)):
        k_tile = tl.load(K_block_ptr, boundary_check=(1, 0), padding_option="zero")
        v_tile = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
        s_tile = tl.dot(q_tile, k_tile) * softmax_scale

        if IS_CAUSAL:
            # Identify the global row and column indices for the current tiles.
            start_q = query_tile_index * Q_TILE_SIZE
            start_k = k_tile_idx * K_TILE_SIZE

            # Create relative row/col indices for the current block
            rows = start_q + tl.arange(0, Q_TILE_SIZE)
            cols = start_k + tl.arange(0, K_TILE_SIZE)

            # Mask: True where Key index > Query index
            mask = rows[:, None] < cols[None, :]
            s_tile = tl.where(mask, float("-inf"), s_tile)

        mi_new = tl.maximum(Mi, tl.max(s_tile, axis=1))
        P_tile = tl.exp(s_tile - mi_new[:, None])
        alpha = tl.exp(Mi - mi_new)
        Li = alpha * Li + tl.sum(P_tile, axis=1)
        Oi = Oi * alpha[:, None]
        Oi = tl.dot(P_tile.to(v_tile.dtype), v_tile, acc=Oi)
        Mi = mi_new

        # Advance K and V block pointers
        K_block_ptr = tl.advance(K_block_ptr, (0, K_TILE_SIZE))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))

    # Final normalization before write-back.
    Oi = Oi / Li[:, None]
    l_sum_exp = Mi + tl.log(Li)

    # Write back to HBM (only once per program execution).
    O_block_ptr = tl.make_block_ptr(
        O_ptr + seq_index * stride_ob,
        shape=(NUM_QUERIES, D_MODEL),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D_MODEL),
        order=(1, 0),
    )
    tl.store(O_block_ptr, Oi.to(O_ptr.dtype.element_ty), boundary_check=(0, 1))

    off_l = (
        seq_index * NUM_QUERIES
        + query_tile_index * Q_TILE_SIZE
        + tl.arange(0, Q_TILE_SIZE)
    )
    tl.store(L_ptr + off_l, l_sum_exp, mask=off_l < (seq_index + 1) * NUM_QUERIES)


backward_preprocess_triton_configs = [
    # flash_backward_preprocess_kernel does not require loading a lot of data,
    # it should manage running with larger tiles.
    triton.Config({"Q_TILE_SIZE": 128}, num_stages=2, num_warps=4),
    triton.Config({"Q_TILE_SIZE": 256}, num_stages=2, num_warps=4),
    triton.Config({"Q_TILE_SIZE": 512}, num_stages=2, num_warps=8),
]


@triton.autotune(
    configs=backward_preprocess_triton_configs,
    # When any of these inputs change then Triton will run autotuning.
    key=["D_MODEL"],
)
@triton.jit
def flash_backward_preprocess_kernel(
    O_ptr,
    dO_ptr,
    D_ptr,
    stride_O0,
    stride_O1,
    stride_O2,
    NUM_QUERIES: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    D_MODEL: tl.constexpr,
):
    query_tile_index = tl.program_id(0)
    seq_idx = tl.program_id(1)

    query_tile_offset = query_tile_index * Q_TILE_SIZE
    O_block_ptr = tl.make_block_ptr(
        base=O_ptr + seq_idx * stride_O0,
        shape=(NUM_QUERIES, D_MODEL),
        strides=(stride_O1, stride_O2),
        offsets=(query_tile_offset, 0),
        block_shape=(Q_TILE_SIZE, D_MODEL),
        order=(1, 0),
    )
    dO_block_ptr = tl.make_block_ptr(
        base=dO_ptr + seq_idx * stride_O0,
        shape=(NUM_QUERIES, D_MODEL),
        strides=(stride_O1, stride_O2),
        offsets=(query_tile_offset, 0),
        block_shape=(Q_TILE_SIZE, D_MODEL),
        order=(1, 0),
    )

    O = tl.load(O_block_ptr, boundary_check=(0,)).to(tl.float32)
    dO = tl.load(dO_block_ptr, boundary_check=(0,)).to(tl.float32)
    D = tl.sum((O * dO), axis=1)
    queries_offset = query_tile_offset + tl.arange(0, Q_TILE_SIZE)
    off_d = seq_idx * NUM_QUERIES + queries_offset
    tl.store(D_ptr + off_d, D, mask=queries_offset < NUM_QUERIES)


backward_triton_configs = [
    # Backward is more register-heavy, keep tiles slightly smaller.
    triton.Config({"Q_TILE_SIZE": 64, "K_TILE_SIZE": 64}, num_stages=3, num_warps=8),
    triton.Config({"Q_TILE_SIZE": 64, "K_TILE_SIZE": 32}, num_stages=3, num_warps=4),
    triton.Config({"Q_TILE_SIZE": 32, "K_TILE_SIZE": 64}, num_stages=3, num_warps=4),
    triton.Config({"Q_TILE_SIZE": 32, "K_TILE_SIZE": 32}, num_stages=2, num_warps=4),
]


@triton.autotune(
    configs=backward_triton_configs,
    # When any of these inputs change then Triton will run autotuning.
    key=["D_MODEL", "IS_CAUSAL"],
    # Since we use atomic_add, we need to make sure Triton zeroes this tensor
    # whenever autotuning.
    reset_to_zero=["dQ_ptr"],
)
@triton.jit
def flash_backward_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    softmax_scale,
    L_ptr,
    D_ptr,
    dO_ptr,
    dQ_ptr,
    dK_ptr,
    dV_ptr,
    stride_O0,
    stride_O1,
    stride_O2,
    NUM_QUERIES: tl.constexpr,
    NUM_KEYS: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    D_MODEL: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    key_tile_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)

    # Pointers for the current sequence.
    Q_seq_ptr = Q_ptr + seq_idx * stride_O0
    K_seq_ptr = K_ptr + seq_idx * stride_O0
    V_seq_ptr = V_ptr + seq_idx * stride_O0
    dO_seq_ptr = dO_ptr + seq_idx * stride_O0

    # Block pointers.
    key_tile_offset = key_tile_idx * K_TILE_SIZE
    K_block_ptr = tl.make_block_ptr(
        base=K_seq_ptr,
        shape=(NUM_KEYS, D_MODEL),
        strides=(stride_O1, stride_O2),
        offsets=(key_tile_offset, 0),
        block_shape=(K_TILE_SIZE, D_MODEL),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V_seq_ptr,
        shape=(NUM_KEYS, D_MODEL),
        strides=(stride_O1, stride_O2),
        offsets=(key_tile_offset, 0),
        block_shape=(K_TILE_SIZE, D_MODEL),
        order=(1, 0),
    )
    Q_block_ptr = tl.make_block_ptr(
        base=Q_seq_ptr,
        shape=(NUM_QUERIES, D_MODEL),
        strides=(stride_O1, stride_O2),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D_MODEL),
        order=(1, 0),
    )
    dO_block_ptr = tl.make_block_ptr(
        base=dO_seq_ptr,
        shape=(NUM_QUERIES, D_MODEL),
        strides=(stride_O1, stride_O2),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D_MODEL),
        order=(1, 0),
    )

    Kj = tl.load(K_block_ptr, boundary_check=(0,))
    Vj = tl.load(V_block_ptr, boundary_check=(0,))

    # Accumulators for dK and dV (in registers).
    dKj = tl.zeros([K_TILE_SIZE, D_MODEL], dtype=tl.float32)
    dVj = tl.zeros([K_TILE_SIZE, D_MODEL], dtype=tl.float32)
    for i in range(0, NUM_QUERIES, Q_TILE_SIZE):
        Qi = tl.load(Q_block_ptr, boundary_check=(0,))
        dOi = tl.load(dO_block_ptr, boundary_check=(0,))

        off_ld = seq_idx * NUM_QUERIES + i + tl.arange(0, Q_TILE_SIZE)
        Li = tl.load(L_ptr + off_ld, mask=off_ld < (seq_idx + 1) * NUM_QUERIES)
        Di = tl.load(D_ptr + off_ld, mask=off_ld < (seq_idx + 1) * NUM_QUERIES)

        queries_offset = i + tl.arange(0, Q_TILE_SIZE)
        Si = tl.dot(Qi, tl.trans(Kj)) * softmax_scale
        if IS_CAUSAL:
            keys_offset = key_tile_offset + tl.arange(0, K_TILE_SIZE)
            Si = tl.where(
                queries_offset[:, None] < keys_offset[None, :], float("-inf"), Si
            )

        Pi = tl.exp(Si - Li[:, None])
        dVj += tl.dot(tl.trans(Pi.to(Qi.dtype)), dOi)
        dPi = tl.dot(dOi, tl.trans(Vj))
        dS = Pi * (dPi - Di[:, None]) * softmax_scale

        # Atomic Add for dQ.
        rk = tl.arange(0, D_MODEL)
        off_dq = (
            seq_idx * stride_O0
            + queries_offset[:, None] * stride_O1
            + rk[None, :] * stride_O2
        )
        tl.atomic_add(
            dQ_ptr + off_dq,
            tl.dot(dS.to(Qi.dtype), Kj),
            mask=queries_offset[:, None] < NUM_QUERIES,
        )
        dKj += tl.dot(tl.trans(dS.to(Qi.dtype)), Qi)

        Q_block_ptr = tl.advance(Q_block_ptr, [Q_TILE_SIZE, 0])
        dO_block_ptr = tl.advance(dO_block_ptr, [Q_TILE_SIZE, 0])

    # Finally, write dK and dV to HBM.
    dk_block_ptr = tl.make_block_ptr(
        base=dK_ptr + seq_idx * stride_O0,
        shape=(NUM_KEYS, D_MODEL),
        strides=(stride_O1, stride_O2),
        offsets=(key_tile_offset, 0),
        block_shape=(K_TILE_SIZE, D_MODEL),
        order=(1, 0),
    )
    dv_block_ptr = tl.make_block_ptr(
        base=dV_ptr + seq_idx * stride_O0,
        shape=(NUM_KEYS, D_MODEL),
        strides=(stride_O1, stride_O2),
        offsets=(key_tile_offset, 0),
        block_shape=(K_TILE_SIZE, D_MODEL),
        order=(1, 0),
    )
    tl.store(dk_block_ptr, dKj.to(Kj.dtype), boundary_check=(0,))
    tl.store(dv_block_ptr, dVj.to(Vj.dtype), boundary_check=(0,))


class TritonFlashAttention(torch.autograd.Function):
    """Flash attention implementation."""

    @staticmethod
    def forward(
        ctx,
        Q: Float[Tensor, "... queries d"],
        K: Float[Tensor, "... keys d"],
        V: Float[Tensor, "... keys d"],
        is_causal: bool = False,
    ) -> Float[Tensor, "... queries d"]:
        # There is not torch computation here, but making sure we don't track
        # residuals makes it future-proof.
        with torch.no_grad():
            assert Q.shape[:-2] == K.shape[:-2] == V.shape[:-2], "Batch/Head mismatch"
            assert Q.shape[-1] == K.shape[-1] == V.shape[-1], "Embedding D mismatch"
            assert K.shape[-2] == V.shape[-2], "Sequence length Nk mismatch"
            check_cuda(Q, "Q")
            check_cuda(K, "K")
            check_cuda(V, "V")

            ctx.is_causal = is_causal  # Needed for the backward.
            ctx.Q_shape = Q.shape
            ctx.K_shape = K.shape

            # Reshape the inputs to 3D tensors to simplify the algorithm.
            # contiguous() is needed to make sure the tl.make_block_ptr API to
            # function correctly.
            Q_flat = rearrange(Q, "... q d -> (...) q d").contiguous()
            K_flat = rearrange(K, "... k d -> (...) k d").contiguous()
            V_flat = rearrange(V, "... k d -> (...) k d").contiguous()

            num_sequences, num_queries, d_model = Q_flat.shape
            n_keys = K_flat.shape[1]
            softmax_scale = 1.0 / math.sqrt(d_model)

            O_flat = torch.zeros_like(Q_flat)
            L_flat = torch.zeros(
                (num_sequences, num_queries), device=Q.device, dtype=torch.float32
            )

            # The Q_TILE_SIZE is dynamically set by Triton's autotune.
            grid = lambda META: (
                triton.cdiv(num_queries, META["Q_TILE_SIZE"]),
                num_sequences,
            )
            flash_fwd_kernel[grid](
                Q_ptr=Q_flat,
                K_ptr=K_flat,
                V_ptr=V_flat,
                IS_CAUSAL=is_causal,
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
                NUM_QUERIES=num_queries,
                NUM_KEYS=n_keys,
                softmax_scale=softmax_scale,
                D_MODEL=d_model,
            )

            ctx.save_for_backward(Q_flat, K_flat, V_flat, O_flat, L_flat)
            return O_flat.view(Q.shape)

    @staticmethod
    def backward(ctx, dO):
        # There is not torch computation here, but making sure we don't track
        # residuals makes it future-proof.
        with torch.no_grad():
            Q_flat, K_flat, V_flat, O_flat, L_flat = ctx.saved_tensors
            check_cuda(Q_flat, "Q")
            check_cuda(K_flat, "K")
            check_cuda(V_flat, "V")
            check_cuda(O_flat, "O")
            check_cuda(L_flat, "L")
            check_cuda(dO, "dO")

            dO_flat = rearrange(dO, "... queries d -> (...) queries d").contiguous()

            # Precompute the D tensor.
            num_sequences, num_queries, d_model = Q_flat.shape
            D = torch.empty(
                (num_sequences, num_queries),
                device=Q_flat.device,
                dtype=torch.float32,
            )
            # The Q_TILE_SIZE is dynamically set by Triton's autotune.
            grid_d = lambda META: (
                triton.cdiv(num_queries, META["Q_TILE_SIZE"]),
                num_sequences,
            )
            flash_backward_preprocess_kernel[grid_d](
                O_ptr=O_flat,
                dO_ptr=dO_flat,
                D_ptr=D,
                stride_O0=O_flat.stride(0),
                stride_O1=O_flat.stride(1),
                stride_O2=O_flat.stride(2),
                NUM_QUERIES=num_queries,
                D_MODEL=d_model,
            )

            # Compute the gradients.
            softmax_scale = 1.0 / math.sqrt(d_model)
            dQ_flat = torch.zeros_like(Q_flat)  # num_sequences, queries, d
            dK_flat = torch.zeros_like(K_flat)  # num_sequences, keys, d
            dV_flat = torch.zeros_like(V_flat)  # num_sequences, keys, d
            num_keys = K_flat.shape[1]
            # The K_TILE_SIZE is dynamically set by Triton's autotune.
            grid_bwd = lambda META: (
                triton.cdiv(num_keys, META["K_TILE_SIZE"]),
                num_sequences,
            )
            flash_backward_kernel[grid_bwd](
                Q_ptr=Q_flat,
                K_ptr=K_flat,
                V_ptr=V_flat,
                softmax_scale=softmax_scale,
                L_ptr=L_flat,
                D_ptr=D,
                dO_ptr=dO_flat,
                dQ_ptr=dQ_flat,
                dK_ptr=dK_flat,
                dV_ptr=dV_flat,
                stride_O0=Q_flat.stride(0),
                stride_O1=Q_flat.stride(1),
                stride_O2=Q_flat.stride(2),
                NUM_QUERIES=num_queries,
                NUM_KEYS=num_keys,
                D_MODEL=d_model,
                IS_CAUSAL=ctx.is_causal,
            )

            return (
                dQ_flat.reshape(ctx.Q_shape),
                dK_flat.reshape(ctx.K_shape),
                dV_flat.reshape(ctx.K_shape),
                None,  # No gradient for is_causal
            )
