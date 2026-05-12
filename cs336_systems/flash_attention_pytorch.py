import triton
import math
import torch
from torch import Tensor
from jaxtyping import Float
from einops import rearrange, einsum


def pytorch_flash_attention(
    pid: int,
    Q: Float[Tensor, "n_rows queries d"],
    K: Float[Tensor, "n_rows keys d"],
    V: Float[Tensor, "n_rows keys d"],
    is_causal: bool,
    O: Float[Tensor, "n_rows queries d"],
    L: Float[Tensor, "n_rows queries"],
    Q_TILE_SIZE: int,
    K_TILE_SIZE: int,
    ROW_TILE_SIZE: int,
):
    # Select only the row that this program execution should process.
    start_row_tile_idx = pid * ROW_TILE_SIZE
    Q = Q[start_row_tile_idx : start_row_tile_idx + ROW_TILE_SIZE]
    K = K[start_row_tile_idx : start_row_tile_idx + ROW_TILE_SIZE]
    V = V[start_row_tile_idx : start_row_tile_idx + ROW_TILE_SIZE]

    _, queries, d = Q.shape
    device, dtype = Q.device, Q.dtype
    keys = K.shape[1]
    softmax_scale = 1.0 / math.sqrt(d)

    for q_tile_idx in range(0, queries, Q_TILE_SIZE):
        q_tile = Q[:, q_tile_idx : q_tile_idx + Q_TILE_SIZE]
        n_rows_tile, queries_tile, _ = q_tile.shape
        mi = torch.full(
            (n_rows_tile, queries_tile, 1), float("-inf"), device=device, dtype=dtype
        )
        li = torch.zeros((n_rows_tile, queries_tile, 1), device=device, dtype=dtype)
        oi = torch.zeros((n_rows_tile, queries_tile, d), device=device, dtype=dtype)
        for k_tile_idx in range(0, keys, K_TILE_SIZE):
            k_tile = K[:, k_tile_idx : k_tile_idx + K_TILE_SIZE]
            s_tile = (
                einsum(
                    q_tile,
                    k_tile,
                    "n_rows_tile queries_tile d, n_rows_tile keys_tile d -> n_rows_tile queries_tile keys_tile",
                )
                * softmax_scale
            )
            mi_new = torch.max(mi, torch.max(s_tile, dim=-1, keepdim=True).values)
            p_tile = torch.exp(s_tile - mi_new)

            alpha = torch.exp(mi - mi_new)
            li = alpha * li + torch.sum(p_tile, dim=-1, keepdim=True)
            v_tile = V[:, k_tile_idx : k_tile_idx + K_TILE_SIZE]
            oi = alpha * oi + einsum(
                p_tile,
                v_tile,
                "n_rows_tile queries_tile keys_tile, n_rows_tile keys_tile d -> n_rows_tile queries_tile d",
            )
            mi = mi_new

        # Use same approach as in a triton kernel: use temporary output tensors
        # to hold the final results and write sporadically HBM to minimize data
        # transfers between HBM and SRAM.
        O[
            start_row_tile_idx : start_row_tile_idx + ROW_TILE_SIZE,
            q_tile_idx : q_tile_idx + Q_TILE_SIZE,
        ] = (
            oi / li
        )
        L[
            start_row_tile_idx : start_row_tile_idx + ROW_TILE_SIZE,
            q_tile_idx : q_tile_idx + Q_TILE_SIZE,
        ] = (mi + torch.log(li)).squeeze(-1)


class PyTorchFlashAttention(torch.autograd.Function):
    """Flash attention implementation.

    Used to debug the actual triton implementation."""

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

        ctx.ROWS_TILE_SIZE = 16
        ctx.Q_TILE_SIZE = 64
        ctx.K_TILE_SIZE = 64

        # Reshape the inputs to 3D tensors to simplify the algorithm.
        output_shape = Q.shape
        Q_flat = rearrange(Q, "... queries d -> (...) queries d")
        K_flat = rearrange(K, "... keys d -> (...) keys d")
        V_flat = rearrange(V, "... keys d -> (...) keys d")

        n_rows, queries = Q_flat.shape[0], Q_flat.shape[1]
        device, dtype = Q.device, Q.dtype

        num_programs = triton.cdiv(n_rows, ctx.ROWS_TILE_SIZE)
        O_flat = torch.zeros_like(Q_flat)
        L_flat = torch.zeros((n_rows, queries), device=device, dtype=dtype)
        for pid in range(num_programs):
            pytorch_flash_attention(
                pid,
                Q_flat,
                K_flat,
                V_flat,
                is_causal,
                O_flat,
                L_flat,
                ctx.Q_TILE_SIZE,
                ctx.K_TILE_SIZE,
                ctx.ROWS_TILE_SIZE,
            )

        ctx.save_for_backward(Q_flat, K_flat, V_flat, O_flat, L_flat)
        return O_flat.view(output_shape)

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("Backward pass not yet implemented.")
