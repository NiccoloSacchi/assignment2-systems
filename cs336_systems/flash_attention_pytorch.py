import triton
import math
import torch
from torch import Tensor
from jaxtyping import Float
from einops import rearrange, einsum


def pytorch_flash_attention(
    # query_tile_index and batch_index represent the program IDs in a Triton
    # grid.
    query_tile_index: int,
    seq_index: int,
    Q: Float[Tensor, "n_sequences queries d"],
    K: Float[Tensor, "n_sequences keys d"],
    V: Float[Tensor, "n_sequences keys d"],
    is_causal: bool,
    O: Float[Tensor, "n_sequences queries d"],
    L: Float[Tensor, "n_sequences queries"],
    Q_TILE_SIZE: int,
    K_TILE_SIZE: int,
):
    # Each 'triton program' only processes 1 query tile of 1 sequence.
    Q_seq = Q[seq_index]
    K_seq = K[seq_index]
    V_seq = V[seq_index]

    _, d = Q_seq.shape
    device, dtype = Q_seq.device, Q_seq.dtype
    n_keys = K_seq.shape[0]
    softmax_scale = 1.0 / math.sqrt(d)

    # Load the specific Query tile for this program instance
    q_start = query_tile_index * Q_TILE_SIZE
    q_tile = Q_seq[q_start : q_start + Q_TILE_SIZE, :]
    queries_in_tile = q_tile.shape[0]

    # Initialize accumulators in "SRAM" (Registers).
    mi = torch.full((queries_in_tile, 1), float("-inf"), device=device, dtype=dtype)
    li = torch.zeros((queries_in_tile, 1), device=device, dtype=dtype)
    oi = torch.zeros((queries_in_tile, d), device=device, dtype=dtype)

    # The single loop over key tiles, as required by Algorithm 1
    for k_tile_idx in range(0, n_keys, K_TILE_SIZE):
        k_tile = K_seq[k_tile_idx : k_tile_idx + K_TILE_SIZE, :]
        v_tile = V_seq[k_tile_idx : k_tile_idx + K_TILE_SIZE, :]
        s_tile = (
            einsum(
                q_tile,
                k_tile,
                "queries_tile d, keys_tile d -> queries_tile keys_tile",
            )
            * softmax_scale
        )

        # Optional: Add causal masking logic here
        mi_new = torch.max(mi, torch.max(s_tile, dim=-1, keepdim=True).values)
        p_tile = torch.exp(s_tile - mi_new)
        alpha = torch.exp(mi - mi_new)
        li = alpha * li + torch.sum(p_tile, dim=-1, keepdim=True)
        oi = alpha * oi + einsum(
            p_tile,
            v_tile,
            "queries_tile keys_tile, keys_tile d -> queries_tile d",
        )
        mi = mi_new

    # Use same approach as in a triton kernel: use temporary output tensors
    # to hold the final results and write only once HBM to minimize data
    # transfers between HBM and SRAM.
    O[seq_index, q_start : q_start + queries_in_tile, :] = oi / li
    L[seq_index, q_start : q_start + queries_in_tile] = (mi + torch.log(li)).squeeze(-1)


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

        n_sequences, n_queries, _ = Q_flat.shape
        n_q_tiles = triton.cdiv(n_queries, ctx.Q_TILE_SIZE)

        O_flat = torch.zeros_like(Q_flat)
        L_flat = torch.zeros((n_sequences, n_queries), device=Q.device, dtype=Q.dtype)

        # Launch grid simulation: (Tq, batch_size)
        for seq_index in range(n_sequences):
            for query_tile_index in range(n_q_tiles):
                pytorch_flash_attention(
                    query_tile_index,
                    seq_index,
                    Q_flat,
                    K_flat,
                    V_flat,
                    is_causal,
                    O_flat,
                    L_flat,
                    ctx.Q_TILE_SIZE,
                    ctx.K_TILE_SIZE,
                )

        ctx.save_for_backward(Q_flat, K_flat, V_flat, O_flat, L_flat)
        return O_flat.view(output_shape)
