"""Flash attention implementation.

Used to debug the actual triton implementation. Tested with:
uv run modal run scripts/execute_tests.py --k test_flash_forward_pass_pytorch
uv run modal run scripts/execute_tests.py --k test_flash_backward_pytorch
"""

import math
import torch
from torch import Tensor
from jaxtyping import Float
from einops import rearrange, einsum


def flash_fwd_kernel(
    # query_tile_index and seq_index represent the program IDs in a Triton grid.
    query_tile_index: int,
    seq_index: int,
    Q: Float[Tensor, "n_sequences queries d"],
    K: Float[Tensor, "n_sequences keys d"],
    V: Float[Tensor, "n_sequences keys d"],
    is_causal: bool,
    O: Float[Tensor, "n_sequences queries d"],
    L: Float[Tensor, "n_sequences queries"],
    N_KEYS: int,
    softmax_scale: float,
    D: int,
    Q_TILE_SIZE: int,
    K_TILE_SIZE: int,
):
    """Function pretending to be a triton kernel implementing the forward."""
    # Each 'triton program' only processes 1 query tile of 1 sequence.
    Q_seq = Q[seq_index]
    K_seq = K[seq_index]
    V_seq = V[seq_index]

    device, dtype = Q_seq.device, Q_seq.dtype

    # Load the specific Query tile for this program instance
    q_start = query_tile_index * Q_TILE_SIZE
    q_tile = Q_seq[q_start : q_start + Q_TILE_SIZE, :]
    queries_in_tile = q_tile.shape[0]

    # Initialize accumulators in "SRAM" (Registers).
    mi = torch.full((queries_in_tile, 1), float("-inf"), device=device, dtype=dtype)
    li = torch.zeros((queries_in_tile, 1), device=device, dtype=dtype)
    oi = torch.zeros((queries_in_tile, D), device=device, dtype=dtype)

    for k_tile_idx in range(0, N_KEYS, K_TILE_SIZE):
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


def flash_backward_kernel(
    # query_tile_index and seq_index represent the program IDs in a Triton grid.
    key_tile_index: int,
    seq_index: int,
    Q: Float[Tensor, "n_sequences queries d"],
    K: Float[Tensor, "n_sequences keys d"],
    V: Float[Tensor, "n_sequences keys d"],
    is_causal: bool,
    O: Float[Tensor, "n_sequences queries d"],
    dO: Float[Tensor, "n_sequences queries d"],
    L: Float[Tensor, "n_sequences queries"],
    D: Float[Tensor, "n_sequences queries"],
    softmax_scale: float,
    dQ: Float[Tensor, "n_sequences queries d"],
    dK: Float[Tensor, "n_sequences keys d"],
    dV: Float[Tensor, "n_sequences keys d"],
    Q_TILE_SIZE: int,
    K_TILE_SIZE: int,
):
    """Function pretending to be a triton kernel implementing the backward."""
    # Make sure this function touches only the assigned sequence and key.
    Q_seq = Q[seq_index]
    O_seq = O[seq_index]
    dO_seq = dO[seq_index]
    L_seq = L[seq_index]
    D_seq = D[seq_index]
    k_start = key_tile_index * K_TILE_SIZE
    Kj = K[seq_index, k_start : k_start + K_TILE_SIZE, :]
    Vj = V[seq_index, k_start : k_start + K_TILE_SIZE, :]
    dQ_seq = dQ[seq_index]
    dK_seq = dK[seq_index]
    dV_seq = dV[seq_index]

    # Initialize "in the SRAM" dQ, dK and dV accumulators for this tile.
    dKj = torch.zeros_like(Kj)
    dVj = torch.zeros_like(Vj)

    n_queries = Q_seq.shape[0]
    for q_tile_start in range(0, n_queries, Q_TILE_SIZE):
        Qi = Q_seq[q_tile_start : q_tile_start + Q_TILE_SIZE, :]
        dOi = dO_seq[q_tile_start : q_tile_start + Q_TILE_SIZE, :]
        Li = L_seq[q_tile_start : q_tile_start + Q_TILE_SIZE]
        Di = D_seq[q_tile_start : q_tile_start + Q_TILE_SIZE]

        Si = (
            einsum(
                Qi,
                Kj,
                "queries_tile d, keys_tile d -> queries_tile keys_tile",
            )
            * softmax_scale
        )
        Pi = torch.exp(Si - Li[:, None])
        dVj += einsum(
            Pi,
            dOi,
            "queries_tile keys_tile, queries_tile d -> keys_tile d",
        )
        dPi = einsum(
            dOi,
            Vj,
            "queries_tile d, keys_tile d -> queries_tile keys_tile",
        )
        dSi = (Pi * (dPi - Di[:, None])) * softmax_scale

        # Write back dQ tile "to HBM".
        dQ_seq[q_tile_start : q_tile_start + Q_TILE_SIZE, :] += einsum(
            dSi,
            Kj,
            "queries_tile keys_tile, keys_tile d -> queries_tile d",
        )
        dKj += einsum(
            dSi,
            Qi,
            "queries_tile keys_tile, queries_tile d -> keys_tile d",
        )

    # Write back dK and dV tiles "to HBM".
    dK_seq[k_start : k_start + K_TILE_SIZE, :] = dKj
    dV_seq[k_start : k_start + K_TILE_SIZE, :] = dVj


class PyTorchFlashAttention(torch.autograd.Function):
    """Flash attention implementation.

    Used to debug the actual triton implementation."""

    @staticmethod
    def forward(
        ctx,
        Q: Float[Tensor, "... queries d"],
        K: Float[Tensor, "... keys d"],
        V: Float[Tensor, "... keys d"],
        is_causal: bool = False,
    ) -> Float[Tensor, "... queries d"]:
        # Disabling gradient calculation because:
        # - We are implementing the Flash backward below.
        # - Makes this implementation actually O(N) in memory.
        with torch.no_grad():
            assert Q.shape[:-2] == K.shape[:-2] == V.shape[:-2], "Batch/Head mismatch"
            assert Q.shape[-1] == K.shape[-1] == V.shape[-1], "Embedding D mismatch"
            assert K.shape[-2] == V.shape[-2], "Sequence length Nk mismatch"
            assert not is_causal, "is_causal is not supported"

            ctx.Q_TILE_SIZE = 64
            ctx.K_TILE_SIZE = 64
            ctx.is_causal = is_causal
            ctx.Q_shape = Q.shape
            ctx.K_shape = K.shape

            # Reshape the inputs to 3D tensors to simplify the algorithm.
            Q_flat = rearrange(Q, "... queries d -> (...) queries d")
            K_flat = rearrange(K, "... keys d -> (...) keys d")
            V_flat = rearrange(V, "... keys d -> (...) keys d")

            n_sequences, n_queries, d = Q_flat.shape
            n_keys = K_flat.shape[1]
            softmax_scale = 1.0 / math.sqrt(d)

            O_flat = torch.zeros_like(Q_flat)
            L_flat = torch.zeros(
                (n_sequences, n_queries), device=Q.device, dtype=Q.dtype
            )

            # Launch grid simulation: (#q tiles, #sequences)
            n_q_tiles = math.ceil(n_queries / ctx.Q_TILE_SIZE)
            for seq_index in range(n_sequences):
                for query_tile_index in range(n_q_tiles):
                    flash_fwd_kernel(
                        query_tile_index=query_tile_index,
                        seq_index=seq_index,
                        Q=Q_flat,
                        K=K_flat,
                        V=V_flat,
                        is_causal=is_causal,
                        O=O_flat,
                        L=L_flat,
                        N_KEYS=n_keys,
                        softmax_scale=softmax_scale,
                        D=d,
                        Q_TILE_SIZE=ctx.Q_TILE_SIZE,
                        K_TILE_SIZE=ctx.K_TILE_SIZE,
                    )

            ctx.save_for_backward(Q_flat, K_flat, V_flat, O_flat, L_flat)
            return O_flat.view(Q.shape)

    @staticmethod
    def backward(ctx, dO):
        with torch.no_grad():
            Q_flat, K_flat, V_flat, O_flat, L_flat = ctx.saved_tensors
            dO_flat = rearrange(dO, "... queries d -> (...) queries d")
            assert (
                dO_flat.shape == O_flat.shape
            ), "Output gradient shape and output shape mismatch"

            n_sequences, n_keys, d = K_flat.shape
            n_k_tiles = math.ceil(n_keys / ctx.K_TILE_SIZE)
            dQ_flat = torch.zeros_like(Q_flat)
            dK_flat = torch.zeros_like(K_flat)
            dV_flat = torch.zeros_like(V_flat)
            D = einsum(dO_flat, O_flat, "... n d, ... n d -> ... n")
            softmax_scale = 1.0 / math.sqrt(d)
            for seq_index in range(n_sequences):
                for key_tile_index in range(n_k_tiles):
                    flash_backward_kernel(
                        key_tile_index=key_tile_index,
                        seq_index=seq_index,
                        Q=Q_flat,
                        K=K_flat,
                        V=V_flat,
                        O=O_flat,
                        L=L_flat,
                        dO=dO_flat,
                        D=D,
                        is_causal=ctx.is_causal,
                        softmax_scale=softmax_scale,
                        dQ=dQ_flat,
                        dK=dK_flat,
                        dV=dV_flat,
                        Q_TILE_SIZE=ctx.Q_TILE_SIZE,
                        K_TILE_SIZE=ctx.K_TILE_SIZE,
                    )
        return (
            dQ_flat.reshape(ctx.Q_shape),
            dK_flat.reshape(ctx.K_shape),
            dV_flat.reshape(ctx.K_shape),
            None,  # No gradient for is_causal
        )
