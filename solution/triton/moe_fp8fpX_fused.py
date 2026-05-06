#!/usr/bin/env python3
"""
Helion-optimized MoE FP8 kernel with hybrid approach:
- Host-side: Dequantization (FP8 → BF16) and routing
- Helion kernel: Expert computation (GEMM1 → SwiGLU → GEMM2)

BF16 has the same dynamic range as FP32 (±3.4e38), avoiding the overflow
issues of FP16 (±65504) while still reducing memory vs FP32.

Usage: python moe_fp8bf16_local_e_dq.py [--workload WORKLOAD_FILE]
"""

import torch
import json
import argparse
import os
from pathlib import Path
from safetensors import safe_open

# Set MOE_DEBUG=1 or pass --debug to enable diagnostic prints.
_DEBUG = os.environ.get("MOE_DEBUG", "0") == "1"

import helion
import helion.language as hl

import triton
import triton.language as tl

import pdb

neg_inf = torch.finfo(torch.float32).min


# ===========================================
# Helper Functions (Host-side)
# ===========================================

def dequantize_fp8_activations_bf16(
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    BLOCK: int = 128
) -> torch.Tensor:
    """Dequantize FP8 activations → BF16 with block-scale dequantization."""
    T, H = hidden_states.shape
    A = hidden_states.to(torch.bfloat16)
    A_scale = hidden_states_scale.to(torch.bfloat16)   # [H/BLOCK, T]
    A_scale_TH = A_scale.permute(1, 0).contiguous()    # [T, H/BLOCK]
    A_scale_expanded = (
        A_scale_TH.unsqueeze(-1)
        .repeat(1, 1, BLOCK)
        .reshape(T, H)
        .contiguous()
    )
    return A * A_scale_expanded


def dequantize_fp8_activations_fp16(
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    BLOCK: int = 128
) -> torch.Tensor:
    """Dequantize FP8 activations → FP16 with block-scale dequantization."""
    T, H = hidden_states.shape
    A = hidden_states.to(torch.float16)
    A_scale = hidden_states_scale.to(torch.float16)    # [H/BLOCK, T]
    A_scale_TH = A_scale.permute(1, 0).contiguous()   # [T, H/BLOCK]
    A_scale_expanded = (
        A_scale_TH.unsqueeze(-1)
        .repeat(1, 1, BLOCK)
        .reshape(T, H)
        .contiguous()
    )
    return A * A_scale_expanded


def dequantize_fp8_activations_fp32(
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    BLOCK: int = 128
) -> torch.Tensor:
    """Dequantize FP8 activations → FP32 with block-scale dequantization."""
    T, H = hidden_states.shape
    A = hidden_states.to(torch.float32)
    A_scale = hidden_states_scale.to(torch.float32)   # [H/BLOCK, T]
    A_scale_TH = A_scale.permute(1, 0).contiguous()  # [T, H/BLOCK]
    A_scale_expanded = (
        A_scale_TH.unsqueeze(-1)
        .repeat(1, 1, BLOCK)
        .reshape(T, H)
        .contiguous()
    )
    return A * A_scale_expanded


# Alias for backward compatibility
dequantize_fp8_activations = dequantize_fp8_activations_bf16


def dequantize_fp8_weights_bf16(
    weights: torch.Tensor,
    weights_scale: torch.Tensor,
    BLOCK: int = 128
) -> torch.Tensor:
    """Dequantize FP8 weights → BF16 with block-scale dequantization."""
    W = weights.to(torch.bfloat16)
    S = weights_scale.to(torch.bfloat16)
    S_expanded = torch.repeat_interleave(S, BLOCK, dim=1)
    S_expanded = torch.repeat_interleave(S_expanded, BLOCK, dim=2)
    return W * S_expanded


def dequantize_fp8_weights_fp16(
    weights: torch.Tensor,
    weights_scale: torch.Tensor,
    BLOCK: int = 128
) -> torch.Tensor:
    """Dequantize FP8 weights → FP16 with block-scale dequantization."""
    W = weights.to(torch.float16)
    S = weights_scale.to(torch.float16)
    S_expanded = torch.repeat_interleave(S, BLOCK, dim=1)
    S_expanded = torch.repeat_interleave(S_expanded, BLOCK, dim=2)
    return W * S_expanded


def dequantize_fp8_weights_fp32(
    weights: torch.Tensor,
    weights_scale: torch.Tensor,
    BLOCK: int = 128
) -> torch.Tensor:
    """Dequantize FP8 weights → FP32 with block-scale dequantization."""
    W = weights.to(torch.float32)
    S = weights_scale.to(torch.float32)
    S_expanded = torch.repeat_interleave(S, BLOCK, dim=1)
    S_expanded = torch.repeat_interleave(S_expanded, BLOCK, dim=2)
    return W * S_expanded


# Alias for backward compatibility
dequantize_fp8_weights = dequantize_fp8_weights_bf16


def compute_deepseek_routing(
    routing_logits: torch.Tensor,
    routing_bias: torch.Tensor,
    routed_scaling_factor: float,
    TOP_K: int = 8,
    N_GROUP: int = 8,
    TOPK_GROUP: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    DeepSeek-V3 no-aux routing with grouped top-k selection.

    Args:
        routing_logits: [T, E_global] routing logits
        routing_bias: [E_global] routing bias
        routed_scaling_factor: scaling factor for weights

    Returns:
        topk_idx: [T, TOP_K] indices of selected experts
        weights: [T, E_global] normalized routing weights
    """
    T, E_global = routing_logits.shape

    # Sigmoid — keep in fp32: bf16 precision causes different top-k selections
    logits = routing_logits.to(torch.float32)
    bias = routing_bias.to(torch.float32).reshape(-1)
    s = 1.0 / (1.0 + torch.exp(-logits))
    s_with_bias = s + bias

    # Grouping
    group_size = E_global // N_GROUP
    s_wb_grouped = s_with_bias.view(T, N_GROUP, group_size)

    # Group scores = sum of top-2 values within each group
    top2_vals, _ = torch.topk(s_wb_grouped, k=2, dim=2, largest=True, sorted=False)
    group_scores = top2_vals.sum(dim=2)

    # Select topk_group groups → group mask
    _, group_idx = torch.topk(group_scores, k=TOPK_GROUP, dim=1, largest=True, sorted=False)
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1.0)
    score_mask = group_mask.unsqueeze(2).expand(T, N_GROUP, group_size).reshape(T, E_global)

    # Global top-k (within kept groups)
    scores_pruned = s_with_bias.masked_fill(score_mask == 0, neg_inf)
    _, topk_idx = torch.topk(scores_pruned, k=TOP_K, dim=1, largest=True, sorted=False)

    # Combination weights: use s (without bias) for normalization
    M = torch.zeros_like(s)
    M.scatter_(1, topk_idx, 1.0)
    weights = s * M
    weights_sum = weights.sum(dim=1, keepdim=True) + 1e-20
    weights = (weights / weights_sum) * routed_scaling_factor

    return topk_idx, weights


# ===========================================
# Expert Computation (PyTorch version - to be replaced with Helion)
# ===========================================

@torch.no_grad()
def expert_computation_bf16(
    A: torch.Tensor,           # [T, H] activations
    W13: dict,                 # {le_int: [2*I, H]} BF16 dequantized GEMM1 weights
    W2: dict,                  # {le_int: [H, I]}   BF16 dequantized GEMM2 weights
    topk_idx: torch.Tensor,
    weights: torch.Tensor,
    local_expert_offset: int,
    E_global: int,
) -> torch.Tensor:
    """Expert computation: pure BF16 matmul. Accumulation in FP32."""
    T, H = A.shape
    I = next(iter(W2.values())).shape[1]
    device = A.device
    local_start = int(local_expert_offset)

    temp_output = torch.zeros((T, H), dtype=torch.float32, device=device)

    for le_int, W13_e in W13.items():
        ge = local_start + le_int
        sel_mask = (topk_idx == ge).any(dim=1)
        token_idx = sel_mask.nonzero(as_tuple=False).squeeze(1)
        if token_idx.numel() == 0:  # CPU-side check — no additional GPU sync
            continue
        A_e  = A.index_select(0, token_idx).to(torch.bfloat16)
        W13_e = W13_e.to(torch.bfloat16)
        W2_e  = W2[le_int].to(torch.bfloat16)
        G1 = A_e.matmul(W13_e.t())
        X1, X2 = G1[:, :I], G1[:, I:]
        C = (X2 / (1.0 + torch.exp(-X2))) * X1
        O = C.matmul(W2_e.t())
        w_tok = weights.index_select(0, token_idx)[:, ge]
        temp_output.index_add_(0, token_idx, (O * w_tok.unsqueeze(1)).to(torch.float32))

    return temp_output


@torch.no_grad()
def expert_computation_fp16(
    A: torch.Tensor,           # [T, H] activations
    W13: dict,                 # {le_int: [2*I, H]} FP16 dequantized GEMM1 weights
    W2: dict,                  # {le_int: [H, I]}   FP16 dequantized GEMM2 weights
    topk_idx: torch.Tensor,
    weights: torch.Tensor,
    local_expert_offset: int,
    E_global: int,
) -> torch.Tensor:
    """Expert computation: pure FP16 matmul. Accumulation in FP32."""
    T, H = A.shape
    I = next(iter(W2.values())).shape[1]
    device = A.device
    local_start = int(local_expert_offset)

    temp_output = torch.zeros((T, H), dtype=torch.float32, device=device)

    for le_int, W13_e in W13.items():
        ge = local_start + le_int
        sel_mask = (topk_idx == ge).any(dim=1)
        token_idx = sel_mask.nonzero(as_tuple=False).squeeze(1)
        if token_idx.numel() == 0:  # CPU-side check — no additional GPU sync
            continue
        A_e  = A.index_select(0, token_idx).to(torch.float16)
        W13_e = W13_e.to(torch.float16)
        W2_e  = W2[le_int].to(torch.float16)
        G1 = A_e.matmul(W13_e.t())
        X1, X2 = G1[:, :I], G1[:, I:]
        C = (X2 / (1.0 + torch.exp(-X2))) * X1
        O = C.matmul(W2_e.t())
        w_tok = weights.index_select(0, token_idx)[:, ge]
        temp_output.index_add_(0, token_idx, (O * w_tok.unsqueeze(1)).to(torch.float32))

    return temp_output


@torch.no_grad()
def expert_computation_fp32(
    A: torch.Tensor,           # [T, H] activations
    W13: dict,                 # {le_int: [2*I, H]} dequantized GEMM1 weights
    W2: dict,                  # {le_int: [H, I]}   dequantized GEMM2 weights
    topk_idx: torch.Tensor,
    weights: torch.Tensor,
    local_expert_offset: int,
    E_global: int,
) -> torch.Tensor:
    """Expert computation: FP32 matmul, TF32 disabled."""
    T, H = A.shape
    I = next(iter(W2.values())).shape[1]
    device = A.device
    local_start = int(local_expert_offset)

    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    temp_output = torch.zeros((T, H), dtype=torch.float32, device=device)

    for le_int, W13_e in W13.items():
        ge = local_start + le_int
        sel_mask = (topk_idx == ge).any(dim=1)
        token_idx = sel_mask.nonzero(as_tuple=False).squeeze(1)
        if token_idx.numel() == 0:  # CPU-side check — no additional GPU sync
            continue
        A_e  = A.index_select(0, token_idx).to(torch.float32)
        W13_e = W13_e.to(torch.float32)
        W2_e  = W2[le_int].to(torch.float32)
        G1 = A_e.matmul(W13_e.t())
        X1, X2 = G1[:, :I], G1[:, I:]
        C = (X2 / (1.0 + torch.exp(-X2))) * X1
        O = C.matmul(W2_e.t())
        w_tok = weights.index_select(0, token_idx)[:, ge]
        temp_output.index_add_(0, token_idx, O * w_tok.unsqueeze(1))

    torch.backends.cuda.matmul.allow_tf32 = prev_tf32
    return temp_output


@torch.no_grad()
def expert_computation_tf32(
    A: torch.Tensor,           # [T, H] activations
    W13: dict,                 # {le_int: [2*I, H]} dequantized GEMM1 weights
    W2: dict,                  # {le_int: [H, I]}   dequantized GEMM2 weights
    topk_idx: torch.Tensor,
    weights: torch.Tensor,
    local_expert_offset: int,
    E_global: int,
) -> torch.Tensor:
    """Expert computation: FP32 matmul, TF32 enabled (10-bit mantissa tensor cores)."""
    T, H = A.shape
    I = next(iter(W2.values())).shape[1]
    device = A.device
    local_start = int(local_expert_offset)

    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    temp_output = torch.zeros((T, H), dtype=torch.float32, device=device)

    for le_int, W13_e in W13.items():
        ge = local_start + le_int
        sel_mask = (topk_idx == ge).any(dim=1)
        token_idx = sel_mask.nonzero(as_tuple=False).squeeze(1)
        if token_idx.numel() == 0:  # CPU-side check — no additional GPU sync
            continue
        A_e  = A.index_select(0, token_idx).to(torch.float32)
        W13_e = W13_e.to(torch.float32)
        W2_e  = W2[le_int].to(torch.float32)
        G1 = A_e.matmul(W13_e.t())
        X1, X2 = G1[:, :I], G1[:, I:]
        C = (X2 / (1.0 + torch.exp(-X2))) * X1
        O = C.matmul(W2_e.t())
        w_tok = weights.index_select(0, token_idx)[:, ge]
        temp_output.index_add_(0, token_idx, O * w_tok.unsqueeze(1))

    torch.backends.cuda.matmul.allow_tf32 = prev_tf32
    return temp_output


# ===========================================
# Helion Fused FP8 Dequant + GEMM Kernels
# ===========================================
# These kernels load FP8 weights on-chip, dequantize using block scales in
# registers, and feed directly into tl.dot — no fp32 weight tensor is ever
# written to global memory.
#
# Weight layout (from load_workload_from_jsonl):
#   gemm1_weights      [E, 2*I, H]          fp8_e4m3   (N=2*I=4096, K=H=7168)
#   gemm1_weights_scale[E, 2*I//128, H//128] fp32       (N_s=32, K_s=56)
#   gemm2_weights      [E, H, I]            fp8_e4m3   (N=H=7168, K=I=2048)
#   gemm2_weights_scale[E, H//128, I//128]  fp32       (N_s=56, K_s=16)
#
# BLOCK_K must equal SCALE_BLOCK=128 so one K-tile aligns with one scale column.

# Best config found by autotuning on wl9 (T=11948). Hardcoded to skip autotuning.
# Regenerate with: @helion.kernel(config=None, static_shapes=True) and run once.
hconfig_fp32 = helion.Config(
    block_sizes=[32, 128],
    indexing=['pointer', 'pointer', 'pointer', 'pointer'],
    l2_groupings=[1],
    load_eviction_policies=['', '', ''],
    loop_orders=[[0, 1]],
    num_stages=4,
    num_warps=4,
    pid_type='flat',
    range_flattens=[None, True],
    range_multi_buffers=[None, None],
    range_num_stages=[0, 2],
    range_unroll_factors=[0, 0],
    range_warp_specializes=[None, None],
)
@helion.kernel(config=hconfig_fp32, static_shapes=True)
def _helion_fp8_gemm_fp32(
    A: torch.Tensor,       # [M, K]      fp32 activations
    W_t: torch.Tensor,     # [K, N]      fp8_e4m3 weights, transposed (may be non-contiguous)
    S_kexp: torch.Tensor,  # [K//128, N] scales pre-expanded on N dim
) -> torch.Tensor:
    """
    Fused FP8-dequant GEMM: C = A @ W_t  (fp32 accumulation, TF32 off).
    - W is stored [K, N] (transposed view, possibly non-contiguous) so the inner tile is W_t[tile_k, tile_n].
    - Scale is [K//128, N]: for each K-tile, one row of scales covers all N columns.
    - tile_k.begin // 128 selects the correct scale row.
    """
    M, K = A.shape
    N = W_t.shape[1]
    C = torch.empty([M, N], dtype=torch.float32, device=A.device)

    for tile_m, tile_n in hl.tile([M, N]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(K, block_size=128):
            a_tile = A[tile_m, tile_k]                    # [bM, 128] fp32
            w_tile = W_t[tile_k, tile_n]                  # [128, bN] fp8
            k_s    = tile_k.begin // 128                  # scalar scale-row index
            s_row  = S_kexp[k_s : k_s + 1, tile_n]       # [1, bN]   fp32
            w_dq   = w_tile.to(torch.float32) * s_row     # [128, bN] fp32 — on-chip
            acc    = hl.dot(a_tile, w_dq, acc=acc)        # [bM,128] @ [128,bN] → [bM,bN]
        C[tile_m, tile_n] = acc

    return C

# Best config found by autotuning on wl9 (T=11948). Hardcoded to skip autotuning.
hconfig_tf32 = helion.Config(
    block_sizes=[32, 128],
    indexing=['pointer', 'pointer', 'pointer', 'pointer'],
    l2_groupings=[32],
    load_eviction_policies=['last', 'last', 'first'],
    loop_orders=[[0, 1]],
    num_stages=4,
    num_warps=4,
    pid_type='flat',
    range_flattens=[None, False],
    range_multi_buffers=[None, None],
    range_num_stages=[0, 0],
    range_unroll_factors=[0, 0],
    range_warp_specializes=[None, False],
)
@helion.kernel(config=hconfig_tf32, static_shapes=True)
def _helion_fp8_gemm_tf32(
    A: torch.Tensor,       # [M, K]      fp32 activations
    W_t: torch.Tensor,     # [K, N]      fp8_e4m3 weights, transposed (may be non-contiguous)
    S_kexp: torch.Tensor,  # [K//128, N] scales pre-expanded on N dim
) -> torch.Tensor:
    """
    Fused FP8-dequant GEMM: C = A @ W_t  (fp32 accumulation, TF32 on).
    Identical to _helion_fp8_gemm_fp32; TF32 is a global compute flag set by the caller.
    """
    M, K = A.shape
    N = W_t.shape[1]
    C = torch.empty([M, N], dtype=torch.float32, device=A.device)

    for tile_m, tile_n in hl.tile([M, N]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(K, block_size=128):
            a_tile = A[tile_m, tile_k]
            w_tile = W_t[tile_k, tile_n]
            k_s    = tile_k.begin // 128
            s_row  = S_kexp[k_s : k_s + 1, tile_n]
            w_dq   = w_tile.to(torch.float32) * s_row
            acc    = hl.dot(a_tile, w_dq, acc=acc)
        C[tile_m, tile_n] = acc

    return C


hconfig_fp8_native = helion.Config(
    block_sizes=[64, 128],
    # 5 tensor accesses: A (load), A_scale_T (load), W_t (load), S_kexp (load), C (store)
    indexing=['pointer', 'pointer', 'pointer', 'pointer', 'pointer'],
    l2_groupings=[32],
    load_eviction_policies=['last', 'last', 'first', 'first'],
    loop_orders=[[0, 1]],
    num_stages=5,
    num_warps=8,
    pid_type='flat',
    range_flattens=[None, False],
    range_multi_buffers=[None, None],
    range_num_stages=[0, 0],
    range_unroll_factors=[0, 0],
    range_warp_specializes=[None, False],
)
@helion.kernel(config=hconfig_fp8_native, static_shapes=True)
def _helion_fp8_gemm_native(
    A: torch.Tensor,          # [M, K]      fp8_e4m3fn activations (NOT pre-dequantized)
    A_scale_T: torch.Tensor,  # [M, K//128] fp32 per-token per-block activation scales (transposed)
    W_t: torch.Tensor,        # [K, N]      fp8_e4m3fn weights (transposed view)
    S_kexp: torch.Tensor,     # [K//128, N] fp32 weight scales (pre-expanded, same as S_kexp in fp32/tf32 kernels)
) -> torch.Tensor:            # [M, N]      fp32
    """
    Native FP8 tensor-core GEMM with per-128-element block scaling.

    tile_k is fixed at 128 = BLOCK_SIZE so each K-tile covers exactly one scale block.
    For each K-tile:
      partial = dot(A_fp8_tile, W_fp8_tile)           # FP8 tensor cores, fp32 accumulator
      acc    += partial * A_scale_T[m, k_s] * S_kexp[k_s, n]  # per-token × per-weight-block
    """
    M, K = A.shape
    N = W_t.shape[1]
    C = torch.empty([M, N], dtype=torch.float32, device=A.device)

    for tile_m, tile_n in hl.tile([M, N]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(K, block_size=128):
            a_tile = A[tile_m, tile_k]                    # [bM, 128] fp8
            w_tile = W_t[tile_k, tile_n]                  # [128, bN] fp8
            k_s    = tile_k.begin // 128

            partial = hl.dot(a_tile, w_tile)              # [bM, bN] fp32 via FP8 tensor cores

            s_a = A_scale_T[tile_m, k_s : k_s + 1]       # [bM, 1]  per-token scale
            s_w = S_kexp[k_s : k_s + 1, tile_n]          # [1, bN]  per-weight-block scale
            acc = acc + partial * s_a * s_w               # broadcast → [bM, bN]

        C[tile_m, tile_n] = acc

    return C


# Original config from commit 539e99b (pre-autotune, static_shapes=True).
# Autotuned variants (autotune_local.py / autotune_modal.py) did not improve over this.
hconfig_swiglu_fp32 = helion.Config(
    block_sizes=[32, 128],
    # 8 load/store ops: A, W13_val, S13_val, W13_gate, S13_gate, W2, S2, O(store)
    indexing=['pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer'],
    l2_groupings=[1],
    # 7 load eviction policies (loads only, not the store)
    load_eviction_policies=['', '', '', '', '', '', ''],
    loop_orders=[[0, 1]],
    num_stages=3,
    num_warps=4,
    pid_type='flat',
    range_flattens=[None, False, False],
    range_multi_buffers=[None, None, None],
    range_num_stages=[0, 0, 0],
    range_unroll_factors=[0, 0, 0],
    range_warp_specializes=[None, False, False],
)
@helion.kernel(config=hconfig_swiglu_fp32, static_shapes=True)
def _helion_fp8_swiglu_fused_fp32(
    A: torch.Tensor,             # [Te, H]       fp32 activations
    W13_val_t: torch.Tensor,     # [H,  I]       fp8_e4m3, value half of W13, transposed
    S13_val_kexp: torch.Tensor,  # [H//128, I]   fp32 scales pre-expanded
    W13_gate_t: torch.Tensor,    # [H,  I]       fp8_e4m3, gate half of W13, transposed
    S13_gate_kexp: torch.Tensor, # [H//128, I]   fp32 scales pre-expanded
    W2_t: torch.Tensor,          # [I,  H_out]   fp8_e4m3, transposed
    S2_kexp: torch.Tensor,       # [I//128, H_out] fp32 scales pre-expanded
) -> torch.Tensor:               # [Te, H_out]   fp32
    """
    Fused: O = (A @ W13_val_t * silu(A @ W13_gate_t)) @ W2_t  (fp32, no TF32).

    W13 is pre-split by the caller into val [H, I] and gate [H, I] halves to avoid
    tile+offset indexing (which produces wrong shapes in Helion's Triton codegen).

    Tiles over (Te, H_out); reduces I without writing the [Te, I] intermediates to
    global memory.  Per 128-wide tile_k of I:
      1. Accumulate both GEMM1 halves over the H dimension on-chip.
      2. Apply SwiGLU.
      3. Dot into the output accumulator for GEMM2.
    """
    Te, H = A.shape
    I = W2_t.shape[0]
    H_out = W2_t.shape[1]
    O = torch.empty([Te, H_out], dtype=torch.float32, device=A.device)

    for tile_m, tile_n in hl.tile([Te, H_out]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)

        for tile_k in hl.tile(I, block_size=128):
            # --- GEMM1: accumulate val and gate halves on-chip ---
            g1_val  = hl.zeros([tile_m, tile_k], dtype=torch.float32)
            g1_gate = hl.zeros([tile_m, tile_k], dtype=torch.float32)

            for tile_h in hl.tile(H, block_size=128):
                a_tile = A[tile_m, tile_h]                               # [bM, 128] fp32
                h_s    = tile_h.begin // 128                             # H-block → scale row

                w13_val  = W13_val_t[tile_h, tile_k]                     # [128, bK] fp8
                s13_val  = S13_val_kexp[h_s : h_s + 1, tile_k]          # [1,   bK] fp32
                g1_val   = hl.dot(a_tile, w13_val.to(torch.float32) * s13_val, acc=g1_val)

                w13_gate = W13_gate_t[tile_h, tile_k]                    # [128, bK] fp8
                s13_gate = S13_gate_kexp[h_s : h_s + 1, tile_k]         # [1,   bK] fp32
                g1_gate  = hl.dot(a_tile, w13_gate.to(torch.float32) * s13_gate, acc=g1_gate)

            # --- SwiGLU ---
            c_e = (g1_gate / (1.0 + torch.exp(-g1_gate))) * g1_val      # [bM, bK]

            # --- GEMM2 ---
            k_s2 = tile_k.begin // 128                                   # I-block → scale row
            w2   = W2_t[tile_k, tile_n]                                  # [bK, bN] fp8
            s2   = S2_kexp[k_s2 : k_s2 + 1, tile_n]                     # [1,  bN] fp32
            acc  = hl.dot(c_e, w2.to(torch.float32) * s2, acc=acc)

        O[tile_m, tile_n] = acc

    return O



@helion.kernel(config=None, static_shapes=True)
def _helion_fp8_grouped_swiglu_fused_fp32(
    A_flat:        torch.Tensor,   # [N, H]               fp32  (N padded per expert to _GROUPED_TILE_M)
    expert_ids:    torch.Tensor,   # [N]                  int32 0-based index into the flat weight rows
    W13_val_flat:  torch.Tensor,   # [E*H, I]             fp8_e4m3  expert e at rows [e*H, (e+1)*H)
    S13_val_flat:  torch.Tensor,   # [E*(H//128), I]      fp32      expert e at rows [e*H_128, (e+1)*H_128)
    W13_gate_flat: torch.Tensor,   # [E*H, I]             fp8_e4m3
    S13_gate_flat: torch.Tensor,   # [E*(H//128), I]      fp32
    W2_flat:       torch.Tensor,   # [E*I, H_out]         fp8_e4m3  expert e at rows [e*I, (e+1)*I)
    S2_flat:       torch.Tensor,   # [E*(I//128), H_out]  fp32      expert e at rows [e*I_128, (e+1)*I_128)
) -> torch.Tensor:                 # [N, H_out]           fp32
    """
    Grouped GEMM version of _helion_fp8_swiglu_fused_fp32.

    All selected local experts are processed in a single kernel launch.  The
    flat token array is sorted by expert; each expert's segment is padded to a
    multiple of _GROUPED_TILE_M so every M-tile belongs to exactly one expert.

    Weight/scale tensors are 2D with experts concatenated along dim-0 to avoid
    3D batch indexing (unsupported by Helion's current codegen).  Dynamic row
    offsets are computed as `le * H + tile_h.begin`, using the same
    dynamic-slice pattern already used in the single-expert kernel for scales.

    Tiling:  (N, H_out) → outer dims
              I          → reduction axis (outer loop, 128-wide tiles)
              H          → contraction axis (inner loop, 128-wide tiles)
    """
    N, H   = A_flat.shape
    I      = W13_val_flat.shape[1]
    H_out  = W2_flat.shape[1]
    H_128  = H // 128
    I_128  = I // 128
    O = torch.empty([N, H_out], dtype=torch.float32, device=A_flat.device)

    for tile_m, tile_n in hl.tile([N, H_out]):
        le  = expert_ids[tile_m.begin]           # scalar: 0-based expert index for this M-tile
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)

        for tile_k in hl.tile(I, block_size=128):
            # Absolute row offsets into W2_flat / S2_flat for this expert + k-tile
            k_abs   = le * I     + tile_k.begin
            k_s_abs = le * I_128 + tile_k.begin // 128

            g1_val  = hl.zeros([tile_m, tile_k], dtype=torch.float32)
            g1_gate = hl.zeros([tile_m, tile_k], dtype=torch.float32)

            for tile_h in hl.tile(H, block_size=128):
                a_tile  = A_flat[tile_m, tile_h]                               # [bM, 128] fp32

                # Absolute row offsets into W13_*_flat / S13_*_flat
                h_abs   = le * H     + tile_h.begin
                h_s_abs = le * H_128 + tile_h.begin // 128

                w13_v   = W13_val_flat [h_abs   : h_abs   + 128, tile_k]      # [128, bK] fp8
                s13_v   = S13_val_flat [h_s_abs : h_s_abs + 1,   tile_k]      # [1,   bK] fp32
                g1_val  = hl.dot(a_tile, w13_v.to(torch.float32) * s13_v,  acc=g1_val)

                w13_g   = W13_gate_flat[h_abs   : h_abs   + 128, tile_k]      # [128, bK] fp8
                s13_g   = S13_gate_flat[h_s_abs : h_s_abs + 1,   tile_k]      # [1,   bK] fp32
                g1_gate = hl.dot(a_tile, w13_g.to(torch.float32) * s13_g, acc=g1_gate)

            # SwiGLU
            c_e = (g1_gate / (1.0 + torch.exp(-g1_gate))) * g1_val            # [bM, bK]

            # GEMM2
            w2_tile = W2_flat[k_abs   : k_abs   + 128, tile_n]                # [128, bN] fp8
            s2_tile = S2_flat[k_s_abs : k_s_abs + 1,   tile_n]               # [1,   bN] fp32
            acc     = hl.dot(c_e, w2_tile.to(torch.float32) * s2_tile, acc=acc)

        O[tile_m, tile_n] = acc

    return O

# Tile-M size for the grouped GEMM wrapper.  Each expert's token segment is
# padded to a multiple of this value so no M-tile straddles an expert boundary.
# Must match (or divide) the BM block size Helion actually chooses at compile
# time.  64 is a safe default for H100/A100 fp8 workloads.
_GROUPED_TILE_M = 64

# Tile-M for the raw Triton grouped path.  32 halves register pressure vs 64
# (BLOCK_M=64 needs ~768 fp32 regs/thread, pushing against H100's 256-reg limit
# per warp; 32 fits comfortably and keeps better occupancy).
_TRITON_GROUPED_TILE_M = 64

# ── Module-level weight cache for triton_grouped_fp32 ─────────────────────────
# Building the flat, contiguous weight tensors from the per-expert fp8 dicts
# takes ~3 GB of D2D copies (all experts, val + gate + W2).  Since model weights
# are constant across decode steps we cache them keyed by the identity of the
def _get_triton_flat_weights(
    gemm1_weights:       torch.Tensor,   # [E, 2*I, H]          fp8
    gemm1_weights_scale: torch.Tensor,   # [E, 2*I//128, H//128] fp32
    gemm2_weights:       torch.Tensor,   # [E, H,   I]          fp8
    gemm2_weights_scale: torch.Tensor,   # [E, H//128, I//128]  fp32
    le_list: list,                       # sorted local expert indices
):
    """
    Build the flat 2-D weight/scale tensors needed by
    _triton_fp8_grouped_swiglu_fused_fp32.

    Uses a single permute+reshape per tensor (one GPU kernel each) instead of
    a Python loop of per-expert .T.contiguous() calls followed by torch.cat.
    """
    idx = torch.tensor(le_list, dtype=torch.long, device=gemm1_weights.device)

    # gemm1_weights[idx]: [E_sel, 2*I, H] → permute(0,2,1) → [E_sel, H, 2*I]
    #                      → reshape → [E_sel*H, 2*I]
    W13_flat = gemm1_weights[idx].permute(0, 2, 1).reshape(-1, gemm1_weights.shape[1]).contiguous()

    # gemm1_weights_scale[idx]: [E_sel, 2*I//128, H//128] → permute → [E_sel, H//128, 2*I//128]
    #                            → reshape → [E_sel*H//128, 2*I//128]
    S13_flat = gemm1_weights_scale[idx].permute(0, 2, 1).reshape(-1, gemm1_weights_scale.shape[1]).contiguous()

    # gemm2_weights[idx]: [E_sel, H, I] → permute(0,2,1) → [E_sel, I, H]
    #                      → reshape → [E_sel*I, H]
    W2_flat = gemm2_weights[idx].permute(0, 2, 1).reshape(-1, gemm2_weights.shape[1]).contiguous()

    # gemm2_weights_scale[idx]: [E_sel, H//128, I//128] → permute → [E_sel, I//128, H//128]
    #                            → reshape → [E_sel*I//128, H//128]
    S2_flat = gemm2_weights_scale[idx].permute(0, 2, 1).reshape(-1, gemm2_weights_scale.shape[1]).contiguous()

    return (W13_flat, S13_flat, W2_flat, S2_flat)

# =============================================================================
# Raw Triton grouped GEMM (Scheme C) — avoids Helion's data-dependent-slice
# limitation by using tl.program_id and explicit pointer arithmetic.
# =============================================================================

@triton.jit
def _triton_fp8_grouped_swiglu_kernel(
    # ── pointers ──────────────────────────────────────────────────────────────
    A_ptr,                  # fp32   [N, H]
    Eid_ptr,                # int32  [N]               0-based expert index per token
    W13_ptr,                # fp8    [E*H,       2*I]  val at cols [:I], gate at cols [I:]
    S13_ptr,                # fp32   [E*H_128, 2*I_128] val scales [:I_128], gate [I_128:]
    W2_ptr,                 # fp8    [E*I,       H_out] GEMM2 weights
    S2_ptr,                 # fp32   [E*I_128,   H_128] GEMM2 scales  (H_128 = H//128)
    O_ptr,                  # fp32   [N, H_out]
    # ── dimensions ────────────────────────────────────────────────────────────
    N, H, I, H_out,
    H_128, I_128,
    # ── row strides (all tensors row-major, inner stride = 1) ─────────────────
    stride_an,              # = H
    stride_w13n,            # = 2*I      (combined val+gate)
    stride_s13n,            # = 2*I//128 (combined val+gate scales)
    stride_w2n,             # = H_out
    stride_s2n,             # = H//128   (compact scales)
    stride_on,              # = H_out
    # ── tile sizes (compile-time) ──────────────────────────────────────────────
    BLOCK_M: tl.constexpr,        # token tile  — must divide _GROUPED_TILE_M
    SCALE_BLOCK_N: tl.constexpr,  # H_out tile  — set to 128 (= FP8 scale block)
    SCALE_BLOCK_K: tl.constexpr,  # I tile      — set to 128 (= FP8 scale block)
    SCALE_BLOCK_H: tl.constexpr,  # H tile      — set to 128 (= FP8 scale block)
):
    """
    One CTA handles one [BLOCK_M, SCALE_BLOCK_N] output tile.

    All BLOCK_M tokens share the same expert (input sorted+padded invariant).
    Expert id is loaded as a scalar; absolute offsets into flat 2-D weight/scale
    tensors are computed as  le*H + h_start  etc. — ordinary integer arithmetic
    that raw Triton handles but Helion's type propagation cannot.

    Scale structure: each 128×128 block of the weight matrix has one fp32 scale.
    With SCALE_BLOCK_H = SCALE_BLOCK_K = SCALE_BLOCK_N = 128 every tile maps to
    exactly one scale block, so dequant is a single scalar load + multiply after tl.dot.
    """
    # Grid is (H_out//SCALE_BLOCK_N, N//BLOCK_M): pid_0=N-tile varies fastest.
    # CUDA launches CTAs in x-major order, so all 56 N-tiles for M-tile 0
    # (expert 0) start before M-tile 1.  This keeps the ~14 MB of weight tiles
    # per expert warm in the 24 MB L2 across all N-tiles, rather than loading
    # 32 experts × 14 MB = 448 MB simultaneously and thrashing the cache.
    pid_n = tl.program_id(0)   # H_out tile — varies fastest → expert-major scheduling
    pid_m = tl.program_id(1)   # token-row (M) tile

    offs_m = pid_m * BLOCK_M       + tl.arange(0, BLOCK_M)        # [BLOCK_M] token rows
    offs_n = pid_n * SCALE_BLOCK_N + tl.arange(0, SCALE_BLOCK_N)  # [SCALE_BLOCK_N] output cols

    # Load expert index (scalar) — all tokens in this M-tile share one expert.
    le = tl.load(Eid_ptr + pid_m * BLOCK_M).to(tl.int32)

    # Base row offsets in the flat expert-concatenated tensors.
    h_base   = le * H        # W13*_flat  expert le rows start here
    h_s_base = le * H_128    # S13*_flat  expert le scale rows start here
    k_base   = le * I        # W2_flat    expert le rows start here
    k_s_base = le * I_128    # S2_flat    expert le scale rows start here

    acc = tl.zeros((BLOCK_M, SCALE_BLOCK_N), dtype=tl.float32)

    # ── outer loop: I-tiles (K-dim of GEMM2 / output-dim of GEMM1) ───────────
    for k_start in range(0, I, SCALE_BLOCK_K):
        k_abs   = k_base   + k_start
        k_s_abs = k_s_base + k_start // SCALE_BLOCK_K  # scale row for this I-block

        offs_k = k_start + tl.arange(0, SCALE_BLOCK_K)

        g1_val  = tl.zeros((BLOCK_M, SCALE_BLOCK_K), dtype=tl.float32)
        g1_gate = tl.zeros((BLOCK_M, SCALE_BLOCK_K), dtype=tl.float32)

        # ── inner loop: H-tiles (K-dim of GEMM1) ─────────────────────────────
        for h_start in range(0, H, SCALE_BLOCK_H):
            h_abs   = h_base   + h_start
            h_s_abs = h_s_base + h_start // SCALE_BLOCK_H  # scale row for this H-block

            offs_h_local = h_start + tl.arange(0, SCALE_BLOCK_H)  # A  col indices (0-based in H)
            offs_h_abs   = h_abs   + tl.arange(0, SCALE_BLOCK_H)  # W13 row indices (absolute)

            # A tile [BLOCK_M, BLOCK_H] fp32 — caller casts A_flat to fp32 before launch
            a_tile = tl.load(
                A_ptr + offs_m[:, None] * stride_an + offs_h_local[None, :]
            )

            # W13 val  cols [offs_k],   gate cols [I + offs_k] — same row stride 2*I
            w13v = tl.load(
                W13_ptr + offs_h_abs[:, None] * stride_w13n + offs_k[None, :]
            ).to(tl.float32)
            s13v = tl.load(S13_ptr + h_s_abs * stride_s13n + k_start // SCALE_BLOCK_K)
            g1_val = g1_val + tl.dot(a_tile, w13v, input_precision="tf32", out_dtype=tl.float32) * s13v

            w13g = tl.load(
                W13_ptr + offs_h_abs[:, None] * stride_w13n + I + offs_k[None, :]
            ).to(tl.float32)
            s13g = tl.load(S13_ptr + h_s_abs * stride_s13n + I_128 + k_start // SCALE_BLOCK_K)
            g1_gate = g1_gate + tl.dot(a_tile, w13g, input_precision="tf32", out_dtype=tl.float32) * s13g

        # ── SwiGLU: val * SiLU(gate) ─────────────────────────────────────────
        c_e = (g1_gate / (1.0 + tl.exp(-g1_gate))) * g1_val   # [BLOCK_M, SCALE_BLOCK_K]

        # ── GEMM2: c_e @ W2_tile, dequant by scalar s2 ───────────────────────
        offs_k_abs = k_abs + tl.arange(0, SCALE_BLOCK_K)

        w2 = tl.load(
            W2_ptr + offs_k_abs[:, None] * stride_w2n + offs_n[None, :]
        ).to(tl.float32)                                        # [SCALE_BLOCK_K, SCALE_BLOCK_N]

        # compact scale: col index = pid_n (each pid_n covers exactly SCALE_BLOCK_N=128 H_out cols)
        s2 = tl.load(S2_ptr + k_s_abs * stride_s2n + pid_n)

        acc = acc + tl.dot(c_e, w2, input_precision="tf32", out_dtype=tl.float32) * s2

    # ── store output ──────────────────────────────────────────────────────────
    tl.store(
        O_ptr + offs_m[:, None] * stride_on + offs_n[None, :],
        acc,
    )


def _triton_fp8_grouped_swiglu_fused_fp32(
    A_flat:    torch.Tensor,   # [N, H]           fp32
    expert_ids: torch.Tensor,  # [N]              int32
    W13_flat:  torch.Tensor,   # [E*H,   2*I]     fp8_e4m3  val cols [:I], gate cols [I:]
    S13_flat:  torch.Tensor,   # [E*H//128, 2*I//128] fp32  val cols [:I//128], gate [I//128:]
    W2_flat:   torch.Tensor,   # [E*I,   H_out]   fp8_e4m3
    S2_flat:   torch.Tensor,   # [E*I//128, H//128] fp32
) -> torch.Tensor:             # [N, H_out]        fp32
    N, H   = A_flat.shape
    I      = W13_flat.shape[1] // 2   # combined tensor has 2*I cols
    H_out  = W2_flat.shape[1]
    H_128  = H // 128
    I_128  = I // 128

    O = torch.empty((N, H_out), dtype=torch.float32, device=A_flat.device)

    BLOCK_M       = _TRITON_GROUPED_TILE_M  # 64 — better occupancy on B200 vs 32
    SCALE_BLOCK_N = 128                     # = FP8 scale block → one scalar scale per N-tile
    SCALE_BLOCK_K = 128                     # = FP8 scale block → one scalar scale per K-tile
    SCALE_BLOCK_H = 128                     # = FP8 scale block → one scalar scale per H-tile

    # (H_out//SCALE_BLOCK_N, N//BLOCK_M): pid_0 = N-tile varies fastest in CUDA,
    # so all N-tiles for the same M-tile (expert) execute in the first wave.
    grid = (triton.cdiv(H_out, SCALE_BLOCK_N), triton.cdiv(N, BLOCK_M))
    if _DEBUG:
        print(f"[triton kernel] grid={grid}  total_CTAs={grid[0]*grid[1]}  N={N}  BLOCK_M={BLOCK_M}  H_out={H_out}")

    _triton_fp8_grouped_swiglu_kernel[grid](
        A_flat, expert_ids,
        W13_flat, S13_flat,
        W2_flat, S2_flat,
        O,
        N, H, I, H_out, H_128, I_128,
        A_flat.stride(0),
        W13_flat.stride(0), S13_flat.stride(0),
        W2_flat.stride(0),  S2_flat.stride(0),
        O.stride(0),
        BLOCK_M=BLOCK_M, SCALE_BLOCK_N=SCALE_BLOCK_N,
        SCALE_BLOCK_K=SCALE_BLOCK_K, SCALE_BLOCK_H=SCALE_BLOCK_H,
        num_warps=8, num_stages=2,
    )
    return O


@torch.no_grad()
def expert_computation_triton_grouped_fp32(
    A: torch.Tensor,
    gemm1_weights:       torch.Tensor,   # [E_local, 2*I, H]          fp8
    gemm1_weights_scale: torch.Tensor,   # [E_local, 2*I//128, H//128] fp32
    gemm2_weights:       torch.Tensor,   # [E_local, H,   I]          fp8
    gemm2_weights_scale: torch.Tensor,   # [E_local, H//128, I//128]  fp32
    topk_idx: torch.Tensor,
    weights: torch.Tensor,
    local_expert_offset: int,
    E_global: int,
    selected_local: torch.Tensor,        # 1-D local expert indices that were chosen
) -> torch.Tensor:
    """
    Expert computation — single Triton grouped GEMM kernel (Scheme C, fp32).

    Accepts the raw stacked weight tensors directly (no per-expert dict), so the
    caller skips the dict-building loop.  _get_triton_flat_weights uses a single
    permute+reshape per tensor instead of E individual .T.contiguous() + cat.
    """
    T, H = A.shape
    device = A.device
    local_start = int(local_expert_offset)
    le_list = sorted(int(le.item()) for le in selected_local)

    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False

    temp_output = torch.zeros((T, H), dtype=torch.float32, device=device)

    # ── pre-sort tokens by expert ─────────────────────────────────────────────
    TOP_K = topk_idx.shape[1]
    token_ids_flat  = torch.arange(T, device=device).unsqueeze(1).expand(-1, TOP_K).reshape(-1)
    expert_ids_flat = topk_idx.reshape(-1)
    sort_order      = expert_ids_flat.argsort(stable=True)
    sorted_token_ids  = token_ids_flat[sort_order]
    sorted_expert_ids = expert_ids_flat[sort_order]
    expert_counts     = torch.bincount(sorted_expert_ids, minlength=E_global)
    expert_offsets_gpu = torch.zeros(E_global + 1, dtype=torch.long, device=device)
    expert_offsets_gpu[1:] = expert_counts.cumsum(0)
    expert_offsets_list = expert_offsets_gpu.cpu().tolist()

    # ── flat 2-D weight/scale tensors (cached across calls) ───────────────────
    W13_flat, S13_flat, W2_flat, S2_flat = _get_triton_flat_weights(
        gemm1_weights, gemm1_weights_scale,
        gemm2_weights, gemm2_weights_scale,
        le_list)

    # ── padded flat token arrays ──────────────────────────────────────────────
    flat_token_ids, flat_kernel_eids, flat_weights_list = [], [], []

    for stacked_idx, le in enumerate(le_list):
        ge    = local_start + le
        start = expert_offsets_list[ge]
        end   = expert_offsets_list[ge + 1]
        Te    = end - start
        if Te == 0:
            continue

        pad  = (-Te) % _TRITON_GROUPED_TILE_M
        toks = sorted_token_ids[start:end]
        if pad:
            toks = torch.cat([toks, toks[:1].expand(pad)])

        flat_token_ids.append(toks)
        flat_kernel_eids.append(
            torch.full((Te + pad,), stacked_idx, dtype=torch.int32, device=device))  # 0-based into le_list

        ge_w = weights[sorted_token_ids[start:end], ge]
        if pad:
            ge_w = torch.cat([ge_w, ge_w.new_zeros(pad)])
        flat_weights_list.append(ge_w)

    if not flat_token_ids:
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32
        return temp_output

    local_token_ids   = torch.cat(flat_token_ids)
    kernel_expert_ids = torch.cat(flat_kernel_eids)
    w_flat            = torch.cat(flat_weights_list)

    if _DEBUG:
        print(f"DEBUG expert_computation_triton_grouped_fp32 () flat token arrays:")
        print(f"  local_token_ids   : {local_token_ids.shape}  {local_token_ids.dtype}")
        print(f"  kernel_expert_ids : {kernel_expert_ids.shape}  {kernel_expert_ids.dtype}")
        print(f"  w_flat            : {w_flat.shape}  {w_flat.dtype}")

    A_flat = A[local_token_ids].to(torch.float32)  # kernel expects fp32; activations arrive as bf16

    if _DEBUG:
        print(f"DEBUG _triton_fp8_grouped_swiglu_fused_fp32 inputs:")
        print(f"  A_flat            : {A_flat.shape}  {A_flat.dtype}")
        print(f"  kernel_expert_ids : {kernel_expert_ids.shape}  {kernel_expert_ids.dtype}")
        print(f"  W13_flat          : {W13_flat.shape}  {W13_flat.dtype}")
        print(f"  S13_flat          : {S13_flat.shape}  {S13_flat.dtype}")
        print(f"  W2_flat           : {W2_flat.shape}  {W2_flat.dtype}")
        print(f"  S2_flat           : {S2_flat.shape}  {S2_flat.dtype}")

    # ── single Triton kernel launch ───────────────────────────────────────────
    O_flat = _triton_fp8_grouped_swiglu_fused_fp32(
        A_flat, kernel_expert_ids,
        W13_flat, S13_flat,
        W2_flat,  S2_flat,
    )

    temp_output.index_add_(0, local_token_ids, O_flat * w_flat.unsqueeze(1))

    torch.backends.cuda.matmul.allow_tf32 = prev_tf32
    return temp_output


@torch.no_grad()
def expert_computation_helion_fp32(
    A: torch.Tensor,              # [T, H]  fp32 activations (already dequantized)
    W13_fp8: dict,                # {le_int: [2*I, H]}    fp8 gemm1 weights
    W13_scale: dict,              # {le_int: [2*I//128, H//128]}  fp32 scales
    W2_fp8: dict,                 # {le_int: [H, I]}      fp8 gemm2 weights
    W2_scale: dict,               # {le_int: [H//128, I//128]}    fp32 scales
    topk_idx: torch.Tensor,
    weights: torch.Tensor,
    local_expert_offset: int,
    E_global: int,
) -> torch.Tensor:
    """
    Expert computation using fused FP8 dequant + GEMM (Helion, fp32/TF32 off).
    Replaces the separate weight_dequant → expert_computation_fp32 pair.
    """
    T, H = A.shape
    device = A.device
    local_start = int(local_expert_offset)

    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False

    temp_output = torch.zeros((T, H), dtype=torch.float32, device=device)

    # Pre-sort tokens by expert: one GPU sync (expert_offsets.cpu()) instead of
    # one sync per expert (nonzero).  sorted_token_ids[start:end] is a GPU view.
    TOP_K = topk_idx.shape[1]
    token_ids_flat = torch.arange(T, device=device).unsqueeze(1).expand(-1, TOP_K).reshape(-1)
    expert_ids_flat = topk_idx.reshape(-1)
    sort_order = expert_ids_flat.argsort(stable=True)
    sorted_token_ids = token_ids_flat[sort_order]           # [T*TOP_K] GPU
    sorted_expert_ids = expert_ids_flat[sort_order]         # [T*TOP_K] GPU
    expert_counts = torch.bincount(sorted_expert_ids, minlength=E_global)  # [E_global]
    expert_offsets_gpu = torch.zeros(E_global + 1, dtype=torch.long, device=device)
    expert_offsets_gpu[1:] = expert_counts.cumsum(0)
    # .tolist() = one sync + one transfer; indexing a Python list has zero C++ dispatch
    expert_offsets_list = expert_offsets_gpu.cpu().tolist()

    # Precompute scale tensors (constant across calls, no per-expert sync)
    # S_kexp shape must be [K//128, N]:
    #   W13_scale[le]: [N//128, K//128] → .T → repeat_interleave(128, dim=1) → [K//128, N]
    #   W2_scale[le]:  [N//128, K//128] → .T → repeat_interleave(128, dim=1) → [K//128, N]
    S13_cache = {le: torch.repeat_interleave(s.T.contiguous(), 128, dim=1) for le, s in W13_scale.items()}
    S2_cache  = {le: torch.repeat_interleave(s.T.contiguous(), 128, dim=1) for le, s in W2_scale.items()}

    for le_int, W13_e in W13_fp8.items():
        ge = local_start + le_int
        start = expert_offsets_list[ge]
        end   = expert_offsets_list[ge + 1]
        if start == end:
            continue
        token_idx = sorted_token_ids[start:end]             # GPU slice, no sync
        A_e = A.index_select(0, token_idx).to(torch.float32)  # [Te, H]

        I = W2_fp8[le_int].shape[1]
        W13_t      = W13_e.T                         # [H, 2*I] fp8, non-contiguous view
        W13_val_t  = W13_t[:, :I]                    # [H, I]   fp8, no copy
        W13_gate_t = W13_t[:, I:]                    # [H, I]   fp8, no copy
        W2_t       = W2_fp8[le_int].T                # [I, H]   fp8, non-contiguous view
        S13_kexp     = S13_cache[le_int]             # [H//128, 2*I]
        S13_val_kexp  = S13_kexp[:, :I]              # [H//128, I] no copy
        S13_gate_kexp = S13_kexp[:, I:]              # [H//128, I] no copy

        if _DEBUG:
            print(f"DEBUG: le_int={le_int}, ge={ge}, start={start}, end={end}, token_idx={token_idx}")
            print(f"DEBUG: A_e.shape={A_e.shape}, W13_val_t.shape={W13_val_t.shape}, W13_gate_t.shape={W13_gate_t.shape}, W2_t.shape={W2_t.shape}, S13_val_kexp.shape={S13_val_kexp.shape}, S13_gate_kexp.shape={S13_gate_kexp.shape}, S2_cache[le_int].shape={S2_cache[le_int].shape}")

        # Fused GEMM1 + SwiGLU + GEMM2: no [Te, I] intermediate in global memory
        O = _helion_fp8_swiglu_fused_fp32(
            A_e, W13_val_t, S13_val_kexp, W13_gate_t, S13_gate_kexp, W2_t, S2_cache[le_int]
        )

        w_tok = weights.index_select(0, token_idx)[:, ge]
        temp_output.index_add_(0, token_idx, O * w_tok.unsqueeze(1))

    torch.backends.cuda.matmul.allow_tf32 = prev_tf32
    return temp_output


@torch.no_grad()
def expert_computation_helion_grouped_fp32(
    A: torch.Tensor,
    W13_fp8: dict,
    W13_scale: dict,
    W2_fp8: dict,
    W2_scale: dict,
    topk_idx: torch.Tensor,
    weights: torch.Tensor,
    local_expert_offset: int,
    E_global: int,
) -> torch.Tensor:
    """
    Expert computation — single grouped GEMM kernel, all selected local experts
    in one launch (Scheme C, fp32 / TF32 off).

    Key differences from expert_computation_helion_fp32:
      • Weight dicts are stacked into [E_sel, H, I] / [E_sel, I, H] tensors.
      • Tokens for all local experts are gathered into one flat array A_flat.
      • _helion_fp8_grouped_swiglu_fused_fp32 processes all experts in a single
        kernel launch — one CTA grid covers the full (N_flat × H_out) space.
      • Each expert's segment in the flat array is padded to a multiple of
        _GROUPED_TILE_M so no M-tile straddles an expert boundary; padding rows
        carry zero routing weight and contribute nothing to the output.
      • A single index_add_ scatters all results back into temp_output.
    """
    T, H = A.shape
    device = A.device
    local_start = int(local_expert_offset)
    le_list = sorted(W13_fp8.keys())    # selected local expert indices, 0-based within shard
    E_sel = len(le_list)

    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False

    temp_output = torch.zeros((T, H), dtype=torch.float32, device=device)

    # ── pre-sort tokens by expert (same as helion_fp32) ──────────────────────
    TOP_K = topk_idx.shape[1]
    token_ids_flat  = torch.arange(T, device=device).unsqueeze(1).expand(-1, TOP_K).reshape(-1)
    expert_ids_flat = topk_idx.reshape(-1)
    sort_order      = expert_ids_flat.argsort(stable=True)
    sorted_token_ids  = token_ids_flat[sort_order]           # [T*TOP_K]
    sorted_expert_ids = expert_ids_flat[sort_order]          # [T*TOP_K], global ids
    expert_counts     = torch.bincount(sorted_expert_ids, minlength=E_global)
    expert_offsets_gpu = torch.zeros(E_global + 1, dtype=torch.long, device=device)
    expert_offsets_gpu[1:] = expert_counts.cumsum(0)
    expert_offsets_list = expert_offsets_gpu.cpu().tolist()  # one sync

    # ── flat 2D weight tensors (experts concatenated along dim-0) ──────────────
    # Helion's codegen only handles 2D tensor indexing.  Instead of stacking into
    # [E, H, I] and using a dynamic batch index le inside the kernel, we cat into
    # [E*H, I] and compute absolute row offsets: le*H + tile_h.begin.
    #
    # W13_fp8[le]: [2*I, H] → .T → [H, 2*I]; val = [:, :I], gate = [:, I:]
    # W2_fp8[le]:  [H, I]   → .T → [I, H]
    I = W2_fp8[le_list[0]].shape[1]
    W13_val_flat  = torch.cat(
        [W13_fp8[le].T.contiguous()[:, :I].contiguous() for le in le_list], dim=0
    )  # [E*H, I]   fp8
    W13_gate_flat = torch.cat(
        [W13_fp8[le].T.contiguous()[:, I:].contiguous() for le in le_list], dim=0
    )  # [E*H, I]   fp8
    W2_flat = torch.cat(
        [W2_fp8[le].T.contiguous() for le in le_list], dim=0
    )  # [E*I, H]   fp8

    # Scale tensors: same repeat_interleave expansion as helion_fp32, then cat.
    # W13_scale[le]: [2*I//128, H//128] → .T → [H//128, 2*I//128]
    #                → repeat_interleave(128, dim=1) → [H//128, 2*I]
    S13_val_flat  = torch.cat(
        [torch.repeat_interleave(W13_scale[le].T.contiguous(), 128, dim=1)[:, :I] for le in le_list],
        dim=0,
    )  # [E*(H//128), I]   fp32
    S13_gate_flat = torch.cat(
        [torch.repeat_interleave(W13_scale[le].T.contiguous(), 128, dim=1)[:, I:] for le in le_list],
        dim=0,
    )  # [E*(H//128), I]   fp32

    # W2_scale[le]: [H//128, I//128] → .T → [I//128, H//128]
    #               → repeat_interleave(128, dim=1) → [I//128, H]
    S2_flat = torch.cat(
        [torch.repeat_interleave(W2_scale[le].T.contiguous(), 128, dim=1) for le in le_list],
        dim=0,
    )  # [E*(I//128), H]   fp32

    # ── build padded flat token arrays ───────────────────────────────────────
    # Each expert's segment is padded to a multiple of _GROUPED_TILE_M so that
    # no M-tile straddles an expert boundary.  Padding rows reuse a real token
    # index but are assigned routing weight 0 → they contribute nothing.
    flat_token_ids   = []
    flat_kernel_eids = []   # 0-based index into stacked weight tensors
    flat_weights_list = []

    for stacked_idx, le in enumerate(le_list):
        ge    = local_start + le
        start = expert_offsets_list[ge]
        end   = expert_offsets_list[ge + 1]
        Te    = end - start
        if Te == 0:
            continue

        pad  = (-Te) % _GROUPED_TILE_M
        toks = sorted_token_ids[start:end]          # GPU slice, no sync
        if pad:
            toks = torch.cat([toks, toks[:1].expand(pad)])

        flat_token_ids.append(toks)
        flat_kernel_eids.append(
            torch.full((Te + pad,), stacked_idx, dtype=torch.int32, device=device)
        )

        # Routing weights for real tokens; zero for padding rows.
        ge_w = weights[sorted_token_ids[start:end], ge]   # [Te]
        if pad:
            ge_w = torch.cat([ge_w, ge_w.new_zeros(pad)])
        flat_weights_list.append(ge_w)

    if not flat_token_ids:
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32
        return temp_output

    local_token_ids   = torch.cat(flat_token_ids)     # [N_flat]
    kernel_expert_ids = torch.cat(flat_kernel_eids)   # [N_flat], 0-based stacked idx
    w_flat            = torch.cat(flat_weights_list)  # [N_flat]

    # ── gather activations ────────────────────────────────────────────────────
    A_flat = A[local_token_ids]   # [N_flat, H] fp32

    # ── single kernel launch for all experts ──────────────────────────────────
    O_flat = _helion_fp8_grouped_swiglu_fused_fp32(
        A_flat,         kernel_expert_ids,
        W13_val_flat,   S13_val_flat,
        W13_gate_flat,  S13_gate_flat,
        W2_flat,        S2_flat,
    )  # [N_flat, H]

    # ── weighted scatter-add ──────────────────────────────────────────────────
    temp_output.index_add_(0, local_token_ids, O_flat * w_flat.unsqueeze(1))

    torch.backends.cuda.matmul.allow_tf32 = prev_tf32
    return temp_output


@torch.no_grad()
def expert_computation_helion_tf32(
    A: torch.Tensor,
    W13_fp8: dict,
    W13_scale: dict,
    W2_fp8: dict,
    W2_scale: dict,
    topk_idx: torch.Tensor,
    weights: torch.Tensor,
    local_expert_offset: int,
    E_global: int,
) -> torch.Tensor:
    """
    Expert computation using fused FP8 dequant + GEMM (Helion, TF32 on).
    Same as helion_fp32 but allows TF32 tensor-core acceleration.
    """
    T, H = A.shape
    I = next(iter(W2_fp8.values())).shape[1]
    device = A.device
    local_start = int(local_expert_offset)

    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True

    temp_output = torch.zeros((T, H), dtype=torch.float32, device=device)

    # Pre-sort tokens by expert: one GPU sync instead of one per expert.
    TOP_K = topk_idx.shape[1]
    token_ids_flat = torch.arange(T, device=device).unsqueeze(1).expand(-1, TOP_K).reshape(-1)
    expert_ids_flat = topk_idx.reshape(-1)
    sort_order = expert_ids_flat.argsort(stable=True)
    sorted_token_ids = token_ids_flat[sort_order]
    sorted_expert_ids = expert_ids_flat[sort_order]
    expert_counts = torch.bincount(sorted_expert_ids, minlength=E_global)
    expert_offsets_gpu = torch.zeros(E_global + 1, dtype=torch.long, device=device)
    expert_offsets_gpu[1:] = expert_counts.cumsum(0)
    expert_offsets_list = expert_offsets_gpu.cpu().tolist()  # one sync, Python list = zero dispatch

    S13_cache = {le: torch.repeat_interleave(s.T.contiguous(), 128, dim=1) for le, s in W13_scale.items()}
    S2_cache  = {le: torch.repeat_interleave(s.T.contiguous(), 128, dim=1) for le, s in W2_scale.items()}

    for le_int, W13_e in W13_fp8.items():
        ge = local_start + le_int
        start = expert_offsets_list[ge]
        end   = expert_offsets_list[ge + 1]
        if start == end:
            continue
        token_idx = sorted_token_ids[start:end]             # GPU slice, no sync
        A_e = A.index_select(0, token_idx).to(torch.float32)

        W13_t = W13_e.T
        W2_t  = W2_fp8[le_int].T

        G1  = _helion_fp8_gemm_tf32(A_e, W13_t, S13_cache[le_int])
        X1, X2 = G1[:, :I], G1[:, I:]
        C_e = (X2 / (1.0 + torch.exp(-X2))) * X1

        O = _helion_fp8_gemm_tf32(C_e, W2_t, S2_cache[le_int])

        w_tok = weights.index_select(0, token_idx)[:, ge]
        temp_output.index_add_(0, token_idx, O * w_tok.unsqueeze(1))

    torch.backends.cuda.matmul.allow_tf32 = prev_tf32
    return temp_output


def expert_computation_helion_fp8(
    A_fp8: torch.Tensor,        # [T, H]        fp8_e4m3fn  raw hidden_states (NOT dequantized)
    A_scale: torch.Tensor,      # [H//128, T]   fp32        hidden_states_scale
    W13_fp8: dict,              # {le_int: [2*I, H]}        fp8
    W13_scale: dict,            # {le_int: [2*I//128, H//128]}  fp32
    W2_fp8: dict,               # {le_int: [H, I]}          fp8
    W2_scale: dict,             # {le_int: [H//128, I//128]}    fp32
    topk_idx: torch.Tensor,
    weights: torch.Tensor,
    local_expert_offset: int,
    E_global: int,
) -> torch.Tensor:
    """
    Expert computation with native FP8 tensor cores for GEMM1.

    Activations stay in FP8 — no dequantization.  Each K=128 tile of GEMM1 uses
    FP8 tensor cores, then scales by A_scale[k_block, token] * W_scale[k_block, n_block].
    GEMM2 uses FP32 C_e with on-chip FP8 weight dequant (same as helion_tf32).
    """
    T, H = A_fp8.shape
    I_dim = next(iter(W2_fp8.values())).shape[1]
    device = A_fp8.device
    local_start = int(local_expert_offset)

    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True

    temp_output = torch.zeros((T, H), dtype=torch.float32, device=device)

    TOP_K = topk_idx.shape[1]
    token_ids_flat = torch.arange(T, device=device).unsqueeze(1).expand(-1, TOP_K).reshape(-1)
    expert_ids_flat = topk_idx.reshape(-1)
    sort_order = expert_ids_flat.argsort(stable=True)
    sorted_token_ids = token_ids_flat[sort_order]
    sorted_expert_ids = expert_ids_flat[sort_order]
    expert_counts = torch.bincount(sorted_expert_ids, minlength=E_global)
    expert_offsets_gpu = torch.zeros(E_global + 1, dtype=torch.long, device=device)
    expert_offsets_gpu[1:] = expert_counts.cumsum(0)
    expert_offsets_list = expert_offsets_gpu.cpu().tolist()

    # Pre-expand weight scales: [N//128, K//128] → .T → repeat_interleave(128) → [K//128, N]
    S13_cache = {le: torch.repeat_interleave(s.T.contiguous(), 128, dim=1) for le, s in W13_scale.items()}
    S2_cache  = {le: torch.repeat_interleave(s.T.contiguous(), 128, dim=1) for le, s in W2_scale.items()}

    # Transpose activation scale for easier per-token indexing in kernel:
    # [H//128, T] → [T, H//128]  so A_scale_T[token, k_block] = scalar
    A_scale_T = A_scale.T.contiguous()

    for le_int, W13_e in W13_fp8.items():
        ge = local_start + le_int
        start = expert_offsets_list[ge]
        end   = expert_offsets_list[ge + 1]
        if start == end:
            continue
        token_idx = sorted_token_ids[start:end]

        # Select expert tokens — keep FP8, no dequantization
        A_e_fp8     = A_fp8.index_select(0, token_idx)         # [Te, H]       fp8
        A_e_scale_T = A_scale_T.index_select(0, token_idx)     # [Te, H//128]  fp32

        # GEMM1: FP8 × FP8 → FP32 (native FP8 tensor cores) — val and gate together
        W13_t = W13_e.T                                         # [H, 2*I]  fp8
        G1 = _helion_fp8_gemm_native(A_e_fp8, A_e_scale_T, W13_t, S13_cache[le_int])
        # G1: [Te, 2*I]  fp32

        # SwiGLU
        X1, X2 = G1[:, :I_dim], G1[:, I_dim:]
        C_e = (X2 / (1.0 + torch.exp(-X2))) * X1               # [Te, I]   fp32

        # GEMM2: FP32 × FP8-dequant (helion_tf32 style — TF32 set above)
        W2_t = W2_fp8[le_int].T                                 # [I, H]    fp8
        O = _helion_fp8_gemm_tf32(C_e, W2_t, S2_cache[le_int]) # [Te, H]   fp32

        w_tok = weights.index_select(0, token_idx)[:, ge]
        temp_output.index_add_(0, token_idx, O * w_tok.unsqueeze(1))

    torch.backends.cuda.matmul.allow_tf32 = prev_tf32
    return temp_output


# ===========================================
# Main Kernel (Hybrid Approach)
# ===========================================

# Dispatch maps — looked up at call time so monkey-patching works
_ACT_DTYPE_MAP = {
    'bf16': torch.bfloat16,
    'fp16': torch.float16,
    'fp32': torch.float32,
}
# tf32 uses fp32 storage for weights (tf32 is a compute mode, not a storage dtype)
# helion_fp32 / helion_tf32: fused dequant inside Helion kernel — no pre-dequant step
_WEIGHT_DTYPE_MAP = {
    'bf16': torch.bfloat16,
    'fp16': torch.float16,
    'fp32': torch.float32,
    'tf32': torch.float32,
    'helion_fp32': None,   # no pre-dequant; raw FP8 passed to expert fn
    'helion_tf32': None,
    'helion_grouped_fp32': None,
    'triton_grouped_fp32': None,
    'helion_fp8': None,    # native FP8 tensor cores; raw FP8 + scales passed directly
}
_WEIGHT_DEQUANT_SUFFIX = {
    'bf16': 'bf16',
    'fp16': 'fp16',
    'fp32': 'fp32',
    'tf32': 'fp32',
    'helion_fp32': None,
    'helion_tf32': None,
    'helion_grouped_fp32': None,
    'triton_grouped_fp32': None,
    'helion_fp8': None,
}
# compute_dtype values that skip the host-side weight dequant phase
_FUSED_COMPUTE_DTYPES = {'helion_fp32', 'helion_tf32', 'helion_grouped_fp32', 'triton_grouped_fp32', 'helion_fp8'}


@torch.no_grad()
def kernel(
    routing_logits: torch.Tensor,
    routing_bias: torch.Tensor,
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    local_expert_offset: int,
    routed_scaling_factor: float,
    output: torch.Tensor,
    **kwargs,  # absorbs act_dtype / compute_dtype; ignored by the benchmark builder
) -> torch.Tensor:
    """
    Hybrid MoE kernel with configurable dequantization and compute dtypes.

    act_dtype:     controls dequantize_fp8_activations_{act_dtype}  (bf16 | fp16 | fp32)
    compute_dtype: controls dequantize_fp8_weights_{yy} + expert_computation_{compute_dtype}
                   (bf16 | fp16 | fp32 | tf32 | helion_fp32 | helion_tf32 | helion_fp8)

    When called by the benchmark runner (no kwargs), defaults to helion_fp8 (native FP8 tensor cores).
    """
    act_dtype     = kwargs.get('act_dtype', 'fp32')
    compute_dtype = kwargs.get('compute_dtype', 'helion_fp8')
    import sys
    _mod = sys.modules[__name__]

    if _DEBUG:
        print(f"DEBUG kernel() input shapes:")
        print(f"  act_dtype={act_dtype}, compute_dtype={compute_dtype}")
        print(f"  routing_logits shape: {routing_logits.shape} ({routing_logits.dtype})")
        print(f"  routing_bias shape: {routing_bias.shape} ({routing_bias.dtype})")
        print(f"  hidden_states shape: {hidden_states.shape} ({hidden_states.dtype})")
        print(f"  hidden_states_scale shape: {hidden_states_scale.shape} ({hidden_states_scale.dtype})")
        print(f"  gemm1_weights shape: {gemm1_weights.shape} ({gemm1_weights.dtype})")
        print(f"  gemm1_weights_scale shape: {gemm1_weights_scale.shape} ({gemm1_weights_scale.dtype})")
        print(f"  gemm2_weights shape: {gemm2_weights.shape} ({gemm2_weights.dtype})")
        print(f"  gemm2_weights_scale shape: {gemm2_weights_scale.shape} ({gemm2_weights_scale.dtype})")
        print(f"  local_expert_offset: {local_expert_offset}")
        print(f"  routed_scaling_factor: {routed_scaling_factor}")
        print(f"  output shape: {output.shape} ({output.dtype})")

    # Constants
    H = 7168
    I = 2048
    E_local = gemm1_weights.shape[0]
    E_global = routing_logits.shape[1]
    BLOCK = 128
    local_start = int(local_expert_offset)

    # Resolve dispatch functions at call time (supports monkey-patching)
    act_dequant_fn = getattr(_mod, f'dequantize_fp8_activations_{act_dtype}')
    expert_computation = getattr(_mod, f'expert_computation_{compute_dtype}')
    weight_dtype = _WEIGHT_DTYPE_MAP[compute_dtype]
    # Fused dtypes skip host-side weight dequant; no weight_dequant_fn needed
    if compute_dtype not in _FUSED_COMPUTE_DTYPES:
        w_suffix = _WEIGHT_DEQUANT_SUFFIX[compute_dtype]
        weight_dequant_fn = getattr(_mod, f'dequantize_fp8_weights_{w_suffix}')

    # Phase 1a: Dequantize activations FP8 → act_dtype
    # helion_fp8 skips dequantization — raw FP8 hidden_states passed directly to expert fn
    if compute_dtype != 'helion_fp8':
        A = act_dequant_fn(hidden_states, hidden_states_scale, BLOCK)

    # Phase 2: Routing (Host-side) — keep in FP32 (BF16/FP16 routing changes topk selection)
    topk_idx, weights = compute_deepseek_routing(
        routing_logits,
        routing_bias,
        routed_scaling_factor,
        TOP_K=8,
        N_GROUP=8,
        TOPK_GROUP=4,
    )
    if _DEBUG:
        print(f"DEBUG kernel() routing output:")
        print(f"  topk_idx shape: {topk_idx.shape}")
        print(f"  weights shape: {weights.shape}")

    # Phase 1b/1c: Select which local experts were chosen
    # KEY OPTIMIZATION: Only dequantize weights for selected experts (~2-4x speedup)
    selected_global = torch.unique(topk_idx)  # [num_selected]
    selected_local = selected_global[(selected_global >= local_start) &
                                     (selected_global < local_start + E_local)] - local_start
    if _DEBUG:
        print(f"DEBUG kernel() expert selection:")
        print(f"  len(selected_global): {len(selected_global)}")
        print(f"  len(selected_local): {len(selected_local)}")

    if len(selected_local) == 0:
        # No local experts selected, return zeros
        output.zero_()
        return output

    # Phase 3: Expert Computation
    if compute_dtype == 'triton_grouped_fp32':
        # Fast path: pass stacked tensors directly — no per-expert dict loop.
        if _DEBUG:
            le0 = int(selected_local[0].item())
            print(f"DEBUG kernel() activations:")
            print(f"  A                  : {A.shape}  {A.dtype}")
            print(f"DEBUG kernel() weight shapes (le={le0}):")
            print(f"  gemm1_weights      : {gemm1_weights[le0].shape}  {gemm1_weights.dtype}")
            print(f"  gemm1_weights_scale: {gemm1_weights_scale[le0].shape}  {gemm1_weights_scale.dtype}")
            print(f"  gemm2_weights      : {gemm2_weights[le0].shape}  {gemm2_weights.dtype}")
            print(f"  gemm2_weights_scale: {gemm2_weights_scale[le0].shape}  {gemm2_weights_scale.dtype}")
            print(f"DEBUG kernel() routing:")
            print(f"  topk_idx       : {topk_idx.shape}  {topk_idx.dtype}")
            print(f"  weights        : {weights.shape}  {weights.dtype}")
            print(f"  local_expert_offset: {local_expert_offset}")
            print(f"  E_global       : {E_global}")
            print(f"  selected_local : {selected_local.shape}  values={selected_local.tolist()}")
        temp_output = expert_computation_triton_grouped_fp32(
            A, gemm1_weights, gemm1_weights_scale, gemm2_weights, gemm2_weights_scale,
            topk_idx, weights, local_expert_offset, E_global, selected_local,
        )
    elif compute_dtype == 'helion_fp8':
        # Native FP8 path: activations stay in FP8, no dequantization
        W13_fp8_dict  = {}
        W13_scale_dict = {}
        W2_fp8_dict   = {}
        W2_scale_dict  = {}
        for le in selected_local:
            le_int = int(le.item())
            W13_fp8_dict[le_int]   = gemm1_weights[le_int]
            W13_scale_dict[le_int] = gemm1_weights_scale[le_int]
            W2_fp8_dict[le_int]    = gemm2_weights[le_int]
            W2_scale_dict[le_int]  = gemm2_weights_scale[le_int]

        temp_output = expert_computation_helion_fp8(
            hidden_states, hidden_states_scale,
            W13_fp8_dict, W13_scale_dict, W2_fp8_dict, W2_scale_dict,
            topk_idx, weights, local_expert_offset, E_global,
        )
    elif compute_dtype in _FUSED_COMPUTE_DTYPES:
        # Helion fused path: build per-expert dicts, pass to helion kernel
        W13_fp8_dict  = {}
        W13_scale_dict = {}
        W2_fp8_dict   = {}
        W2_scale_dict  = {}
        for le in selected_local:
            le_int = int(le.item())
            W13_fp8_dict[le_int]   = gemm1_weights[le_int]        # [2*I, H]  fp8
            W13_scale_dict[le_int] = gemm1_weights_scale[le_int]  # [2*I//128, H//128]
            W2_fp8_dict[le_int]    = gemm2_weights[le_int]        # [H, I]    fp8
            W2_scale_dict[le_int]  = gemm2_weights_scale[le_int]  # [H//128, I//128]

        if _DEBUG:
            le0 = next(iter(W13_fp8_dict))
            print(f"DEBUG weight shapes (le={le0}):")
            print(f"  W13_fp8  : {W13_fp8_dict[le0].shape}  {W13_fp8_dict[le0].dtype}")
            print(f"  W13_scale: {W13_scale_dict[le0].shape}  {W13_scale_dict[le0].dtype}")
            print(f"  W2_fp8   : {W2_fp8_dict[le0].shape}  {W2_fp8_dict[le0].dtype}")
            print(f"  W2_scale : {W2_scale_dict[le0].shape}  {W2_scale_dict[le0].dtype}")

        temp_output = expert_computation(
            A, W13_fp8_dict, W13_scale_dict, W2_fp8_dict, W2_scale_dict,
            topk_idx, weights, local_expert_offset, E_global,
        )
    else:
        # Separate dequant path (original behaviour for bf16/fp16/fp32/tf32)
        W13_dict = {}
        W2_dict  = {}
        for le in selected_local:
            le_int = int(le.item())
            W13_dict[le_int] = weight_dequant_fn(
                gemm1_weights[le_int:le_int+1],
                gemm1_weights_scale[le_int:le_int+1],
                BLOCK
            )[0]
            W2_dict[le_int] = weight_dequant_fn(
                gemm2_weights[le_int:le_int+1],
                gemm2_weights_scale[le_int:le_int+1],
                BLOCK
            )[0]

        temp_output = expert_computation(
            A, W13_dict, W2_dict, topk_idx, weights,
            local_expert_offset, E_global,
        )

    # Convert to output dtype
    output.copy_(temp_output)

    return output


# ===========================================
# Workload Loading (same as original)
# ===========================================

def load_tensor_from_safetensors(path: str, tensor_key: str) -> torch.Tensor:
    """Load a specific tensor from a safetensors file."""
    with safe_open(path, framework="pt", device="cpu") as f:
        return f.get_tensor(tensor_key)


def create_random_tensor(shape: tuple, dtype: torch.dtype) -> torch.Tensor:
    """Create a random tensor with the specified shape and dtype."""
    if dtype == torch.float8_e4m3fn:
        return torch.randn(shape, dtype=torch.float32).to(dtype)
    else:
        return torch.randn(shape, dtype=dtype)


def load_workload_from_jsonl(workload_line: dict, base_path: Path):
    """Load a workload from a JSONL entry."""
    inputs_spec = workload_line["workload"]["inputs"]
    axes = workload_line["workload"]["axes"]
    seq_len = axes["seq_len"]

    H = 7168
    I = 2048
    E_local = 32
    num_hidden_blocks = 56
    num_intermediate_blocks = 16
    num_gemm1_out_blocks = 32

    inputs = {}

    for name, spec in inputs_spec.items():
        if spec["type"] == "safetensors":
            path = base_path / spec["path"]
            tensor_key = spec["tensor_key"]
            inputs[name] = load_tensor_from_safetensors(str(path), tensor_key)
        elif spec["type"] == "random":
            if name == "hidden_states":
                inputs[name] = create_random_tensor((seq_len, H), torch.float8_e4m3fn)
            elif name == "hidden_states_scale":
                inputs[name] = create_random_tensor((num_hidden_blocks, seq_len), torch.float32)
            elif name == "gemm1_weights":
                inputs[name] = create_random_tensor((E_local, 2 * I, H), torch.float8_e4m3fn)
            elif name == "gemm1_weights_scale":
                inputs[name] = create_random_tensor((E_local, num_gemm1_out_blocks, num_hidden_blocks), torch.float32)
            elif name == "gemm2_weights":
                inputs[name] = create_random_tensor((E_local, H, I), torch.float8_e4m3fn)
            elif name == "gemm2_weights_scale":
                inputs[name] = create_random_tensor((E_local, num_hidden_blocks, num_intermediate_blocks), torch.float32)
        elif spec["type"] == "scalar":
            inputs[name] = spec["value"]

    return inputs


def main():
    parser = argparse.ArgumentParser(
        description="Execute Helion-optimized MoE FP8 kernel (dequantize to BF16)"
    )
    parser.add_argument(
        "--workload",
        type=str,
        default="workloads/moe/moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048.jsonl",
        help="Path to the workload JSONL file"
    )
    parser.add_argument("--workload-index", type=int, default=0, help="Which workload to run (line number)")
    parser.add_argument("--all-workloads", action="store_true", help="Run ALL workloads in the file")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmarking")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3, help="Number of warmup iterations")
    parser.add_argument("--profile", action="store_true", help="Run Kineto profiler on one kernel call")
    parser.add_argument("--profile-dir", type=str, default="./profiles", help="Directory to save Kineto trace JSON")
    parser.add_argument("--debug", action="store_true", help="Enable diagnostic prints")
    parser.add_argument(
        "--act-dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"],
        help="Activation dequantization dtype (dequantize_fp8_activations_xx)"
    )
    parser.add_argument(
        "--compute-dtype", type=str, default="tf32",
        choices=["bf16", "fp16", "fp32", "tf32", "helion_fp32", "helion_tf32", "helion_grouped_fp32", "triton_grouped_fp32"],
        help=(
            "Weight dequantization + expert compute dtype. "
            "bf16/fp16/fp32/tf32: separate dequant then GEMM. "
            "helion_fp32/helion_tf32: fused FP8-dequant inside Helion GEMM kernel (no fp32 weight materialised)."
        )
    )

    args = parser.parse_args()

    # Propagate --debug to the module-level flag so kernel() picks it up too.
    if args.debug:
        import sys
        sys.modules[__name__]._DEBUG = True

    # Load workload
    script_dir = Path(__file__).parent
    workload_path = Path(args.workload)

    if not workload_path.is_absolute():
        candidate = script_dir / args.workload
        if candidate.exists():
            workload_path = candidate
        else:
            dataset_path = os.environ.get('FIB_DATASET_PATH')
            if dataset_path:
                candidate = Path(dataset_path) / args.workload
                if candidate.exists():
                    workload_path = candidate

            # Try common locations
            if not workload_path.exists():
                common_paths = [
                    Path.home() / "ai" / "mlsys26-contest" / args.workload,
                    Path.home() / "ai" / "flashinfer-trace" / args.workload,
                    Path("/home/sgoswami/ai/mlsys26-contest") / args.workload,
                ]
                for candidate in common_paths:
                    if candidate.exists():
                        workload_path = candidate
                        break

    if not workload_path.exists():
        print(f"Error: Workload file not found: {workload_path}")
        return

    print(f"Loading workload from: {workload_path}")
    print(f"Device: {args.device}")

    # Determine which workloads to run
    if args.all_workloads:
        with open(workload_path, 'r') as f:
            total_workloads = sum(1 for _ in f)
        workload_indices = list(range(total_workloads))
        print(f"Running ALL {total_workloads} workloads")
    else:
        workload_indices = [args.workload_index]
        print(f"Running workload index: {args.workload_index}")

    base_path = workload_path.parent.parent.parent

    # Storage for results
    all_results = []

    # Run each workload
    for wl_idx in workload_indices:
        if args.all_workloads:
            print(f"\n{'='*80}")
            print(f"WORKLOAD {wl_idx} / {total_workloads - 1}")
            print(f"{'='*80}")

        # Load workload data
        with open(workload_path, 'r') as f:
            for i, line in enumerate(f):
                if i == wl_idx:
                    workload_data = json.loads(line)
                    break

        inputs = load_workload_from_jsonl(workload_data, base_path)

        # Move to device
        device = torch.device(args.device)
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                inputs[key] = value.to(device)

        # Allocate output
        seq_len = inputs["hidden_states"].shape[0]
        H = 7168
        inputs["output"] = torch.zeros((seq_len, H), dtype=torch.bfloat16, device=device)

        if _DEBUG:
            print(f"\nInput shapes:")
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: {value.shape} ({value.dtype})")
                else:
                    print(f"  {key}: {value}")

        kernel_kwargs = dict(act_dtype=args.act_dtype, compute_dtype=args.compute_dtype)

        # Warmup
        for _ in range(args.warmup):
            _ = kernel(**inputs, **kernel_kwargs)

        # Benchmark
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for _ in range(args.iterations):
            output = kernel(**inputs, **kernel_kwargs)
        end_event.record()

        torch.cuda.synchronize()
        elapsed_ms = start_event.elapsed_time(end_event)
        avg_time_ms = elapsed_ms / args.iterations

        print(f"\nBenchmark results ({args.iterations} iterations):")
        print(f"  Total time: {elapsed_ms:.3f} ms")
        print(f"  Average time: {avg_time_ms:.3f} ms")

        # Kineto profiling
        if args.profile:
            from torch.profiler import profile, record_function, ProfilerActivity
            from datetime import datetime
            import re

            profile_dir = Path(args.profile_dir)
            profile_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            trace_name = f"moe_fp8fpX_at_{args.act_dtype}_ct_{args.compute_dtype}_wl{wl_idx}_{ts}"
            trace_path = profile_dir / f"{trace_name}.json"

            _prof_warmup = 17
            _prof_active = 3
            print(f"\nRunning Kineto profiler ({_prof_warmup} warmup + {_prof_active} captured iters)...")
            for _ in range(_prof_warmup):
                kernel(**inputs, **kernel_kwargs)
            torch.cuda.synchronize()

            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                with_stack=False,
                acc_events=True,
            ) as prof:
                for i in range(_prof_active):
                    with record_function(f"iteration {i+_prof_warmup}"):
                        kernel(**inputs, **kernel_kwargs)
                torch.cuda.synchronize()

            prof.export_chrome_trace(str(trace_path))
            print(f"  Chrome trace saved: {trace_path}")
            print(f"  Open with: https://ui.perfetto.dev  or  chrome://tracing")

            print("\nTop 20 ops by CUDA time (self):")
            key_avgs = prof.key_averages()
            print(key_avgs.table(sort_by="self_device_time_total", row_limit=20))

            table_str = key_avgs.table(sort_by="self_device_time_total", row_limit=0)
            cpu_m  = re.search(r"Self CPU time total:\s*([\d.]+)(m?s)", table_str)
            cuda_m = re.search(r"Self CUDA time total:\s*([\d.]+)(m?s)", table_str)
            def _to_ms(val, unit): return float(val) if unit == 'ms' else float(val) / 1e3
            if cpu_m and cuda_m:
                print(f"{'TOTAL (self)':>55}  {_to_ms(cpu_m.group(1), cpu_m.group(2)):>12.3f}ms"
                      f"  {_to_ms(cuda_m.group(1), cuda_m.group(2)):>12.3f}ms")

    if _DEBUG:
        print(f"\nOutput shape: {output.shape} ({output.dtype})")
        print(f"Output stats:")
        print(f"  Min: {output.min().item():.6f}")
        print(f"  Max: {output.max().item():.6f}")
        print(f"  Mean: {output.mean().item():.6f}")
        print(f"  Std: {output.std().item():.6f}")


if __name__ == "__main__":
    main()
