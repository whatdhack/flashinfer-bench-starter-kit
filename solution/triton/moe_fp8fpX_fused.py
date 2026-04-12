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
}
_WEIGHT_DEQUANT_SUFFIX = {
    'bf16': 'bf16',
    'fp16': 'fp16',
    'fp32': 'fp32',
    'tf32': 'fp32',
    'helion_fp32': None,
    'helion_tf32': None,
}
# compute_dtype values that skip the host-side weight dequant phase
_FUSED_COMPUTE_DTYPES = {'helion_fp32', 'helion_tf32'}


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
                   (bf16 | fp16 | fp32 | tf32 | helion_fp32 | helion_tf32)

    When called by the benchmark runner (no kwargs), defaults to fp32 activations + helion_tf32.
    """
    act_dtype     = kwargs.get('act_dtype', 'fp32')
    compute_dtype = kwargs.get('compute_dtype', 'helion_tf32')
    import sys
    _mod = sys.modules[__name__]

    if _DEBUG:
        print(f"act_dtype={act_dtype}, compute_dtype={compute_dtype}")
        print(f"routing_logits shape: {routing_logits.shape} ({routing_logits.dtype})")
        print(f"routing_bias shape: {routing_bias.shape} ({routing_bias.dtype})")
        print(f"hidden_states shape: {hidden_states.shape} ({hidden_states.dtype})")
        print(f"hidden_states_scale shape: {hidden_states_scale.shape} ({hidden_states_scale.dtype})")
        print(f"gemm1_weights shape: {gemm1_weights.shape} ({gemm1_weights.dtype})")
        print(f"gemm1_weights_scale shape: {gemm1_weights_scale.shape} ({gemm1_weights_scale.dtype})")
        print(f"gemm2_weights shape: {gemm2_weights.shape} ({gemm2_weights.dtype})")
        print(f"gemm2_weights_scale shape: {gemm2_weights_scale.shape} ({gemm2_weights_scale.dtype})")
        print(f"local_expert_offset: {local_expert_offset}")
        print(f"routed_scaling_factor: {routed_scaling_factor}")
        print(f"output shape: {output.shape} ({output.dtype})")

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
    A = act_dequant_fn(hidden_states, hidden_states_scale, BLOCK)
    if _DEBUG:
        print(f"A dtype: {A.dtype}, shape: {A.shape}")

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
        print(f"topk_idx shape: {topk_idx.shape}")
        print(f"weights shape: {weights.shape}")

    # Phase 1b/1c: Select which local experts were chosen
    # KEY OPTIMIZATION: Only dequantize weights for selected experts (~2-4x speedup)
    selected_global = torch.unique(topk_idx)  # [num_selected]
    selected_local = selected_global[(selected_global >= local_start) &
                                     (selected_global < local_start + E_local)] - local_start
    if _DEBUG:
        print(f"len(selected_global): {len(selected_global)}")
        print(f"len(selected_local): {len(selected_local)}")

    if len(selected_local) == 0:
        # No local experts selected, return zeros
        output.zero_()
        return output

    # Phase 3: Expert Computation
    if compute_dtype in _FUSED_COMPUTE_DTYPES:
        # Fused path: pass raw FP8 slices + scales; dequant happens inside the Helion kernel
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
        choices=["bf16", "fp16", "fp32", "tf32", "helion_fp32", "helion_tf32"],
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
