"""
Autotune _helion_fp8_swiglu_fused_fp32 locally using all 19 workload shapes.

All workloads share: H=7168, I=2048, E=32 experts, TOP_K=8.
Only T (total tokens) varies.  Per-expert token count Te ≈ T*TOP_K/E.

Workloads (T values from moe_fp8fpX_fused_20260412_132050.log):
  T = 1, 7, 14, 15, 16, 32, 52, 53, 54, 55, 56, 57, 58, 59, 62, 80, 901, 11948, 14107
  → unique avg Te (rounded, ≥1): 1, 2, 4, 13, 14, 15, 16, 20, 225, 2987, 3527

Usage:
    python scripts/autotune_local.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import helion
import helion.language as hl

# Fixed across all workloads
H     = 7168
I     = 2048
E     = 32
TOP_K = 8

# All 19 workload T values → unique avg Te = max(1, round(T*TOP_K/E))
# With static_shapes=False, one config is tuned across all shapes.
# The autotuner benchmarks using all Te values as a representative workload mix.
_T_VALUES = [1, 7, 14, 15, 16, 32, 52, 53, 54, 55, 56, 57, 58, 59, 62, 80, 901, 11948, 14107]
TE_VALUES = sorted(set(max(1, round(T * TOP_K / E)) for T in _T_VALUES))
# → [1, 2, 4, 13, 14, 15, 16, 20, 225, 2987, 3527]


@helion.kernel(config=None, static_shapes=False)
def _autotune_swiglu(
    A: torch.Tensor,             # [Te, H]       fp32
    W13_val_t: torch.Tensor,     # [H,  I]       fp8_e4m3
    S13_val_kexp: torch.Tensor,  # [H//128, I]   fp32
    W13_gate_t: torch.Tensor,    # [H,  I]       fp8_e4m3
    S13_gate_kexp: torch.Tensor, # [H//128, I]   fp32
    W2_t: torch.Tensor,          # [I,  H]       fp8_e4m3
    S2_kexp: torch.Tensor,       # [I//128, H]   fp32
) -> torch.Tensor:
    Te, H_ = A.shape
    I_ = W2_t.shape[0]
    H_out = W2_t.shape[1]
    O = torch.empty([Te, H_out], dtype=torch.float32, device=A.device)

    for tile_m, tile_n in hl.tile([Te, H_out]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)

        for tile_k in hl.tile(I_, block_size=128):
            g1_val  = hl.zeros([tile_m, tile_k], dtype=torch.float32)
            g1_gate = hl.zeros([tile_m, tile_k], dtype=torch.float32)

            for tile_h in hl.tile(H_, block_size=128):
                a_tile = A[tile_m, tile_h]
                h_s    = tile_h.begin // 128

                w13_val  = W13_val_t[tile_h, tile_k]
                s13_val  = S13_val_kexp[h_s : h_s + 1, tile_k]
                g1_val   = hl.dot(a_tile, w13_val.to(torch.float32) * s13_val, acc=g1_val)

                w13_gate = W13_gate_t[tile_h, tile_k]
                s13_gate = S13_gate_kexp[h_s : h_s + 1, tile_k]
                g1_gate  = hl.dot(a_tile, w13_gate.to(torch.float32) * s13_gate, acc=g1_gate)

            c_e = (g1_gate / (1.0 + torch.exp(-g1_gate))) * g1_val

            k_s2 = tile_k.begin // 128
            w2   = W2_t[tile_k, tile_n]
            s2   = S2_kexp[k_s2 : k_s2 + 1, tile_n]
            acc  = hl.dot(c_e, w2.to(torch.float32) * s2, acc=acc)

        O[tile_m, tile_n] = acc

    return O


def main():
    assert torch.cuda.is_available(), "CUDA required"
    dev = torch.device("cuda")

    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")
    print(f"Autotuning for Te values: {TE_VALUES}")
    print(f"H={H}, I={I}, E={E}, TOP_K={TOP_K}\n")

    # Build fixed weight tensors (shared across all Te)
    W13_full   = torch.randn(2 * I, H, device=dev).to(torch.float8_e4m3fn)
    W13_t      = W13_full.T                          # [H, 2*I]
    W13_val_t  = W13_t[:, :I].contiguous()           # [H, I]
    W13_gate_t = W13_t[:, I:].contiguous()           # [H, I]

    S13_full      = torch.randn(H // 128, 2 * I, device=dev)
    S13_val_kexp  = S13_full[:, :I].contiguous()     # [H//128, I]
    S13_gate_kexp = S13_full[:, I:].contiguous()     # [H//128, I]

    W2      = torch.randn(H, I, device=dev).to(torch.float8_e4m3fn)
    W2_t    = W2.T.contiguous()                      # [I, H]
    S2_kexp = torch.randn(I // 128, H, device=dev)   # [I//128, H]

    # With static_shapes=False, trigger autotuning with the first call.
    # Helion benchmarks internally; subsequent calls with any Te reuse the config.
    print(f"Triggering autotuner (static_shapes=False — single config for all Te)...")
    A = torch.randn(TE_VALUES[len(TE_VALUES) // 2], H, device=dev, dtype=torch.float32)
    _autotune_swiglu(A, W13_val_t, S13_val_kexp,
                     W13_gate_t, S13_gate_kexp,
                     W2_t, S2_kexp)

    bound = _autotune_swiglu.bind((A, W13_val_t, S13_val_kexp,
                                   W13_gate_t, S13_gate_kexp,
                                   W2_t, S2_kexp))
    cfg = bound._config
    print(f"\nBest config: {cfg!r}")

    print("\n" + "="*60)
    print(f"SUMMARY ({gpu_name}) — paste into moe_fp8fpX_fused.py")
    print("="*60)
    print(f"\nhconfig_swiglu_fp32 = {cfg!r}")


if __name__ == "__main__":
    main()
