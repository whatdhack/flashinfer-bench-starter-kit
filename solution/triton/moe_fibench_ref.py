import torch


@torch.no_grad()
def run(
    routing_logits: torch.Tensor,       # [T, E_global=256]         float32
    routing_bias: torch.Tensor,         # [E_global=256]            float32
    hidden_states: torch.Tensor,        # [T, H=7168]               fp8 e4m3fn
    hidden_states_scale: torch.Tensor,  # [H/128=56, T]             float32
    gemm1_weights: torch.Tensor,        # [E_local=32, 2*I=4096, H=7168]   fp8 e4m3fn
    gemm1_weights_scale: torch.Tensor,  # [E_local=32, 2*I/128=32, H/128=56] float32
    gemm2_weights: torch.Tensor,        # [E_local=32, H=7168, I=2048]     fp8 e4m3fn
    gemm2_weights_scale: torch.Tensor,  # [E_local=32, H/128=56, I/128=16]  float32
    local_expert_offset: int,           # first global expert index on this rank
    routed_scaling_factor: float,       # scalar multiplier on routing weights
):
    """
    • FP8 block-scale dequantization: float ≈ fp8 * scale
    • DeepSeek-V3 no-aux routing:
        s = sigmoid(logits)
        s_with_bias = s + bias
        group by n_group=8; per group take top-2 sum → pick topk_group=4 groups
        on the kept groups, take global top_k=8 experts
        combine with weights derived from s (without bias), normalized and
        scaled by routed_scaling_factor
    • Local computation:
        only experts in [local_expert_offset, local_expert_offset + E_local) are
        computed on this rank (GEMM1 → SwiGLU → GEMM2), then per-token weighted
        accumulation.
    """

    # Fixed DeepSeek-V3/R1 geometry
    H = 7168
    I = 2048
    E_local = gemm1_weights.shape[0] # El
    
    SCALE_BLOCK = 128
    E_global = routing_logits.shape[1] #Eg
    T = routing_logits.shape[0]

    assert H == 7168, "hidden_size must be 7168" 
    assert I == 2048, "intermediate_size must be 2048"
    assert E_global == 256, "num_experts must be 256"
    assert E_local == 32, "num_local_experts must be 32"

    # Routing constants
    TOP_K = 8
    N_GROUP = 8 # Eg/El
    TOPK_GROUP = 4

    # Block counts
    num_hidden_blocks = H // SCALE_BLOCK          # 56
    num_intermediate_blocks = I // SCALE_BLOCK    # 16
    num_gemm1_out_blocks = (2 * I) // SCALE_BLOCK # 32

    # Shape checks
    assert hidden_states.shape == (T, H)
    assert hidden_states_scale.shape == (num_hidden_blocks, T)
    assert gemm1_weights.shape == (E_local, 2 * I, H)
    assert gemm1_weights_scale.shape == (E_local, num_gemm1_out_blocks, num_hidden_blocks)
    assert gemm2_weights.shape == (E_local, H, I)
    assert gemm2_weights_scale.shape == (E_local, num_hidden_blocks, num_intermediate_blocks)
    assert routing_bias.shape[-1] == E_global

    device = hidden_states.device

    # 1) FP8 block-scale dequantization
    # hidden_states: [T, H], scale: [H/128, T] (transposed layout)
    A_fp32 = hidden_states.to(torch.float32)
    A_scale = hidden_states_scale.to(torch.float32)                # [H/128, T]
    A_scale_TH = A_scale.permute(1, 0).contiguous()            # [T, H/128]
    A_scale_expanded = (
        A_scale_TH.unsqueeze(-1)
        .repeat(1, 1, SCALE_BLOCK)                                   # [T, H/128, 128]
        .reshape(T, H)                                         # [T, H]
        .contiguous()
    )
    A = A_fp32 * A_scale_expanded                              # [T, H] float32

    # W13: [E_local, 2I, H], scale: [E_local, (2I)/128, H/128]
    W13_fp32 = gemm1_weights.to(torch.float32)
    S13 = gemm1_weights_scale.to(torch.float32)
    S13_expanded = torch.repeat_interleave(S13, SCALE_BLOCK, dim=1)  # [El, 2I, H/128]
    S13_expanded = torch.repeat_interleave(S13_expanded, SCALE_BLOCK, dim=2)  # [El, 2I, H]
    W13 = W13_fp32 * S13_expanded                              # [El, 2I, H] float32

    # W2: [E_local, H, I], scale: [E_local, H/128, I/128]
    W2_fp32 = gemm2_weights.to(torch.float32)
    S2 = gemm2_weights_scale.to(torch.float32)
    S2_expanded = torch.repeat_interleave(S2, SCALE_BLOCK, dim=1)    # [El, H, I/128]
    S2_expanded = torch.repeat_interleave(S2_expanded, SCALE_BLOCK, dim=2)    # [El, H, I]
    W2 = W2_fp32 * S2_expanded                                 # [El, H, I] float32

    # 2) No-aux routing
    logits = routing_logits.to(torch.float32)                      # [T, Eg]
    bias = routing_bias.to(torch.float32).reshape(-1)              # [Eg]

    # Sigmoid
    s = 1.0 / (1.0 + torch.exp(-logits))                       # [T, Eg]
    s_with_bias = s + bias                                     # [T, Eg] (broadcast)

    # Grouping
    group_size = E_global // N_GROUP # 32 = El
    s_wb_grouped = s_with_bias.view(T, N_GROUP, group_size)    # [T, 8, 32] = [T, Eg/El, El]

    # Group scores = sum of top-2 values within each group
    top2_vals, _ = torch.topk(s_wb_grouped, k=2, dim=2, largest=True, sorted=False)  # [T, 8, 2]
    group_scores = top2_vals.sum(dim=2)                        # [T, 8]

    # Select topk_group groups → group mask
    _, group_idx = torch.topk(group_scores, k=TOPK_GROUP, dim=1, largest=True, sorted=False)  # [T, 4]
    group_mask = torch.zeros_like(group_scores)                # [T, 8]
    group_mask.scatter_(1, group_idx, 1.0)
    score_mask = group_mask.unsqueeze(2).expand(T, N_GROUP, group_size).reshape(T, E_global)  # [T, Eg]

    # Global top-k (within kept groups), based on s_with_bias
    neg_inf = torch.finfo(torch.float32).min
    scores_pruned = s_with_bias.masked_fill(score_mask == 0, neg_inf)                  # [T, Eg]
    _, topk_idx = torch.topk(scores_pruned, k=TOP_K, dim=1, largest=True, sorted=False)  # [T, 8]

    # Combination weights: use s (without bias) for normalization
    M = torch.zeros_like(s)                                    # [T, Eg]
    M.scatter_(1, topk_idx, 1.0)                               # 0/1 mask
    weights = s * M                                            # [T, Eg]
    weights_sum = weights.sum(dim=1, keepdim=True) + 1e-20     # [T, 1]
    weights = (weights / weights_sum) * routed_scaling_factor  # [T, Eg]

    # 3) Local expert compute and accumulation
    output = torch.zeros((T, H), dtype=torch.float32, device=device)

    local_start = int(local_expert_offset)

    # For each local expert: find selected tokens, run GEMM1→SwiGLU→GEMM2, accumulate by weights
    for le in range(E_local):
        ge = local_start + le
        if ge < 0 or ge >= E_global:
            continue

        # Tokens that selected this global expert ge in their top-k
        sel_mask_per_token = (topk_idx == ge).any(dim=1)       # [T] bool
        if not sel_mask_per_token.any():
            continue

        token_idx = torch.nonzero(sel_mask_per_token, as_tuple=False).squeeze(1)  # [Tk]
        Tk = token_idx.numel()

        # Gather inputs and weights for this expert
        A_e = A.index_select(0, token_idx)                     # [Tk, H]
        W13_e = W13[le]                                        # [2I, H]
        W2_e = W2[le]                                          # [H, I]

        # GEMM1: [Tk, H] @ [H, 2I] = [Tk, 2I]
        G1 = A_e.matmul(W13_e.t())                             # [Tk, 2I]

        # SwiGLU: split and apply silu(x) = x / (1 + exp(-x))
        X1 = G1[:, :I]                                         # [Tk, I]
        X2 = G1[:, I:]                                         # [Tk, I]
        silu_X2 = X2 / (1.0 + torch.exp(-X2))                  # [Tk, I]
        C = silu_X2 * X1                                       # [Tk, I]

        # GEMM2: [Tk, I] @ [I, H] = [Tk, H]
        O = C.matmul(W2_e.t())                                 # [Tk, H]

        # Accumulate with per-token routing weights for this expert
        w_tok = weights.index_select(0, token_idx)[:, ge]      # [Tk]
        output.index_add_(0, token_idx, O * w_tok.unsqueeze(1))  # [Tk,H] * [Tk,1]

    return output.to(torch.bfloat16)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    T = 7          # number of tokens
    H = 7168
    I = 2048
    E_local = 32 # El
    E_global = 256 # Eg
    SCALE_BLOCK = 128

    num_hidden_blocks = H // SCALE_BLOCK           # 56
    num_intermediate_blocks = I // SCALE_BLOCK     # 16
    num_gemm1_out_blocks = (2 * I) // SCALE_BLOCK  # 32

    print(f"Device: {device}  |  T={T}, H={H}, I={I}, E_local={E_local}, E_global={E_global}")
    print("Allocating inputs...")

    routing_logits = torch.randn(T, E_global, device=device)
    routing_bias = torch.zeros(E_global, device=device)
    hidden_states = torch.randn(T, H, device=device).to(torch.float8_e4m3fn)
    hidden_states_scale = torch.ones(num_hidden_blocks, T, device=device)

    # Use torch.empty with fp8 dtype directly to avoid large float32 intermediates
    gemm1_weights = torch.empty(E_local, 2 * I, H, dtype=torch.float8_e4m3fn, device=device)
    gemm1_weights_scale = torch.ones(E_local, num_gemm1_out_blocks, num_hidden_blocks, device=device)
    gemm2_weights = torch.empty(E_local, H, I, dtype=torch.float8_e4m3fn, device=device)
    gemm2_weights_scale = torch.ones(E_local, num_hidden_blocks, num_intermediate_blocks, device=device)

    local_expert_offset = 0
    routed_scaling_factor = 1.0

    print("Running...")
    output = run(
        routing_logits=routing_logits,
        routing_bias=routing_bias,
        hidden_states=hidden_states,
        hidden_states_scale=hidden_states_scale,
        gemm1_weights=gemm1_weights,
        gemm1_weights_scale=gemm1_weights_scale,
        gemm2_weights=gemm2_weights,
        gemm2_weights_scale=gemm2_weights_scale,
        local_expert_offset=local_expert_offset,
        routed_scaling_factor=routed_scaling_factor,
    )

    print(f"Output shape : {output.shape}")
    print(f"Output dtype : {output.dtype}")
    print(f"Output[0, :8]: {output[0, :8]}")


if __name__ == "__main__":
    main()