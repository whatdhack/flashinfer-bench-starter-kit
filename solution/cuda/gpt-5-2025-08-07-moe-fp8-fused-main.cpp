#include "gpt-5-2025-08-07-moe-fp8-fused-kernel.h"

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublas_v2.h>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <cstring>

#define CUBLAS_CHECK(status) \
  do { \
    cublasStatus_t st__ = (status); \
    if (st__ != CUBLAS_STATUS_SUCCESS) { \
      fprintf(stderr, "cuBLAS Error %d at %s:%d\n", int(st__), __FILE__, __LINE__); \
    } \
  } while (0)

static inline void check_input(const torch::Tensor& t, c10::ScalarType dtype, const std::vector<int64_t>& shape_prefix) {
  TORCH_CHECK(t.is_cuda(), "Tensor must be CUDA");
  TORCH_CHECK(t.scalar_type() == dtype, "Unexpected dtype");
  TORCH_CHECK(t.dim() >= (int)shape_prefix.size(), "Unexpected rank");
  for (size_t i = 0; i < shape_prefix.size(); ++i) {
    if (shape_prefix[i] >= 0) {
      TORCH_CHECK(t.size(i) == shape_prefix[i], "Unexpected size at dim ", i);
    }
  }
}

torch::Tensor run(
    torch::Tensor routing_logits,        // [T, 256], float32
    torch::Tensor routing_bias,          // [256], bfloat16 (all zeros for no bias)
    torch::Tensor hidden_states,         // [T, 7168], float8_e4m3fn
    torch::Tensor hidden_states_scale,   // [56, T], float32
    torch::Tensor gemm1_weights,         // [32, 4096, 7168], float8_e4m3fn
    torch::Tensor gemm1_weights_scale,   // [32, 32, 56], float32
    torch::Tensor gemm2_weights,         // [32, 7168, 2048], float8_e4m3fn
    torch::Tensor gemm2_weights_scale,   // [32, 56, 16], float32
    int64_t local_expert_offset,         // int
    double routed_scaling_factor         // float
) {
  TORCH_CHECK(routing_logits.is_cuda(), "routing_logits must be CUDA");
  TORCH_CHECK(hidden_states.is_cuda(), "hidden_states must be CUDA");
  TORCH_CHECK(hidden_states_scale.is_cuda(), "hidden_states_scale must be CUDA");
  TORCH_CHECK(gemm1_weights.is_cuda() && gemm1_weights_scale.is_cuda(), "gemm1 weights must be CUDA");
  TORCH_CHECK(gemm2_weights.is_cuda() && gemm2_weights_scale.is_cuda(), "gemm2 weights must be CUDA");
  TORCH_CHECK(routing_bias.is_cuda(), "routing_bias must be CUDA");

  const int64_t T = routing_logits.size(0);
  TORCH_CHECK(routing_logits.size(1) == NUM_EXPERTS_GLOBAL, "routing_logits shape mismatch");
  TORCH_CHECK(hidden_states.size(0) == T && hidden_states.size(1) == HIDDEN_SIZE, "hidden_states shape mismatch");
  TORCH_CHECK(hidden_states_scale.size(0) == NUM_HIDDEN_BLOCKS && hidden_states_scale.size(1) == T, "hidden_states_scale shape mismatch");
  TORCH_CHECK(gemm1_weights.size(0) == NUM_LOCAL_EXPERTS &&
              gemm1_weights.size(1) == GEMM1_OUT_SIZE &&
              gemm1_weights.size(2) == HIDDEN_SIZE, "gemm1_weights shape mismatch");
  TORCH_CHECK(gemm1_weights_scale.sizes() == torch::IntArrayRef({NUM_LOCAL_EXPERTS, NUM_GEMM1_OUT_BLOCKS, NUM_HIDDEN_BLOCKS}), "gemm1_weights_scale shape mismatch");
  TORCH_CHECK(gemm2_weights.sizes() == torch::IntArrayRef({NUM_LOCAL_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE}), "gemm2_weights shape mismatch");
  TORCH_CHECK(gemm2_weights_scale.sizes() == torch::IntArrayRef({NUM_LOCAL_EXPERTS, NUM_HIDDEN_BLOCKS, NUM_INTERMEDIATE_BLOCKS}), "gemm2_weights_scale shape mismatch");
  TORCH_CHECK(routing_bias.size(0) == NUM_EXPERTS_GLOBAL, "routing_bias size mismatch");

  c10::cuda::CUDAGuard device_guard(routing_logits.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // Cast routing bias to float32 (device)
  auto routing_bias_f32 = routing_bias.to(torch::kFloat32).contiguous();
  auto routing_logits_f32 = routing_logits.contiguous(); // already float32

  // 1) Hidden states FP8 -> float32 using PyTorch conversion, then apply block scale
  auto A_fp32 = hidden_states.to(torch::kFloat32).contiguous();
  TORCH_CHECK(A_fp32.size(0) == T && A_fp32.size(1) == HIDDEN_SIZE, "A_fp32 shape mismatch");
  auto hs_scale_c = hidden_states_scale.contiguous();
  launch_apply_hidden_block_scale(
      A_fp32.data_ptr<float>(),
      hs_scale_c.data_ptr<float>(),
      (int)T, stream);

  // 2) Routing: compute topk indices and weights
  auto topk_idx = torch::empty({T, ROUTE_TOP_K}, torch::dtype(torch::kInt32).device(routing_logits.device()));
  auto topk_w = torch::empty({T, ROUTE_TOP_K}, torch::dtype(torch::kFloat32).device(routing_logits.device()));
  launch_noaux_routing_topk8(
      routing_logits_f32.data_ptr<float>(),
      routing_bias_f32.data_ptr<float>(),
      (int)T,
      static_cast<float>(routed_scaling_factor),
      topk_idx.data_ptr<int>(),
      topk_w.data_ptr<float>(),
      stream);

  // 3) Build local assignments for experts in [local_expert_offset, local_expert_offset + 32)
  auto counts = torch::zeros({NUM_LOCAL_EXPERTS}, torch::dtype(torch::kInt32).device(routing_logits.device()));
  launch_count_local_assignments(
      topk_idx.data_ptr<int>(),
      (int)T,
      (int)local_expert_offset,
      counts.data_ptr<int>(),
      stream);

  // Sync to read counts on host
  CUDA_CHECK(cudaStreamSynchronize(stream));
  auto counts_cpu = counts.cpu();
  auto counts_ptr = counts_cpu.data_ptr<int>();
  std::vector<int> h_counts(NUM_LOCAL_EXPERTS);
  int total_assign = 0;
  int max_Tk = 0;
  for (int i = 0; i < NUM_LOCAL_EXPERTS; ++i) {
    h_counts[i] = counts_ptr[i];
    total_assign += h_counts[i];
    max_Tk = std::max(max_Tk, h_counts[i]);
  }
  std::vector<int> h_offsets(NUM_LOCAL_EXPERTS + 1, 0);
  for (int i = 0; i < NUM_LOCAL_EXPERTS; ++i) h_offsets[i + 1] = h_offsets[i] + h_counts[i];

  // Allocate assignment buffers and fill
  auto d_offsets = torch::empty({NUM_LOCAL_EXPERTS}, torch::dtype(torch::kInt32).device(routing_logits.device()));
  CUDA_CHECK(cudaMemcpyAsync(d_offsets.data_ptr<int>(), h_offsets.data(), sizeof(int) * NUM_LOCAL_EXPERTS, cudaMemcpyHostToDevice, stream));
  auto token_ids = torch::empty({std::max(1, total_assign)}, torch::dtype(torch::kInt32).device(routing_logits.device()));
  auto token_wts = torch::empty({std::max(1, total_assign)}, torch::dtype(torch::kFloat32).device(routing_logits.device()));
  launch_fill_local_assignments(
      topk_idx.data_ptr<int>(),
      topk_w.data_ptr<float>(),
      (int)T,
      (int)local_expert_offset,
      d_offsets.data_ptr<int>(),
      token_ids.data_ptr<int>(),
      token_wts.data_ptr<float>(),
      stream);

  // 4) Output buffer (float32 accumulation)
  auto output_f32 = torch::zeros({T, HIDDEN_SIZE}, torch::dtype(torch::kFloat32).device(routing_logits.device()));

  // 5) cuBLAS handle
  cublasHandle_t handle = nullptr;
  CUBLAS_CHECK(cublasCreate(&handle));
  CUBLAS_CHECK(cublasSetStream(handle, stream));

  // 6) Per-expert processing
  // Workspace sized by max_Tk
  int Tk_max = std::max(1, max_Tk);
  auto A_tok = torch::empty({Tk_max, HIDDEN_SIZE}, torch::dtype(torch::kFloat32).device(routing_logits.device()));
  auto G1 = torch::empty({Tk_max, GEMM1_OUT_SIZE}, torch::dtype(torch::kFloat32).device(routing_logits.device()));
  auto C = torch::empty({Tk_max, INTERMEDIATE_SIZE}, torch::dtype(torch::kFloat32).device(routing_logits.device()));
  auto Otmp = torch::empty({Tk_max, HIDDEN_SIZE}, torch::dtype(torch::kFloat32).device(routing_logits.device()));

  const float alpha = 1.0f, beta0 = 0.0f;

  for (int le = 0; le < NUM_LOCAL_EXPERTS; ++le) {
    int Tk = h_counts[le];
    if (Tk == 0) continue;

    int start = h_offsets[le];
    const int* d_token_ids_le = token_ids.data_ptr<int>() + start;
    const float* d_token_w_le = token_wts.data_ptr<float>() + start;

    // Gather A_tok [Tk, H]
    launch_gather_rows(
        A_fp32.data_ptr<float>(),
        d_token_ids_le,
        (int)T, (int)Tk, HIDDEN_SIZE,
        A_tok.data_ptr<float>(),
        stream);

    // Dequantize W13 for this local expert: take slice [le, :, :]
    auto w13_fp8 = gemm1_weights.select(0, le).contiguous(); // [4096, 7168] float8
    auto w13_f32 = w13_fp8.to(torch::kFloat32).contiguous(); // decode fp8 -> float32
    auto s13 = gemm1_weights_scale.select(0, le).contiguous(); // [32, 56] float32
    // Apply 128x128 block scale
    launch_apply_block_scale_128x128(
        w13_f32.data_ptr<float>(),
        GEMM1_OUT_SIZE, HIDDEN_SIZE,
        s13.data_ptr<float>(),
        NUM_GEMM1_OUT_BLOCKS, NUM_HIDDEN_BLOCKS,
        stream);

    // GEMM1: G1[Tk, 4096] = A_tok[Tk, 7168] @ W13^T [7168, 4096]
    // Column-major trick: C_cm(4096 x Tk) = (W13_cm^T)(4096x7168) * (A_cm)(7168xTk)
    CUBLAS_CHECK(cublasSgemm(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        GEMM1_OUT_SIZE,        // m = 4096
        Tk,                    // n = Tk
        HIDDEN_SIZE,           // k = 7168
        &alpha,
        w13_f32.data_ptr<float>(), HIDDEN_SIZE,    // A: (7168 x 4096), lda=7168
        A_tok.data_ptr<float>(), HIDDEN_SIZE,      // B: (7168 x Tk),  ldb=7168
        &beta0,
        G1.data_ptr<float>(),  GEMM1_OUT_SIZE));   // C: (4096 x Tk),  ldc=4096

    // SwiGLU: C = silu(G1[:, I:]) * G1[:, :I]
    launch_swiglu(G1.data_ptr<float>(), Tk, C.data_ptr<float>(), stream);

    // Dequantize W2 for this expert: [7168, 2048] row-major
    auto w2_fp8 = gemm2_weights.select(0, le).contiguous();   // [7168, 2048], fp8
    auto w2_f32 = w2_fp8.to(torch::kFloat32).contiguous();    // [7168, 2048], row-major
    auto s2 = gemm2_weights_scale.select(0, le).contiguous(); // [56, 16]
    launch_apply_block_scale_128x128(
        w2_f32.data_ptr<float>(),
        HIDDEN_SIZE, INTERMEDIATE_SIZE,
        s2.data_ptr<float>(),
        NUM_HIDDEN_BLOCKS, NUM_INTERMEDIATE_BLOCKS,
        stream);

    // GEMM2: Otmp[Tk, 7168] = C[Tk, 2048] @ W2^T [2048, 7168]
    // Interpret w2_f32 row-major [7168, 2048] as column-major [2048, 7168], then transpose in GEMM.
    CUBLAS_CHECK(cublasSgemm(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        HIDDEN_SIZE,            // m = 7168
        Tk,                     // n = Tk
        INTERMEDIATE_SIZE,      // k = 2048
        &alpha,
        w2_f32.data_ptr<float>(), INTERMEDIATE_SIZE, // A: (2048 x 7168) col-major, lda=2048, op(T)->(7168 x 2048)
        C.data_ptr<float>(),      INTERMEDIATE_SIZE, // B: (2048 x Tk),  ldb=2048
        &beta0,
        Otmp.data_ptr<float>(), HIDDEN_SIZE));       // C: (7168 x Tk),  ldc=7168

    // Accumulate weighted add to output
    launch_accumulate_weighted_add(
        Otmp.data_ptr<float>(),
        d_token_ids_le,
        d_token_w_le,
        Tk, HIDDEN_SIZE,
        output_f32.data_ptr<float>(),
        stream);
  }

  // Destroy cuBLAS
  CUBLAS_CHECK(cublasDestroy(handle));

  // Convert to BF16 for output
  auto output_bf16 = output_f32.to(torch::kBFloat16);

  return output_bf16;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &run,
        "moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048 (B200-optimized)");
}
