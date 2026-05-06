#include "gpt-5-2025-08-07-moe-fp8-fused-kernel.h"
#include <cstdio>
#include <cmath>

#ifndef CUDART_INF_F
#define CUDART_INF_F (__int_as_float(0x7f800000))
#endif

// Warp reduce max
__device__ __forceinline__ float warp_max(float v) {
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    float other = __shfl_down_sync(0xffffffffu, v, offset);
    v = fmaxf(v, other);
  }
  return v;
}

// 1) No-aux routing kernel
// One block per token (T), 8 warps per block (256 threads), one warp per group
__global__ void noaux_routing_topk8_kernel(
    const float* __restrict__ logits,   // [T, 256]
    const float* __restrict__ bias,     // [256]
    int T,
    float routed_scaling_factor,
    int* __restrict__ topk_idx,         // [T, 8]
    float* __restrict__ topk_w) {       // [T, 8]

  __shared__ float group_scores[ROUTE_NUM_GROUP]; // 8
  __shared__ unsigned int keep_group_mask;        // bitmask of 8 groups
  __shared__ float warpCandVal[ROUTE_NUM_GROUP * ROUTE_TOP_K];        // 8*8 = 64
  __shared__ int   warpCandIdx[ROUTE_NUM_GROUP * ROUTE_TOP_K];
  __shared__ float warpCandSNoBias[ROUTE_NUM_GROUP * ROUTE_TOP_K];

  int t = blockIdx.x;
  if (t >= T) return;

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5; // 0..7
  const int e = warp * ROUTE_GROUP_SIZE + lane;  // expert index 0..255

  // Load and compute s and s_with_bias
  float l = logits[t * NUM_EXPERTS_GLOBAL + e];
  float s = 1.f / (1.f + __expf(-l));
  float sb = s + bias[e];

  // Compute group top-2 sum within warp
  float v = sb;
  float m1 = warp_max(v);
  unsigned mask1 = __ballot_sync(0xffffffffu, v == m1);
  int idx1_lane = __ffs(mask1) - 1;
  float v2 = (lane == idx1_lane) ? -CUDART_INF_F : v;
  float m2 = warp_max(v2);
  if (lane == 0) {
    group_scores[warp] = m1 + m2;
  }
  __syncthreads();

  // Select top-4 groups on a single thread
  if (threadIdx.x == 0) {
    float temp_scores[ROUTE_NUM_GROUP];
    #pragma unroll
    for (int g = 0; g < ROUTE_NUM_GROUP; ++g) temp_scores[g] = group_scores[g];
    unsigned int mask_bits = 0u;
    #pragma unroll
    for (int j = 0; j < ROUTE_TOPK_GROUP; ++j) {
      int best = 0;
      float bestv = temp_scores[0];
      #pragma unroll
      for (int g = 1; g < ROUTE_NUM_GROUP; ++g) {
        if (temp_scores[g] > bestv) { bestv = temp_scores[g]; best = g; }
      }
      mask_bits |= (1u << best);
      temp_scores[best] = -CUDART_INF_F;
    }
    keep_group_mask = mask_bits;
  }
  __syncthreads();

  // Prune unkept groups by setting -inf, keep sb for kept groups
  bool keep = ((keep_group_mask >> warp) & 1u) != 0u;
  float cur = keep ? sb : -CUDART_INF_F;

  // Compute top-8 within this warp (group)
  #pragma unroll
  for (int j = 0; j < ROUTE_TOP_K; ++j) {
    float m = warp_max(cur);
    unsigned msk = __ballot_sync(0xffffffffu, cur == m);
    int max_lane = __ffs(msk) - 1;
    float s_no_bias_sel = __shfl_sync(0xffffffffu, s, max_lane);
    if (lane == 0) {
      int base = warp * ROUTE_TOP_K + j;
      warpCandVal[base] = m;
      warpCandIdx[base] = warp * ROUTE_GROUP_SIZE + max_lane;
      warpCandSNoBias[base] = s_no_bias_sel;
    }
    if (lane == max_lane) cur = -CUDART_INF_F;
  }
  __syncthreads();

  // Merge 64 candidates to top-8 globally
  if (threadIdx.x == 0) {
    float temp_val[ROUTE_NUM_GROUP * ROUTE_TOP_K];
    int   temp_idx[ROUTE_NUM_GROUP * ROUTE_TOP_K];
    float temp_snb[ROUTE_NUM_GROUP * ROUTE_TOP_K];

    #pragma unroll
    for (int i = 0; i < ROUTE_NUM_GROUP * ROUTE_TOP_K; ++i) {
      temp_val[i] = warpCandVal[i];
      temp_idx[i] = warpCandIdx[i];
      temp_snb[i] = warpCandSNoBias[i];
    }

    float sel_s[ROUTE_TOP_K];
    int sel_idx[ROUTE_TOP_K];

    #pragma unroll
    for (int j = 0; j < ROUTE_TOP_K; ++j) {
      int best_i = 0;
      float best_v = temp_val[0];
      #pragma unroll
      for (int i = 1; i < ROUTE_NUM_GROUP * ROUTE_TOP_K; ++i) {
        if (temp_val[i] > best_v) { best_v = temp_val[i]; best_i = i; }
      }
      sel_idx[j] = temp_idx[best_i];
      sel_s[j] = temp_snb[best_i];
      temp_val[best_i] = -CUDART_INF_F;
    }

    // Normalize weights using s (no bias)
    float sumw = 0.f;
    #pragma unroll
    for (int j = 0; j < ROUTE_TOP_K; ++j) sumw += sel_s[j];
    sumw = fmaxf(sumw, 1e-20f);
    #pragma unroll
    for (int j = 0; j < ROUTE_TOP_K; ++j) {
      float w = (sel_s[j] / sumw) * routed_scaling_factor;
      topk_idx[t * ROUTE_TOP_K + j] = sel_idx[j];
      topk_w[t * ROUTE_TOP_K + j] = w;
    }
  }
}

// 2) Hidden block scale application (in-place)
__global__ void apply_hidden_block_scale_kernel(
    float* __restrict__ A,            // [T, H]
    const float* __restrict__ S,      // [H/128, T] in row-major
    int T, int H) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int N = T * H;
  for (int i = idx; i < N; i += blockDim.x * gridDim.x) {
    int t = i / H;
    int h = i - t * H;
    int hb = h >> 7; // h/128
    float sc = S[hb * T + t];
    A[i] *= sc;
  }
}

// 3) Apply 128x128 block scale to 2D matrix (in-place)
__global__ void apply_block_scale_128x128_kernel(
    float* __restrict__ M,     // [rows, cols]
    int rows, int cols,
    const float* __restrict__ S,// [rows/128, cols/128]
    int Sb_rows, int Sb_cols) {

  int blk_row = blockIdx.y; // 0..rows/128 - 1
  int blk_col = blockIdx.x; // 0..cols/128 - 1
  float scale = S[blk_row * Sb_cols + blk_col];

  int row_base = blk_row * BLOCK_SIZE_128;
  int col_base = blk_col * BLOCK_SIZE_128;

  int tx = threadIdx.x; // 0..31
  int ty = threadIdx.y; // 0..7

  // Fully cover the 128x128 tile using 32x8 threads
  for (int r = ty; r < BLOCK_SIZE_128; r += blockDim.y) {
    int row = row_base + r;
    float* row_ptr = M + row * cols;
    for (int c = tx; c < BLOCK_SIZE_128; c += blockDim.x) {
      int col = col_base + c;
      row_ptr[col] *= scale;
    }
  }
}

// 4) Count assignments per local expert
__global__ void count_local_assignments_kernel(
    const int* __restrict__ topk_idx,   // [T, 8]
    int T,
    int local_expert_offset,
    int* __restrict__ counts) {         // [32]
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= T) return;
  int base = t * ROUTE_TOP_K;
  #pragma unroll
  for (int k = 0; k < ROUTE_TOP_K; ++k) {
    int ge = topk_idx[base + k];
    int le = ge - local_expert_offset;
    if ((unsigned)le < (unsigned)NUM_LOCAL_EXPERTS) {
      atomicAdd(&counts[le], 1);
    }
  }
}

// 5) Fill assignments using prefix offsets
__global__ void fill_local_assignments_kernel(
    const int* __restrict__ topk_idx,   // [T, 8]
    const float* __restrict__ topk_w,   // [T, 8]
    int T,
    int local_expert_offset,
    int* __restrict__ offsets_inout,    // [32], running counters
    int* __restrict__ token_ids_out,    // [total]
    float* __restrict__ token_w_out) {  // [total]
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= T) return;
  int base = t * ROUTE_TOP_K;
  #pragma unroll
  for (int k = 0; k < ROUTE_TOP_K; ++k) {
    int ge = topk_idx[base + k];
    int le = ge - local_expert_offset;
    if ((unsigned)le < (unsigned)NUM_LOCAL_EXPERTS) {
      int pos = atomicAdd(&offsets_inout[le], 1);
      token_ids_out[pos] = t;
      token_w_out[pos] = topk_w[base + k];
    }
  }
}

// 6) Gather rows [T,H] -> [Tk,H]
__global__ void gather_rows_kernel(
    const float* __restrict__ A,     // [T, H]
    const int* __restrict__ token_ids,// [Tk]
    int /*T*/, int Tk, int H,
    float* __restrict__ A_out) {     // [Tk, H]
  int row = blockIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= Tk || col >= H) return;
  int t = token_ids[row];
  A_out[row * H + col] = A[t * H + col];
}

// 7) SwiGLU kernel
__global__ void swiglu_kernel(
    const float* __restrict__ G1, // [Tk, 4096]
    int Tk,
    float* __restrict__ C) {      // [Tk, 2048]
  int row = blockIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= Tk || col >= INTERMEDIATE_SIZE) return;
  const float* g1_row = G1 + row * GEMM1_OUT_SIZE;
  float x1 = g1_row[col];
  float x2 = g1_row[col + INTERMEDIATE_SIZE];
  float silu = x2 / (1.0f + __expf(-x2));
  C[row * INTERMEDIATE_SIZE + col] = silu * x1;
}

// 8) Accumulate O into output with weights
__global__ void accumulate_weighted_add_kernel(
    const float* __restrict__ O,       // [Tk, H]
    const int* __restrict__ token_ids, // [Tk]
    const float* __restrict__ weights, // [Tk]
    int Tk, int H,
    float* __restrict__ output) {      // [T, H]
  int row = blockIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= Tk || col >= H) return;
  int t = token_ids[row];
  float w = weights[row];
  float val = O[row * H + col] * w;
  output[t * H + col] += val;
}

// Launchers

void launch_noaux_routing_topk8(
    const float* routing_logits,
    const float* routing_bias,
    int T,
    float routed_scaling_factor,
    int* topk_idx,
    float* topk_w,
    cudaStream_t stream) {

  dim3 block(ROUTE_NUM_GROUP * 32); // 8 warps
  dim3 grid(T);
  noaux_routing_topk8_kernel<<<grid, block, 0, stream>>>(
      routing_logits, routing_bias, T, routed_scaling_factor, topk_idx, topk_w);
  CUDA_CHECK(cudaGetLastError());
}

void launch_apply_hidden_block_scale(
    float* A_fp32,
    const float* hs_scale,
    int T,
    cudaStream_t stream) {
  int H = HIDDEN_SIZE;
  int64_t N64 = static_cast<int64_t>(T) * H;
  int threads = 256;
  int blocks = static_cast<int>((N64 + threads - 1) / threads);
  blocks = max(1, min(blocks, 65535));
  apply_hidden_block_scale_kernel<<<blocks, threads, 0, stream>>>(A_fp32, hs_scale, T, H);
  CUDA_CHECK(cudaGetLastError());
}

void launch_apply_block_scale_128x128(
    float* M, int rows, int cols,
    const float* S, int S_rows, int S_cols,
    cudaStream_t stream) {

  dim3 grid(S_cols, S_rows);   // blocks in [cols/128, rows/128]
  dim3 block(32, 8);           // 256 threads
  apply_block_scale_128x128_kernel<<<grid, block, 0, stream>>>(M, rows, cols, S, S_rows, S_cols);
  CUDA_CHECK(cudaGetLastError());
}

void launch_count_local_assignments(
    const int* topk_idx, int T, int local_expert_offset,
    int* counts, cudaStream_t stream) {
  int threads = 256;
  int blocks = (T + threads - 1) / threads;
  count_local_assignments_kernel<<<blocks, threads, 0, stream>>>(
      topk_idx, T, local_expert_offset, counts);
  CUDA_CHECK(cudaGetLastError());
}

void launch_fill_local_assignments(
    const int* topk_idx, const float* topk_w, int T, int local_expert_offset,
    int* offsets_inout, int* token_ids_out, float* token_w_out,
    cudaStream_t stream) {
  int threads = 256;
  int blocks = (T + threads - 1) / threads;
  fill_local_assignments_kernel<<<blocks, threads, 0, stream>>>(
      topk_idx, topk_w, T, local_expert_offset, offsets_inout, token_ids_out, token_w_out);
  CUDA_CHECK(cudaGetLastError());
}

void launch_gather_rows(
    const float* A, const int* token_ids, int /*T*/, int Tk, int H,
    float* A_out, cudaStream_t stream) {
  dim3 block(256);
  dim3 grid((H + block.x - 1) / block.x, Tk);
  if (Tk > 0) {
    gather_rows_kernel<<<grid, block, 0, stream>>>(A, token_ids, 0, Tk, H, A_out);
    CUDA_CHECK(cudaGetLastError());
  }
}

void launch_swiglu(
    const float* G1, int Tk, float* C, cudaStream_t stream) {
  dim3 block(256);
  dim3 grid((INTERMEDIATE_SIZE + block.x - 1) / block.x, Tk);
  if (Tk > 0) {
    swiglu_kernel<<<grid, block, 0, stream>>>(G1, Tk, C);
    CUDA_CHECK(cudaGetLastError());
  }
}

void launch_accumulate_weighted_add(
    const float* O, const int* token_ids, const float* weights, int Tk, int H,
    float* output, cudaStream_t stream) {
  dim3 block(256);
  dim3 grid((H + block.x - 1) / block.x, Tk);
  if (Tk > 0) {
    accumulate_weighted_add_kernel<<<grid, block, 0, stream>>>(
        O, token_ids, weights, Tk, H, output);
    CUDA_CHECK(cudaGetLastError());
  }
}
