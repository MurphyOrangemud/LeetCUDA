/**
 * @file test.cu
 * @brief peusdo code of flash attention for my own understanding
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

#define CP_ASYNC_BULK_COMMIT_GROUP()                                           \
  asm volatile("cp.async.bulk.commit_group;\n" ::)
#define CP_ASYNC_BULK_WAIT_ALL() asm volatile("cp.async.bulk.wait_all;\n" ::)
#define CP_ASYNC_BULK_WAIT_GROUP(n)                                            \
  asm volatile("cp.async.bulk.wait_group %0;\n" ::"n"(n))
#define CP_ASYNC_BULK_S2G(dst, src, bytes)                                         \
  asm volatile(                                                                \
      "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], "      \
      "%2;\n" ::"l"(dst),                                                      \
      "r"(src), "n"(bytes))
#define CP_ASYNC_BULK_TENSOR_2D_S2G(tensorMap, coord_i, coord_j, src)         \
  asm volatile(                                                                \
      "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];\n"   \
      ::"l"(tensorMap), "r"(coord_i), "r"(coord_j), "r"(src))
#define CP_ASYNC_BULK_TENSOR_2D_G2S(dst, tensorMap, coord_i, coord_j, mbar)    \
  asm volatile( \
    "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n" \
    ::"r"(dst), "l"(tensorMap), "r"(coord_i), "r"(coord_j), "l"(mbar))

// gmem -> smem
#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)
#define CP_ASYNC_WAIT_ALL() asm volatile("cp.async.wait_all;\n" ::)
#define CP_ASYNC_WAIT_GROUP(n)                                                 \
  asm volatile("cp.async.wait_group %0;\n" ::"n"(n))
// ca(cache all, L1 + L2): support 4, 8, 16 bytes, cg(cache global, L2): only
// support 16 bytes.
#define CP_ASYNC_CA(dst, src, bytes)                                           \
  asm volatile(                                                                \
      "cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst),       \
      "l"(src), "n"(bytes))
#define CP_ASYNC_CG(dst, src, bytes)                                           \
  asm volatile(                                                                \
      "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst),       \
      "l"(src), "n"(bytes))

#define MBARRIER_INIT(mbar, count) \
  asm volatile( \
    ".reg .pred P_OUT;\n" \
    "mbarrier.init.b64 [%0], %1;\n"::"l"(mbar), "r"(count))

#define MBARRIER_EXPECT(mbar, count) \
  asm volatile( \
    "mbarrier.expect_tx.b64 [%0], %1;\n"::"l"(mbar), "r"(count))

#define MBARRIER_TRY_WAIT(mbar, result, phase) \
  asm volatile( \
    "mbarrier.try_wait.parity.b64 P_OUT, [%1], %2;\n" \
    "selp.b32 %0, 1, 0, P_OUT;\n":"=r"(result):"l"(mbar),"r"(phase))

// ldmatrix
#define LDMATRIX_X1(R, addr)                                                   \
  asm volatile("ldmatrix.sync.aligned.x1.m8n8.shared.b16 {%0}, [%1];\n"        \
               : "=r"(R)                                                       \
               : "r"(addr))
#define LDMATRIX_X2(R0, R1, addr)                                              \
  asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n"    \
               : "=r"(R0), "=r"(R1)                                            \
               : "r"(addr))
#define LDMATRIX_X4(R0, R1, R2, R3, addr)                                      \
  asm volatile(                                                                \
      "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"     \
      : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3)                                 \
      : "r"(addr))
#define LDMATRIX_X1_T(R, addr)                                                 \
  asm volatile("ldmatrix.sync.aligned.x1.trans.m8n8.shared.b16 {%0}, [%1];\n"  \
               : "=r"(R)                                                       \
               : "r"(addr))
#define LDMATRIX_X2_T(R0, R1, addr)                                            \
  asm volatile(                                                                \
      "ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, [%2];\n"       \
      : "=r"(R0), "=r"(R1)                                                     \
      : "r"(addr))
#define LDMATRIX_X4_T(R0, R1, R2, R3, addr)                                    \
  asm volatile(                                                                \
      "ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0, %1, %2, %3}, "      \
      "[%4];\n"                                                                \
      : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3)                                 \
      : "r"(addr))
// stmatrix: requires sm_90 or higher.
#define STMATRIX_X1(addr, R)                                                   \
  asm volatile(                                                                \
      "stmatrix.sync.aligned.x1.m8n8.shared.b16 [%0], {%1};\n" ::"r"(addr),    \
      "r"(R))
#define STMATRIX_X2(addr, R0, R1)                                              \
  asm volatile(                                                                \
      "stmatrix.sync.aligned.x2.m8n8.shared.b16 [%0], {%1, %2};\n" ::"r"(      \
          addr),                                                               \
      "r"(R0), "r"(R1))
#define STMATRIX_X4(addr, R0, R1, R2, R3)                                      \
  asm volatile(                                                                \
      "stmatrix.sync.aligned.x4.m8n8.shared.b16 [%0], {%1, %2, %3, %4};\n" ::  \
          "r"(addr),                                                           \
      "r"(R0), "r"(R1), "r"(R2), "r"(R3))
#define STMATRIX_X1_T(addr, R)                                                 \
  asm volatile(                                                                \
      "stmatrix.sync.aligned.x1.trans.m8n8.shared.b16 [%0], {%1};\n" ::"r"(    \
          addr),                                                               \
      "r"(R))
#define STMATRIX_X2_T(addr, R0, R1)                                            \
  asm volatile(                                                                \
      "stmatrix.sync.aligned.x2.trans.m8n8.shared.b16 [%0], {%1, %2};\n" ::    \
          "r"(addr),                                                           \
      "r"(R0), "r"(R1))
#define STMATRIX_X4_T(addr, R0, R1, R2, R3)                                    \
  asm volatile(                                                                \
      "stmatrix.sync.aligned.x4.trans.m8n8.shared.b16 [%0], {%1, %2, %3, "     \
      "%4};\n" ::"r"(addr),                                                    \
      "r"(R0), "r"(R1), "r"(R2), "r"(R3))
// mma m16n8k16
#define HMMA16816(RD0, RD1, RA0, RA1, RA2, RA3, RB0, RB1, RC0, RC1)            \
  asm volatile(                                                                \
      "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, "  \
      "%4, %5}, {%6, %7}, {%8, %9};\n"                                         \
      : "=r"(RD0), "=r"(RD1)                                                   \
      : "r"(RA0), "r"(RA1), "r"(RA2), "r"(RA3), "r"(RB0), "r"(RB1), "r"(RC0),  \
        "r"(RC1))


template <const int kWarpSize = 32>
__device__ __forceinline__ float warp_reduce_max_f32(float val) {
#pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val = max(val, __shfl_xor_sync(0xffffffff, val, mask));
  }
  return val;
}

template <const int kWarpSize = 32>
__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
#pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask);
  }
  return val;
}

// Q K V O size (N, d), l size (N), m size (N)
// N 1024
// d 128
template <const int N, const int d>
__global__ void kernel(
    const __grid_constant__ CUtensorMap Q_map, 
    const __grid_constant__ CUtensorMap K_map, 
    const __grid_constant__ CUtensorMap V_map,
    const __grid_constant__ CUtensorMap O_map,
    half *Q, half *K, half *V, half *O, half *l, half *m) {
  constexpr int M = 32 * 1024; // * sizeof(half)
  constexpr int Bc = M / (4 * d);     // 64
  constexpr int Br = min(M / (4 * d), d);   // 64
  constexpr int Tr = N / Br;
  constexpr int Tc = N / Bc;
  constexpr int KV_offset = Bc * d;
  constexpr int QO_offset = Br * d;

  constexpr int MMA_m = 16; // Br dim
  constexpr int MMA_k = 16; // d dim
  constexpr int MMA_n = 8; // Bc dim
  constexpr int MMA_TILE_m = 2; // warp_m
  constexpr int MMA_TILE_n = 4; // warp_n
  constexpr int WARP_TILE_m = 2;  // Br / (MMA_m * MMA_TILE_m)
  constexpr int WARP_TILE_n = 2;  // Bc / (MMA_n * MMA_TILE_n)

  __shared__ __align__(128) half smem[M];
  __shared__ __align__(128) half smem_l[Br];
  __shared__ __align__(128) half smem_m[Br];
  __shared__ __align__(128) half smem_s[Br * Bc];
  half *Kj = smem;
  half *Vj = smem + KV_offset;
  half *Qi = smem + KV_offset * 2;
  half *Oi = smem + KV_offset * 2 + QO_offset;
  half *li = smem_l;
  half *mi = smem_m;

  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int warp_id = tid / 32;
  const int lane_id = tid % 32;
  const int warp_m = warp_id % 2;   // 0,1
  const int warp_n = warp_id / 2;   // 0,1,2,3

  __shared__ uint64_t mbarrier;
  if (tid == 0) {
    MBARRIER_INIT(&mbarrier, 0);
  }

  int phase = 0;
  for (int j = 0; j < Tc; ++j) {
    // load Kj, Vj to shmem (Bc, d)
    uint32_t smem_k_addr = __cvta_generic_to_shared(Kj);
    uint32_t smem_v_addr = __cvta_generic_to_shared(Vj);

    if (tid == 0) {
      MBARRIER_EXPECT(&mbarrier, 2);
      CP_ASYNC_BULK_TENSOR_2D_G2S(smem_k_addr, &K_map, 0, j * Bc, &mbarrier);
      CP_ASYNC_BULK_TENSOR_2D_G2S(smem_v_addr, &V_map, 0, j * Bc, &mbarrier);
    }

    for (int i = 0; i < Tr; ++i) {
      // load Qi, Oi, li, mi to shmem
      uint32_t smem_q_addr = __cvta_generic_to_shared(Qi);
      uint32_t smem_o_addr = __cvta_generic_to_shared(Oi);
      uint32_t smem_l_addr = __cvta_generic_to_shared(li);
      uint32_t smem_m_addr = __cvta_generic_to_shared(mi);
      if (tid == 0) {
        MBARRIER_EXPECT(&mbarrier, 2);
        CP_ASYNC_BULK_TENSOR_2D_G2S(smem_q_addr, &Q_map, 0, i * Br, &mbarrier);
        CP_ASYNC_BULK_TENSOR_2D_G2S(smem_o_addr, &O_map, 0, i * Br, &mbarrier);
      }

#pragma unroll
      for (int k = 0; k < Br; k += 8) {
        CP_ASYNC_CG(smem_l_addr + k * sizeof(half), &l[i * Br + k], 16);
        CP_ASYNC_CG(smem_m_addr + k * sizeof(half), &m[i * Br + k], 16);
      }

      uint32_t result;
      do {
        MBARRIER_TRY_WAIT(&mbarrier, result, phase);
      } while (!result);
      phase ^= 1;

      // compute Sij = Qi * trans(Kj) Qi (Br * d) Kj (Bc * d)
      // Br = MMA_m(16) * MMA_TILE_m(2) * WARP_TILE(2)
      // Bc = MMA_n(8) * MMA_TILE_n(4) * WARP_TILE(2)
      uint32_t RQ[WARP_TILE_m][4];    // RA
      uint32_t RK[WARP_TILE_n][2];    // RB
      uint32_t RS[WARP_TILE_m][WARP_TILE_n][2]; // RC
#pragma unroll
      for (int k_stage = 0; k_stage < d / MMA_k; ++k_stage) {
#pragma unroll
        for (int m_stage = 0; m_stage < WARP_TILE_m; ++m_stage) {
          uint32_t smem_q_m_warp = m_stage * MMA_m * MMA_TILE_m + warp_m * MMA_m;
          uint32_t smem_q_m_lane = smem_q_m_warp + lane_id % 16;
          uint32_t smem_q_k_warp = k_stage * MMA_k;
          uint32_t smem_q_k_lane = smem_q_k_warp + (lane_id / 16) * 8;
          uint32_t smem_q_ptr = smem_q_addr + (smem_q_m_lane * d + smem_q_k_lane) * sizeof(half);
          LDMATRIX_X4(RQ[m_stage][0], RQ[m_stage][1], RQ[m_stage][2], RQ[m_stage][3], smem_q_ptr);
        }

#pragma unroll
        for (int n_stage = 0; n_stage < WARP_TILE_n; ++n_stage) {
          uint32_t smem_k_n_warp = n_stage * MMA_n * MMA_TILE_n + warp_n * MMA_n;
          uint32_t smem_k_n_lane = smem_k_n_warp + lane_id % 16;
          uint32_t smem_k_k_warp = k_stage * MMA_k;
          uint32_t smem_k_k_lane = smem_k_k_warp;
          uint32_t smem_k_ptr = smem_k_addr + (smem_k_n_lane * d + smem_q_k_lane) * sizeof(half);
          LDMATRIX_X2(RK[n_stage][0], RK[n_stage][1], smem_k_ptr);
        }

#pragma unroll
        for (int m_stage = 0; m_stage < WARP_TILE_m; ++m_stage) {
#pragma unroll
          for (int n_stage = 0; n_stage < WARP_TILE_n; ++n_stage) {
            HMMA16816(RS[m_stage][n_stage][0], RS[m_stage][n_stage][1], 
              RQ[m_stage][0], RQ[m_stage][1], RQ[m_stage][2], RQ[m_stage][3], 
              RK[n_stage][0], RK[n_stage][1], RS[m_stage][n_stage][0], RS[m_stage][n_stage][1]);
          }
        }
      }
      // store back to Sij (Br * Bc)
      uint32_t smem_s_addr = __cvta_generic_to_shared(smem_s);
      for (int m_stage = 0; m_stage < WARP_TILE_m; ++m_stage) {
        for (int n_stage = 0; n_stage < WARP_TILE_n; ++n_stage) {
          uint32_t smem_s_m_warp = m_stage * MMA_m * MMA_TILE_m + warp_m * MMA_m;
          uint32_t smem_s_n_warp = n_stage * MMA_n * MMA_TILE_n + warp_n * MMA_n;
          uint32_t smem_s_ptr = smem_s_addr + ((smem_s_m_warp + lane_id % 16) * MMA_n + smem_s_n_warp) * sizeof(half);
          STMATRIX_X2(smem_s_ptr, RS[m_stage][n_stage][0], RS[m_stage][n_stage][1]);
        }
      }

      // compute mij = rowmax(Sij)
      // compute Pij = exp(Sij - mi)
      // compute lij = rowsum(Pij)
#pragma unroll
      for (int row = 0; row < Br; row += 4) {
        __shared__ float max_m[4], sum[4];
        float max_m_1, max_m_2;
        float sum_1, sum_2;
        float max_v;
        float sum_v;
        if (warp_id % 2 == 0) {
          max_m_1 = __half2float(smem_s[(row + warp_id / 2) * Bc + lane_id]);
          max_m_1 = warp_reduce_max_f32<32>(max_m_1);
          if (lane_id == 0) {
            max_m[warp_id / 2] = max_m_1;
          }
        } else {
          max_m_2 = __half2float(smem_s[(row + warp_id / 2) * Bc + lane_id + 32]);
          max_m_2 = warp_reduce_max_f32<32>(max_m_2);
        }
        __syncthreads();
        if (warp_id % 2 && lane == 0) {
          max_m[warp_id / 2] = max(max_m[warp_id / 2], max_m_2);
        }
        __syncthreads();

        if (warp_id % 2 == 0) {
          sum_1 = __expf(__half2float(smem_s[(row + warp_id / 2) * Bc + lane_id]) - max_m[warp_id / 2]);
          sum_1 = warp_reduce_sum_f32<32>(sum_1);
          if (lane_id == 0) {
            sum[warp_id / 2] = sum_1;
          }
        } else {
          sum_2 = __expf(__half2float(smem_s[(row + warp_id / 2) * Bc + lane_id + 32]) - max_m[warp_id / 2]);
          sum_2 = warp_reduce_sum_f32<32>(sum_2);
        }
        __syncthreads();
        if (warp_id % 2 && lane == 0) {
          sum[warp_id / 2] += sum_2;
        }
        __syncthreads();

        // compute li = exp(mi_old - mi) * li + exp(mij - mi)
        float exp_1 = __expf(mi[row + warp_id / 2] - max(__half2float(mi[row + warp_id / 2]), max_m[warp_id / 2]));
        float exp_2 = __expf(max_m[warp_id / 2] - max(__half2float(mi[row + warp_id / 2]), max_m[warp_id / 2]));
        if (warp_id % 2 == 0 && lane_id == 0) {
          li[row + warp_id / 2] = __float2half(exp_1 * li[row + warp_id / 2] + exp_2);
        }

        // compute Pij
        smem_s[(row + warp_id / 2) * Bc + lane_id + (warp_id % 2) * 32] = __float2half(__expf(__half2float(smem_s[(row + warp_id / 2) * Bc + lane_id + (warp_id % 2) * 32]) - max_m[warp_id / 2]));
        // compute Pij[Br][Bc] * Vj[Bc][y] -> Oi[Br][y]
        // compute Oi = diag(li)^-1 * (diag(li) * exp(mi_old - mi) * Oi + exp(mij - mi) * Pij * Vj)
        __shared__ float pvsums[4][2];
        float scale = __half2float(li[row + warp_id / 2]);
        float scale_inv = 1.0f / scale;
        for (int o_d = 0; o_d < d; ++o_d) {
          float pvsum = __half2float(hmul(smem_s[(row + warp_id / 2) * Bc + lane_id + (warp_id % 2) * 32], Vj[lane_id + (warp_id % 2) * 32]));
          pvsum = warp_reduce_sum_f32<32>(pvsum);
          if (lane_id == 0) {
            pvsum[warp_id / 2][warp_id % 2] = pvsum;
          }
          __syncthreads();
          Oi[(row + warp_id / 2) * d + o_d] = __float2half(scale_inv * (scale * exp_1 * Oi[(row + warp_id / 2) * d + o_d]) + exp_2 * (pvsum[warp_id / 2][0] + pvsum[warp_id / 2][1]));
        }

        // compute mi = max(mi, mij)
        if (warp_id % 2 == 0 && lane_id == 0) {
          mi[row + warp_id / 2] = __float2half(max(__half2float(mi[row + warp_id / 2]), max_m[warp_id / 2]));
        }
      }

      // write Oi to gmem
      if (tid == 0) {
        uint32_t smem_o_addr = __cvta_generic_to_shared(Oi);
        CP_ASYNC_BULK_TENSOR_2D_S2G(&O_map, 0, i * Br, smem_o_addr);
        CP_ASYNC_BULK_COMMIT_GROUP();
      }

      // write li, mi to gmem
#pragma unroll
      for (int k = 0; k < Br; k += 4) {
        LDST128BITS(smem_l[k]) = LDST128BITS(l[i * Br + k]);
        LDST128BITS(smem_m[k]) = LDST128BITS(m[i * Br + k]);
      }

      CP_ASYNC_BULK_WAIT_GROUP(0);
      __syncthreads();
    }
  }
}

#define CUDA_CHECK(condition) \
  { \
    cudaError_t err = condition; \
    if (err != cudaSuccess) { \
      printf("CUDA Error from %s: %s\n", #condition, cudaGetErrorString(err)); \
    } \
  }

void launch(const int N, const int d) {
  // We only consider when d == 128
  constexpr int M = 32 * 1024; // * sizeof(half)
  constexpr int Bc = M / (4 * d);     // 64
  constexpr int Br = min(M / (4 * d), d);   // 64

  half *Q, *K, *V, *O, *l, *m;
  half *dev_Q, *dev_K, *dev_V, *dev_O, *dev_l, *dev_m;

  Q = (half *)malloc(N * d * sizeof(half));
  K = (half *)malloc(N * d * sizeof(half));
  V = (half *)malloc(N * d * sizeof(half));
  O = (half *)malloc(N * d * sizeof(half));
  l = (half *)malloc(N * sizeof(half));
  m = (half *)malloc(N * sizeof(half));

  // fill Q, K, V with random numbers

  // initialize O, l, m
  memset(O, 0, N * d * sizeof(half));
  memset(l, 0, N * sizeof(half));
  for (int i = 0; i < N; i++) {
    m[i] = __float2half(-FLT_MAX);
  }

  CUDA_CHECK(cudaMalloc(&dev_Q, N * d * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&dev_K, N * d * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&dev_V, N * d * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&dev_O, N * d * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&dev_l, N * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&dev_m, N * sizeof(half)));

  CUDA_CHECK(cudaMemcpy(dev_Q, Q, N * d * sizeof(half), cudaMemcpyDefault));
  CUDA_CHECK(cudaMemcpy(dev_K, K, N * d * sizeof(half), cudaMemcpyDefault));
  CUDA_CHECK(cudaMemcpy(dev_V, V, N * d * sizeof(half), cudaMemcpyDefault));
  CUDA_CHECK(cudaMemcpy(dev_O, O, N * d * sizeof(half), cudaMemcpyDefault));
  CUDA_CHECK(cudaMemcpy(dev_l, l, N * sizeof(half), cudaMemcpyDefault));
  CUDA_CHECK(cudaMemcpy(dev_m, m, N * sizeof(half), cudaMemcpyDefault));

  CUtensorMap Q_map, K_map, V_map, O_map;

  {
    // Q map
    cuuint64_t global_dim[] = {d, N};
    cuuint64_t global_strides[] = {d * sizeof(half)};
    cuuint64_t box_dim[] = {d, Br};
    cuuint32_t element_strides[] = {1, 1};
    cuTensorMapEncodeTiled(&Q_map, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 2, dev_Q, 
        global_dim, global_strides, box_dim, element_strides, 
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE, 
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  }
  {
    // K map
    cuuint64_t global_dim[] = {d, N};
    cuuint64_t global_strides[] = {d * sizeof(half)};
    cuuint64_t box_dim[] = {d, Bc};
    cuuint32_t element_strides[] = {1, 1};
    cuTensorMapEncodeTiled(&K_map, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 2, dev_K, 
        global_dim, global_strides, box_dim, element_strides, 
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE, 
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  }
  {
    // V map
    cuuint64_t global_dim[] = {d, N};
    cuuint64_t global_strides[] = {d * sizeof(half)};
    cuuint64_t box_dim[] = {d, Bc};
    cuuint32_t element_strides[] = {1, 1};
    cuTensorMapEncodeTiled(&V_map, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 2, dev_V, 
        global_dim, global_strides, box_dim, element_strides, 
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE, 
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  }
  {
    // O map
    cuuint64_t global_dim[] = {d, N};
    cuuint64_t global_strides[] = {d * sizeof(half)};
    cuuint64_t box_dim[] = {d, Br};
    cuuint32_t element_strides[] = {1, 1};
    cuTensorMapEncodeTiled(&O_map, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 2, dev_O, 
        global_dim, global_strides, box_dim, element_strides, 
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE, 
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  }

  dim3 block(256);
  kernel<<<1, 256>>>(Q_map, K_map, V_map, O_map, dev_Q, dev_K, dev_V, dev_O, dev_l, dev_m);

  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  printf("%s\n", cudaGetErrorString(err));
}
