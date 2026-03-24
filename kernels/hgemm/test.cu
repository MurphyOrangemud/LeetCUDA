#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <stdio.h>

#include <torch/extension.h>
#include <torch/types.h>

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


__constant__ CUtensorMap tma_a;
__constant__ CUtensorMap tma_b;
__constant__ CUtensorMap tma_c;

__global__ void test_kernel() {
  int tid = threadIdx.x;

  __shared__ half s_a[16][16];
  __shared__ half s_b[16][8];
  __shared__ half s_c[16][8];

  if (tid == 0) {
    for (int i = 0; i < 16; ++i) {
      for (int j = 0; j < 16; ++j) {
        s_a[i][j] = __float2half(1.0f);
      }
    }
    for (int i = 0; i < 16; ++i) {
      for (int j = 0; j < 8; ++j) {
        s_b[i][j] = __float2half(1.0f);
      }
    }
  }
  __syncthreads();

  // test
  if (tid == 0) {
    for (int i = 0; i < 16; ++i) {
      printf("[");
      for (int j = 0; j < 16; ++j) {
        if (j) printf(", ");
        printf("%.2f", __half2float(s_a[i][j]));
      }
      printf("]\n");
    }
    for (int i = 0; i < 16; ++i) {
      printf("[");
      for (int j = 0; j < 8; ++j) {
        if (j) printf(", ");
        printf("%.2f", __half2float(s_b[i][j]));
      }
      printf("]\n");
    }
  }
  __syncthreads();

  const int warp_id = tid / 32;
  const int lane_id = tid % 32;
  const int warp_m = warp_id % 2;
  const int warp_n = warp_id / 2;

  if (warp_id == 0) {
    uint32_t RA[4];
    uint32_t RB[2];
    uint32_t RC[2];

    RC[0] = 0;
    RC[1] = 0;

    uint32_t smem_a = __cvta_generic_to_shared(s_a);
    uint32_t smem_b = __cvta_generic_to_shared(s_b);
    uint32_t smem_c = __cvta_generic_to_shared(s_c);

    LDMATRIX_X4(RA[0], RA[1], RA[2], RA[3], smem_a);
    LDMATRIX_X2_T(RB[0], RB[1], smem_b);
    __syncthreads();

    // test 1st load
    printf("%x %x %x %x\n", RA[0], RA[1], RA[2], RA[3]);

    HMMA16816(RC[0], RC[1], RA[0], RA[1], RA[2], RA[3], RB[0], RB[1], RC[0], RC[1]);
    __syncthreads();

    uint32_t smem_c_ptr = smem_c + (lane_id % 16) * 8 * sizeof(half);
    STMATRIX_X2(smem_c_ptr, RC[0], RC[1]);
    __syncthreads();
  }

  // test
  if (tid == 0) {
    for (int i = 0; i < 16; ++i) {
      printf("[");
      for (int j = 0; j < 8; ++j) {
        if (j) printf(", ");
        printf("%.2f", __half2float(s_c[i][j]));
      }
      printf("]\n");
    }
  }
  __syncthreads();
}

void launch() {
  dim3 block(32);
  dim3 grid(1);
  test_kernel<<<grid, block>>>();
  cudaDeviceSynchronize();
}

// __global__ void test_kernel(half *a, half *b, half *c, int M, int N, int K) {
//   int bx = blockIdx.x;
//   int by = blockIdx.y;
//   int tx = threadIdx.x;
//   int ty = threadIdx.y;
//   int tid = tx * blockDim.y + ty;

//   // printf("a: %.2f, %.2f\nb: %.2f, %.2f\n", __half2float(a[0]), __half2float(a[M * K - 1]), __half2float(b[0]), __half2float(b[K * N - 1]));  

//   const int BM = 128;
//   const int BN = 128;
//   const int BK = 16;

//   __shared__ __align__(128) half s_a[2][BM][BK];
//   __shared__ __align__(128) half s_b[2][BK][BN];

//   int load_gmem_a_bulk_m = by * BM;
//   // int load_gmem_a_bulk_addr = load_gmem_a_bulk_m * K;
//   int load_gmem_b_bulk_n = bx * BN;
//   // int load_gmem_b_bulk_addr = load_gmem_b_bulk_n;
//   __shared__ __align__(128) half s_c[BM][BN];

//   __shared__ uint64_t mbarrier;
//   uint32_t phase = 0;
//   if (tid == 0) {
//     MBARRIER_INIT(&mbarrier, 0);
//   }

//   {
//     // copy phase-0
//     if (tid == 0) {
//       uint32_t shared_a_addr = __cvta_generic_to_shared(s_a[phase]);
//       uint32_t shared_b_addr = __cvta_generic_to_shared(s_b[phase]);

//       MBARRIER_EXPECT(&mbarrier, 2);
//       CP_ASYNC_BULK_TENSOR_2D_G2S(shared_a_addr, &tma_a, load_gmem_a_bulk_m, 0, &mbarrier);
//       CP_ASYNC_BULK_TENSOR_2D_G2S(shared_b_addr, &tma_b, 0, load_gmem_b_bulk_n, &mbarrier);
//     }

//     uint32_t result;
//     do {
//       MBARRIER_TRY_WAIT(&mbarrier, result, phase);
//     } while (!result);

//     __syncthreads();
//     phase ^= 1;
//   }

//   for (int k = 1; k < K / BK; ++k) {
//     if (tid == 0) {
//       uint32_t shared_a_addr = __cvta_generic_to_shared(s_a[phase]);
//       uint32_t shared_b_addr = __cvta_generic_to_shared(s_b[phase]);

//       MBARRIER_EXPECT(&mbarrier, 2);
//       CP_ASYNC_BULK_TENSOR_2D_G2S(shared_a_addr, &tma_a, load_gmem_a_bulk_m, k * BK, &mbarrier);
//       CP_ASYNC_BULK_TENSOR_2D_G2S(shared_b_addr, &tma_b, k * BK, load_gmem_b_bulk_n, &mbarrier);
//     }
    
//     // for simplicity, we only execute this by one thread
//     if (tid == 0) {
//       for (int i = 0; i < BM; ++i) {
//         for (int j = 0; j < BN; ++j) {
//           half sum = __float2half(0.0f);
//           for (int k = 0; k < BK; ++k) {
//             sum = __hadd(sum, __hmul(s_a[phase][i][k], s_b[phase][k][j]));
//           }
//           s_c[i][j] = __hadd(sum, s_c[i][j]);
//         }
//       }
//     }

//     uint32_t result;
//     do {
//       MBARRIER_TRY_WAIT(&mbarrier, result, phase);
//     } while (!result);

//     __syncthreads();
//     phase ^= 1;
//   }

//   // calculate last block
//   {
//     if (tid == 0) {
//       for (int i = 0; i < BM; ++i) {
//         for (int j = 0; j < BN; ++j) {
//           half sum = __float2half(0.0f);
//           for (int k = 0; k < BK; ++k) {
//             sum = __hadd(sum, __hmul(s_a[phase][i][k], s_b[phase][k][j]));
//           }
//           s_c[i][j] = __hadd(sum, s_c[i][j]);
//         }
//       }
//     }
//   }

//   // verify first copy
//   // printf("a: %.2f, %.2f\nb: %.2f, %.2f\n", __half2float(s_a[0][0]), __half2float(s_a[BM-1][BK-1]), __half2float(s_b[0][0]), __half2float(s_b[BK-1][BN-1]));
  
//   __syncthreads();
//   // verify before copy
//   // printf("shmem c: %.2f, %.2f\n", __half2float(s_c[0][0]), __half2float(s_c[BM-1][BN-1]));

//   if (tid == 0) {
//     uint32_t shared_c_addr = __cvta_generic_to_shared(s_c);
//     CP_ASYNC_BULK_TENSOR_2D_S2G(&tma_c, load_gmem_a_bulk_m, load_gmem_b_bulk_n, shared_c_addr);
//     CP_ASYNC_BULK_COMMIT_GROUP();
//   }

//   CP_ASYNC_BULK_WAIT_GROUP(0);
//   __syncthreads();

//   // verify after copy from device side
//   printf("c: %.2f, %.2f\n", __half2float(c[0]), __half2float(c[M * N - 1]));
// }

// void launch() {
//   const int M = 1024;
//   const int N = 1024;
//   const int K = 1024;
//   const int BM = 128;
//   const int BN = 128;
//   const int BK = 16;
//   half *a = (half *)malloc(M * K * sizeof(half));
//   half *b = (half *)malloc(K * N * sizeof(half));

//   for (int i = 0; i < M; ++i) {
//     for (int j = 0; j < K; ++j) {
//       a[i * K + j] = __float2half(1.0f);
//     }
//   }

//   for (int i = 0; i < K; ++i) {
//     for (int j = 0; j < N; ++j) {
//       b[i * N + j] = __float2half(1.0f);
//     }
//   }

//   printf("a: %.2f, %.2f\nb: %.2f, %.2f\n", __half2float(a[0]), __half2float(a[M * K - 1]), __half2float(b[0]), __half2float(b[K * N - 1]));  

//   half *dev_a = nullptr;
//   half *dev_b = nullptr;
//   cudaMalloc(&dev_a, M * K * sizeof(half));
//   cudaMalloc(&dev_b, K * N * sizeof(half));
//   cudaMemcpy(dev_a, a, M * K * sizeof(half), cudaMemcpyDefault);
//   cudaMemcpy(dev_b, b, K * N * sizeof(half), cudaMemcpyDefault);

//   half *dev_c = nullptr;
//   cudaMalloc(&dev_c, M * N * sizeof(half));
  
//   {
//     CUtensorMap tma_desc;
//     cuuint64_t global_dim[] = {K, M};
//     cuuint64_t global_strides[] = {K * sizeof(half)};
//     cuuint32_t box_dim[] = {BK, BM};
//     cuuint32_t element_strides[] = {1, 1};
//     cuTensorMapEncodeTiled(&tma_desc, 
//       CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 2, dev_a, 
//       global_dim, global_strides, box_dim, element_strides, 
//       CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, 
//       CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
//     cudaMemcpyToSymbol(tma_a, &tma_desc, sizeof(CUtensorMap));
//   }

//   {
//     CUtensorMap tma_desc;
//     cuuint64_t global_dim[] = {N, K};
//     cuuint64_t global_strides[] = {N * sizeof(half)};
//     cuuint32_t box_dim[] = {BN, BK};
//     cuuint32_t element_strides[] = {1, 1};
//     cuTensorMapEncodeTiled(&tma_desc, 
//       CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 2, dev_b, 
//       global_dim, global_strides, box_dim, element_strides, 
//       CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, 
//       CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
//     cudaMemcpyToSymbol(tma_b, &tma_desc, sizeof(CUtensorMap));
//   }
  
//   {
//     CUtensorMap tma_desc;
//     cuuint64_t global_dim[] = {N, M};
//     cuuint64_t global_strides[] = {N * sizeof(half)};
//     cuuint32_t box_dim[] = {BN, BM};
//     cuuint32_t element_strides[] = {1, 1};
//     cuTensorMapEncodeTiled(&tma_desc, 
//       CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 2, dev_c, 
//       global_dim, global_strides, box_dim, element_strides, 
//       CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, 
//       CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
//     cudaMemcpyToSymbol(tma_c, &tma_desc, sizeof(CUtensorMap));
//   }

//   dim3 block(256);
//   dim3 grid(N / BN, M / BM);
//   test_kernel<<<grid, block>>>(dev_a, dev_b, dev_c, M, N, K);

//   cudaDeviceSynchronize();

//   cudaError_t err = cudaGetLastError();
//   printf("%s\n", cudaGetErrorString(err));

//   half *c = (half *)malloc(M * N * sizeof(half));
//   cudaMemcpy(c, dev_c, M * N * sizeof(half), cudaMemcpyDefault);
//   printf("%.2f, %.2f\n", __half2float(c[0]), __half2float(c[M * N - 1]));
// }

PYBIND11_MODULE(test, m) {
  m.def("launch", &launch, "launch");
}
