# SGEMM

## HGEMM/SGEMM Supported Matrix

|CUDA Cores|Sliced K(Loop over K)|Tile Block|Tile Thread|
|:---:|:---:|:---:|:---:|
|✔️|✔️|✔️|✔️|
|**WMMA(m16n16k16)**|**MMA(m16n8k16)**|**Pack LDST(128 bits)**|**SMEM Padding**|
|✔️|✔️|✔️|✔️|
|**Copy Async**|**Tile MMA(More Threads)**|**Tile Warp(More Values)**|**Multi Stages**|
|✔️|✔️|✔️|✔️|
|**Reg Double Buffers**|**Block Swizzle**|**Warp Swizzle**|**Collective Store(Reg Reuse&Warp Shfl)**|
|✔️|✔️|✔️|✔️|
|**Row Major(NN)**|**Col Major(TN)**|**SGEMM TF32**|**SMEM Swizzle/Permuted**|
|✔️|✔️|✔️|❔|


## 0x00 说明

包含以下内容：

- [X] sgemm_naive_f32_kernel (naive)
- [X] sgemm_sliced_k_f32_kernel (sliced_k with smem)
- [X] sgemm_t_8x8_sliced_k_f32x4_kernel (thread tile 8x8)
- [X] sgemm_t_8x8_sliced_k_f32x4_bcf_kernel (bank conflicts free)
- [X] sgemm_t_8x8_sliced_k_f32x4_bcf_dbuf_kernel (bank conflicts free, double buffers)
- [X] sgemm_t_8x8_sliced_k16_f32x4_bcf_dbuf_kernel (double buffers, k16)
- [X] sgemm_t_8x8_sliced_k16_f32x4_bcf_dbuf_async_kernel (double buffers, k16, copy async)
- [X] sgemm_wmma_m16n16k8_mma4x2_warp2x4_stages (WMMA, Tile MMA/Warp, Copy Async, Stage, Pad, Block swizzle)
- [X] PyTorch bindings

## 目前性能
目前在L20上，CUDA Cores FP32(L20 FP32/TF32理论算力为59.8 TFLOPS) 的实现能达到cuBLAS大概85%~90%左右的性能(TFLOPS)，部分size下会超过cuBLAS。已知问题为bank conflicts没有完全消除，目前通过padding的方式缓解bank conflicts会导致shared memory浪费，也会影响SM occupancy。而Tensor Cores TF32的实现，只能达到cuBLAS TF32大概80%左右的性能，尚有较大差距。目前未手工实现smem swizzle(受限于WMMA API的灵活性以及本人的能力)，后续将会尝试通过MMA PTX实现smem swizzle/permuted。另外，当前TF32的实现依赖额外的FP32转TF32的kernel，对整体性能有影响。

## 共享内存 Bank Conflicts

含义：在访问shared memory时，因多个线程读写同一个Bank中的不同数据地址时，导致shared memory 并发读写 退化 成顺序读写的现象叫做Bank Conflict；

![](https://github.com/PaddleJitLab/CUDATutorial/blob/develop/docs/09_optimize_reduce/02_bank_conflict/images/ef322be7c3e5b6b9be69d2b90e88083f50569a58a97129f348e483b946ab4edf.png)

SM调度单位为一个warp（一个warp内32个Thread），shared_memory 可以 被一个warp中的所有（32个）线程进行访问，shared_memory 映射到大小相等的32个Bank上，Bank的数据读取带宽为32bit / cycle (4 bytes)，因此，主要需要考虑一个Warp内32线程的访问共享内存时的bank冲突。
对于多个线程读取同一个Bank数据时（不同地址），硬件把内存读写请求，拆分成 conflict-free requests，进行顺序读写，此时将会触发多次内存事务。特别地，当一个warp中的所有线程读写同一个地址时，会触发broadcast机制，此时不会退化成顺序读写。上面提到触发broadcast机制的条件是all threads acess same address，但在翻阅cuda-c-programming-guide以及最新版本的[NVProfGuide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html) 时，发现只要是多个thread 读写就会触发broadcast（不需要All）。

- 多个线程读同一个数据时，仅有一个线程读，然后broadcast到其他线程
- 多个线程写同一个数据时，仅会有一个线程写成功

NVIDIA的[文章](https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/)中指出，我们还可以通过 `cudaDeviceSetSharedMemConfig()` 函数设置默认Bank Size（默认为4 bytes）来避免bank conflicts，可设置为cudaSharedMemBankSizeFourByte或者cudaSharedMemBankSizeEightByte。对于某些场景来说，设置cudaSharedMemBankSizeEightByte或许更加合适，比如使用double数据类型时。

```C
cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
```

## 双缓冲 Double Buffers

本仓库实现的SGEMM Double Buffers策略如下：1）主循环从bk = 1 开始，第一次数据加载在主循环之前，最后一次计算在主循环之后，这是pipeline 的特点决定的；2）由于计算和下一次访存使用的Shared Memory不同，因此主循环中每次循环只需要一次__syncthreads()即可，对比非double buffers版本，总共节省了 ((K + BK - 1) / BK) - 1 次block内的同步操作。比如，bk=1时，FFMA计算使用的是s_a[0]和s_b[0]，因此，和s_a[1]和s_b[1]的加载是没有依赖关系的。FFMA计算，从global内存到s_a[1]和s_b[1]和HFMA计算可以并行。s_a[1]和s_b[1]用于加载下一块BK需要的数据到共享内存；3）由于GPU不能向CPU那样支持乱序执行，主循环中需要先将下一次循环计算需要的Gloabal Memory中的数据load 到寄存器，然后进行本次计算，之后再将load到寄存器中的数据写到Shared Memory，这样在LDG指令向Global Memory做load时，不会影响后续HFMA及其它运算指令的 launch 执行，也就达到了Double Buffers的目的。

```C
  // 1）主循环从bk = 1 开始，第一次数据加载在主循环之前，最后一次计算在主循环之后，这是pipeline 的特点决定的；
  // 2）由于计算和下一次访存使用的Shared Memory不同，因此主循环中每次循环只需要一次__syncthreads()即可
  // 3）由于GPU不能向CPU那样支持乱序执行，主循环中需要先将下一次循环计算需要的Gloabal Memory中的数据load
  // 到寄存器，然后进行本次计算，之后再将load到寄存器中的数据写到Shared Memory，这样在LDG指令向Global
  // Memory做load时，不会影响后续FFMA及其它运算指令的 launch 执行，也就达到了Double Buffering的目的。

  // bk = 0 is loading here, buffer 0

  {
    int load_a_gmem_k = load_a_smem_k;
    int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
    int load_b_gmem_k = load_b_smem_k;
    int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;
    FLOAT4(r_load_a[0]) = FLOAT4(a[load_a_gmem_addr]);
    FLOAT4(r_load_b[0]) = FLOAT4(b[load_b_gmem_addr]);

    s_a[0][load_a_smem_k + 0][load_a_smem_m] = r_load_a[0];
    s_a[0][load_a_smem_k + 1][load_a_smem_m] = r_load_a[1];
    s_a[0][load_a_smem_k + 2][load_a_smem_m] = r_load_a[2];
    s_a[0][load_a_smem_k + 3][load_a_smem_m] = r_load_a[3];
    FLOAT4(s_b[0][load_b_smem_k][load_b_smem_n]) = FLOAT4(r_load_b[0]);
  }
  // Without this synchronization, accuracy may occasionally be abnormal.
  __syncthreads();

  // bk start from 1，需要注意的是，虽然 bk 从 1 开始，但实际上 bk=1时，使用的是
  // 第0块BK中的数据（已经加载到共享内存s_a[0]和s_b[0]）；bk=2时，实际计算的是第1块
  // BK中的数据。其余以此类推，这个循环结束后，剩下最后一块BK大小的数据需要计算。
  for (int bk = 1; bk < (K + BK - 1) / BK; bk++) {

    int smem_sel = (bk - 1) & 1;
    int smem_sel_next = bk & 1;

    int load_a_gmem_k = bk * BK + load_a_smem_k;
    int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
    int load_b_gmem_k = bk * BK + load_b_smem_k;
    int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;
    FLOAT4(r_load_a[0]) = FLOAT4(a[load_a_gmem_addr]);
    FLOAT4(r_load_b[0]) = FLOAT4(b[load_b_gmem_addr]);

    #pragma unroll
    for (int tk = 0; tk < BK; tk++) {
      FLOAT4(r_comp_a[0]) = FLOAT4(s_a[smem_sel][tk][ty * TM / 2     ]);
      FLOAT4(r_comp_a[4]) = FLOAT4(s_a[smem_sel][tk][ty * TM / 2 + BM / 2]);
      FLOAT4(r_comp_b[0]) = FLOAT4(s_b[smem_sel][tk][tx * TN / 2     ]);
      FLOAT4(r_comp_b[4]) = FLOAT4(s_b[smem_sel][tk][tx * TN / 2 + BN / 2]);

      #pragma unroll
      for (int tm = 0; tm < TM; tm++) {
        #pragma unroll
        for (int tn = 0; tn < TN; tn++) {
          // r_c[tm][tn] += r_comp_a[tm] * r_comp_b[tn];
          r_c[tm][tn] = __fmaf_rn(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
        }
      }
    }

    // 对比非double buffers版本，此处不需要__syncthreads()，总共节省了
    // ((K + BK - 1) / BK) - 1 次block内的同步操作。比如，bk=1时，HFMA计算
    // 使用的是s_a[0]和s_b[0]，因此，和s_a[1]和s_b[1]的加载是没有依赖关系的。
    // 从global内存到s_a[1]和s_b[1]和HFMA计算可以并行。s_a[1]和s_b[1]用于
    // 加载下一块BK需要的数据到共享内存。
    s_a[smem_sel_next][load_a_smem_k + 0][load_a_smem_m] = r_load_a[0];
    s_a[smem_sel_next][load_a_smem_k + 1][load_a_smem_m] = r_load_a[1];
    s_a[smem_sel_next][load_a_smem_k + 2][load_a_smem_m] = r_load_a[2];
    s_a[smem_sel_next][load_a_smem_k + 3][load_a_smem_m] = r_load_a[3];
    FLOAT4(s_b[smem_sel_next][load_b_smem_k][load_b_smem_n]) = FLOAT4(r_load_b[0]);

    __syncthreads();
  }

  // 计算剩下最后一块BK
  #pragma unroll
  for (int tk = 0; tk < BK; tk++) {
    FLOAT4(r_comp_a[0]) = FLOAT4(s_a[1][tk][ty * TM / 2     ]);
    FLOAT4(r_comp_a[4]) = FLOAT4(s_a[1][tk][ty * TM / 2 + BM / 2]);
    FLOAT4(r_comp_b[0]) = FLOAT4(s_b[1][tk][tx * TN / 2     ]);
    FLOAT4(r_comp_b[4]) = FLOAT4(s_b[1][tk][tx * TN / 2 + BN / 2]);

    #pragma unroll
    for (int tm = 0; tm < TM; tm++) {
      #pragma unroll
      for (int tn = 0; tn < TN; tn++) {
        // r_c[tm][tn] += r_comp_a[tm] * r_comp_b[tn];
        r_c[tm][tn] = __fmaf_rn(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
      }
    }
  }
```

## 参考文献

- [CUDA编程概念】一、什么是bank conflict？](https://zhuanlan.zhihu.com/p/659142274)
- [解决 bank conflict](https://github.com/PaddleJitLab/CUDATutorial/blob/develop/docs/09_optimize_reduce/02_bank_conflict/README.md)
- [Bank Conflict free 的几种方式](https://zhuanlan.zhihu.com/p/722286440)
- [Using Shared Memory in CUDA C/C++](https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/)
- [CUDA（三）：通用矩阵乘法：从入门到熟练](https://zhuanlan.zhihu.com/p/657632577)

## 测试

```bash
# 只测试Ada架构 不指定默认编译所有架构 耗时较长: Volta, Ampere, Ada, Hopper, ...
export TORCH_CUDA_ARCH_LIST=Ada
python3 sgemm.py
```
输出:

```bash
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=4096, K=2048
                  out_f32x4(t8x8sk): ['70.6019897', '26.1625347'], time:2.428984ms, swizzle: NOOP, TFLOPS: 28.29 (+0.00%)
                 out_f32x4(t8x8bcf): ['70.6019897', '26.1625347'], time:2.112817ms, swizzle: NOOP, TFLOPS: 32.53 (+14.96%)
                out_f32x4(t8x8dbuf): ['70.6019897', '26.1625347'], time:1.877713ms, swizzle: NOOP, TFLOPS: 36.60 (+12.52%)
                    out_f32(cublas): ['70.6019897', '26.1625347'], time:2.229022ms, swizzle: NOOP, TFLOPS: 30.83
                         out_f32_th: ['70.6019897', '26.1625347'], time:1.778435ms, swizzle: NOOP, TFLOPS: 38.64 (+5.58%)
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['70.5943985', '26.1725273'], time:2.035927ms, swizzle: NOOP, TFLOPS: 33.75
    out_tf32(mma2x4+warp2x4+stage2): ['70.5943985', '26.1725273'], time:1.670312ms, swizzle: NOOP, TFLOPS: 41.14 (+6.47%)
  out_tf32(mma2x4+...+stage3+dsmem): ['70.5943985', '26.1725273'], time:1.820373ms, swizzle: NOOP, TFLOPS: 37.75
  out_tf32(mma2x4+...+stage2+dsmem): ['70.5943985', '26.1725273'], time:1.646137ms, swizzle: NOOP, TFLOPS: 41.75 (+1.47%)
out_tf32(mma2x4+...+stage3+swizzle): ['70.5943985', '26.1725273'], time:2.027678ms, swizzle: 512 , TFLOPS: 33.89
out_tf32(mma2x4+...+stage2+swizzle): ['70.5943985', '26.1725273'], time:1.640319ms, swizzle: 512 , TFLOPS: 41.89 (+0.35%)
 out_tf32(...+stage3+dsmem+swizzle): ['70.5943985', '26.1725273'], time:1.807355ms, swizzle: 512 , TFLOPS: 38.02
 out_tf32(...+stage2+dsmem+swizzle): ['70.5943985', '26.1725273'], time:1.627850ms, swizzle: 512 , TFLOPS: 42.21 (+0.77%)
              out_tf32(cublas+tf32): ['70.5943985', '26.1725273'], time:7.086372ms, swizzle: NOOP, TFLOPS: 9.70
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=4096, K=4096
                  out_f32x4(t8x8sk): ['151.780014', '4.5990448 '], time:4.822254ms, swizzle: NOOP, TFLOPS: 28.50 (+0.00%)
                 out_f32x4(t8x8bcf): ['151.780014', '4.5990448 '], time:4.319739ms, swizzle: NOOP, TFLOPS: 31.82 (+11.63%)
                out_f32x4(t8x8dbuf): ['151.780014', '4.5990448 '], time:3.906702ms, swizzle: NOOP, TFLOPS: 35.18 (+10.57%)
                    out_f32(cublas): ['151.780014', '4.5990448 '], time:4.850530ms, swizzle: NOOP, TFLOPS: 28.33
                         out_f32_th: ['151.780014', '4.5990448 '], time:3.584909ms, swizzle: NOOP, TFLOPS: 38.34 (+8.98%)
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['151.794143', '4.5965395 '], time:4.346919ms, swizzle: NOOP, TFLOPS: 31.62
    out_tf32(mma2x4+warp2x4+stage2): ['151.794143', '4.5965395 '], time:3.493309ms, swizzle: NOOP, TFLOPS: 39.34 (+2.62%)
  out_tf32(mma2x4+...+stage3+dsmem): ['151.794143', '4.5965395 '], time:3.765821ms, swizzle: NOOP, TFLOPS: 36.50
  out_tf32(mma2x4+...+stage2+dsmem): ['151.794143', '4.5965395 '], time:3.599095ms, swizzle: NOOP, TFLOPS: 38.19
out_tf32(mma2x4+...+stage3+swizzle): ['151.794143', '4.5965395 '], time:4.048442ms, swizzle: 512 , TFLOPS: 33.95
out_tf32(mma2x4+...+stage2+swizzle): ['151.794143', '4.5965395 '], time:3.320336ms, swizzle: 512 , TFLOPS: 41.39 (+5.21%)
 out_tf32(...+stage3+dsmem+swizzle): ['151.794143', '4.5965395 '], time:3.658032ms, swizzle: 512 , TFLOPS: 37.57
 out_tf32(...+stage2+dsmem+swizzle): ['151.794143', '4.5965395 '], time:3.310155ms, swizzle: 512 , TFLOPS: 41.52 (+0.31%)
              out_tf32(cublas+tf32): ['151.794143', '4.5965395 '], time:2.807903ms, swizzle: NOOP, TFLOPS: 48.95 (+17.89%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=4096, K=8192
                  out_f32x4(t8x8sk): ['118.496635', '44.2837791'], time:9.974384ms, swizzle: NOOP, TFLOPS: 27.56 (+0.00%)
                 out_f32x4(t8x8bcf): ['118.496635', '44.2837791'], time:8.764767ms, swizzle: NOOP, TFLOPS: 31.36 (+13.80%)
                out_f32x4(t8x8dbuf): ['118.496635', '44.2837791'], time:8.941769ms, swizzle: NOOP, TFLOPS: 30.74
                    out_f32(cublas): ['118.496635', '44.2837791'], time:7.849812ms, swizzle: NOOP, TFLOPS: 35.02 (+11.66%)
                         out_f32_th: ['118.496635', '44.2837791'], time:7.393693ms, swizzle: NOOP, TFLOPS: 37.18 (+6.17%)
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['118.526184', '44.2716636'], time:8.627605ms, swizzle: NOOP, TFLOPS: 31.86
    out_tf32(mma2x4+warp2x4+stage2): ['118.526184', '44.2716636'], time:6.934285ms, swizzle: NOOP, TFLOPS: 39.64 (+6.63%)
  out_tf32(mma2x4+...+stage3+dsmem): ['118.526184', '44.2716636'], time:7.462024ms, swizzle: NOOP, TFLOPS: 36.84
  out_tf32(mma2x4+...+stage2+dsmem): ['118.526184', '44.2716636'], time:6.970906ms, swizzle: NOOP, TFLOPS: 39.43
out_tf32(mma2x4+...+stage3+swizzle): ['118.526184', '44.2716636'], time:8.261394ms, swizzle: 512 , TFLOPS: 33.27
out_tf32(mma2x4+...+stage2+swizzle): ['118.526184', '44.2716636'], time:6.864094ms, swizzle: 512 , TFLOPS: 40.05 (+1.02%)
 out_tf32(...+stage3+dsmem+swizzle): ['118.526184', '44.2716636'], time:7.449316ms, swizzle: 512 , TFLOPS: 36.90
 out_tf32(...+stage2+dsmem+swizzle): ['118.526184', '44.2716636'], time:6.867933ms, swizzle: 512 , TFLOPS: 40.02
              out_tf32(cublas+tf32): ['118.526184', '44.2716636'], time:5.459380ms, swizzle: NOOP, TFLOPS: 50.35 (+25.73%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=8192, K=2048
                  out_f32x4(t8x8sk): ['70.5972366', '26.1622695'], time:4.638457ms, swizzle: NOOP, TFLOPS: 29.63 (+0.00%)
                 out_f32x4(t8x8bcf): ['70.5972366', '26.1622695'], time:4.083228ms, swizzle: NOOP, TFLOPS: 33.66 (+13.60%)
                out_f32x4(t8x8dbuf): ['70.5972366', '26.1622695'], time:3.705859ms, swizzle: NOOP, TFLOPS: 37.09 (+10.18%)
                    out_f32(cublas): ['70.5972366', '26.1622695'], time:4.071259ms, swizzle: NOOP, TFLOPS: 33.76
                         out_f32_th: ['70.5972366', '26.1622695'], time:3.648686ms, swizzle: NOOP, TFLOPS: 37.67 (+1.57%)
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['70.5943985', '26.1725273'], time:3.987336ms, swizzle: NOOP, TFLOPS: 34.47
    out_tf32(mma2x4+warp2x4+stage2): ['70.5943985', '26.1725273'], time:3.204703ms, swizzle: NOOP, TFLOPS: 42.89 (+13.85%)
  out_tf32(mma2x4+...+stage3+dsmem): ['70.5943985', '26.1725273'], time:3.465056ms, swizzle: NOOP, TFLOPS: 39.66
  out_tf32(mma2x4+...+stage2+dsmem): ['70.5943985', '26.1725273'], time:3.179168ms, swizzle: NOOP, TFLOPS: 43.23 (+0.80%)
out_tf32(mma2x4+...+stage3+swizzle): ['70.5943985', '26.1725273'], time:3.828763ms, swizzle: 1024, TFLOPS: 35.90
out_tf32(mma2x4+...+stage2+swizzle): ['70.5943985', '26.1725273'], time:3.141665ms, swizzle: 1024, TFLOPS: 43.75 (+1.19%)
 out_tf32(...+stage3+dsmem+swizzle): ['70.5943985', '26.1725273'], time:3.441977ms, swizzle: 1024, TFLOPS: 39.93
 out_tf32(...+stage2+dsmem+swizzle): ['70.5943985', '26.1725273'], time:3.152799ms, swizzle: 1024, TFLOPS: 43.59
              out_tf32(cublas+tf32): ['70.5943985', '26.1725273'], time:2.859544ms, swizzle: NOOP, TFLOPS: 48.06 (+9.87%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=8192, K=4096
                  out_f32x4(t8x8sk): ['151.801406', '4.59161139'], time:9.912538ms, swizzle: NOOP, TFLOPS: 27.73 (+0.00%)
                 out_f32x4(t8x8bcf): ['151.801406', '4.59161139'], time:8.917999ms, swizzle: NOOP, TFLOPS: 30.82 (+11.15%)
                out_f32x4(t8x8dbuf): ['151.801406', '4.59161139'], time:8.958077ms, swizzle: NOOP, TFLOPS: 30.68
                    out_f32(cublas): ['151.801406', '4.59161139'], time:7.909870ms, swizzle: NOOP, TFLOPS: 34.75 (+12.75%)
                         out_f32_th: ['151.801406', '4.59161139'], time:7.236218ms, swizzle: NOOP, TFLOPS: 37.99 (+9.31%)
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['151.794143', '4.5965395 '], time:7.893776ms, swizzle: NOOP, TFLOPS: 34.82
    out_tf32(mma2x4+warp2x4+stage2): ['151.794143', '4.5965395 '], time:6.559514ms, swizzle: NOOP, TFLOPS: 41.91 (+10.32%)
  out_tf32(mma2x4+...+stage3+dsmem): ['151.794143', '4.5965395 '], time:6.930255ms, swizzle: NOOP, TFLOPS: 39.66
  out_tf32(mma2x4+...+stage2+dsmem): ['151.794143', '4.5965395 '], time:6.577444ms, swizzle: NOOP, TFLOPS: 41.79
out_tf32(mma2x4+...+stage3+swizzle): ['151.794143', '4.5965395 '], time:7.675647ms, swizzle: 1024, TFLOPS: 35.81
out_tf32(mma2x4+...+stage2+swizzle): ['151.794143', '4.5965395 '], time:6.308770ms, swizzle: 1024, TFLOPS: 43.57 (+3.97%)
 out_tf32(...+stage3+dsmem+swizzle): ['151.794143', '4.5965395 '], time:6.884336ms, swizzle: 1024, TFLOPS: 39.93
 out_tf32(...+stage2+dsmem+swizzle): ['151.794143', '4.5965395 '], time:6.305503ms, swizzle: 1024, TFLOPS: 43.59 (+0.05%)
              out_tf32(cublas+tf32): ['151.794143', '4.5965395 '], time:5.328726ms, swizzle: NOOP, TFLOPS: 51.58 (+18.33%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=8192, K=8192
                  out_f32x4(t8x8sk): ['118.518661', '44.2836265'], time:20.20986ms, swizzle: NOOP, TFLOPS: 27.20 (+0.00%)
                 out_f32x4(t8x8bcf): ['118.518661', '44.2836265'], time:18.03719ms, swizzle: NOOP, TFLOPS: 30.48 (+12.05%)
                out_f32x4(t8x8dbuf): ['118.518661', '44.2836265'], time:18.61379ms, swizzle: NOOP, TFLOPS: 29.53
                    out_f32(cublas): ['118.518661', '44.2836265'], time:15.54746ms, swizzle: NOOP, TFLOPS: 35.36 (+16.01%)
                         out_f32_th: ['118.518661', '44.2836265'], time:15.30375ms, swizzle: NOOP, TFLOPS: 35.92 (+1.59%)
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['118.526184', '44.2716636'], time:15.66731ms, swizzle: NOOP, TFLOPS: 35.09
    out_tf32(mma2x4+warp2x4+stage2): ['118.526184', '44.2716636'], time:13.19141ms, swizzle: NOOP, TFLOPS: 41.68 (+16.01%)
  out_tf32(mma2x4+...+stage3+dsmem): ['118.526184', '44.2716636'], time:13.83848ms, swizzle: NOOP, TFLOPS: 39.73
  out_tf32(mma2x4+...+stage2+dsmem): ['118.526184', '44.2716636'], time:13.15524ms, swizzle: NOOP, TFLOPS: 41.79 (+0.27%)
out_tf32(mma2x4+...+stage3+swizzle): ['118.526184', '44.2716636'], time:15.49148ms, swizzle: 1024, TFLOPS: 35.49
out_tf32(mma2x4+...+stage2+swizzle): ['118.526184', '44.2716636'], time:12.80868ms, swizzle: 1024, TFLOPS: 42.92 (+2.71%)
 out_tf32(...+stage3+dsmem+swizzle): ['118.526184', '44.2716636'], time:13.90929ms, swizzle: 1024, TFLOPS: 39.52
 out_tf32(...+stage2+dsmem+swizzle): ['118.526184', '44.2716636'], time:12.78388ms, swizzle: 1024, TFLOPS: 43.00 (+0.19%)
              out_tf32(cublas+tf32): ['118.526184', '44.2716636'], time:10.33768ms, swizzle: NOOP, TFLOPS: 53.18 (+23.66%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=16384, K=2048
                  out_f32x4(t8x8sk): ['70.5972366', '26.1622695'], time:9.941315ms, swizzle: NOOP, TFLOPS: 27.65 (+0.00%)
                 out_f32x4(t8x8bcf): ['70.5972366', '26.1622695'], time:9.267258ms, swizzle: NOOP, TFLOPS: 29.66 (+7.27%)
                out_f32x4(t8x8dbuf): ['70.5972366', '26.1622695'], time:9.232449ms, swizzle: NOOP, TFLOPS: 29.77 (+0.38%)
                    out_f32(cublas): ['70.5972366', '26.1622695'], time:7.846927ms, swizzle: NOOP, TFLOPS: 35.03 (+17.66%)
                         out_f32_th: ['70.5972366', '26.1622695'], time:7.085800ms, swizzle: NOOP, TFLOPS: 38.79 (+10.74%)
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['70.5943985', '26.1725273'], time:7.701039ms, swizzle: NOOP, TFLOPS: 35.69
    out_tf32(mma2x4+warp2x4+stage2): ['70.5943985', '26.1725273'], time:6.537389ms, swizzle: NOOP, TFLOPS: 42.05 (+8.39%)
  out_tf32(mma2x4+...+stage3+dsmem): ['70.5943985', '26.1725273'], time:6.712508ms, swizzle: NOOP, TFLOPS: 40.95
  out_tf32(mma2x4+...+stage2+dsmem): ['70.5943985', '26.1725273'], time:6.550049ms, swizzle: NOOP, TFLOPS: 41.97
out_tf32(mma2x4+...+stage3+swizzle): ['70.5943985', '26.1725273'], time:7.554650ms, swizzle: 2048, TFLOPS: 36.39
out_tf32(mma2x4+...+stage2+swizzle): ['70.5943985', '26.1725273'], time:6.168079ms, swizzle: 2048, TFLOPS: 44.56 (+5.99%)
 out_tf32(...+stage3+dsmem+swizzle): ['70.5943985', '26.1725273'], time:6.722187ms, swizzle: 2048, TFLOPS: 40.89
 out_tf32(...+stage2+dsmem+swizzle): ['70.5943985', '26.1725273'], time:6.171321ms, swizzle: 2048, TFLOPS: 44.54
              out_tf32(cublas+tf32): ['70.5943985', '26.1725273'], time:5.131006ms, swizzle: NOOP, TFLOPS: 53.57 (+20.21%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=16384, K=4096
                  out_f32x4(t8x8sk): ['151.799118', '4.6021018 '], time:20.19996ms, swizzle: NOOP, TFLOPS: 27.22 (+0.00%)
                 out_f32x4(t8x8bcf): ['151.799118', '4.6021018 '], time:18.53487ms, swizzle: NOOP, TFLOPS: 29.66 (+8.98%)
                out_f32x4(t8x8dbuf): ['151.799118', '4.6021018 '], time:18.93479ms, swizzle: NOOP, TFLOPS: 29.03
                    out_f32(cublas): ['151.799118', '4.6021018 '], time:14.90321ms, swizzle: NOOP, TFLOPS: 36.89 (+24.37%)
                         out_f32_th: ['151.799118', '4.6021018 '], time:14.38026ms, swizzle: NOOP, TFLOPS: 38.23 (+3.64%)
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['151.794143', '4.5965395 '], time:15.34090ms, swizzle: NOOP, TFLOPS: 35.84
    out_tf32(mma2x4+warp2x4+stage2): ['151.794143', '4.5965395 '], time:12.95042ms, swizzle: NOOP, TFLOPS: 42.45 (+11.04%)
  out_tf32(mma2x4+...+stage3+dsmem): ['151.794143', '4.5965395 '], time:13.73360ms, swizzle: NOOP, TFLOPS: 40.03
  out_tf32(mma2x4+...+stage2+dsmem): ['151.794143', '4.5965395 '], time:12.93442ms, swizzle: NOOP, TFLOPS: 42.50 (+0.12%)
out_tf32(mma2x4+...+stage3+swizzle): ['151.794143', '4.5965395 '], time:15.03224ms, swizzle: 2048, TFLOPS: 36.57
out_tf32(mma2x4+...+stage2+swizzle): ['151.794143', '4.5965395 '], time:12.34993ms, swizzle: 2048, TFLOPS: 44.51 (+4.73%)
 out_tf32(...+stage3+dsmem+swizzle): ['151.794143', '4.5965395 '], time:13.40029ms, swizzle: 2048, TFLOPS: 41.03
 out_tf32(...+stage2+dsmem+swizzle): ['151.794143', '4.5965395 '], time:12.32724ms, swizzle: 2048, TFLOPS: 44.60 (+0.18%)
              out_tf32(cublas+tf32): ['151.794143', '4.5965395 '], time:9.960341ms, swizzle: NOOP, TFLOPS: 55.19 (+23.76%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=16384, K=8192
                  out_f32x4(t8x8sk): ['118.513626', '44.2889137'], time:40.22870ms, swizzle: NOOP, TFLOPS: 27.33 (+0.00%)
                 out_f32x4(t8x8bcf): ['118.513626', '44.2889137'], time:39.04280ms, swizzle: NOOP, TFLOPS: 28.16 (+3.04%)
                out_f32x4(t8x8dbuf): ['118.513626', '44.2889137'], time:39.80977ms, swizzle: NOOP, TFLOPS: 27.62
                    out_f32(cublas): ['118.513626', '44.2889137'], time:28.38425ms, swizzle: NOOP, TFLOPS: 38.74 (+37.55%)
                         out_f32_th: ['118.513626', '44.2889137'], time:29.08875ms, swizzle: NOOP, TFLOPS: 37.80
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['118.526184', '44.2716636'], time:30.07037ms, swizzle: NOOP, TFLOPS: 36.56
    out_tf32(mma2x4+warp2x4+stage2): ['118.526184', '44.2716636'], time:26.02388ms, swizzle: NOOP, TFLOPS: 42.25 (+9.07%)
  out_tf32(mma2x4+...+stage3+dsmem): ['118.526184', '44.2716636'], time:27.45041ms, swizzle: NOOP, TFLOPS: 40.05
  out_tf32(mma2x4+...+stage2+dsmem): ['118.526184', '44.2716636'], time:26.32236ms, swizzle: NOOP, TFLOPS: 41.77
out_tf32(mma2x4+...+stage3+swizzle): ['118.526184', '44.2716636'], time:30.09891ms, swizzle: 2048, TFLOPS: 36.53
out_tf32(mma2x4+...+stage2+swizzle): ['118.526184', '44.2716636'], time:24.76131ms, swizzle: 2048, TFLOPS: 44.40 (+5.10%)
 out_tf32(...+stage3+dsmem+swizzle): ['118.526184', '44.2716636'], time:26.82106ms, swizzle: 2048, TFLOPS: 40.99
 out_tf32(...+stage2+dsmem+swizzle): ['118.526184', '44.2716636'], time:24.67982ms, swizzle: 2048, TFLOPS: 44.55 (+0.33%)
              out_tf32(cublas+tf32): ['118.526184', '44.2716636'], time:19.58444ms, swizzle: NOOP, TFLOPS: 56.14 (+26.02%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=4096, K=2048
                  out_f32x4(t8x8sk): ['70.5949554', '26.1727619'], time:4.644012ms, swizzle: NOOP, TFLOPS: 29.59 (+0.00%)
                 out_f32x4(t8x8bcf): ['70.5949554', '26.1727619'], time:4.165029ms, swizzle: NOOP, TFLOPS: 33.00 (+11.50%)
                out_f32x4(t8x8dbuf): ['70.5949554', '26.1727619'], time:3.532195ms, swizzle: NOOP, TFLOPS: 38.91 (+17.92%)
                    out_f32(cublas): ['70.5949554', '26.1727619'], time:4.056715ms, swizzle: NOOP, TFLOPS: 33.88
                         out_f32_th: ['70.5949554', '26.1727619'], time:3.668260ms, swizzle: NOOP, TFLOPS: 37.47
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['70.5943985', '26.1725273'], time:4.008388ms, swizzle: NOOP, TFLOPS: 34.29
    out_tf32(mma2x4+warp2x4+stage2): ['70.5943985', '26.1725273'], time:3.218698ms, swizzle: NOOP, TFLOPS: 42.70 (+9.74%)
  out_tf32(mma2x4+...+stage3+dsmem): ['70.5943985', '26.1725273'], time:3.489041ms, swizzle: NOOP, TFLOPS: 39.39
  out_tf32(mma2x4+...+stage2+dsmem): ['70.5943985', '26.1725273'], time:3.196096ms, swizzle: NOOP, TFLOPS: 43.00 (+0.71%)
out_tf32(mma2x4+...+stage3+swizzle): ['70.5943985', '26.1725273'], time:3.782248ms, swizzle: 512 , TFLOPS: 36.34
out_tf32(mma2x4+...+stage2+swizzle): ['70.5943985', '26.1725273'], time:3.096580ms, swizzle: 512 , TFLOPS: 44.38 (+3.21%)
 out_tf32(...+stage3+dsmem+swizzle): ['70.5943985', '26.1725273'], time:3.394317ms, swizzle: 512 , TFLOPS: 40.49
 out_tf32(...+stage2+dsmem+swizzle): ['70.5943985', '26.1725273'], time:3.095269ms, swizzle: 512 , TFLOPS: 44.40 (+0.04%)
              out_tf32(cublas+tf32): ['70.5943985', '26.1725273'], time:11.76311ms, swizzle: NOOP, TFLOPS: 11.68
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=4096, K=4096
                  out_f32x4(t8x8sk): ['151.796371', '4.59689951'], time:9.283566ms, swizzle: NOOP, TFLOPS: 29.61 (+0.00%)
                 out_f32x4(t8x8bcf): ['151.796371', '4.59689951'], time:8.359241ms, swizzle: NOOP, TFLOPS: 32.88 (+11.06%)
                out_f32x4(t8x8dbuf): ['151.796371', '4.59689951'], time:7.493996ms, swizzle: NOOP, TFLOPS: 36.68 (+11.55%)
                    out_f32(cublas): ['151.796371', '4.59689951'], time:7.483124ms, swizzle: NOOP, TFLOPS: 36.73 (+0.15%)
                         out_f32_th: ['151.796371', '4.59689951'], time:7.139444ms, swizzle: NOOP, TFLOPS: 38.50 (+4.81%)
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['151.794143', '4.5965395 '], time:7.942914ms, swizzle: NOOP, TFLOPS: 34.61
    out_tf32(mma2x4+warp2x4+stage2): ['151.794143', '4.5965395 '], time:6.454420ms, swizzle: NOOP, TFLOPS: 42.59 (+10.61%)
  out_tf32(mma2x4+...+stage3+dsmem): ['151.794143', '4.5965395 '], time:7.018256ms, swizzle: NOOP, TFLOPS: 39.17
  out_tf32(mma2x4+...+stage2+dsmem): ['151.794143', '4.5965395 '], time:6.443977ms, swizzle: NOOP, TFLOPS: 42.66 (+0.16%)
out_tf32(mma2x4+...+stage3+swizzle): ['151.794143', '4.5965395 '], time:7.723641ms, swizzle: 512 , TFLOPS: 35.59
out_tf32(mma2x4+...+stage2+swizzle): ['151.794143', '4.5965395 '], time:6.369042ms, swizzle: 512 , TFLOPS: 43.16 (+1.18%)
 out_tf32(...+stage3+dsmem+swizzle): ['151.794143', '4.5965395 '], time:6.931543ms, swizzle: 512 , TFLOPS: 39.66
 out_tf32(...+stage2+dsmem+swizzle): ['151.794143', '4.5965395 '], time:6.361842ms, swizzle: 512 , TFLOPS: 43.21 (+0.11%)
              out_tf32(cublas+tf32): ['151.794143', '4.5965395 '], time:5.284237ms, swizzle: NOOP, TFLOPS: 52.02 (+20.39%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=4096, K=8192
                  out_f32x4(t8x8sk): ['118.532104', '44.2729606'], time:19.66500ms, swizzle: NOOP, TFLOPS: 27.96 (+0.00%)
                 out_f32x4(t8x8bcf): ['118.532104', '44.2729606'], time:17.24970ms, swizzle: NOOP, TFLOPS: 31.87 (+14.00%)
                out_f32x4(t8x8dbuf): ['118.532104', '44.2729606'], time:17.30856ms, swizzle: NOOP, TFLOPS: 31.76
                    out_f32(cublas): ['118.532104', '44.2729606'], time:15.01247ms, swizzle: NOOP, TFLOPS: 36.62 (+14.90%)
                         out_f32_th: ['118.532104', '44.2729606'], time:14.77088ms, swizzle: NOOP, TFLOPS: 37.22 (+1.64%)
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['118.526184', '44.2716636'], time:15.61958ms, swizzle: NOOP, TFLOPS: 35.20
    out_tf32(mma2x4+warp2x4+stage2): ['118.526184', '44.2716636'], time:13.11204ms, swizzle: NOOP, TFLOPS: 41.93 (+12.65%)
  out_tf32(mma2x4+...+stage3+dsmem): ['118.526184', '44.2716636'], time:13.86370ms, swizzle: NOOP, TFLOPS: 39.65
  out_tf32(mma2x4+...+stage2+dsmem): ['118.526184', '44.2716636'], time:13.01887ms, swizzle: NOOP, TFLOPS: 42.23 (+0.72%)
out_tf32(mma2x4+...+stage3+swizzle): ['118.526184', '44.2716636'], time:15.49036ms, swizzle: 512 , TFLOPS: 35.49
out_tf32(mma2x4+...+stage2+swizzle): ['118.526184', '44.2716636'], time:12.93551ms, swizzle: 512 , TFLOPS: 42.50 (+0.64%)
 out_tf32(...+stage3+dsmem+swizzle): ['118.526184', '44.2716636'], time:13.91084ms, swizzle: 512 , TFLOPS: 39.52
 out_tf32(...+stage2+dsmem+swizzle): ['118.526184', '44.2716636'], time:12.87522ms, swizzle: 512 , TFLOPS: 42.70 (+0.47%)
              out_tf32(cublas+tf32): ['118.526184', '44.2716636'], time:10.32779ms, swizzle: NOOP, TFLOPS: 53.23 (+24.67%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=8192, K=2048
                  out_f32x4(t8x8sk): ['70.5949554', '26.1727619'], time:9.005260ms, swizzle: NOOP, TFLOPS: 30.52 (+0.00%)
                 out_f32x4(t8x8bcf): ['70.5949554', '26.1727619'], time:8.109664ms, swizzle: NOOP, TFLOPS: 33.90 (+11.04%)
                out_f32x4(t8x8dbuf): ['70.5949554', '26.1727619'], time:7.237076ms, swizzle: NOOP, TFLOPS: 37.98 (+12.06%)
                    out_f32(cublas): ['70.5949554', '26.1727619'], time:7.283616ms, swizzle: NOOP, TFLOPS: 37.74
                         out_f32_th: ['70.5949554', '26.1727619'], time:7.025599ms, swizzle: NOOP, TFLOPS: 39.13 (+3.01%)
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['70.5943985', '26.1725273'], time:7.638692ms, swizzle: NOOP, TFLOPS: 35.98
    out_tf32(mma2x4+warp2x4+stage2): ['70.5943985', '26.1725273'], time:6.153583ms, swizzle: NOOP, TFLOPS: 44.67 (+14.17%)
  out_tf32(mma2x4+...+stage3+dsmem): ['70.5943985', '26.1725273'], time:6.675100ms, swizzle: NOOP, TFLOPS: 41.18
  out_tf32(mma2x4+...+stage2+dsmem): ['70.5943985', '26.1725273'], time:6.140279ms, swizzle: NOOP, TFLOPS: 44.77 (+0.22%)
out_tf32(mma2x4+...+stage3+swizzle): ['70.5943985', '26.1725273'], time:7.350254ms, swizzle: 1024, TFLOPS: 37.40
out_tf32(mma2x4+...+stage2+swizzle): ['70.5943985', '26.1725273'], time:6.009721ms, swizzle: 1024, TFLOPS: 45.74 (+2.17%)
 out_tf32(...+stage3+dsmem+swizzle): ['70.5943985', '26.1725273'], time:6.560659ms, swizzle: 1024, TFLOPS: 41.90
 out_tf32(...+stage2+dsmem+swizzle): ['70.5943985', '26.1725273'], time:6.008577ms, swizzle: 1024, TFLOPS: 45.75 (+0.02%)
              out_tf32(cublas+tf32): ['70.5943985', '26.1725273'], time:5.121445ms, swizzle: NOOP, TFLOPS: 53.67 (+17.32%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=8192, K=4096
                  out_f32x4(t8x8sk): ['151.796371', '4.59689951'], time:19.40293ms, swizzle: NOOP, TFLOPS: 28.33 (+0.00%)
                 out_f32x4(t8x8bcf): ['151.796371', '4.59689951'], time:17.21770ms, swizzle: NOOP, TFLOPS: 31.93 (+12.69%)
                out_f32x4(t8x8dbuf): ['151.796371', '4.59689951'], time:17.95308ms, swizzle: NOOP, TFLOPS: 30.62
                    out_f32(cublas): ['151.796371', '4.59689951'], time:14.42518ms, swizzle: NOOP, TFLOPS: 38.11 (+19.36%)
                         out_f32_th: ['151.796371', '4.59689951'], time:14.29438ms, swizzle: NOOP, TFLOPS: 38.46 (+0.92%)
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['151.794143', '4.5965395 '], time:14.90476ms, swizzle: NOOP, TFLOPS: 36.88
    out_tf32(mma2x4+warp2x4+stage2): ['151.794143', '4.5965395 '], time:12.51502ms, swizzle: NOOP, TFLOPS: 43.93 (+14.22%)
  out_tf32(mma2x4+...+stage3+dsmem): ['151.794143', '4.5965395 '], time:13.19789ms, swizzle: NOOP, TFLOPS: 41.65
  out_tf32(mma2x4+...+stage2+dsmem): ['151.794143', '4.5965395 '], time:12.53654ms, swizzle: NOOP, TFLOPS: 43.85
out_tf32(mma2x4+...+stage3+swizzle): ['151.794143', '4.5965395 '], time:14.80431ms, swizzle: 1024, TFLOPS: 37.13
out_tf32(mma2x4+...+stage2+swizzle): ['151.794143', '4.5965395 '], time:12.12592ms, swizzle: 1024, TFLOPS: 45.34 (+3.21%)
 out_tf32(...+stage3+dsmem+swizzle): ['151.794143', '4.5965395 '], time:13.21063ms, swizzle: 1024, TFLOPS: 41.61
 out_tf32(...+stage2+dsmem+swizzle): ['151.794143', '4.5965395 '], time:12.12511ms, swizzle: 1024, TFLOPS: 45.34 (+0.01%)
              out_tf32(cublas+tf32): ['151.794143', '4.5965395 '], time:10.02106ms, swizzle: NOOP, TFLOPS: 54.86 (+21.00%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=8192, K=8192
                  out_f32x4(t8x8sk): ['118.532104', '44.2729606'], time:39.05200ms, swizzle: NOOP, TFLOPS: 28.16 (+0.00%)
                 out_f32x4(t8x8bcf): ['118.532104', '44.2729606'], time:36.05434ms, swizzle: NOOP, TFLOPS: 30.50 (+8.31%)
                out_f32x4(t8x8dbuf): ['118.532104', '44.2729606'], time:36.42346ms, swizzle: NOOP, TFLOPS: 30.19
                    out_f32(cublas): ['118.532104', '44.2729606'], time:28.22470ms, swizzle: NOOP, TFLOPS: 38.96 (+27.74%)
                         out_f32_th: ['118.532104', '44.2729606'], time:28.45404ms, swizzle: NOOP, TFLOPS: 38.64
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['118.526184', '44.2716636'], time:29.65857ms, swizzle: NOOP, TFLOPS: 37.07
    out_tf32(mma2x4+warp2x4+stage2): ['118.526184', '44.2716636'], time:25.09703ms, swizzle: NOOP, TFLOPS: 43.81 (+12.46%)
  out_tf32(mma2x4+...+stage3+dsmem): ['118.526184', '44.2716636'], time:26.67160ms, swizzle: NOOP, TFLOPS: 41.22
  out_tf32(mma2x4+...+stage2+dsmem): ['118.526184', '44.2716636'], time:25.22740ms, swizzle: NOOP, TFLOPS: 43.58
out_tf32(mma2x4+...+stage3+swizzle): ['118.526184', '44.2716636'], time:29.67340ms, swizzle: 1024, TFLOPS: 37.05
out_tf32(mma2x4+...+stage2+swizzle): ['118.526184', '44.2716636'], time:24.31735ms, swizzle: 1024, TFLOPS: 45.22 (+3.21%)
 out_tf32(...+stage3+dsmem+swizzle): ['118.526184', '44.2716636'], time:26.41408ms, swizzle: 1024, TFLOPS: 41.63
 out_tf32(...+stage2+dsmem+swizzle): ['118.526184', '44.2716636'], time:24.30074ms, swizzle: 1024, TFLOPS: 45.25 (+0.07%)
              out_tf32(cublas+tf32): ['118.526184', '44.2716636'], time:19.56663ms, swizzle: NOOP, TFLOPS: 56.19 (+24.19%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=16384, K=2048
                  out_f32x4(t8x8sk): ['70.5949554', '26.1727619'], time:19.93403ms, swizzle: NOOP, TFLOPS: 27.58 (+0.00%)
                 out_f32x4(t8x8bcf): ['70.5949554', '26.1727619'], time:17.85275ms, swizzle: NOOP, TFLOPS: 30.79 (+11.66%)
                out_f32x4(t8x8dbuf): ['70.5949554', '26.1727619'], time:17.60568ms, swizzle: NOOP, TFLOPS: 31.23 (+1.40%)
                    out_f32(cublas): ['70.5949554', '26.1727619'], time:14.66460ms, swizzle: NOOP, TFLOPS: 37.49 (+20.06%)
                         out_f32_th: ['70.5949554', '26.1727619'], time:14.66336ms, swizzle: NOOP, TFLOPS: 37.49 (+0.01%)
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['70.5943985', '26.1725273'], time:14.75033ms, swizzle: NOOP, TFLOPS: 37.27
    out_tf32(mma2x4+warp2x4+stage2): ['70.5943985', '26.1725273'], time:12.68918ms, swizzle: NOOP, TFLOPS: 43.32 (+15.56%)
  out_tf32(mma2x4+...+stage3+dsmem): ['70.5943985', '26.1725273'], time:13.28039ms, swizzle: NOOP, TFLOPS: 41.40
  out_tf32(mma2x4+...+stage2+dsmem): ['70.5943985', '26.1725273'], time:12.78223ms, swizzle: NOOP, TFLOPS: 43.01
out_tf32(mma2x4+...+stage3+swizzle): ['70.5943985', '26.1725273'], time:14.66119ms, swizzle: 2048, TFLOPS: 37.50
out_tf32(mma2x4+...+stage2+swizzle): ['70.5943985', '26.1725273'], time:11.99231ms, swizzle: 2048, TFLOPS: 45.84 (+5.81%)
 out_tf32(...+stage3+dsmem+swizzle): ['70.5943985', '26.1725273'], time:13.03169ms, swizzle: 2048, TFLOPS: 42.19
 out_tf32(...+stage2+dsmem+swizzle): ['70.5943985', '26.1725273'], time:11.96327ms, swizzle: 2048, TFLOPS: 45.95 (+0.24%)
              out_tf32(cublas+tf32): ['70.5943985', '26.1725273'], time:9.859824ms, swizzle: NOOP, TFLOPS: 55.76 (+21.33%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=16384, K=4096
                  out_f32x4(t8x8sk): ['151.796371', '4.59689951'], time:40.03288ms, swizzle: NOOP, TFLOPS: 27.47 (+0.00%)
                 out_f32x4(t8x8bcf): ['151.796371', '4.59689951'], time:39.52372ms, swizzle: NOOP, TFLOPS: 27.82 (+1.29%)
                out_f32x4(t8x8dbuf): ['151.796371', '4.59689951'], time:37.59534ms, swizzle: NOOP, TFLOPS: 29.25 (+5.13%)
                    out_f32(cublas): ['151.796371', '4.59689951'], time:27.83019ms, swizzle: NOOP, TFLOPS: 39.51 (+35.09%)
                         out_f32_th: ['151.796371', '4.59689951'], time:27.95956ms, swizzle: NOOP, TFLOPS: 39.33
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['151.794143', '4.5965395 '], time:29.30724ms, swizzle: NOOP, TFLOPS: 37.52
    out_tf32(mma2x4+warp2x4+stage2): ['151.794143', '4.5965395 '], time:25.27904ms, swizzle: NOOP, TFLOPS: 43.49 (+10.09%)
  out_tf32(mma2x4+...+stage3+dsmem): ['151.794143', '4.5965395 '], time:27.31575ms, swizzle: NOOP, TFLOPS: 40.25
  out_tf32(mma2x4+...+stage2+dsmem): ['151.794143', '4.5965395 '], time:25.58822ms, swizzle: NOOP, TFLOPS: 42.97
out_tf32(mma2x4+...+stage3+swizzle): ['151.794143', '4.5965395 '], time:29.27069ms, swizzle: 2048, TFLOPS: 37.56
out_tf32(mma2x4+...+stage2+swizzle): ['151.794143', '4.5965395 '], time:23.81775ms, swizzle: 2048, TFLOPS: 46.16 (+6.14%)
 out_tf32(...+stage3+dsmem+swizzle): ['151.794143', '4.5965395 '], time:26.00069ms, swizzle: 2048, TFLOPS: 42.29
 out_tf32(...+stage2+dsmem+swizzle): ['151.794143', '4.5965395 '], time:23.87239ms, swizzle: 2048, TFLOPS: 46.06
              out_tf32(cublas+tf32): ['151.794143', '4.5965395 '], time:19.24333ms, swizzle: NOOP, TFLOPS: 57.14 (+23.77%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=16384, K=8192
                  out_f32x4(t8x8sk): ['118.532104', '44.2729606'], time:81.30698ms, swizzle: NOOP, TFLOPS: 27.05 (+0.00%)
                 out_f32x4(t8x8bcf): ['118.532104', '44.2729606'], time:75.78270ms, swizzle: NOOP, TFLOPS: 29.02 (+7.29%)
                out_f32x4(t8x8dbuf): ['118.532104', '44.2729606'], time:75.56617ms, swizzle: NOOP, TFLOPS: 29.10 (+0.29%)
                    out_f32(cublas): ['118.532104', '44.2729606'], time:56.42166ms, swizzle: NOOP, TFLOPS: 38.97 (+33.93%)
                         out_f32_th: ['118.532104', '44.2729606'], time:57.50610ms, swizzle: NOOP, TFLOPS: 38.24
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['118.526184', '44.2716636'], time:58.45718ms, swizzle: NOOP, TFLOPS: 37.62
    out_tf32(mma2x4+warp2x4+stage2): ['118.526184', '44.2716636'], time:51.36411ms, swizzle: NOOP, TFLOPS: 42.81 (+9.85%)
  out_tf32(mma2x4+...+stage3+dsmem): ['118.526184', '44.2716636'], time:53.86862ms, swizzle: NOOP, TFLOPS: 40.82
  out_tf32(mma2x4+...+stage2+dsmem): ['118.526184', '44.2716636'], time:51.22380ms, swizzle: NOOP, TFLOPS: 42.93 (+0.27%)
out_tf32(mma2x4+...+stage3+swizzle): ['118.526184', '44.2716636'], time:58.32481ms, swizzle: 2048, TFLOPS: 37.70
out_tf32(mma2x4+...+stage2+swizzle): ['118.526184', '44.2716636'], time:47.85780ms, swizzle: 2048, TFLOPS: 45.95 (+7.03%)
 out_tf32(...+stage3+dsmem+swizzle): ['118.526184', '44.2716636'], time:51.81453ms, swizzle: 2048, TFLOPS: 42.44
 out_tf32(...+stage2+dsmem+swizzle): ['118.526184', '44.2716636'], time:47.76165ms, swizzle: 2048, TFLOPS: 46.04 (+0.20%)
              out_tf32(cublas+tf32): ['118.526184', '44.2716636'], time:38.08858ms, swizzle: NOOP, TFLOPS: 57.73 (+25.40%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=16384, N=4096, K=2048
                  out_f32x4(t8x8sk): ['70.5949554', '26.1727619'], time:9.190845ms, swizzle: NOOP, TFLOPS: 29.91 (+0.00%)
                 out_f32x4(t8x8bcf): ['70.5949554', '26.1727619'], time:8.345413ms, swizzle: NOOP, TFLOPS: 32.94 (+10.13%)
                out_f32x4(t8x8dbuf): ['70.5949554', '26.1727619'], time:7.679963ms, swizzle: NOOP, TFLOPS: 35.79 (+8.66%)
                    out_f32(cublas): ['70.5949554', '26.1727619'], time:7.500529ms, swizzle: NOOP, TFLOPS: 36.65 (+2.39%)
                         out_f32_th: ['70.5949554', '26.1727619'], time:7.146787ms, swizzle: NOOP, TFLOPS: 38.46 (+4.95%)
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['70.5943985', '26.1725273'], time:7.968235ms, swizzle: NOOP, TFLOPS: 34.50
    out_tf32(mma2x4+warp2x4+stage2): ['70.5943985', '26.1725273'], time:6.254506ms, swizzle: NOOP, TFLOPS: 43.95 (+14.27%)
  out_tf32(mma2x4+...+stage3+dsmem): ['70.5943985', '26.1725273'], time:6.782460ms, swizzle: NOOP, TFLOPS: 40.53
  out_tf32(mma2x4+...+stage2+dsmem): ['70.5943985', '26.1725273'], time:6.247973ms, swizzle: NOOP, TFLOPS: 43.99 (+0.10%)
out_tf32(mma2x4+...+stage3+swizzle): ['70.5943985', '26.1725273'], time:7.488203ms, swizzle: 512 , TFLOPS: 36.71
out_tf32(mma2x4+...+stage2+swizzle): ['70.5943985', '26.1725273'], time:6.200075ms, swizzle: 512 , TFLOPS: 44.33 (+0.77%)
 out_tf32(...+stage3+dsmem+swizzle): ['70.5943985', '26.1725273'], time:6.759619ms, swizzle: 512 , TFLOPS: 40.66
 out_tf32(...+stage2+dsmem+swizzle): ['70.5943985', '26.1725273'], time:6.231451ms, swizzle: 512 , TFLOPS: 44.11
              out_tf32(cublas+tf32): ['70.5943985', '26.1725273'], time:5.184912ms, swizzle: NOOP, TFLOPS: 53.01 (+19.58%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=16384, N=4096, K=4096
                  out_f32x4(t8x8sk): ['151.796371', '4.59689951'], time:18.67318ms, swizzle: NOOP, TFLOPS: 29.44 (+0.00%)
                 out_f32x4(t8x8bcf): ['151.796371', '4.59689951'], time:16.58837ms, swizzle: NOOP, TFLOPS: 33.14 (+12.57%)
                out_f32x4(t8x8dbuf): ['151.796371', '4.59689951'], time:16.44637ms, swizzle: NOOP, TFLOPS: 33.43 (+0.86%)
                    out_f32(cublas): ['151.796371', '4.59689951'], time:14.57281ms, swizzle: NOOP, TFLOPS: 37.72 (+12.86%)
                         out_f32_th: ['151.796371', '4.59689951'], time:14.51504ms, swizzle: NOOP, TFLOPS: 37.87 (+0.40%)
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['151.794143', '4.5965395 '], time:15.47667ms, swizzle: NOOP, TFLOPS: 35.52
    out_tf32(mma2x4+warp2x4+stage2): ['151.794143', '4.5965395 '], time:12.47291ms, swizzle: NOOP, TFLOPS: 44.08 (+16.37%)
  out_tf32(mma2x4+...+stage3+dsmem): ['151.794143', '4.5965395 '], time:13.44106ms, swizzle: NOOP, TFLOPS: 40.90
  out_tf32(mma2x4+...+stage2+dsmem): ['151.794143', '4.5965395 '], time:12.39275ms, swizzle: NOOP, TFLOPS: 44.36 (+0.65%)
out_tf32(mma2x4+...+stage3+swizzle): ['151.794143', '4.5965395 '], time:14.96281ms, swizzle: 512 , TFLOPS: 36.74
out_tf32(mma2x4+...+stage2+swizzle): ['151.794143', '4.5965395 '], time:12.40277ms, swizzle: 512 , TFLOPS: 44.33
 out_tf32(...+stage3+dsmem+swizzle): ['151.794143', '4.5965395 '], time:13.47801ms, swizzle: 512 , TFLOPS: 40.79
 out_tf32(...+stage2+dsmem+swizzle): ['151.794143', '4.5965395 '], time:12.43972ms, swizzle: 512 , TFLOPS: 44.19
              out_tf32(cublas+tf32): ['151.794143', '4.5965395 '], time:10.85467ms, swizzle: NOOP, TFLOPS: 50.65 (+14.17%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=16384, N=4096, K=8192
                  out_f32x4(t8x8sk): ['118.532104', '44.2729606'], time:38.72056ms, swizzle: NOOP, TFLOPS: 28.40 (+0.00%)
                 out_f32x4(t8x8bcf): ['118.532104', '44.2729606'], time:34.69905ms, swizzle: NOOP, TFLOPS: 31.69 (+11.59%)
                out_f32x4(t8x8dbuf): ['118.532104', '44.2729606'], time:36.12399ms, swizzle: NOOP, TFLOPS: 30.44
                    out_f32(cublas): ['118.532104', '44.2729606'], time:28.58903ms, swizzle: NOOP, TFLOPS: 38.46 (+21.37%)
                         out_f32_th: ['118.532104', '44.2729606'], time:28.67548ms, swizzle: NOOP, TFLOPS: 38.34
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['118.526184', '44.2716636'], time:30.18698ms, swizzle: NOOP, TFLOPS: 36.42
    out_tf32(mma2x4+warp2x4+stage2): ['118.526184', '44.2716636'], time:25.13649ms, swizzle: NOOP, TFLOPS: 43.74 (+13.74%)
  out_tf32(mma2x4+...+stage3+dsmem): ['118.526184', '44.2716636'], time:26.80773ms, swizzle: NOOP, TFLOPS: 41.01
  out_tf32(mma2x4+...+stage2+dsmem): ['118.526184', '44.2716636'], time:25.33824ms, swizzle: NOOP, TFLOPS: 43.39
out_tf32(mma2x4+...+stage3+swizzle): ['118.526184', '44.2716636'], time:30.06722ms, swizzle: 512 , TFLOPS: 36.57
out_tf32(mma2x4+...+stage2+swizzle): ['118.526184', '44.2716636'], time:25.04580ms, swizzle: 512 , TFLOPS: 43.90 (+0.36%)
 out_tf32(...+stage3+dsmem+swizzle): ['118.526184', '44.2716636'], time:26.84135ms, swizzle: 512 , TFLOPS: 40.96
 out_tf32(...+stage2+dsmem+swizzle): ['118.526184', '44.2716636'], time:24.95355ms, swizzle: 512 , TFLOPS: 44.06 (+0.37%)
              out_tf32(cublas+tf32): ['118.526184', '44.2716636'], time:19.74155ms, swizzle: NOOP, TFLOPS: 55.70 (+26.40%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=16384, N=8192, K=2048
                  out_f32x4(t8x8sk): ['70.5949554', '26.1727619'], time:18.36364ms, swizzle: NOOP, TFLOPS: 29.94 (+0.00%)
                 out_f32x4(t8x8bcf): ['70.5949554', '26.1727619'], time:16.34912ms, swizzle: NOOP, TFLOPS: 33.63 (+12.32%)
                out_f32x4(t8x8dbuf): ['70.5949554', '26.1727619'], time:14.82284ms, swizzle: NOOP, TFLOPS: 37.09 (+10.30%)
                    out_f32(cublas): ['70.5949554', '26.1727619'], time:14.45541ms, swizzle: NOOP, TFLOPS: 38.03 (+2.54%)
                         out_f32_th: ['70.5949554', '26.1727619'], time:14.56203ms, swizzle: NOOP, TFLOPS: 37.75
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['70.5943985', '26.1725273'], time:14.91312ms, swizzle: NOOP, TFLOPS: 36.86
    out_tf32(mma2x4+warp2x4+stage2): ['70.5943985', '26.1725273'], time:12.08066ms, swizzle: NOOP, TFLOPS: 45.51 (+19.66%)
  out_tf32(mma2x4+...+stage3+dsmem): ['70.5943985', '26.1725273'], time:13.07072ms, swizzle: NOOP, TFLOPS: 42.06
  out_tf32(mma2x4+...+stage2+dsmem): ['70.5943985', '26.1725273'], time:12.01016ms, swizzle: NOOP, TFLOPS: 45.77 (+0.59%)
out_tf32(mma2x4+...+stage3+swizzle): ['70.5943985', '26.1725273'], time:14.58058ms, swizzle: 1024, TFLOPS: 37.70
out_tf32(mma2x4+...+stage2+swizzle): ['70.5943985', '26.1725273'], time:12.01112ms, swizzle: 1024, TFLOPS: 45.77
 out_tf32(...+stage3+dsmem+swizzle): ['70.5943985', '26.1725273'], time:13.10825ms, swizzle: 1024, TFLOPS: 41.94
 out_tf32(...+stage2+dsmem+swizzle): ['70.5943985', '26.1725273'], time:12.03854ms, swizzle: 1024, TFLOPS: 45.67
              out_tf32(cublas+tf32): ['70.5943985', '26.1725273'], time:9.944319ms, swizzle: NOOP, TFLOPS: 55.28 (+20.77%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=16384, N=8192, K=4096
                  out_f32x4(t8x8sk): ['151.796371', '4.59689951'], time:39.44745ms, swizzle: NOOP, TFLOPS: 27.87 (+0.00%)
                 out_f32x4(t8x8bcf): ['151.796371', '4.59689951'], time:35.19003ms, swizzle: NOOP, TFLOPS: 31.24 (+12.10%)
                out_f32x4(t8x8dbuf): ['151.796371', '4.59689951'], time:36.57977ms, swizzle: NOOP, TFLOPS: 30.06
                    out_f32(cublas): ['151.796371', '4.59689951'], time:27.93822ms, swizzle: NOOP, TFLOPS: 39.36 (+25.96%)
                         out_f32_th: ['151.796371', '4.59689951'], time:27.93700ms, swizzle: NOOP, TFLOPS: 39.36 (+0.00%)
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['151.794143', '4.5965395 '], time:29.24573ms, swizzle: NOOP, TFLOPS: 37.60
    out_tf32(mma2x4+warp2x4+stage2): ['151.794143', '4.5965395 '], time:24.57020ms, swizzle: NOOP, TFLOPS: 44.75 (+13.70%)
  out_tf32(mma2x4+...+stage3+dsmem): ['151.794143', '4.5965395 '], time:26.55055ms, swizzle: NOOP, TFLOPS: 41.41
  out_tf32(mma2x4+...+stage2+dsmem): ['151.794143', '4.5965395 '], time:24.88572ms, swizzle: NOOP, TFLOPS: 44.18
out_tf32(mma2x4+...+stage3+swizzle): ['151.794143', '4.5965395 '], time:29.28466ms, swizzle: 1024, TFLOPS: 37.55
out_tf32(mma2x4+...+stage2+swizzle): ['151.794143', '4.5965395 '], time:23.89683ms, swizzle: 1024, TFLOPS: 46.01 (+2.82%)
 out_tf32(...+stage3+dsmem+swizzle): ['151.794143', '4.5965395 '], time:26.11415ms, swizzle: 1024, TFLOPS: 42.10
 out_tf32(...+stage2+dsmem+swizzle): ['151.794143', '4.5965395 '], time:23.87890ms, swizzle: 1024, TFLOPS: 46.05 (+0.08%)
              out_tf32(cublas+tf32): ['151.794143', '4.5965395 '], time:19.27731ms, swizzle: NOOP, TFLOPS: 57.04 (+23.87%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=16384, N=8192, K=8192
                  out_f32x4(t8x8sk): ['118.532104', '44.2729606'], time:79.11319ms, swizzle: NOOP, TFLOPS: 27.80 (+0.00%)
                 out_f32x4(t8x8bcf): ['118.532104', '44.2729606'], time:70.98405ms, swizzle: NOOP, TFLOPS: 30.98 (+11.45%)
                out_f32x4(t8x8dbuf): ['118.532104', '44.2729606'], time:71.76809ms, swizzle: NOOP, TFLOPS: 30.64
                    out_f32(cublas): ['118.532104', '44.2729606'], time:55.91969ms, swizzle: NOOP, TFLOPS: 39.32 (+26.94%)
                         out_f32_th: ['118.532104', '44.2729606'], time:56.78405ms, swizzle: NOOP, TFLOPS: 38.73
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['118.526184', '44.2716636'], time:58.23874ms, swizzle: NOOP, TFLOPS: 37.76
    out_tf32(mma2x4+warp2x4+stage2): ['118.526184', '44.2716636'], time:49.20217ms, swizzle: NOOP, TFLOPS: 44.69 (+13.65%)
  out_tf32(mma2x4+...+stage3+dsmem): ['118.526184', '44.2716636'], time:53.33271ms, swizzle: NOOP, TFLOPS: 41.23
  out_tf32(mma2x4+...+stage2+dsmem): ['118.526184', '44.2716636'], time:49.59840ms, swizzle: NOOP, TFLOPS: 44.34
out_tf32(mma2x4+...+stage3+swizzle): ['118.526184', '44.2716636'], time:58.33761ms, swizzle: 1024, TFLOPS: 37.69
out_tf32(mma2x4+...+stage2+swizzle): ['118.526184', '44.2716636'], time:47.81997ms, swizzle: 1024, TFLOPS: 45.99 (+2.89%)
 out_tf32(...+stage3+dsmem+swizzle): ['118.526184', '44.2716636'], time:51.88267ms, swizzle: 1024, TFLOPS: 42.38
 out_tf32(...+stage2+dsmem+swizzle): ['118.526184', '44.2716636'], time:47.80828ms, swizzle: 1024, TFLOPS: 46.00 (+0.02%)
              out_tf32(cublas+tf32): ['118.526184', '44.2716636'], time:38.11509ms, swizzle: NOOP, TFLOPS: 57.69 (+25.43%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=16384, N=16384, K=2048
                  out_f32x4(t8x8sk): ['70.5949554', '26.1727619'], time:40.08102ms, swizzle: NOOP, TFLOPS: 27.43 (+0.00%)
                 out_f32x4(t8x8bcf): ['70.5949554', '26.1727619'], time:39.66226ms, swizzle: NOOP, TFLOPS: 27.72 (+1.06%)
                out_f32x4(t8x8dbuf): ['70.5949554', '26.1727619'], time:36.46554ms, swizzle: NOOP, TFLOPS: 30.15 (+8.77%)
                    out_f32(cublas): ['70.5949554', '26.1727619'], time:28.34019ms, swizzle: NOOP, TFLOPS: 38.80 (+28.67%)
                         out_f32_th: ['70.5949554', '26.1727619'], time:28.30972ms, swizzle: NOOP, TFLOPS: 38.84 (+0.11%)
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['70.5943985', '26.1725273'], time:28.73399ms, swizzle: NOOP, TFLOPS: 38.27
    out_tf32(mma2x4+warp2x4+stage2): ['70.5943985', '26.1725273'], time:25.33073ms, swizzle: NOOP, TFLOPS: 43.41 (+11.76%)
  out_tf32(mma2x4+...+stage3+dsmem): ['70.5943985', '26.1725273'], time:26.69138ms, swizzle: NOOP, TFLOPS: 41.19
  out_tf32(mma2x4+...+stage2+dsmem): ['70.5943985', '26.1725273'], time:25.41232ms, swizzle: NOOP, TFLOPS: 43.27
out_tf32(mma2x4+...+stage3+swizzle): ['70.5943985', '26.1725273'], time:28.79602ms, swizzle: 2048, TFLOPS: 38.18
out_tf32(mma2x4+...+stage2+swizzle): ['70.5943985', '26.1725273'], time:23.39887ms, swizzle: 2048, TFLOPS: 46.99 (+8.26%)
 out_tf32(...+stage3+dsmem+swizzle): ['70.5943985', '26.1725273'], time:25.56235ms, swizzle: 2048, TFLOPS: 43.01
 out_tf32(...+stage2+dsmem+swizzle): ['70.5943985', '26.1725273'], time:23.46084ms, swizzle: 2048, TFLOPS: 46.87
              out_tf32(cublas+tf32): ['70.5943985', '26.1725273'], time:19.40128ms, swizzle: NOOP, TFLOPS: 56.67 (+20.60%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=16384, N=16384, K=4096
                  out_f32x4(t8x8sk): ['151.796371', '4.59689951'], time:81.40509ms, swizzle: NOOP, TFLOPS: 27.01 (+0.00%)
                 out_f32x4(t8x8bcf): ['151.796371', '4.59689951'], time:75.39424ms, swizzle: NOOP, TFLOPS: 29.17 (+7.97%)
                out_f32x4(t8x8dbuf): ['151.796371', '4.59689951'], time:75.67217ms, swizzle: NOOP, TFLOPS: 29.06
                    out_f32(cublas): ['151.796371', '4.59689951'], time:55.54578ms, swizzle: NOOP, TFLOPS: 39.59 (+35.73%)
                         out_f32_th: ['151.796371', '4.59689951'], time:56.35116ms, swizzle: NOOP, TFLOPS: 39.02
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['151.794143', '4.5965395 '], time:57.64467ms, swizzle: NOOP, TFLOPS: 38.15
    out_tf32(mma2x4+warp2x4+stage2): ['151.794143', '4.5965395 '], time:50.40433ms, swizzle: NOOP, TFLOPS: 43.63 (+10.20%)
  out_tf32(mma2x4+...+stage3+dsmem): ['151.794143', '4.5965395 '], time:53.50663ms, swizzle: NOOP, TFLOPS: 41.10
  out_tf32(mma2x4+...+stage2+dsmem): ['151.794143', '4.5965395 '], time:50.22649ms, swizzle: NOOP, TFLOPS: 43.78 (+0.35%)
out_tf32(mma2x4+...+stage3+swizzle): ['151.794143', '4.5965395 '], time:57.27660ms, swizzle: 2048, TFLOPS: 38.39
out_tf32(mma2x4+...+stage2+swizzle): ['151.794143', '4.5965395 '], time:46.61462ms, swizzle: 2048, TFLOPS: 47.17 (+7.75%)
 out_tf32(...+stage3+dsmem+swizzle): ['151.794143', '4.5965395 '], time:50.91807ms, swizzle: 2048, TFLOPS: 43.19
 out_tf32(...+stage2+dsmem+swizzle): ['151.794143', '4.5965395 '], time:46.73092ms, swizzle: 2048, TFLOPS: 47.06
              out_tf32(cublas+tf32): ['151.794143', '4.5965395 '], time:38.29209ms, swizzle: NOOP, TFLOPS: 57.43 (+21.73%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=16384, N=16384, K=8192
                  out_f32x4(t8x8sk): ['118.532104', '44.2729606'], time:162.8879ms, swizzle: NOOP, TFLOPS: 27.00 (+0.00%)
                 out_f32x4(t8x8bcf): ['118.532104', '44.2729606'], time:151.1848ms, swizzle: NOOP, TFLOPS: 29.09 (+7.74%)
                out_f32x4(t8x8dbuf): ['118.532104', '44.2729606'], time:151.3025ms, swizzle: NOOP, TFLOPS: 29.07
                    out_f32(cublas): ['118.532104', '44.2729606'], time:112.4181ms, swizzle: NOOP, TFLOPS: 39.12 (+34.48%)
                         out_f32_th: ['118.532104', '44.2729606'], time:112.4917ms, swizzle: NOOP, TFLOPS: 39.10
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['118.526184', '44.2716636'], time:115.7331ms, swizzle: NOOP, TFLOPS: 38.00
    out_tf32(mma2x4+warp2x4+stage2): ['118.526184', '44.2716636'], time:100.3637ms, swizzle: NOOP, TFLOPS: 43.82 (+12.01%)
  out_tf32(mma2x4+...+stage3+dsmem): ['118.526184', '44.2716636'], time:106.3712ms, swizzle: NOOP, TFLOPS: 41.35
  out_tf32(mma2x4+...+stage2+dsmem): ['118.526184', '44.2716636'], time:102.4972ms, swizzle: NOOP, TFLOPS: 42.91
out_tf32(mma2x4+...+stage3+swizzle): ['118.526184', '44.2716636'], time:114.2313ms, swizzle: 2048, TFLOPS: 38.50
out_tf32(mma2x4+...+stage2+swizzle): ['118.526184', '44.2716636'], time:93.91186ms, swizzle: 2048, TFLOPS: 46.83 (+6.87%)
 out_tf32(...+stage3+dsmem+swizzle): ['118.526184', '44.2716636'], time:101.5390ms, swizzle: 2048, TFLOPS: 43.31
 out_tf32(...+stage2+dsmem+swizzle): ['118.526184', '44.2716636'], time:93.69635ms, swizzle: 2048, TFLOPS: 46.94 (+0.23%)
              out_tf32(cublas+tf32): ['118.526184', '44.2716636'], time:75.96850ms, swizzle: NOOP, TFLOPS: 57.89 (+23.34%)
----------------------------------------------------------------------------------------------------------------------------------
```

增加了async版本的sgemm_t_8x8_k_sliced_bcf_dbuf_kernel，和async kernels的benchmark。但是async版本在小尺寸的M，N上性能反而下降，在大尺寸的M，N上性能持平。推测是syncthreads和wait带来的额外开销。
对于latency hiding，需要profile。

``` bash
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=4096, K=2048
                  out_f32x4(t8x8sk): ['-47.051422', '-4.6909046'], time:1.653313ms, swizzle: NOOP, TFLOPS: 41.56 (+0.00%)
                 out_f32x4(t8x8bcf): ['-47.051422', '-4.6909046'], time:1.429057ms, swizzle: NOOP, TFLOPS: 48.09 (+15.69%)
                out_f32x4(t8x8dbuf): ['-47.051422', '-4.6909046'], time:1.287746ms, swizzle: NOOP, TFLOPS: 53.36 (+10.97%)
           out_f32x4(t8x8dbufasync): ['-47.051422', '-4.6909046'], time:1.395559ms, swizzle: NOOP, TFLOPS: 49.24 
             out_f32x4(k16t8x4dbuf): ['-47.051422', '-4.6909046'], time:2.099990ms, swizzle: NOOP, TFLOPS: 32.72 
        out_f32x4(k16t8x4dbufasync): ['-47.051422', '-4.6909046'], time:2.024650ms, swizzle: NOOP, TFLOPS: 33.94 
              out_f32x4(k168x8dbuf): ['-47.051422', '-4.6909046'], time:1.614928ms, swizzle: NOOP, TFLOPS: 42.55 
         out_f32x4(k168x8dbufasync): ['-47.051422', '-4.6909046'], time:1.569557ms, swizzle: NOOP, TFLOPS: 43.78 
             out_f32x4(k168x16dbuf): ['-47.051422', '-4.6909046'], time:2.767300ms, swizzle: NOOP, TFLOPS: 24.83 
        out_f32x4(k168x16dbufasync): ['-47.051422', '-4.6909046'], time:2.170276ms, swizzle: NOOP, TFLOPS: 31.66 
                    out_f32(cublas): ['-47.051399', '-4.6908912'], time:1.352572ms, swizzle: NOOP, TFLOPS: 50.81 
                         out_f32_th: ['-47.051399', '-4.6908912'], time:1.332497ms, swizzle: NOOP, TFLOPS: 51.57 
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['-47.060657', '-4.6654434'], time:1.203894ms, swizzle: NOOP, TFLOPS: 57.08 (+6.97%)
    out_tf32(mma2x4+warp2x4+stage2): ['-47.060657', '-4.6654434'], time:1.086020ms, swizzle: NOOP, TFLOPS: 63.28 (+10.85%)
  out_tf32(mma2x4+...+stage3+dsmem): ['-47.060657', '-4.6654434'], time:1.161718ms, swizzle: NOOP, TFLOPS: 59.15 
  out_tf32(mma2x4+...+stage2+dsmem): ['-47.060657', '-4.6654434'], time:1.055979ms, swizzle: NOOP, TFLOPS: 65.08 (+2.84%)
out_tf32(mma2x4+...+stage3+swizzle): ['-47.060657', '-4.6654434'], time:1.194834ms, swizzle: 512 , TFLOPS: 57.51 
out_tf32(mma2x4+...+stage2+swizzle): ['-47.060657', '-4.6654434'], time:1.054072ms, swizzle: 512 , TFLOPS: 65.19 (+0.18%)
 out_tf32(...+stage3+dsmem+swizzle): ['-47.060657', '-4.6654434'], time:1.120901ms, swizzle: 512 , TFLOPS: 61.31 
 out_tf32(...+stage2+dsmem+swizzle): ['-47.060657', '-4.6654434'], time:1.042556ms, swizzle: 512 , TFLOPS: 65.91 (+1.10%)
              out_tf32(cublas+tf32): ['-47.060657', '-4.6654434'], time:0.820159ms, swizzle: NOOP, TFLOPS: 83.79 (+27.12%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=4096, K=4096
                  out_f32x4(t8x8sk): ['-75.344055', '-55.511753'], time:3.243613ms, swizzle: NOOP, TFLOPS: 42.37 (+0.00%)
                 out_f32x4(t8x8bcf): ['-75.344055', '-55.511753'], time:3.052854ms, swizzle: NOOP, TFLOPS: 45.02 (+6.25%)
                out_f32x4(t8x8dbuf): ['-75.344055', '-55.511753'], time:2.610349ms, swizzle: NOOP, TFLOPS: 52.65 (+16.95%)
           out_f32x4(t8x8dbufasync): ['-75.344055', '-55.511753'], time:3.417229ms, swizzle: NOOP, TFLOPS: 40.22 
             out_f32x4(k16t8x4dbuf): ['-75.344055', '-55.511753'], time:4.194879ms, swizzle: NOOP, TFLOPS: 32.76 
        out_f32x4(k16t8x4dbufasync): ['-75.344055', '-55.511753'], time:3.727173ms, swizzle: NOOP, TFLOPS: 36.87 
              out_f32x4(k168x8dbuf): ['-75.344055', '-55.511753'], time:3.168344ms, swizzle: NOOP, TFLOPS: 43.38 
         out_f32x4(k168x8dbufasync): ['-75.344055', '-55.511753'], time:3.142857ms, swizzle: NOOP, TFLOPS: 43.73 
             out_f32x4(k168x16dbuf): ['-75.344055', '-55.511753'], time:5.627894ms, swizzle: NOOP, TFLOPS: 24.42 
        out_f32x4(k168x16dbufasync): ['-75.344055', '-55.511753'], time:3.943181ms, swizzle: NOOP, TFLOPS: 34.85 
                    out_f32(cublas): ['-75.344055', '-55.511753'], time:2.519941ms, swizzle: NOOP, TFLOPS: 54.54 (+3.59%)
                         out_f32_th: ['-75.344055', '-55.511753'], time:2.490329ms, swizzle: NOOP, TFLOPS: 55.19 (+1.19%)
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['-75.353981', '-55.498649'], time:2.808666ms, swizzle: NOOP, TFLOPS: 48.93 
    out_tf32(mma2x4+warp2x4+stage2): ['-75.353981', '-55.498649'], time:2.244997ms, swizzle: NOOP, TFLOPS: 61.22 (+10.93%)
  out_tf32(mma2x4+...+stage3+dsmem): ['-75.353981', '-55.498649'], time:2.446985ms, swizzle: NOOP, TFLOPS: 56.17 
  out_tf32(mma2x4+...+stage2+dsmem): ['-75.353981', '-55.498649'], time:2.155327ms, swizzle: NOOP, TFLOPS: 63.77 (+4.16%)
out_tf32(mma2x4+...+stage3+swizzle): ['-75.353981', '-55.498649'], time:2.397131ms, swizzle: 512 , TFLOPS: 57.33 
out_tf32(mma2x4+...+stage2+swizzle): ['-75.353981', '-55.498649'], time:2.129530ms, swizzle: 512 , TFLOPS: 64.54 (+1.21%)
 out_tf32(...+stage3+dsmem+swizzle): ['-75.353981', '-55.498649'], time:2.275061ms, swizzle: 512 , TFLOPS: 60.41 
 out_tf32(...+stage2+dsmem+swizzle): ['-75.353981', '-55.498649'], time:2.107858ms, swizzle: 512 , TFLOPS: 65.20 (+1.03%)
              out_tf32(cublas+tf32): ['-75.353981', '-55.498649'], time:1.578330ms, swizzle: NOOP, TFLOPS: 87.08 (+33.55%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=4096, K=8192
                  out_f32x4(t8x8sk): ['-16.721485', '-50.357620'], time:6.698060ms, swizzle: NOOP, TFLOPS: 41.04 (+0.00%)
                 out_f32x4(t8x8bcf): ['-16.721485', '-50.357620'], time:6.412386ms, swizzle: NOOP, TFLOPS: 42.87 (+4.46%)
                out_f32x4(t8x8dbuf): ['-16.721485', '-50.357620'], time:6.217098ms, swizzle: NOOP, TFLOPS: 44.21 (+3.14%)
           out_f32x4(t8x8dbufasync): ['-16.721485', '-50.357620'], time:6.670618ms, swizzle: NOOP, TFLOPS: 41.21 
             out_f32x4(k16t8x4dbuf): ['-16.721485', '-50.357620'], time:8.158040ms, swizzle: NOOP, TFLOPS: 33.69 
        out_f32x4(k16t8x4dbufasync): ['-16.721485', '-50.357620'], time:7.761693ms, swizzle: NOOP, TFLOPS: 35.41 
              out_f32x4(k168x8dbuf): ['-16.721485', '-50.357620'], time:6.530690ms, swizzle: NOOP, TFLOPS: 42.09 
         out_f32x4(k168x8dbufasync): ['-16.721485', '-50.357620'], time:6.360816ms, swizzle: NOOP, TFLOPS: 43.21 
             out_f32x4(k168x16dbuf): ['-16.721485', '-50.357620'], time:9.227609ms, swizzle: NOOP, TFLOPS: 29.79 
        out_f32x4(k168x16dbufasync): ['-16.721485', '-50.357620'], time:7.617187ms, swizzle: NOOP, TFLOPS: 36.09 
                    out_f32(cublas): ['-16.721569', '-50.357555'], time:4.938554ms, swizzle: NOOP, TFLOPS: 55.66 (+25.89%)
                         out_f32_th: ['-16.721569', '-50.357555'], time:5.078554ms, swizzle: NOOP, TFLOPS: 54.13 
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['-16.728071', '-50.355793'], time:5.454444ms, swizzle: NOOP, TFLOPS: 50.40 
    out_tf32(mma2x4+warp2x4+stage2): ['-16.728071', '-50.355793'], time:4.362177ms, swizzle: NOOP, TFLOPS: 63.01 (+13.21%)
  out_tf32(mma2x4+...+stage3+dsmem): ['-16.728071', '-50.355793'], time:4.665756ms, swizzle: NOOP, TFLOPS: 58.91 
  out_tf32(mma2x4+...+stage2+dsmem): ['-16.728071', '-50.355793'], time:4.371380ms, swizzle: NOOP, TFLOPS: 62.88 
out_tf32(mma2x4+...+stage3+swizzle): ['-16.728071', '-50.355793'], time:4.898905ms, swizzle: 512 , TFLOPS: 56.11 
out_tf32(mma2x4+...+stage2+swizzle): ['-16.728071', '-50.355793'], time:4.408669ms, swizzle: 512 , TFLOPS: 62.35 
 out_tf32(...+stage3+dsmem+swizzle): ['-16.728071', '-50.355793'], time:4.650902ms, swizzle: 512 , TFLOPS: 59.10 
 out_tf32(...+stage2+dsmem+swizzle): ['-16.728071', '-50.355793'], time:4.371666ms, swizzle: 512 , TFLOPS: 62.88 
              out_tf32(cublas+tf32): ['-16.728071', '-50.355793'], time:3.123378ms, swizzle: NOOP, TFLOPS: 88.01 (+39.66%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=8192, K=2048
                  out_f32x4(t8x8sk): ['-47.065979', '-4.6792573'], time:3.200268ms, swizzle: NOOP, TFLOPS: 42.95 (+0.00%)
                 out_f32x4(t8x8bcf): ['-47.065979', '-4.6792573'], time:3.118252ms, swizzle: NOOP, TFLOPS: 44.08 (+2.63%)
                out_f32x4(t8x8dbuf): ['-47.065979', '-4.6792573'], time:2.853918ms, swizzle: NOOP, TFLOPS: 48.16 (+9.26%)
           out_f32x4(t8x8dbufasync): ['-47.065979', '-4.6792573'], time:3.396224ms, swizzle: NOOP, TFLOPS: 40.47 
             out_f32x4(k16t8x4dbuf): ['-47.065979', '-4.6792573'], time:4.110789ms, swizzle: NOOP, TFLOPS: 33.43 
        out_f32x4(k16t8x4dbufasync): ['-47.065979', '-4.6792573'], time:3.669071ms, swizzle: NOOP, TFLOPS: 37.46 
              out_f32x4(k168x8dbuf): ['-47.065979', '-4.6792573'], time:3.091526ms, swizzle: NOOP, TFLOPS: 44.46 
         out_f32x4(k168x8dbufasync): ['-47.065979', '-4.6792573'], time:3.105211ms, swizzle: NOOP, TFLOPS: 44.26 
             out_f32x4(k168x16dbuf): ['-47.065979', '-4.6792573'], time:5.140590ms, swizzle: NOOP, TFLOPS: 26.74 
        out_f32x4(k168x16dbufasync): ['-47.065979', '-4.6792573'], time:4.109025ms, swizzle: NOOP, TFLOPS: 33.45 
                    out_f32(cublas): ['-47.065979', '-4.6792573'], time:2.359175ms, swizzle: NOOP, TFLOPS: 58.26 (+20.97%)
                         out_f32_th: ['-47.065979', '-4.6792573'], time:2.490186ms, swizzle: NOOP, TFLOPS: 55.19 
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['-47.060657', '-4.6654434'], time:2.863430ms, swizzle: NOOP, TFLOPS: 48.00 
    out_tf32(mma2x4+warp2x4+stage2): ['-47.060657', '-4.6654434'], time:2.298736ms, swizzle: NOOP, TFLOPS: 59.79 (+2.63%)
  out_tf32(mma2x4+...+stage3+dsmem): ['-47.060657', '-4.6654434'], time:2.299714ms, swizzle: NOOP, TFLOPS: 59.76 
  out_tf32(mma2x4+...+stage2+dsmem): ['-47.060657', '-4.6654434'], time:2.129697ms, swizzle: NOOP, TFLOPS: 64.53 (+7.94%)
out_tf32(mma2x4+...+stage3+swizzle): ['-47.060657', '-4.6654434'], time:2.347970ms, swizzle: 1024, TFLOPS: 58.54 
out_tf32(mma2x4+...+stage2+swizzle): ['-47.060657', '-4.6654434'], time:2.062988ms, swizzle: 1024, TFLOPS: 66.62 (+3.23%)
 out_tf32(...+stage3+dsmem+swizzle): ['-47.060657', '-4.6654434'], time:2.216172ms, swizzle: 1024, TFLOPS: 62.02 
 out_tf32(...+stage2+dsmem+swizzle): ['-47.060657', '-4.6654434'], time:2.051973ms, swizzle: 1024, TFLOPS: 66.98 (+0.54%)
              out_tf32(cublas+tf32): ['-47.060657', '-4.6654434'], time:1.613497ms, swizzle: NOOP, TFLOPS: 85.18 (+27.18%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=8192, K=4096
                  out_f32x4(t8x8sk): ['-75.357116', '-55.511447'], time:6.868934ms, swizzle: NOOP, TFLOPS: 40.02 (+0.00%)
                 out_f32x4(t8x8bcf): ['-75.357116', '-55.511447'], time:6.521439ms, swizzle: NOOP, TFLOPS: 42.15 (+5.33%)
                out_f32x4(t8x8dbuf): ['-75.357116', '-55.511447'], time:6.251287ms, swizzle: NOOP, TFLOPS: 43.97 (+4.32%)
           out_f32x4(t8x8dbufasync): ['-75.357116', '-55.511447'], time:6.762623ms, swizzle: NOOP, TFLOPS: 40.65 
             out_f32x4(k16t8x4dbuf): ['-75.357116', '-55.511447'], time:8.405470ms, swizzle: NOOP, TFLOPS: 32.70 
        out_f32x4(k16t8x4dbufasync): ['-75.357116', '-55.511447'], time:9.169816ms, swizzle: NOOP, TFLOPS: 29.98 
              out_f32x4(k168x8dbuf): ['-75.357116', '-55.511447'], time:5.990910ms, swizzle: NOOP, TFLOPS: 45.88 (+4.35%)
         out_f32x4(k168x8dbufasync): ['-75.357116', '-55.511447'], time:6.424546ms, swizzle: NOOP, TFLOPS: 42.79 
             out_f32x4(k168x16dbuf): ['-75.357116', '-55.511447'], time:9.044480ms, swizzle: NOOP, TFLOPS: 30.39 
        out_f32x4(k168x16dbufasync): ['-75.357116', '-55.511447'], time:7.610058ms, swizzle: NOOP, TFLOPS: 36.12 
                    out_f32(cublas): ['-75.357116', '-55.511447'], time:4.855775ms, swizzle: NOOP, TFLOPS: 56.61 (+23.38%)
                         out_f32_th: ['-75.357116', '-55.511447'], time:5.052828ms, swizzle: NOOP, TFLOPS: 54.40 
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['-75.353981', '-55.498649'], time:5.542612ms, swizzle: NOOP, TFLOPS: 49.59 
    out_tf32(mma2x4+warp2x4+stage2): ['-75.353981', '-55.498649'], time:4.239749ms, swizzle: NOOP, TFLOPS: 64.83 (+14.53%)
  out_tf32(mma2x4+...+stage3+dsmem): ['-75.353981', '-55.498649'], time:4.467272ms, swizzle: NOOP, TFLOPS: 61.53 
  out_tf32(mma2x4+...+stage2+dsmem): ['-75.353981', '-55.498649'], time:4.190254ms, swizzle: NOOP, TFLOPS: 65.60 (+1.18%)
out_tf32(mma2x4+...+stage3+swizzle): ['-75.353981', '-55.498649'], time:4.713654ms, swizzle: 1024, TFLOPS: 58.32 
out_tf32(mma2x4+...+stage2+swizzle): ['-75.353981', '-55.498649'], time:4.154992ms, swizzle: 1024, TFLOPS: 66.16 (+0.85%)
 out_tf32(...+stage3+dsmem+swizzle): ['-75.353981', '-55.498649'], time:4.455137ms, swizzle: 1024, TFLOPS: 61.70 
 out_tf32(...+stage2+dsmem+swizzle): ['-75.353981', '-55.498649'], time:4.135012ms, swizzle: 1024, TFLOPS: 66.48 (+0.48%)
              out_tf32(cublas+tf32): ['-75.353981', '-55.498649'], time:3.150415ms, swizzle: NOOP, TFLOPS: 87.25 (+31.25%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=8192, K=8192
                  out_f32x4(t8x8sk): ['-16.720695', '-50.355293'], time:14.29526ms, swizzle: NOOP, TFLOPS: 38.46 (+0.00%)
                 out_f32x4(t8x8bcf): ['-16.720695', '-50.355293'], time:13.28203ms, swizzle: NOOP, TFLOPS: 41.39 (+7.63%)
                out_f32x4(t8x8dbuf): ['-16.720695', '-50.355293'], time:13.20524ms, swizzle: NOOP, TFLOPS: 41.63 (+0.58%)
           out_f32x4(t8x8dbufasync): ['-16.720695', '-50.355293'], time:13.30392ms, swizzle: NOOP, TFLOPS: 41.32 
             out_f32x4(k16t8x4dbuf): ['-16.720695', '-50.355293'], time:16.79301ms, swizzle: NOOP, TFLOPS: 32.74 
        out_f32x4(k16t8x4dbufasync): ['-16.720695', '-50.355293'], time:18.09563ms, swizzle: NOOP, TFLOPS: 30.38 
              out_f32x4(k168x8dbuf): ['-16.720695', '-50.355293'], time:12.59877ms, swizzle: NOOP, TFLOPS: 43.64 (+4.81%)
         out_f32x4(k168x8dbufasync): ['-16.720695', '-50.355293'], time:13.02719ms, swizzle: NOOP, TFLOPS: 42.20 
             out_f32x4(k168x16dbuf): ['-16.720695', '-50.355293'], time:16.90580ms, swizzle: NOOP, TFLOPS: 32.52 
        out_f32x4(k168x16dbufasync): ['-16.720695', '-50.355293'], time:15.43076ms, swizzle: NOOP, TFLOPS: 35.63 
                    out_f32(cublas): ['-16.720695', '-50.355293'], time:9.804368ms, swizzle: NOOP, TFLOPS: 56.07 (+28.50%)
                         out_f32_th: ['-16.720695', '-50.355293'], time:10.04364ms, swizzle: NOOP, TFLOPS: 54.74 
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['-16.728071', '-50.355793'], time:9.903883ms, swizzle: NOOP, TFLOPS: 55.51 
    out_tf32(mma2x4+warp2x4+stage2): ['-16.728071', '-50.355793'], time:8.380341ms, swizzle: NOOP, TFLOPS: 65.60 (+16.99%)
  out_tf32(mma2x4+...+stage3+dsmem): ['-16.728071', '-50.355793'], time:8.950352ms, swizzle: NOOP, TFLOPS: 61.42 
  out_tf32(mma2x4+...+stage2+dsmem): ['-16.728071', '-50.355793'], time:8.470273ms, swizzle: NOOP, TFLOPS: 64.90 
out_tf32(mma2x4+...+stage3+swizzle): ['-16.728071', '-50.355793'], time:9.469151ms, swizzle: 1024, TFLOPS: 58.06 
out_tf32(mma2x4+...+stage2+swizzle): ['-16.728071', '-50.355793'], time:8.425903ms, swizzle: 1024, TFLOPS: 65.25 
 out_tf32(...+stage3+dsmem+swizzle): ['-16.728071', '-50.355793'], time:8.962416ms, swizzle: 1024, TFLOPS: 61.34 
 out_tf32(...+stage2+dsmem+swizzle): ['-16.728071', '-50.355793'], time:8.350539ms, swizzle: 1024, TFLOPS: 65.83 (+0.36%)
              out_tf32(cublas+tf32): ['-16.728071', '-50.355793'], time:6.200599ms, swizzle: NOOP, TFLOPS: 88.66 (+34.67%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=16384, K=2048
                  out_f32x4(t8x8sk): ['-47.065979', '-4.6792573'], time:6.394529ms, swizzle: NOOP, TFLOPS: 42.99 (+0.00%)
                 out_f32x4(t8x8bcf): ['-47.065979', '-4.6792573'], time:6.849741ms, swizzle: NOOP, TFLOPS: 40.13 
                out_f32x4(t8x8dbuf): ['-47.065979', '-4.6792573'], time:6.262230ms, swizzle: NOOP, TFLOPS: 43.89 (+2.11%)
           out_f32x4(t8x8dbufasync): ['-47.065979', '-4.6792573'], time:7.025456ms, swizzle: NOOP, TFLOPS: 39.13 
             out_f32x4(k16t8x4dbuf): ['-47.065979', '-4.6792573'], time:8.348727ms, swizzle: NOOP, TFLOPS: 32.92 
        out_f32x4(k16t8x4dbufasync): ['-47.065979', '-4.6792573'], time:9.451246ms, swizzle: NOOP, TFLOPS: 29.08 
              out_f32x4(k168x8dbuf): ['-47.065979', '-4.6792573'], time:6.234622ms, swizzle: NOOP, TFLOPS: 44.09 (+0.44%)
         out_f32x4(k168x8dbufasync): ['-47.065979', '-4.6792573'], time:6.695890ms, swizzle: NOOP, TFLOPS: 41.05 
             out_f32x4(k168x16dbuf): ['-47.065979', '-4.6792573'], time:9.861803ms, swizzle: NOOP, TFLOPS: 27.87 
        out_f32x4(k168x16dbufasync): ['-47.065979', '-4.6792573'], time:7.761311ms, swizzle: NOOP, TFLOPS: 35.42 
                    out_f32(cublas): ['-47.065979', '-4.6792573'], time:4.867625ms, swizzle: NOOP, TFLOPS: 56.47 (+28.08%)
                         out_f32_th: ['-47.065979', '-4.6792573'], time:5.006027ms, swizzle: NOOP, TFLOPS: 54.91 
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['-47.060657', '-4.6654434'], time:5.473732ms, swizzle: NOOP, TFLOPS: 50.22 
    out_tf32(mma2x4+warp2x4+stage2): ['-47.060657', '-4.6654434'], time:4.336214ms, swizzle: NOOP, TFLOPS: 63.39 (+12.26%)
  out_tf32(mma2x4+...+stage3+dsmem): ['-47.060657', '-4.6654434'], time:5.289101ms, swizzle: NOOP, TFLOPS: 51.97 
  out_tf32(mma2x4+...+stage2+dsmem): ['-47.060657', '-4.6654434'], time:4.218578ms, swizzle: NOOP, TFLOPS: 65.16 (+2.79%)
out_tf32(mma2x4+...+stage3+swizzle): ['-47.060657', '-4.6654434'], time:4.637432ms, swizzle: 2048, TFLOPS: 59.27 
out_tf32(mma2x4+...+stage2+swizzle): ['-47.060657', '-4.6654434'], time:4.089260ms, swizzle: 2048, TFLOPS: 67.22 (+3.16%)
 out_tf32(...+stage3+dsmem+swizzle): ['-47.060657', '-4.6654434'], time:4.384636ms, swizzle: 2048, TFLOPS: 62.69 
 out_tf32(...+stage2+dsmem+swizzle): ['-47.060657', '-4.6654434'], time:4.077291ms, swizzle: 2048, TFLOPS: 67.42 (+0.29%)
              out_tf32(cublas+tf32): ['-47.060657', '-4.6654434'], time:3.176498ms, swizzle: NOOP, TFLOPS: 86.53 (+28.36%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=16384, K=4096
                  out_f32x4(t8x8sk): ['-75.351844', '-55.497821'], time:14.36460ms, swizzle: NOOP, TFLOPS: 38.27 (+0.00%)
                 out_f32x4(t8x8bcf): ['-75.351844', '-55.497821'], time:13.49749ms, swizzle: NOOP, TFLOPS: 40.73 (+6.42%)
                out_f32x4(t8x8dbuf): ['-75.351844', '-55.497821'], time:13.25161ms, swizzle: NOOP, TFLOPS: 41.49 (+1.86%)
           out_f32x4(t8x8dbufasync): ['-75.351844', '-55.497821'], time:13.69035ms, swizzle: NOOP, TFLOPS: 40.16 
             out_f32x4(k16t8x4dbuf): ['-75.351844', '-55.497821'], time:17.26102ms, swizzle: NOOP, TFLOPS: 31.85 
        out_f32x4(k16t8x4dbufasync): ['-75.351844', '-55.497821'], time:18.61145ms, swizzle: NOOP, TFLOPS: 29.54 
              out_f32x4(k168x8dbuf): ['-75.351844', '-55.497821'], time:13.22119ms, swizzle: NOOP, TFLOPS: 41.58 (+0.23%)
         out_f32x4(k168x8dbufasync): ['-75.351844', '-55.497821'], time:13.41955ms, swizzle: NOOP, TFLOPS: 40.97 
             out_f32x4(k168x16dbuf): ['-75.351844', '-55.497821'], time:17.28107ms, swizzle: NOOP, TFLOPS: 31.81 
        out_f32x4(k168x16dbufasync): ['-75.351844', '-55.497821'], time:15.59402ms, swizzle: NOOP, TFLOPS: 35.25 
                    out_f32(cublas): ['-75.351844', '-55.497821'], time:9.901237ms, swizzle: NOOP, TFLOPS: 55.52 (+33.53%)
                         out_f32_th: ['-75.351844', '-55.497821'], time:10.19232ms, swizzle: NOOP, TFLOPS: 53.94 
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['-75.353981', '-55.498649'], time:10.35170ms, swizzle: NOOP, TFLOPS: 53.11 
    out_tf32(mma2x4+warp2x4+stage2): ['-75.353981', '-55.498649'], time:8.374333ms, swizzle: NOOP, TFLOPS: 65.65 (+18.23%)
  out_tf32(mma2x4+...+stage3+dsmem): ['-75.353981', '-55.498649'], time:10.28804ms, swizzle: NOOP, TFLOPS: 53.44 
  out_tf32(mma2x4+...+stage2+dsmem): ['-75.353981', '-55.498649'], time:8.439445ms, swizzle: NOOP, TFLOPS: 65.14 
out_tf32(mma2x4+...+stage3+swizzle): ['-75.353981', '-55.498649'], time:9.309530ms, swizzle: 2048, TFLOPS: 59.05 
out_tf32(mma2x4+...+stage2+swizzle): ['-75.353981', '-55.498649'], time:8.211350ms, swizzle: 2048, TFLOPS: 66.95 (+1.98%)
 out_tf32(...+stage3+dsmem+swizzle): ['-75.353981', '-55.498649'], time:8.787727ms, swizzle: 2048, TFLOPS: 62.56 
 out_tf32(...+stage2+dsmem+swizzle): ['-75.353981', '-55.498649'], time:8.178305ms, swizzle: 2048, TFLOPS: 67.22 (+0.40%)
              out_tf32(cublas+tf32): ['-75.353981', '-55.498649'], time:6.253337ms, swizzle: NOOP, TFLOPS: 87.91 (+30.78%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=4096, N=16384, K=8192
                  out_f32x4(t8x8sk): ['-16.718025', '-50.343330'], time:28.96327ms, swizzle: NOOP, TFLOPS: 37.96 (+0.00%)
                 out_f32x4(t8x8bcf): ['-16.718025', '-50.343330'], time:27.37829ms, swizzle: NOOP, TFLOPS: 40.16 (+5.79%)
                out_f32x4(t8x8dbuf): ['-16.718025', '-50.343330'], time:27.48036ms, swizzle: NOOP, TFLOPS: 40.01 
           out_f32x4(t8x8dbufasync): ['-16.718025', '-50.343330'], time:27.06847ms, swizzle: NOOP, TFLOPS: 40.62 (+1.14%)
             out_f32x4(k16t8x4dbuf): ['-16.718025', '-50.343330'], time:34.03635ms, swizzle: NOOP, TFLOPS: 32.30 
        out_f32x4(k16t8x4dbufasync): ['-16.718025', '-50.343330'], time:36.87658ms, swizzle: NOOP, TFLOPS: 29.82 
              out_f32x4(k168x8dbuf): ['-16.718025', '-50.343330'], time:26.87468ms, swizzle: NOOP, TFLOPS: 40.91 (+0.72%)
         out_f32x4(k168x8dbufasync): ['-16.718025', '-50.343330'], time:26.75187ms, swizzle: NOOP, TFLOPS: 41.10 (+0.46%)
             out_f32x4(k168x16dbuf): ['-16.718025', '-50.343330'], time:33.39214ms, swizzle: NOOP, TFLOPS: 32.93 
        out_f32x4(k168x16dbufasync): ['-16.718025', '-50.343330'], time:31.22949ms, swizzle: NOOP, TFLOPS: 35.21 
                    out_f32(cublas): ['-16.718025', '-50.343330'], time:20.30217ms, swizzle: NOOP, TFLOPS: 54.16 (+31.77%)
                         out_f32_th: ['-16.718025', '-50.343330'], time:20.12820ms, swizzle: NOOP, TFLOPS: 54.63 (+0.86%)
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['-16.728071', '-50.355793'], time:20.35386ms, swizzle: NOOP, TFLOPS: 54.02 
    out_tf32(mma2x4+warp2x4+stage2): ['-16.728071', '-50.355793'], time:18.72282ms, swizzle: NOOP, TFLOPS: 58.73 (+7.51%)
  out_tf32(mma2x4+...+stage3+dsmem): ['-16.728071', '-50.355793'], time:20.34897ms, swizzle: NOOP, TFLOPS: 54.03 
  out_tf32(mma2x4+...+stage2+dsmem): ['-16.728071', '-50.355793'], time:19.02282ms, swizzle: NOOP, TFLOPS: 57.80 
out_tf32(mma2x4+...+stage3+swizzle): ['-16.728071', '-50.355793'], time:18.62988ms, swizzle: 2048, TFLOPS: 59.02 (+0.50%)
out_tf32(mma2x4+...+stage2+swizzle): ['-16.728071', '-50.355793'], time:16.49153ms, swizzle: 2048, TFLOPS: 66.67 (+12.97%)
 out_tf32(...+stage3+dsmem+swizzle): ['-16.728071', '-50.355793'], time:17.55268ms, swizzle: 2048, TFLOPS: 62.64 
 out_tf32(...+stage2+dsmem+swizzle): ['-16.728071', '-50.355793'], time:16.38042ms, swizzle: 2048, TFLOPS: 67.12 (+0.68%)
              out_tf32(cublas+tf32): ['-16.728071', '-50.355793'], time:12.42010ms, swizzle: NOOP, TFLOPS: 88.53 (+31.89%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=4096, K=2048
                  out_f32x4(t8x8sk): ['-47.060710', '-4.6656251'], time:3.080534ms, swizzle: NOOP, TFLOPS: 44.62 (+0.00%)
                 out_f32x4(t8x8bcf): ['-47.060710', '-4.6656251'], time:3.528785ms, swizzle: NOOP, TFLOPS: 38.95 
                out_f32x4(t8x8dbuf): ['-47.060710', '-4.6656251'], time:2.910304ms, swizzle: NOOP, TFLOPS: 47.22 (+5.85%)
           out_f32x4(t8x8dbufasync): ['-47.060710', '-4.6656251'], time:3.334546ms, swizzle: NOOP, TFLOPS: 41.22 
             out_f32x4(k16t8x4dbuf): ['-47.060710', '-4.6656251'], time:4.221796ms, swizzle: NOOP, TFLOPS: 32.55 
        out_f32x4(k16t8x4dbufasync): ['-47.060710', '-4.6656251'], time:3.730630ms, swizzle: NOOP, TFLOPS: 36.84 
              out_f32x4(k168x8dbuf): ['-47.060710', '-4.6656251'], time:3.112673ms, swizzle: NOOP, TFLOPS: 44.15 
         out_f32x4(k168x8dbufasync): ['-47.060710', '-4.6656251'], time:3.068518ms, swizzle: NOOP, TFLOPS: 44.79 
             out_f32x4(k168x16dbuf): ['-47.060710', '-4.6656251'], time:5.026984ms, swizzle: NOOP, TFLOPS: 27.34 
        out_f32x4(k168x16dbufasync): ['-47.060710', '-4.6656251'], time:4.177665ms, swizzle: NOOP, TFLOPS: 32.90 
                    out_f32(cublas): ['-47.060710', '-4.6656251'], time:2.326631ms, swizzle: NOOP, TFLOPS: 59.07 (+25.09%)
                         out_f32_th: ['-47.060710', '-4.6656251'], time:2.408289ms, swizzle: NOOP, TFLOPS: 57.07 
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['-47.060657', '-4.6654434'], time:2.780580ms, swizzle: NOOP, TFLOPS: 49.43 
    out_tf32(mma2x4+warp2x4+stage2): ['-47.060657', '-4.6654434'], time:2.171826ms, swizzle: NOOP, TFLOPS: 63.28 (+7.13%)
  out_tf32(mma2x4+...+stage3+dsmem): ['-47.060657', '-4.6654434'], time:2.275156ms, swizzle: NOOP, TFLOPS: 60.41 
  out_tf32(mma2x4+...+stage2+dsmem): ['-47.060657', '-4.6654434'], time:2.080130ms, swizzle: NOOP, TFLOPS: 66.07 (+4.41%)
out_tf32(mma2x4+...+stage3+swizzle): ['-47.060657', '-4.6654434'], time:2.327227ms, swizzle: 512 , TFLOPS: 59.06 
out_tf32(mma2x4+...+stage2+swizzle): ['-47.060657', '-4.6654434'], time:2.056837ms, swizzle: 512 , TFLOPS: 66.82 (+1.13%)
 out_tf32(...+stage3+dsmem+swizzle): ['-47.060657', '-4.6654434'], time:2.202868ms, swizzle: 512 , TFLOPS: 62.39 
 out_tf32(...+stage2+dsmem+swizzle): ['-47.060657', '-4.6654434'], time:2.043676ms, swizzle: 512 , TFLOPS: 67.25 (+0.64%)
              out_tf32(cublas+tf32): ['-47.060657', '-4.6654434'], time:1.612544ms, swizzle: NOOP, TFLOPS: 85.23 (+26.74%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=4096, K=4096
                  out_f32x4(t8x8sk): ['-75.354446', '-55.499485'], time:6.656289ms, swizzle: NOOP, TFLOPS: 41.30 (+0.00%)
                 out_f32x4(t8x8bcf): ['-75.354446', '-55.499485'], time:6.301641ms, swizzle: NOOP, TFLOPS: 43.62 (+5.63%)
                out_f32x4(t8x8dbuf): ['-75.354446', '-55.499485'], time:6.097555ms, swizzle: NOOP, TFLOPS: 45.08 (+3.35%)
           out_f32x4(t8x8dbufasync): ['-75.354446', '-55.499485'], time:6.662106ms, swizzle: NOOP, TFLOPS: 41.26 
             out_f32x4(k16t8x4dbuf): ['-75.354446', '-55.499485'], time:7.767581ms, swizzle: NOOP, TFLOPS: 35.39 
        out_f32x4(k16t8x4dbufasync): ['-75.354446', '-55.499485'], time:7.323813ms, swizzle: NOOP, TFLOPS: 37.53 
              out_f32x4(k168x8dbuf): ['-75.354446', '-55.499485'], time:6.245303ms, swizzle: NOOP, TFLOPS: 44.01 
         out_f32x4(k168x8dbufasync): ['-75.354446', '-55.499485'], time:6.255793ms, swizzle: NOOP, TFLOPS: 43.94 
             out_f32x4(k168x16dbuf): ['-75.354446', '-55.499485'], time:9.866786ms, swizzle: NOOP, TFLOPS: 27.86 
        out_f32x4(k168x16dbufasync): ['-75.354446', '-55.499485'], time:7.652211ms, swizzle: NOOP, TFLOPS: 35.92 
                    out_f32(cublas): ['-75.354446', '-55.499485'], time:4.790568ms, swizzle: NOOP, TFLOPS: 57.38 (+27.28%)
                         out_f32_th: ['-75.354446', '-55.499485'], time:4.997372ms, swizzle: NOOP, TFLOPS: 55.00 
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['-75.353981', '-55.498649'], time:5.286908ms, swizzle: NOOP, TFLOPS: 51.99 
    out_tf32(mma2x4+warp2x4+stage2): ['-75.353981', '-55.498649'], time:4.236364ms, swizzle: NOOP, TFLOPS: 64.89 (+13.08%)
  out_tf32(mma2x4+...+stage3+dsmem): ['-75.353981', '-55.498649'], time:4.481673ms, swizzle: NOOP, TFLOPS: 61.33 
  out_tf32(mma2x4+...+stage2+dsmem): ['-75.353981', '-55.498649'], time:4.164934ms, swizzle: NOOP, TFLOPS: 66.00 (+1.72%)
out_tf32(mma2x4+...+stage3+swizzle): ['-75.353981', '-55.498649'], time:4.735803ms, swizzle: 512 , TFLOPS: 58.04 
out_tf32(mma2x4+...+stage2+swizzle): ['-75.353981', '-55.498649'], time:4.232597ms, swizzle: 512 , TFLOPS: 64.94 
 out_tf32(...+stage3+dsmem+swizzle): ['-75.353981', '-55.498649'], time:4.483127ms, swizzle: 512 , TFLOPS: 61.31 
 out_tf32(...+stage2+dsmem+swizzle): ['-75.353981', '-55.498649'], time:4.207682ms, swizzle: 512 , TFLOPS: 65.33 
              out_tf32(cublas+tf32): ['-75.353981', '-55.498649'], time:3.143763ms, swizzle: NOOP, TFLOPS: 87.44 (+32.48%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=4096, K=8192
                  out_f32x4(t8x8sk): ['-16.729681', '-50.358272'], time:13.89679ms, swizzle: NOOP, TFLOPS: 39.56 (+0.00%)
                 out_f32x4(t8x8bcf): ['-16.729681', '-50.358272'], time:13.10465ms, swizzle: NOOP, TFLOPS: 41.95 (+6.04%)
                out_f32x4(t8x8dbuf): ['-16.729681', '-50.358272'], time:12.84394ms, swizzle: NOOP, TFLOPS: 42.80 (+2.03%)
           out_f32x4(t8x8dbufasync): ['-16.729681', '-50.358272'], time:12.92002ms, swizzle: NOOP, TFLOPS: 42.55 
             out_f32x4(k16t8x4dbuf): ['-16.729681', '-50.358272'], time:15.91892ms, swizzle: NOOP, TFLOPS: 34.53 
        out_f32x4(k16t8x4dbufasync): ['-16.729681', '-50.358272'], time:15.60695ms, swizzle: NOOP, TFLOPS: 35.23 
              out_f32x4(k168x8dbuf): ['-16.729681', '-50.358272'], time:12.74766ms, swizzle: NOOP, TFLOPS: 43.13 (+0.76%)
         out_f32x4(k168x8dbufasync): ['-16.729681', '-50.358272'], time:12.74528ms, swizzle: NOOP, TFLOPS: 43.13 (+0.02%)
             out_f32x4(k168x16dbuf): ['-16.729681', '-50.358272'], time:17.47632ms, swizzle: NOOP, TFLOPS: 31.46 
        out_f32x4(k168x16dbufasync): ['-16.729681', '-50.358272'], time:15.45002ms, swizzle: NOOP, TFLOPS: 35.58 
                    out_f32(cublas): ['-16.729681', '-50.358272'], time:9.442520ms, swizzle: NOOP, TFLOPS: 58.22 (+34.98%)
                         out_f32_th: ['-16.729681', '-50.358272'], time:10.19515ms, swizzle: NOOP, TFLOPS: 53.92 
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['-16.728071', '-50.355793'], time:10.15954ms, swizzle: NOOP, TFLOPS: 54.11 
    out_tf32(mma2x4+warp2x4+stage2): ['-16.728071', '-50.355793'], time:8.410191ms, swizzle: NOOP, TFLOPS: 65.37 (+12.27%)
  out_tf32(mma2x4+...+stage3+dsmem): ['-16.728071', '-50.355793'], time:8.984136ms, swizzle: NOOP, TFLOPS: 61.19 
  out_tf32(mma2x4+...+stage2+dsmem): ['-16.728071', '-50.355793'], time:8.436894ms, swizzle: NOOP, TFLOPS: 65.16 
out_tf32(mma2x4+...+stage3+swizzle): ['-16.728071', '-50.355793'], time:9.509420ms, swizzle: 512 , TFLOPS: 57.81 
out_tf32(mma2x4+...+stage2+swizzle): ['-16.728071', '-50.355793'], time:8.570861ms, swizzle: 512 , TFLOPS: 64.14 
 out_tf32(...+stage3+dsmem+swizzle): ['-16.728071', '-50.355793'], time:8.989834ms, swizzle: 512 , TFLOPS: 61.15 
 out_tf32(...+stage2+dsmem+swizzle): ['-16.728071', '-50.355793'], time:8.537983ms, swizzle: 512 , TFLOPS: 64.39 
              out_tf32(cublas+tf32): ['-16.728071', '-50.355793'], time:6.228494ms, swizzle: NOOP, TFLOPS: 88.26 (+35.03%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=8192, K=2048
                  out_f32x4(t8x8sk): ['-47.060710', '-4.6656251'], time:6.798839ms, swizzle: NOOP, TFLOPS: 40.43 (+0.00%)
                 out_f32x4(t8x8bcf): ['-47.060710', '-4.6656251'], time:6.447601ms, swizzle: NOOP, TFLOPS: 42.63 (+5.45%)
                out_f32x4(t8x8dbuf): ['-47.060710', '-4.6656251'], time:6.140947ms, swizzle: NOOP, TFLOPS: 44.76 (+4.99%)
           out_f32x4(t8x8dbufasync): ['-47.060710', '-4.6656251'], time:6.522798ms, swizzle: NOOP, TFLOPS: 42.14 
             out_f32x4(k16t8x4dbuf): ['-47.060710', '-4.6656251'], time:7.807850ms, swizzle: NOOP, TFLOPS: 35.21 
        out_f32x4(k16t8x4dbufasync): ['-47.060710', '-4.6656251'], time:7.288694ms, swizzle: NOOP, TFLOPS: 37.71 
              out_f32x4(k168x8dbuf): ['-47.060710', '-4.6656251'], time:6.241059ms, swizzle: NOOP, TFLOPS: 44.04 
         out_f32x4(k168x8dbufasync): ['-47.060710', '-4.6656251'], time:6.311464ms, swizzle: NOOP, TFLOPS: 43.55 
             out_f32x4(k168x16dbuf): ['-47.060710', '-4.6656251'], time:10.07242ms, swizzle: NOOP, TFLOPS: 27.29 
        out_f32x4(k168x16dbufasync): ['-47.060710', '-4.6656251'], time:7.776546ms, swizzle: NOOP, TFLOPS: 35.35 
                    out_f32(cublas): ['-47.060710', '-4.6656251'], time:4.550719ms, swizzle: NOOP, TFLOPS: 60.40 (+34.94%)
                         out_f32_th: ['-47.060710', '-4.6656251'], time:5.080461ms, swizzle: NOOP, TFLOPS: 54.10 
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['-47.060657', '-4.6654434'], time:5.138754ms, swizzle: NOOP, TFLOPS: 53.49 
    out_tf32(mma2x4+warp2x4+stage2): ['-47.060657', '-4.6654434'], time:4.164814ms, swizzle: NOOP, TFLOPS: 66.00 (+9.27%)
  out_tf32(mma2x4+...+stage3+dsmem): ['-47.060657', '-4.6654434'], time:4.335999ms, swizzle: NOOP, TFLOPS: 63.39 
  out_tf32(mma2x4+...+stage2+dsmem): ['-47.060657', '-4.6654434'], time:4.056859ms, swizzle: NOOP, TFLOPS: 67.76 (+2.66%)
out_tf32(mma2x4+...+stage3+swizzle): ['-47.060657', '-4.6654434'], time:4.586672ms, swizzle: 1024, TFLOPS: 59.93 
out_tf32(mma2x4+...+stage2+swizzle): ['-47.060657', '-4.6654434'], time:4.042649ms, swizzle: 1024, TFLOPS: 67.99 (+0.35%)
 out_tf32(...+stage3+dsmem+swizzle): ['-47.060657', '-4.6654434'], time:4.334664ms, swizzle: 1024, TFLOPS: 63.41 
 out_tf32(...+stage2+dsmem+swizzle): ['-47.060657', '-4.6654434'], time:4.019880ms, swizzle: 1024, TFLOPS: 68.38 (+0.57%)
              out_tf32(cublas+tf32): ['-47.060657', '-4.6654434'], time:3.195333ms, swizzle: NOOP, TFLOPS: 86.02 (+25.80%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=8192, K=4096
                  out_f32x4(t8x8sk): ['-75.354446', '-55.499485'], time:13.96210ms, swizzle: NOOP, TFLOPS: 39.37 (+0.00%)
                 out_f32x4(t8x8bcf): ['-75.354446', '-55.499485'], time:13.20223ms, swizzle: NOOP, TFLOPS: 41.64 (+5.76%)
                out_f32x4(t8x8dbuf): ['-75.354446', '-55.499485'], time:12.84816ms, swizzle: NOOP, TFLOPS: 42.79 (+2.76%)
           out_f32x4(t8x8dbufasync): ['-75.354446', '-55.499485'], time:13.17770ms, swizzle: NOOP, TFLOPS: 41.72 
             out_f32x4(k16t8x4dbuf): ['-75.354446', '-55.499485'], time:16.38050ms, swizzle: NOOP, TFLOPS: 33.56 
        out_f32x4(k16t8x4dbufasync): ['-75.354446', '-55.499485'], time:18.34349ms, swizzle: NOOP, TFLOPS: 29.97 
              out_f32x4(k168x8dbuf): ['-75.354446', '-55.499485'], time:12.71991ms, swizzle: NOOP, TFLOPS: 43.22 (+1.01%)
         out_f32x4(k168x8dbufasync): ['-75.354446', '-55.499485'], time:12.87837ms, swizzle: NOOP, TFLOPS: 42.69 
             out_f32x4(k168x16dbuf): ['-75.354446', '-55.499485'], time:17.24660ms, swizzle: NOOP, TFLOPS: 31.88 
        out_f32x4(k168x16dbufasync): ['-75.354446', '-55.499485'], time:15.59972ms, swizzle: NOOP, TFLOPS: 35.24 
                    out_f32(cublas): ['-75.354446', '-55.499485'], time:10.09151ms, swizzle: NOOP, TFLOPS: 54.48 (+26.05%)
                         out_f32_th: ['-75.354446', '-55.499485'], time:10.17272ms, swizzle: NOOP, TFLOPS: 54.04 
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['-75.353981', '-55.498649'], time:10.22260ms, swizzle: NOOP, TFLOPS: 53.78 
    out_tf32(mma2x4+warp2x4+stage2): ['-75.353981', '-55.498649'], time:8.127856ms, swizzle: NOOP, TFLOPS: 67.64 (+24.16%)
  out_tf32(mma2x4+...+stage3+dsmem): ['-75.353981', '-55.498649'], time:8.685517ms, swizzle: NOOP, TFLOPS: 63.30 
  out_tf32(mma2x4+...+stage2+dsmem): ['-75.353981', '-55.498649'], time:8.162832ms, swizzle: NOOP, TFLOPS: 67.35 
out_tf32(mma2x4+...+stage3+swizzle): ['-75.353981', '-55.498649'], time:9.191656ms, swizzle: 1024, TFLOPS: 59.81 
out_tf32(mma2x4+...+stage2+swizzle): ['-75.353981', '-55.498649'], time:8.137893ms, swizzle: 1024, TFLOPS: 67.56 
 out_tf32(...+stage3+dsmem+swizzle): ['-75.353981', '-55.498649'], time:8.668637ms, swizzle: 1024, TFLOPS: 63.42 
 out_tf32(...+stage2+dsmem+swizzle): ['-75.353981', '-55.498649'], time:8.101415ms, swizzle: 1024, TFLOPS: 67.86 (+0.33%)
              out_tf32(cublas+tf32): ['-75.353981', '-55.498649'], time:6.275963ms, swizzle: NOOP, TFLOPS: 87.60 (+29.09%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=8192, K=8192
                  out_f32x4(t8x8sk): ['-16.729681', '-50.358272'], time:28.35204ms, swizzle: NOOP, TFLOPS: 38.78 (+0.00%)
                 out_f32x4(t8x8bcf): ['-16.729681', '-50.358272'], time:26.40411ms, swizzle: NOOP, TFLOPS: 41.64 (+7.38%)
                out_f32x4(t8x8dbuf): ['-16.729681', '-50.358272'], time:26.32806ms, swizzle: NOOP, TFLOPS: 41.76 (+0.29%)
           out_f32x4(t8x8dbufasync): ['-16.729681', '-50.358272'], time:26.06961ms, swizzle: NOOP, TFLOPS: 42.18 (+0.99%)
             out_f32x4(k16t8x4dbuf): ['-16.729681', '-50.358272'], time:32.76157ms, swizzle: NOOP, TFLOPS: 33.56 
        out_f32x4(k16t8x4dbufasync): ['-16.729681', '-50.358272'], time:36.40823ms, swizzle: NOOP, TFLOPS: 30.20 
              out_f32x4(k168x8dbuf): ['-16.729681', '-50.358272'], time:26.03523ms, swizzle: NOOP, TFLOPS: 42.23 (+0.13%)
         out_f32x4(k168x8dbufasync): ['-16.729681', '-50.358272'], time:25.72710ms, swizzle: NOOP, TFLOPS: 42.74 (+1.20%)
             out_f32x4(k168x16dbuf): ['-16.729681', '-50.358272'], time:33.43932ms, swizzle: NOOP, TFLOPS: 32.88 
        out_f32x4(k168x16dbufasync): ['-16.729681', '-50.358272'], time:31.05635ms, swizzle: NOOP, TFLOPS: 35.40 
                    out_f32(cublas): ['-16.729681', '-50.358272'], time:20.12453ms, swizzle: NOOP, TFLOPS: 54.64 (+27.84%)
                         out_f32_th: ['-16.729681', '-50.358272'], time:20.11132ms, swizzle: NOOP, TFLOPS: 54.67 (+0.07%)
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['-16.728071', '-50.355793'], time:18.97876ms, swizzle: NOOP, TFLOPS: 57.93 (+5.97%)
    out_tf32(mma2x4+warp2x4+stage2): ['-16.728071', '-50.355793'], time:16.63732ms, swizzle: NOOP, TFLOPS: 66.09 (+14.07%)
  out_tf32(mma2x4+...+stage3+dsmem): ['-16.728071', '-50.355793'], time:17.52586ms, swizzle: NOOP, TFLOPS: 62.74 
  out_tf32(mma2x4+...+stage2+dsmem): ['-16.728071', '-50.355793'], time:16.75102ms, swizzle: NOOP, TFLOPS: 65.64 
out_tf32(mma2x4+...+stage3+swizzle): ['-16.728071', '-50.355793'], time:18.45724ms, swizzle: 1024, TFLOPS: 59.57 
out_tf32(mma2x4+...+stage2+swizzle): ['-16.728071', '-50.355793'], time:16.60916ms, swizzle: 1024, TFLOPS: 66.20 (+0.17%)
 out_tf32(...+stage3+dsmem+swizzle): ['-16.728071', '-50.355793'], time:17.33326ms, swizzle: 1024, TFLOPS: 63.43 
 out_tf32(...+stage2+dsmem+swizzle): ['-16.728071', '-50.355793'], time:16.24128ms, swizzle: 1024, TFLOPS: 67.70 (+2.27%)
              out_tf32(cublas+tf32): ['-16.728071', '-50.355793'], time:12.43293ms, swizzle: NOOP, TFLOPS: 88.44 (+30.63%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=16384, K=2048
                  out_f32x4(t8x8sk): ['-47.060710', '-4.6656251'], time:13.98527ms, swizzle: NOOP, TFLOPS: 39.31 (+0.00%)
                 out_f32x4(t8x8bcf): ['-47.060710', '-4.6656251'], time:13.17927ms, swizzle: NOOP, TFLOPS: 41.71 (+6.12%)
                out_f32x4(t8x8dbuf): ['-47.060710', '-4.6656251'], time:13.34328ms, swizzle: NOOP, TFLOPS: 41.20 
           out_f32x4(t8x8dbufasync): ['-47.060710', '-4.6656251'], time:13.56148ms, swizzle: NOOP, TFLOPS: 40.54 
             out_f32x4(k16t8x4dbuf): ['-47.060710', '-4.6656251'], time:16.00973ms, swizzle: NOOP, TFLOPS: 34.34 
        out_f32x4(k16t8x4dbufasync): ['-47.060710', '-4.6656251'], time:18.91307ms, swizzle: NOOP, TFLOPS: 29.07 
              out_f32x4(k168x8dbuf): ['-47.060710', '-4.6656251'], time:12.93704ms, swizzle: NOOP, TFLOPS: 42.49 (+1.87%)
         out_f32x4(k168x8dbufasync): ['-47.060710', '-4.6656251'], time:13.23125ms, swizzle: NOOP, TFLOPS: 41.55 
             out_f32x4(k168x16dbuf): ['-47.060710', '-4.6656251'], time:18.04797ms, swizzle: NOOP, TFLOPS: 30.46 
        out_f32x4(k168x16dbufasync): ['-47.060710', '-4.6656251'], time:15.84594ms, swizzle: NOOP, TFLOPS: 34.69 
                    out_f32(cublas): ['-47.060710', '-4.6656251'], time:10.09936ms, swizzle: NOOP, TFLOPS: 54.43 (+28.10%)
                         out_f32_th: ['-47.060710', '-4.6656251'], time:10.18674ms, swizzle: NOOP, TFLOPS: 53.97 
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['-47.060657', '-4.6654434'], time:10.46772ms, swizzle: NOOP, TFLOPS: 52.52 
    out_tf32(mma2x4+warp2x4+stage2): ['-47.060657', '-4.6654434'], time:8.240175ms, swizzle: NOOP, TFLOPS: 66.72 (+22.56%)
  out_tf32(mma2x4+...+stage3+dsmem): ['-47.060657', '-4.6654434'], time:10.32938ms, swizzle: NOOP, TFLOPS: 53.22 
  out_tf32(mma2x4+...+stage2+dsmem): ['-47.060657', '-4.6654434'], time:8.283329ms, swizzle: NOOP, TFLOPS: 66.37 
out_tf32(mma2x4+...+stage3+swizzle): ['-47.060657', '-4.6654434'], time:9.062314ms, swizzle: 2048, TFLOPS: 60.66 
out_tf32(mma2x4+...+stage2+swizzle): ['-47.060657', '-4.6654434'], time:7.989788ms, swizzle: 2048, TFLOPS: 68.81 (+3.13%)
 out_tf32(...+stage3+dsmem+swizzle): ['-47.060657', '-4.6654434'], time:8.544850ms, swizzle: 2048, TFLOPS: 64.34 
 out_tf32(...+stage2+dsmem+swizzle): ['-47.060657', '-4.6654434'], time:7.977056ms, swizzle: 2048, TFLOPS: 68.92 (+0.16%)
              out_tf32(cublas+tf32): ['-47.060657', '-4.6654434'], time:6.368517ms, swizzle: NOOP, TFLOPS: 86.32 (+25.26%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=16384, K=4096
                  out_f32x4(t8x8sk): ['-75.354446', '-55.499485'], time:29.19404ms, swizzle: NOOP, TFLOPS: 37.66 (+0.00%)
                 out_f32x4(t8x8bcf): ['-75.354446', '-55.499485'], time:27.11408ms, swizzle: NOOP, TFLOPS: 40.55 (+7.67%)
                out_f32x4(t8x8dbuf): ['-75.354446', '-55.499485'], time:27.30100ms, swizzle: NOOP, TFLOPS: 40.27 
           out_f32x4(t8x8dbufasync): ['-75.354446', '-55.499485'], time:26.82816ms, swizzle: NOOP, TFLOPS: 40.98 (+1.07%)
             out_f32x4(k16t8x4dbuf): ['-75.354446', '-55.499485'], time:33.14166ms, swizzle: NOOP, TFLOPS: 33.18 
        out_f32x4(k16t8x4dbufasync): ['-75.354446', '-55.499485'], time:37.29856ms, swizzle: NOOP, TFLOPS: 29.48 
              out_f32x4(k168x8dbuf): ['-75.354446', '-55.499485'], time:26.89886ms, swizzle: NOOP, TFLOPS: 40.88 
         out_f32x4(k168x8dbufasync): ['-75.354446', '-55.499485'], time:26.53365ms, swizzle: NOOP, TFLOPS: 41.44 (+1.11%)
             out_f32x4(k168x16dbuf): ['-75.354446', '-55.499485'], time:33.72852ms, swizzle: NOOP, TFLOPS: 32.60 
        out_f32x4(k168x16dbufasync): ['-75.354446', '-55.499485'], time:31.40311ms, swizzle: NOOP, TFLOPS: 35.01 
                    out_f32(cublas): ['-75.354446', '-55.499485'], time:20.21865ms, swizzle: NOOP, TFLOPS: 54.38 (+31.23%)
                         out_f32_th: ['-75.354446', '-55.499485'], time:20.21381ms, swizzle: NOOP, TFLOPS: 54.39 (+0.02%)
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['-75.353981', '-55.498649'], time:20.28422ms, swizzle: NOOP, TFLOPS: 54.21 
    out_tf32(mma2x4+warp2x4+stage2): ['-75.353981', '-55.498649'], time:16.42227ms, swizzle: NOOP, TFLOPS: 66.95 (+23.09%)
  out_tf32(mma2x4+...+stage3+dsmem): ['-75.353981', '-55.498649'], time:20.09727ms, swizzle: NOOP, TFLOPS: 54.71 
  out_tf32(mma2x4+...+stage2+dsmem): ['-75.353981', '-55.498649'], time:16.80400ms, swizzle: NOOP, TFLOPS: 65.43 
out_tf32(mma2x4+...+stage3+swizzle): ['-75.353981', '-55.498649'], time:18.12791ms, swizzle: 2048, TFLOPS: 60.65 
out_tf32(mma2x4+...+stage2+swizzle): ['-75.353981', '-55.498649'], time:16.12157ms, swizzle: 2048, TFLOPS: 68.20 (+1.87%)
 out_tf32(...+stage3+dsmem+swizzle): ['-75.353981', '-55.498649'], time:17.04490ms, swizzle: 2048, TFLOPS: 64.51 
 out_tf32(...+stage2+dsmem+swizzle): ['-75.353981', '-55.498649'], time:15.99178ms, swizzle: 2048, TFLOPS: 68.75 (+0.81%)
              out_tf32(cublas+tf32): ['-75.353981', '-55.498649'], time:12.52326ms, swizzle: NOOP, TFLOPS: 87.80 (+27.70%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=8192, N=16384, K=8192
                  out_f32x4(t8x8sk): ['-16.729681', '-50.358272'], time:57.26828ms, swizzle: NOOP, TFLOPS: 38.40 (+0.00%)
                 out_f32x4(t8x8bcf): ['-16.729681', '-50.358272'], time:55.07311ms, swizzle: NOOP, TFLOPS: 39.93 (+3.99%)
                out_f32x4(t8x8dbuf): ['-16.729681', '-50.358272'], time:54.52587ms, swizzle: NOOP, TFLOPS: 40.33 (+1.00%)
           out_f32x4(t8x8dbufasync): ['-16.729681', '-50.358272'], time:54.33506ms, swizzle: NOOP, TFLOPS: 40.47 (+0.35%)
             out_f32x4(k16t8x4dbuf): ['-16.729681', '-50.358272'], time:68.61190ms, swizzle: NOOP, TFLOPS: 32.05 
        out_f32x4(k16t8x4dbufasync): ['-16.729681', '-50.358272'], time:74.06039ms, swizzle: NOOP, TFLOPS: 29.69 
              out_f32x4(k168x8dbuf): ['-16.729681', '-50.358272'], time:53.46734ms, swizzle: NOOP, TFLOPS: 41.13 (+1.62%)
         out_f32x4(k168x8dbufasync): ['-16.729681', '-50.358272'], time:53.30424ms, swizzle: NOOP, TFLOPS: 41.25 (+0.31%)
             out_f32x4(k168x16dbuf): ['-16.729681', '-50.358272'], time:67.03381ms, swizzle: NOOP, TFLOPS: 32.80 
        out_f32x4(k168x16dbufasync): ['-16.729681', '-50.358272'], time:62.40849ms, swizzle: NOOP, TFLOPS: 35.24 
                    out_f32(cublas): ['-16.729681', '-50.358272'], time:40.84336ms, swizzle: NOOP, TFLOPS: 53.84 (+30.51%)
                         out_f32_th: ['-16.729681', '-50.358272'], time:40.61832ms, swizzle: NOOP, TFLOPS: 54.14 (+0.55%)
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['-16.728071', '-50.355793'], time:39.70985ms, swizzle: NOOP, TFLOPS: 55.38 (+2.29%)
    out_tf32(mma2x4+warp2x4+stage2): ['-16.728071', '-50.355793'], time:38.30974ms, swizzle: NOOP, TFLOPS: 57.40 (+3.65%)
  out_tf32(mma2x4+...+stage3+dsmem): ['-16.728071', '-50.355793'], time:39.49790ms, swizzle: NOOP, TFLOPS: 55.67 
  out_tf32(mma2x4+...+stage2+dsmem): ['-16.728071', '-50.355793'], time:38.48206ms, swizzle: NOOP, TFLOPS: 57.14 
out_tf32(mma2x4+...+stage3+swizzle): ['-16.728071', '-50.355793'], time:36.25507ms, swizzle: 2048, TFLOPS: 60.65 (+5.67%)
out_tf32(mma2x4+...+stage2+swizzle): ['-16.728071', '-50.355793'], time:32.20772ms, swizzle: 2048, TFLOPS: 68.28 (+12.57%)
 out_tf32(...+stage3+dsmem+swizzle): ['-16.728071', '-50.355793'], time:34.51404ms, swizzle: 2048, TFLOPS: 63.71 
 out_tf32(...+stage2+dsmem+swizzle): ['-16.728071', '-50.355793'], time:32.17818ms, swizzle: 2048, TFLOPS: 68.34 (+0.09%)
              out_tf32(cublas+tf32): ['-16.728071', '-50.355793'], time:24.87213ms, swizzle: NOOP, TFLOPS: 88.41 (+29.37%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=16384, N=4096, K=2048
                  out_f32x4(t8x8sk): ['-47.060710', '-4.6656251'], time:6.460428ms, swizzle: NOOP, TFLOPS: 42.55 (+0.00%)
                 out_f32x4(t8x8bcf): ['-47.060710', '-4.6656251'], time:6.402564ms, swizzle: NOOP, TFLOPS: 42.93 (+0.90%)
                out_f32x4(t8x8dbuf): ['-47.060710', '-4.6656251'], time:5.658197ms, swizzle: NOOP, TFLOPS: 48.58 (+13.16%)
           out_f32x4(t8x8dbufasync): ['-47.060710', '-4.6656251'], time:6.605958ms, swizzle: NOOP, TFLOPS: 41.61 
             out_f32x4(k16t8x4dbuf): ['-47.060710', '-4.6656251'], time:7.782769ms, swizzle: NOOP, TFLOPS: 35.32 
        out_f32x4(k16t8x4dbufasync): ['-47.060710', '-4.6656251'], time:7.396531ms, swizzle: NOOP, TFLOPS: 37.16 
              out_f32x4(k168x8dbuf): ['-47.060710', '-4.6656251'], time:6.217432ms, swizzle: NOOP, TFLOPS: 44.21 
         out_f32x4(k168x8dbufasync): ['-47.060710', '-4.6656251'], time:6.220746ms, swizzle: NOOP, TFLOPS: 44.19 
             out_f32x4(k168x16dbuf): ['-47.060710', '-4.6656251'], time:9.925103ms, swizzle: NOOP, TFLOPS: 27.70 
        out_f32x4(k168x16dbufasync): ['-47.060710', '-4.6656251'], time:7.786130ms, swizzle: NOOP, TFLOPS: 35.30 
                    out_f32(cublas): ['-47.060710', '-4.6656251'], time:4.854559ms, swizzle: NOOP, TFLOPS: 56.62 (+16.55%)
                         out_f32_th: ['-47.060710', '-4.6656251'], time:5.283093ms, swizzle: NOOP, TFLOPS: 52.03 
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['-47.060657', '-4.6654434'], time:5.335474ms, swizzle: NOOP, TFLOPS: 51.52 
    out_tf32(mma2x4+warp2x4+stage2): ['-47.060657', '-4.6654434'], time:4.239010ms, swizzle: NOOP, TFLOPS: 64.84 (+14.52%)
  out_tf32(mma2x4+...+stage3+dsmem): ['-47.060657', '-4.6654434'], time:4.474496ms, swizzle: NOOP, TFLOPS: 61.43 
  out_tf32(mma2x4+...+stage2+dsmem): ['-47.060657', '-4.6654434'], time:4.110980ms, swizzle: NOOP, TFLOPS: 66.86 (+3.11%)
out_tf32(mma2x4+...+stage3+swizzle): ['-47.060657', '-4.6654434'], time:4.676079ms, swizzle: 512 , TFLOPS: 58.78 
out_tf32(mma2x4+...+stage2+swizzle): ['-47.060657', '-4.6654434'], time:4.158711ms, swizzle: 512 , TFLOPS: 66.10 
 out_tf32(...+stage3+dsmem+swizzle): ['-47.060657', '-4.6654434'], time:4.449486ms, swizzle: 512 , TFLOPS: 61.78 
 out_tf32(...+stage2+dsmem+swizzle): ['-47.060657', '-4.6654434'], time:4.144144ms, swizzle: 512 , TFLOPS: 66.33 
              out_tf32(cublas+tf32): ['-47.060657', '-4.6654434'], time:3.213500ms, swizzle: NOOP, TFLOPS: 85.54 (+27.93%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=16384, N=4096, K=4096
                  out_f32x4(t8x8sk): ['-75.354446', '-55.499485'], time:14.16513ms, swizzle: NOOP, TFLOPS: 38.81 (+0.00%)
                 out_f32x4(t8x8bcf): ['-75.354446', '-55.499485'], time:13.22996ms, swizzle: NOOP, TFLOPS: 41.55 (+7.07%)
                out_f32x4(t8x8dbuf): ['-75.354446', '-55.499485'], time:12.91038ms, swizzle: NOOP, TFLOPS: 42.58 (+2.48%)
           out_f32x4(t8x8dbufasync): ['-75.354446', '-55.499485'], time:12.98484ms, swizzle: NOOP, TFLOPS: 42.34 
             out_f32x4(k16t8x4dbuf): ['-75.354446', '-55.499485'], time:15.36393ms, swizzle: NOOP, TFLOPS: 35.78 
        out_f32x4(k16t8x4dbufasync): ['-75.354446', '-55.499485'], time:14.93911ms, swizzle: NOOP, TFLOPS: 36.80 
              out_f32x4(k168x8dbuf): ['-75.354446', '-55.499485'], time:12.74352ms, swizzle: NOOP, TFLOPS: 43.14 (+1.31%)
         out_f32x4(k168x8dbufasync): ['-75.354446', '-55.499485'], time:12.75591ms, swizzle: NOOP, TFLOPS: 43.10 
             out_f32x4(k168x16dbuf): ['-75.354446', '-55.499485'], time:17.13323ms, swizzle: NOOP, TFLOPS: 32.09 
        out_f32x4(k168x16dbufasync): ['-75.354446', '-55.499485'], time:15.61603ms, swizzle: NOOP, TFLOPS: 35.20 
                    out_f32(cublas): ['-75.354446', '-55.499485'], time:9.514045ms, swizzle: NOOP, TFLOPS: 57.78 (+33.94%)
                         out_f32_th: ['-75.354446', '-55.499485'], time:10.21158ms, swizzle: NOOP, TFLOPS: 53.84 
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['-75.353981', '-55.498649'], time:9.873723ms, swizzle: NOOP, TFLOPS: 55.68 
    out_tf32(mma2x4+warp2x4+stage2): ['-75.353981', '-55.498649'], time:8.247256ms, swizzle: NOOP, TFLOPS: 66.66 (+15.36%)
  out_tf32(mma2x4+...+stage3+dsmem): ['-75.353981', '-55.498649'], time:8.824229ms, swizzle: NOOP, TFLOPS: 62.30 
  out_tf32(mma2x4+...+stage2+dsmem): ['-75.353981', '-55.498649'], time:8.270120ms, swizzle: NOOP, TFLOPS: 66.47 
out_tf32(mma2x4+...+stage3+swizzle): ['-75.353981', '-55.498649'], time:9.350538ms, swizzle: 512 , TFLOPS: 58.79 
out_tf32(mma2x4+...+stage2+swizzle): ['-75.353981', '-55.498649'], time:8.396744ms, swizzle: 512 , TFLOPS: 65.47 
 out_tf32(...+stage3+dsmem+swizzle): ['-75.353981', '-55.498649'], time:8.850240ms, swizzle: 512 , TFLOPS: 62.12 
 out_tf32(...+stage2+dsmem+swizzle): ['-75.353981', '-55.498649'], time:8.382391ms, swizzle: 512 , TFLOPS: 65.58 
              out_tf32(cublas+tf32): ['-75.353981', '-55.498649'], time:6.312847ms, swizzle: NOOP, TFLOPS: 87.09 (+30.64%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=16384, N=4096, K=8192
                  out_f32x4(t8x8sk): ['-16.729681', '-50.358272'], time:28.25469ms, swizzle: NOOP, TFLOPS: 38.91 (+0.00%)
                 out_f32x4(t8x8bcf): ['-16.729681', '-50.358272'], time:26.38928ms, swizzle: NOOP, TFLOPS: 41.67 (+7.07%)
                out_f32x4(t8x8dbuf): ['-16.729681', '-50.358272'], time:26.26681ms, swizzle: NOOP, TFLOPS: 41.86 (+0.47%)
           out_f32x4(t8x8dbufasync): ['-16.729681', '-50.358272'], time:25.91352ms, swizzle: NOOP, TFLOPS: 42.43 (+1.36%)
             out_f32x4(k16t8x4dbuf): ['-16.729681', '-50.358272'], time:31.97050ms, swizzle: NOOP, TFLOPS: 34.39 
        out_f32x4(k16t8x4dbufasync): ['-16.729681', '-50.358272'], time:31.64088ms, swizzle: NOOP, TFLOPS: 34.75 
              out_f32x4(k168x8dbuf): ['-16.729681', '-50.358272'], time:25.69801ms, swizzle: NOOP, TFLOPS: 42.79 (+0.84%)
         out_f32x4(k168x8dbufasync): ['-16.729681', '-50.358272'], time:25.85070ms, swizzle: NOOP, TFLOPS: 42.53 
             out_f32x4(k168x16dbuf): ['-16.729681', '-50.358272'], time:33.41231ms, swizzle: NOOP, TFLOPS: 32.91 
        out_f32x4(k168x16dbufasync): ['-16.729681', '-50.358272'], time:31.13510ms, swizzle: NOOP, TFLOPS: 35.31 
                    out_f32(cublas): ['-16.729681', '-50.358272'], time:20.54734ms, swizzle: NOOP, TFLOPS: 53.51 (+25.07%)
                         out_f32_th: ['-16.729681', '-50.358272'], time:20.22862ms, swizzle: NOOP, TFLOPS: 54.35 (+1.58%)
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['-16.728071', '-50.355793'], time:18.90823ms, swizzle: NOOP, TFLOPS: 58.15 (+6.98%)
    out_tf32(mma2x4+warp2x4+stage2): ['-16.728071', '-50.355793'], time:16.62817ms, swizzle: NOOP, TFLOPS: 66.12 (+13.71%)
  out_tf32(mma2x4+...+stage3+dsmem): ['-16.728071', '-50.355793'], time:17.64948ms, swizzle: NOOP, TFLOPS: 62.30 
  out_tf32(mma2x4+...+stage2+dsmem): ['-16.728071', '-50.355793'], time:16.71087ms, swizzle: NOOP, TFLOPS: 65.80 
out_tf32(mma2x4+...+stage3+swizzle): ['-16.728071', '-50.355793'], time:18.88034ms, swizzle: 512 , TFLOPS: 58.24 
out_tf32(mma2x4+...+stage2+swizzle): ['-16.728071', '-50.355793'], time:17.23682ms, swizzle: 512 , TFLOPS: 63.79 
 out_tf32(...+stage3+dsmem+swizzle): ['-16.728071', '-50.355793'], time:17.83206ms, swizzle: 512 , TFLOPS: 61.66 
 out_tf32(...+stage2+dsmem+swizzle): ['-16.728071', '-50.355793'], time:16.86165ms, swizzle: 512 , TFLOPS: 65.21 
              out_tf32(cublas+tf32): ['-16.728071', '-50.355793'], time:12.47577ms, swizzle: NOOP, TFLOPS: 88.13 (+33.28%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=16384, N=8192, K=2048
                  out_f32x4(t8x8sk): ['-47.060710', '-4.6656251'], time:14.22321ms, swizzle: NOOP, TFLOPS: 38.65 (+0.00%)
                 out_f32x4(t8x8bcf): ['-47.060710', '-4.6656251'], time:12.88597ms, swizzle: NOOP, TFLOPS: 42.66 (+10.38%)
                out_f32x4(t8x8dbuf): ['-47.060710', '-4.6656251'], time:12.92362ms, swizzle: NOOP, TFLOPS: 42.54 
           out_f32x4(t8x8dbufasync): ['-47.060710', '-4.6656251'], time:12.95907ms, swizzle: NOOP, TFLOPS: 42.42 
             out_f32x4(k16t8x4dbuf): ['-47.060710', '-4.6656251'], time:15.22619ms, swizzle: NOOP, TFLOPS: 36.11 
        out_f32x4(k16t8x4dbufasync): ['-47.060710', '-4.6656251'], time:14.75358ms, swizzle: NOOP, TFLOPS: 37.26 
              out_f32x4(k168x8dbuf): ['-47.060710', '-4.6656251'], time:12.62526ms, swizzle: NOOP, TFLOPS: 43.54 (+2.06%)
         out_f32x4(k168x8dbufasync): ['-47.060710', '-4.6656251'], time:12.62886ms, swizzle: NOOP, TFLOPS: 43.53 
             out_f32x4(k168x16dbuf): ['-47.060710', '-4.6656251'], time:17.84720ms, swizzle: NOOP, TFLOPS: 30.80 
        out_f32x4(k168x16dbufasync): ['-47.060710', '-4.6656251'], time:15.72022ms, swizzle: NOOP, TFLOPS: 34.97 
                    out_f32(cublas): ['-47.060710', '-4.6656251'], time:10.40380ms, swizzle: NOOP, TFLOPS: 52.84 (+21.35%)
                         out_f32_th: ['-47.060710', '-4.6656251'], time:10.24830ms, swizzle: NOOP, TFLOPS: 53.64 (+1.52%)
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['-47.060657', '-4.6654434'], time:9.918951ms, swizzle: NOOP, TFLOPS: 55.42 (+3.32%)
    out_tf32(mma2x4+warp2x4+stage2): ['-47.060657', '-4.6654434'], time:8.089613ms, swizzle: NOOP, TFLOPS: 67.96 (+22.61%)
  out_tf32(mma2x4+...+stage3+dsmem): ['-47.060657', '-4.6654434'], time:8.570861ms, swizzle: NOOP, TFLOPS: 64.14 
  out_tf32(mma2x4+...+stage2+dsmem): ['-47.060657', '-4.6654434'], time:8.102369ms, swizzle: NOOP, TFLOPS: 67.85 
out_tf32(mma2x4+...+stage3+swizzle): ['-47.060657', '-4.6654434'], time:9.078979ms, swizzle: 1024, TFLOPS: 60.55 
out_tf32(mma2x4+...+stage2+swizzle): ['-47.060657', '-4.6654434'], time:8.036780ms, swizzle: 1024, TFLOPS: 68.40 (+0.66%)
 out_tf32(...+stage3+dsmem+swizzle): ['-47.060657', '-4.6654434'], time:8.579850ms, swizzle: 1024, TFLOPS: 64.08 
 out_tf32(...+stage2+dsmem+swizzle): ['-47.060657', '-4.6654434'], time:8.009862ms, swizzle: 1024, TFLOPS: 68.63 (+0.34%)
              out_tf32(cublas+tf32): ['-47.060657', '-4.6654434'], time:6.366086ms, swizzle: NOOP, TFLOPS: 86.36 (+25.82%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=16384, N=8192, K=4096
                  out_f32x4(t8x8sk): ['-75.354446', '-55.499485'], time:28.24409ms, swizzle: NOOP, TFLOPS: 38.93 (+0.00%)
                 out_f32x4(t8x8bcf): ['-75.354446', '-55.499485'], time:26.64139ms, swizzle: NOOP, TFLOPS: 41.27 (+6.02%)
                out_f32x4(t8x8dbuf): ['-75.354446', '-55.499485'], time:26.63242ms, swizzle: NOOP, TFLOPS: 41.28 (+0.03%)
           out_f32x4(t8x8dbufasync): ['-75.354446', '-55.499485'], time:26.25021ms, swizzle: NOOP, TFLOPS: 41.89 (+1.46%)
             out_f32x4(k16t8x4dbuf): ['-75.354446', '-55.499485'], time:32.84521ms, swizzle: NOOP, TFLOPS: 33.48 
        out_f32x4(k16t8x4dbufasync): ['-75.354446', '-55.499485'], time:36.87822ms, swizzle: NOOP, TFLOPS: 29.81 
              out_f32x4(k168x8dbuf): ['-75.354446', '-55.499485'], time:26.30395ms, swizzle: NOOP, TFLOPS: 41.80 
         out_f32x4(k168x8dbufasync): ['-75.354446', '-55.499485'], time:25.96173ms, swizzle: NOOP, TFLOPS: 42.35 (+1.11%)
             out_f32x4(k168x16dbuf): ['-75.354446', '-55.499485'], time:33.67555ms, swizzle: NOOP, TFLOPS: 32.65 
        out_f32x4(k168x16dbufasync): ['-75.354446', '-55.499485'], time:31.36978ms, swizzle: NOOP, TFLOPS: 35.05 
                    out_f32(cublas): ['-75.354446', '-55.499485'], time:20.76914ms, swizzle: NOOP, TFLOPS: 52.94 (+25.00%)
                         out_f32_th: ['-75.354446', '-55.499485'], time:20.23167ms, swizzle: NOOP, TFLOPS: 54.35 (+2.66%)
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['-75.353981', '-55.498649'], time:18.30868ms, swizzle: NOOP, TFLOPS: 60.05 (+10.50%)
    out_tf32(mma2x4+warp2x4+stage2): ['-75.353981', '-55.498649'], time:16.16692ms, swizzle: NOOP, TFLOPS: 68.01 (+13.25%)
  out_tf32(mma2x4+...+stage3+dsmem): ['-75.353981', '-55.498649'], time:17.26713ms, swizzle: NOOP, TFLOPS: 63.68 
  out_tf32(mma2x4+...+stage2+dsmem): ['-75.353981', '-55.498649'], time:16.43080ms, swizzle: NOOP, TFLOPS: 66.92 
out_tf32(mma2x4+...+stage3+swizzle): ['-75.353981', '-55.498649'], time:18.26791ms, swizzle: 1024, TFLOPS: 60.19 
out_tf32(mma2x4+...+stage2+swizzle): ['-75.353981', '-55.498649'], time:16.33546ms, swizzle: 1024, TFLOPS: 67.31 
 out_tf32(...+stage3+dsmem+swizzle): ['-75.353981', '-55.498649'], time:17.09074ms, swizzle: 1024, TFLOPS: 64.33 
 out_tf32(...+stage2+dsmem+swizzle): ['-75.353981', '-55.498649'], time:16.03906ms, swizzle: 1024, TFLOPS: 68.55 (+0.80%)
              out_tf32(cublas+tf32): ['-75.353981', '-55.498649'], time:12.53089ms, swizzle: NOOP, TFLOPS: 87.74 (+28.00%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=16384, N=8192, K=8192
                  out_f32x4(t8x8sk): ['-16.729681', '-50.358272'], time:55.99205ms, swizzle: NOOP, TFLOPS: 39.27 (+0.00%)
                 out_f32x4(t8x8bcf): ['-16.729681', '-50.358272'], time:53.17327ms, swizzle: NOOP, TFLOPS: 41.36 (+5.30%)
                out_f32x4(t8x8dbuf): ['-16.729681', '-50.358272'], time:52.89652ms, swizzle: NOOP, TFLOPS: 41.57 (+0.52%)
           out_f32x4(t8x8dbufasync): ['-16.729681', '-50.358272'], time:52.42714ms, swizzle: NOOP, TFLOPS: 41.94 (+0.90%)
             out_f32x4(k16t8x4dbuf): ['-16.729681', '-50.358272'], time:65.91427ms, swizzle: NOOP, TFLOPS: 33.36 
        out_f32x4(k16t8x4dbufasync): ['-16.729681', '-50.358272'], time:73.20423ms, swizzle: NOOP, TFLOPS: 30.04 
              out_f32x4(k168x8dbuf): ['-16.729681', '-50.358272'], time:51.68914ms, swizzle: NOOP, TFLOPS: 42.54 (+1.43%)
         out_f32x4(k168x8dbufasync): ['-16.729681', '-50.358272'], time:51.89125ms, swizzle: NOOP, TFLOPS: 42.38 
             out_f32x4(k168x16dbuf): ['-16.729681', '-50.358272'], time:66.85421ms, swizzle: NOOP, TFLOPS: 32.89 
        out_f32x4(k168x16dbufasync): ['-16.729681', '-50.358272'], time:62.23177ms, swizzle: NOOP, TFLOPS: 35.34 
                    out_f32(cublas): ['-16.729681', '-50.358272'], time:41.07894ms, swizzle: NOOP, TFLOPS: 53.53 (+25.83%)
                         out_f32_th: ['-16.729681', '-50.358272'], time:40.48686ms, swizzle: NOOP, TFLOPS: 54.31 (+1.46%)
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['-16.728071', '-50.355793'], time:36.71095ms, swizzle: NOOP, TFLOPS: 59.90 (+10.29%)
    out_tf32(mma2x4+warp2x4+stage2): ['-16.728071', '-50.355793'], time:33.19857ms, swizzle: NOOP, TFLOPS: 66.24 (+10.58%)
  out_tf32(mma2x4+...+stage3+dsmem): ['-16.728071', '-50.355793'], time:35.75634ms, swizzle: NOOP, TFLOPS: 61.50 
  out_tf32(mma2x4+...+stage2+dsmem): ['-16.728071', '-50.355793'], time:33.22067ms, swizzle: NOOP, TFLOPS: 66.19 
out_tf32(mma2x4+...+stage3+swizzle): ['-16.728071', '-50.355793'], time:36.21408ms, swizzle: 1024, TFLOPS: 60.72 
out_tf32(mma2x4+...+stage2+swizzle): ['-16.728071', '-50.355793'], time:32.49535ms, swizzle: 1024, TFLOPS: 67.67 (+2.16%)
 out_tf32(...+stage3+dsmem+swizzle): ['-16.728071', '-50.355793'], time:34.46302ms, swizzle: 1024, TFLOPS: 63.81 
 out_tf32(...+stage2+dsmem+swizzle): ['-16.728071', '-50.355793'], time:32.46705ms, swizzle: 1024, TFLOPS: 67.73 (+0.09%)
              out_tf32(cublas+tf32): ['-16.728071', '-50.355793'], time:24.95450ms, swizzle: NOOP, TFLOPS: 88.12 (+30.10%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=16384, N=16384, K=2048
                  out_f32x4(t8x8sk): ['-47.060710', '-4.6656251'], time:29.20765ms, swizzle: NOOP, TFLOPS: 37.64 (+0.00%)
                 out_f32x4(t8x8bcf): ['-47.060710', '-4.6656251'], time:26.66914ms, swizzle: NOOP, TFLOPS: 41.23 (+9.52%)
                out_f32x4(t8x8dbuf): ['-47.060710', '-4.6656251'], time:26.94776ms, swizzle: NOOP, TFLOPS: 40.80 
           out_f32x4(t8x8dbufasync): ['-47.060710', '-4.6656251'], time:27.04577ms, swizzle: NOOP, TFLOPS: 40.65 
             out_f32x4(k16t8x4dbuf): ['-47.060710', '-4.6656251'], time:32.32545ms, swizzle: NOOP, TFLOPS: 34.01 
        out_f32x4(k16t8x4dbufasync): ['-47.060710', '-4.6656251'], time:37.94088ms, swizzle: NOOP, TFLOPS: 28.98 
              out_f32x4(k168x8dbuf): ['-47.060710', '-4.6656251'], time:26.69541ms, swizzle: NOOP, TFLOPS: 41.19 
         out_f32x4(k168x8dbufasync): ['-47.060710', '-4.6656251'], time:26.52416ms, swizzle: NOOP, TFLOPS: 41.45 (+0.55%)
             out_f32x4(k168x16dbuf): ['-47.060710', '-4.6656251'], time:34.01968ms, swizzle: NOOP, TFLOPS: 32.32 
        out_f32x4(k168x16dbufasync): ['-47.060710', '-4.6656251'], time:31.71181ms, swizzle: NOOP, TFLOPS: 34.67 
                    out_f32(cublas): ['-47.060710', '-4.6656251'], time:20.25883ms, swizzle: NOOP, TFLOPS: 54.27 (+30.93%)
                         out_f32_th: ['-47.060710', '-4.6656251'], time:20.32036ms, swizzle: NOOP, TFLOPS: 54.11 
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['-47.060657', '-4.6654434'], time:20.50101ms, swizzle: NOOP, TFLOPS: 53.63 
    out_tf32(mma2x4+warp2x4+stage2): ['-47.060657', '-4.6654434'], time:16.39070ms, swizzle: NOOP, TFLOPS: 67.08 (+23.60%)
  out_tf32(mma2x4+...+stage3+dsmem): ['-47.060657', '-4.6654434'], time:20.43299ms, swizzle: NOOP, TFLOPS: 53.81 
  out_tf32(mma2x4+...+stage2+dsmem): ['-47.060657', '-4.6654434'], time:16.86718ms, swizzle: NOOP, TFLOPS: 65.19 
out_tf32(mma2x4+...+stage3+swizzle): ['-47.060657', '-4.6654434'], time:18.05207ms, swizzle: 2048, TFLOPS: 60.91 
out_tf32(mma2x4+...+stage2+swizzle): ['-47.060657', '-4.6654434'], time:16.00086ms, swizzle: 2048, TFLOPS: 68.72 (+2.44%)
 out_tf32(...+stage3+dsmem+swizzle): ['-47.060657', '-4.6654434'], time:16.87958ms, swizzle: 2048, TFLOPS: 65.14 
 out_tf32(...+stage2+dsmem+swizzle): ['-47.060657', '-4.6654434'], time:15.81218ms, swizzle: 2048, TFLOPS: 69.54 (+1.19%)
              out_tf32(cublas+tf32): ['-47.060657', '-4.6654434'], time:12.69874ms, swizzle: NOOP, TFLOPS: 86.58 (+24.52%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=16384, N=16384, K=4096
                  out_f32x4(t8x8sk): ['-75.354446', '-55.499485'], time:57.62410ms, swizzle: NOOP, TFLOPS: 38.16 (+0.00%)
                 out_f32x4(t8x8bcf): ['-75.354446', '-55.499485'], time:54.61454ms, swizzle: NOOP, TFLOPS: 40.26 (+5.51%)
                out_f32x4(t8x8dbuf): ['-75.354446', '-55.499485'], time:54.20265ms, swizzle: NOOP, TFLOPS: 40.57 (+0.76%)
           out_f32x4(t8x8dbufasync): ['-75.354446', '-55.499485'], time:53.89254ms, swizzle: NOOP, TFLOPS: 40.80 (+0.58%)
             out_f32x4(k16t8x4dbuf): ['-75.354446', '-55.499485'], time:66.09683ms, swizzle: NOOP, TFLOPS: 33.27 
        out_f32x4(k16t8x4dbufasync): ['-75.354446', '-55.499485'], time:74.76580ms, swizzle: NOOP, TFLOPS: 29.41 
              out_f32x4(k168x8dbuf): ['-75.354446', '-55.499485'], time:53.12254ms, swizzle: NOOP, TFLOPS: 41.40 (+1.45%)
         out_f32x4(k168x8dbufasync): ['-75.354446', '-55.499485'], time:53.05356ms, swizzle: NOOP, TFLOPS: 41.45 (+0.13%)
             out_f32x4(k168x16dbuf): ['-75.354446', '-55.499485'], time:67.35692ms, swizzle: NOOP, TFLOPS: 32.65 
        out_f32x4(k168x16dbufasync): ['-75.354446', '-55.499485'], time:62.84084ms, swizzle: NOOP, TFLOPS: 34.99 
                    out_f32(cublas): ['-75.354446', '-55.499485'], time:40.55628ms, swizzle: NOOP, TFLOPS: 54.22 (+30.81%)
                         out_f32_th: ['-75.354446', '-55.499485'], time:40.47796ms, swizzle: NOOP, TFLOPS: 54.33 (+0.19%)
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['-75.353981', '-55.498649'], time:39.75238ms, swizzle: NOOP, TFLOPS: 55.32 (+1.83%)
    out_tf32(mma2x4+warp2x4+stage2): ['-75.353981', '-55.498649'], time:32.99241ms, swizzle: NOOP, TFLOPS: 66.65 (+20.49%)
  out_tf32(mma2x4+...+stage3+dsmem): ['-75.353981', '-55.498649'], time:39.66310ms, swizzle: NOOP, TFLOPS: 55.44 
  out_tf32(mma2x4+...+stage2+dsmem): ['-75.353981', '-55.498649'], time:32.68809ms, swizzle: NOOP, TFLOPS: 67.27 (+0.93%)
out_tf32(mma2x4+...+stage3+swizzle): ['-75.353981', '-55.498649'], time:35.53545ms, swizzle: 2048, TFLOPS: 61.88 
out_tf32(mma2x4+...+stage2+swizzle): ['-75.353981', '-55.498649'], time:31.71443ms, swizzle: 2048, TFLOPS: 69.34 (+3.07%)
 out_tf32(...+stage3+dsmem+swizzle): ['-75.353981', '-55.498649'], time:33.64064ms, swizzle: 2048, TFLOPS: 65.37 
 out_tf32(...+stage2+dsmem+swizzle): ['-75.353981', '-55.498649'], time:31.82904ms, swizzle: 2048, TFLOPS: 69.09 
              out_tf32(cublas+tf32): ['-75.353981', '-55.498649'], time:25.01533ms, swizzle: NOOP, TFLOPS: 87.91 (+26.78%)
----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------
                                                       M=16384, N=16384, K=8192
                  out_f32x4(t8x8sk): ['-16.729681', '-50.358272'], time:114.8013ms, swizzle: NOOP, TFLOPS: 38.31 (+0.00%)
                 out_f32x4(t8x8bcf): ['-16.729681', '-50.358272'], time:108.9582ms, swizzle: NOOP, TFLOPS: 40.36 (+5.36%)
                out_f32x4(t8x8dbuf): ['-16.729681', '-50.358272'], time:108.0725ms, swizzle: NOOP, TFLOPS: 40.70 (+0.82%)
           out_f32x4(t8x8dbufasync): ['-16.729681', '-50.358272'], time:106.9746ms, swizzle: NOOP, TFLOPS: 41.11 (+1.03%)
             out_f32x4(k16t8x4dbuf): ['-16.729681', '-50.358272'], time:138.7217ms, swizzle: NOOP, TFLOPS: 31.70 
        out_f32x4(k16t8x4dbufasync): ['-16.729681', '-50.358272'], time:148.4906ms, swizzle: NOOP, TFLOPS: 29.62 
              out_f32x4(k168x8dbuf): ['-16.729681', '-50.358272'], time:105.8515ms, swizzle: NOOP, TFLOPS: 41.55 (+1.06%)
         out_f32x4(k168x8dbufasync): ['-16.729681', '-50.358272'], time:106.0791ms, swizzle: NOOP, TFLOPS: 41.46 
             out_f32x4(k168x16dbuf): ['-16.729681', '-50.358272'], time:134.4140ms, swizzle: NOOP, TFLOPS: 32.72 
        out_f32x4(k168x16dbufasync): ['-16.729681', '-50.358272'], time:125.4974ms, swizzle: NOOP, TFLOPS: 35.04 
                    out_f32(cublas): ['-16.729681', '-50.358272'], time:81.04460ms, swizzle: NOOP, TFLOPS: 54.27 (+30.61%)
                         out_f32_th: ['-16.729681', '-50.358272'], time:81.02524ms, swizzle: NOOP, TFLOPS: 54.28 (+0.02%)
--------------------------------------------------------------WMMA----------------------------------------------------------------
    out_tf32(mma2x4+warp2x4+stage3): ['-16.728071', '-50.355793'], time:78.23324ms, swizzle: NOOP, TFLOPS: 56.22 (+3.57%)
    out_tf32(mma2x4+warp2x4+stage2): ['-16.728071', '-50.355793'], time:77.08487ms, swizzle: NOOP, TFLOPS: 57.05 (+1.49%)
  out_tf32(mma2x4+...+stage3+dsmem): ['-16.728071', '-50.355793'], time:78.11026ms, swizzle: NOOP, TFLOPS: 56.31 
  out_tf32(mma2x4+...+stage2+dsmem): ['-16.728071', '-50.355793'], time:77.33864ms, swizzle: NOOP, TFLOPS: 56.87 
out_tf32(mma2x4+...+stage3+swizzle): ['-16.728071', '-50.355793'], time:71.27919ms, swizzle: 2048, TFLOPS: 61.70 (+8.14%)
out_tf32(mma2x4+...+stage2+swizzle): ['-16.728071', '-50.355793'], time:64.17446ms, swizzle: 2048, TFLOPS: 68.53 (+11.07%)
 out_tf32(...+stage3+dsmem+swizzle): ['-16.728071', '-50.355793'], time:67.82913ms, swizzle: 2048, TFLOPS: 64.84 
 out_tf32(...+stage2+dsmem+swizzle): ['-16.728071', '-50.355793'], time:63.59438ms, swizzle: 2048, TFLOPS: 69.16 (+0.91%)
              out_tf32(cublas+tf32): ['-16.728071', '-50.355793'], time:49.69983ms, swizzle: NOOP, TFLOPS: 88.49 (+27.96%)
----------------------------------------------------------------------------------------------------------------------------------
```
