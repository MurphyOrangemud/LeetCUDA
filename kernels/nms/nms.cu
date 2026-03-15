#include <algorithm>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <vector>

#define WARP_SIZE 32
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define CONST_FLOAT4(value) (reinterpret_cast<const float4 *>(&(value))[0])

__global__ void nms_kernel_origin(const float *boxes, const float *scores,
                                  int *keep, int num_boxes,
                                  float iou_threshold) {
  const int threadsPerBlock = blockDim.x;
  const int threadId = threadIdx.x;
  const int blockId = blockIdx.x;
  const int idx = blockId * threadsPerBlock + threadId;

  if (idx >= num_boxes)
    return;

  float x1 = boxes[idx * 4 + 0];
  float y1 = boxes[idx * 4 + 1];
  float x2 = boxes[idx * 4 + 2];
  float y2 = boxes[idx * 4 + 3];
  int suppressed = 0;

  for (int i = 0; i < idx; ++i) {
    if (keep[i] == 0)
      continue;

    float x1_i = boxes[i * 4 + 0];
    float y1_i = boxes[i * 4 + 1];
    float x2_i = boxes[i * 4 + 2];
    float y2_i = boxes[i * 4 + 3];

    float inter_x1 = max(x1, x1_i);
    float inter_y1 = max(y1, y1_i);
    float inter_x2 = min(x2, x2_i);
    float inter_y2 = min(y2, y2_i);
    float inter_w = max(0.0f, inter_x2 - inter_x1);
    float inter_h = max(0.0f, inter_y2 - inter_y1);
    float inter_area = inter_w * inter_h;

    float area = (x2 - x1) * (y2 - y1);
    float area_i = (x2_i - x1_i) * (y2_i - y1_i);
    float iou = inter_area / (area + area_i - inter_area);

    if (iou > iou_threshold) {
      keep[idx] = 0;
      return;
    }
  }
  keep[idx] = 1;
  return;
}

__global__ void nms_kernel_optim_1d(const float *boxes, const float *scores,
                                    int *keep, int num_boxes,
                                    float iou_threshold) {
  const int idx = threadIdx.x + blockIdx.x * blockDim.x;

  __shared__ float4 sbox_x[WARP_SIZE];
  __shared__ float4 sbox_y[WARP_SIZE];

  sbox_x[threadIdx.x] = CONST_FLOAT4(boxes[idx * 4]);
  __syncthreads();

  for (int i = threadIdx.x; i < num_boxes; i += blockDim.x) {
    sbox_y[i] = CONST_FLOAT4(boxes[i * 4]);
    __syncthreads();
#pragma unroll
    for (int mask = WARP_SIZE >> 1; mask >= 1; mask >>= 1) {
      float4 reg_x = sbox_x[threadIdx.x];

      float x1 = reg_x.x;
      float y1 = reg_x.y;
      float x2 = reg_x.z;
      float y2 = reg_x.w;

      float x1_i = __shfl_xor_sync(0xffffffff, x1, mask);
      float y1_i = __shfl_xor_sync(0xffffffff, y1, mask);
      float x2_i = __shfl_xor_sync(0xffffffff, x2, mask);
      float y2_i = __shfl_xor_sync(0xffffffff, y2, mask);

      float inter_x1 = max(x1, x1_i);
      float inter_y1 = max(y1, y1_i);
      float inter_x2 = min(x2, x2_i);
      float inter_y2 = min(y2, y2_i);
      float inter_w = max(0.0f, inter_x2 - inter_x1);
      float inter_h = max(0.0f, inter_y2 - inter_y1);
      float inter_area = inter_w * inter_h;

      float area = (x2 - x1) * (y2 - y1);
      float area_i = (x2_i - x1_i) * (y2_i - y1_i);
      float iou = inter_area / (area + area_i - inter_area);

      if (iou > iou_threshold && i < idx && keep[i]) {
        keep[idx] = 0;
      }
    }
  }
}

__global__ void nms_kernel_optim_2d(const float *boxes, const float *scores,
                                    int *keep, int num_boxes,
                                    float iou_threshold) {
  const int idx_x = threadIdx.x + blockIdx.x * blockDim.x;
  const int idx_y = threadIdx.y + blockIdx.y * blockDim.y;

  if (idx_x < num_boxes && idx_y < idx_x) {
    float4 reg = CONST_FLOAT4(boxes[idx_x * 4]);
    float4 reg_i = CONST_FLOAT4(boxes[idx_y * 4]);

    float inter_x1 = max(reg.x, reg_i.x);
    float inter_y1 = max(reg.y, reg_i.y);
    float inter_x2 = min(reg.z, reg_i.z);
    float inter_y2 = min(reg.w, reg_i.w);
    float inter_w = max(0.0f, inter_x2 - inter_x1);
    float inter_h = max(0.0f, inter_y2 - inter_y1);
    float inter_area = inter_w * inter_h;

    float area = (reg.z - reg.x) * (reg.w - reg.y);
    float area_i = (reg_i.z - reg_i.x) * (reg_i.w - reg_i.y);
    float iou = inter_area / (area + area_i - inter_area);

    if (iou > iou_threshold && keep[idx_y]) {
      keep[idx_x] = 0;
    }
  }
}

#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func)                                   \
  m.def(STRINGFY(func), &func, STRINGFY(func));

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

#define TORCH_BINDING_NMS_1D(name)                                             \
  torch::Tensor nms_##name(torch::Tensor boxes, torch::Tensor scores,          \
                           float iou_threshold) {                              \
    CHECK_TORCH_TENSOR_DTYPE(boxes, torch::kFloat32);                          \
    CHECK_TORCH_TENSOR_DTYPE(scores, torch::kFloat32);                         \
    const int num_boxes = boxes.size(0);                                       \
    auto toption =                                                             \
        torch::TensorOptions().dtype(torch::kInt32).device(boxes.device());    \
    auto keep = torch::empty({boxes.size(0)}, toption);                        \
    dim3 block(WARP_SIZE);                                                     \
    dim3 grid((num_boxes + WARP_SIZE - 1) / WARP_SIZE);                        \
    /* sort boxes by scores */                                                 \
    auto order_t = std::get<1>(                                                \
        scores.sort(/*stable=*/true, /*dim=*/0, /* descending=*/true));        \
    auto boxes_sorted = boxes.index_select(0, order_t).contiguous();           \
                                                                               \
    nms_kernel_##name<<<grid, block>>>(                                        \
        reinterpret_cast<float *>(boxes_sorted.data_ptr()),                    \
        reinterpret_cast<float *>(scores.data_ptr()),                          \
        reinterpret_cast<int *>(keep.data_ptr()), num_boxes, iou_threshold);   \
    auto keep_cpu = keep.to(torch::kCPU);                                      \
                                                                               \
    std::vector<int> keep_indices;                                             \
    auto keep_accessor = keep_cpu.accessor<int, 1>();                          \
    for (int i = 0; i < num_boxes; ++i) {                                      \
      if (keep_accessor[i] == 1) {                                             \
        keep_indices.push_back(i);                                             \
      }                                                                        \
    }                                                                          \
    return torch::tensor(keep_indices,                                         \
                         torch::TensorOptions().dtype(torch::kInt32));         \
  }

TORCH_BINDING_NMS_1D(origin);
TORCH_BINDING_NMS_1D(optim_1d);

torch::Tensor nms_2d(torch::Tensor boxes, torch::Tensor scores,
                     float iou_threshold) {
  CHECK_TORCH_TENSOR_DTYPE(boxes, torch::kFloat32);
  CHECK_TORCH_TENSOR_DTYPE(scores, torch::kFloat32);
  const int num_boxes = boxes.size(0);
  auto toption =
      torch::TensorOptions().dtype(torch::kInt32).device(boxes.device());
  auto keep = torch::ones({boxes.size(0)}, toption);
  dim3 block(WARP_SIZE, WARP_SIZE);
  dim3 grid((num_boxes + WARP_SIZE - 1) / WARP_SIZE,
            (num_boxes + WARP_SIZE - 1) / WARP_SIZE);
  auto order_t = std::get<1>(
      scores.sort(/*stable=*/true, /*dim=*/0, /* descending=*/true));
  auto boxes_sorted = boxes.index_select(0, order_t).contiguous();

  nms_kernel_optim_2d<<<grid, block>>>(
      reinterpret_cast<float *>(boxes_sorted.data_ptr()),
      reinterpret_cast<float *>(scores.data_ptr()),
      reinterpret_cast<int *>(keep.data_ptr()), num_boxes, iou_threshold);

  auto keep_cpu = keep.to(torch::kCPU);

  std::vector<int> keep_indices;
  auto keep_accessor = keep_cpu.accessor<int, 1>();
  for (int i = 0; i < num_boxes; ++i) {
    if (keep_accessor[i] == 1) {
      keep_indices.push_back(i);
    }
  }
  return torch::tensor(keep_indices,
                       torch::TensorOptions().dtype(torch::kInt32));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(nms_origin)
  TORCH_BINDING_COMMON_EXTENSION(nms_optim_1d)
  TORCH_BINDING_COMMON_EXTENSION(nms_2d)
}
