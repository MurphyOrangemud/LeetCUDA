import torch
from torch.utils.cpp_extension import load
import time

torch.set_grad_enabled(False)

# Load the CUDA kernel as a python module
lib = load(
    name="hist_lib",
    sources=["histogram.cu"],
    extra_cuda_cflags=[
        "-O3",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
    ],
    extra_cflags=["-std=c++17"],
)

def run_benchmark(
    perf_func: callable, 
    a: torch.Tensor,
    tag: str,
    warmup: int = 10,
    iters: int = 1000,
    show_all: bool = False,
):
    # warmup
    for i in range(warmup):
        perf_func(a)
    torch.cuda.synchronize()
    start = time.time()
    # iters
    for i in range(iters):
        perf_func(a)
    torch.cuda.synchronize()
    end = time.time()
    total_time = (end - start) * 1000
    mean_time = total_time / iters

    out_info = f"out_{tag}"
    print(f"{out_info:>18} time:{mean_time:.8f}ms")

print("-" * 80)
a = torch.tensor(list(range(10)) * 1000, dtype=torch.int32).cuda()
run_benchmark(lib.histogram_i32, a, "h_i32")
run_benchmark(lib.histogram_i32x4, a, "h_i32x4")
print("-" * 80)
a = torch.tensor([1] * 10000, dtype=torch.int32).cuda()
run_benchmark(lib.histogram_i32, a, "h_i32")
run_benchmark(lib.histogram_i32x4, a, "h_i32x4")
print("-" * 80)

print("-" * 80)
a = torch.tensor(list(range(10)) * 1024, dtype=torch.int32).cuda()
run_benchmark(lib.histogram_i32, a, "h_i32")
run_benchmark(lib.histogram_i32x4, a, "h_i32x4")
print("-" * 80)
a = torch.tensor([1] * 1024 * 10, dtype=torch.int32).cuda()
run_benchmark(lib.histogram_i32, a, "h_i32")
run_benchmark(lib.histogram_i32x4, a, "h_i32x4")
print("-" * 80)

print("-" * 80)
a = torch.tensor(list(range(10)) * 1024 * 1024, dtype=torch.int32).cuda()
run_benchmark(lib.histogram_i32, a, "h_i32")
run_benchmark(lib.histogram_i32x4, a, "h_i32x4")
print("-" * 80)
a = torch.tensor([1] * 1024 * 1024, dtype=torch.int32).cuda()
run_benchmark(lib.histogram_i32, a, "h_i32")
run_benchmark(lib.histogram_i32x4, a, "h_i32x4")
print("-" * 80)

print("-" * 80)
a = torch.tensor(list(range(10)) * 1000 * 1000, dtype=torch.int32).cuda()
run_benchmark(lib.histogram_i32, a, "h_i32")
run_benchmark(lib.histogram_i32x4, a, "h_i32x4")
print("-" * 80)
a = torch.tensor([1] * 1000 * 1000, dtype=torch.int32).cuda()
run_benchmark(lib.histogram_i32, a, "h_i32")
run_benchmark(lib.histogram_i32x4, a, "h_i32x4")
print("-" * 80)

"""
While there's no optimization, i32x4 kernels are expected to be a bit 
faster than i32 kernels because of vectorization.
The 
"""
