from torch.utils.cpp_extension import load

mod = load(
    name="test",
    sources=["test.cu"],
    extra_cuda_cflags=[
        "-gencode=arch=compute_90,code=compute_90",
        "-lcuda",
    ],
    extra_cflags=["-std=c++17"],
    extra_ldflags=["-Wl,--no-as-needed -lcuda"],
    verbose=True
)

mod.launch()

