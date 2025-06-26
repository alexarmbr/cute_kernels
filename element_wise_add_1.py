import torch
import os
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass.cute.runtime import from_dlpack
from functools import partial
from benchmark import benchmark

os.environ["CUTE_DSL_ARCH"] = "sm_90a"

@cute.kernel
def naive_element_wise_add_kernel(
    gA: cute.Tensor,
    gB: cute.Tensor,
    gC: cute.Tensor,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()
    i = bidx * bdim + tidx
    gC[i] = gA[i] + gB[i]
        

@cute.jit
def element_wise_add(
    gA: cute.Tensor,
    gB: cute.Tensor,
    gC: cute.Tensor
):
    NUM_THREADS = 256
    m,n = gA.shape
    NUM_BLOCKS = (m * n) // NUM_THREADS
    naive_element_wise_add_kernel(gA, gB, gC).launch(
        grid = (NUM_BLOCKS, 1, 1),
        block = [NUM_THREADS, 1, 1],
    )

    
if __name__ == "__main__":
    M,N = 2048, 2048

    A = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    C = torch.zeros(M, N, device="cuda", dtype=torch.bfloat16)

    gA = from_dlpack(A)
    gB = from_dlpack(B)
    gC = from_dlpack(C)

    element_wise_add_ = cute.compile(element_wise_add, gA, gB, gC)

    element_wise_add_(gA, gB, gC)

    assert torch.allclose(C, A + B)

    avg_ms = benchmark(partial(element_wise_add_, gA, gB, gC), num_warmups=5, num_iterations=100)
    bytes_per_sec = (3 * A.numel() * 2) / (avg_ms / 1e3)
    gb_per_sec = bytes_per_sec / 1e9
    print(f"achieved memory bandwidth {gb_per_sec:.2f} GB/s")
    