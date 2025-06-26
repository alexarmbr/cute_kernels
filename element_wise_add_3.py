import torch
import os
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass.cute.runtime import from_dlpack
from functools import partial
from benchmark import benchmark

os.environ["CUTE_DSL_ARCH"] = "sm_90a"

# 4035.45 GB/s
@cute.kernel
def element_wise_add_kernel(
    gA: cute.Tensor,
    gB: cute.Tensor,
    gC: cute.Tensor,
    layout_tv: cute.Layout
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()
    gdim, _, _ = cute.arch.grid_dim()

    block_coord = ((None, None), bidx)
    thread_coord = (tidx, (None, None))
    
    gA_tile = gA[block_coord]
    gB_tile = gB[block_coord]
    gC_tile = gC[block_coord]

    tAgA = cute.composition(gA_tile, layout_tv)
    tAgB = cute.composition(gB_tile, layout_tv)
    tAgC = cute.composition(gC_tile, layout_tv)

    gA_thread = tAgA[thread_coord].load()
    gB_thread = tAgB[thread_coord].load()

    tAgC[thread_coord] = gA_thread + gB_thread

@cute.jit
def element_wise_add(
    gA: cute.Tensor,
    gB: cute.Tensor,
    gC: cute.Tensor
):

    thread_layout = cute.make_layout((8, 32), stride=(32, 1))
    val_layout = cute.make_layout((4, 8), stride=(8, 1))

    block_layout, layout_tv = cute.make_layout_tv(thread_layout, val_layout)
    gA = cute.zipped_divide(gA, block_layout)
    gB = cute.zipped_divide(gB, block_layout)
    gC = cute.zipped_divide(gC, block_layout)

    num_blocks = cute.size(gA, mode=[1])
    num_threads = cute.size(thread_layout)

    print(f"num_blocks: {num_blocks}, num_threads: {num_threads}")

    element_wise_add_kernel(gA, gB, gC, layout_tv).launch(
        grid=[num_blocks, 1, 1],
        block=[num_threads, 1, 1],
    )
    
if __name__ == "__main__":
    M,N = 8192, 8192

    A = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    C = torch.zeros(M, N, device="cuda", dtype=torch.bfloat16)

    gA = from_dlpack(A, assumed_align=16)
    gB = from_dlpack(B, assumed_align=16)
    gC = from_dlpack(C, assumed_align=16)

    element_wise_add_ = cute.compile(element_wise_add, gA, gB, gC)
    element_wise_add_(gA, gB, gC)

    assert torch.allclose(C, A + B)
    avg_ms = benchmark(partial(element_wise_add_, gA, gB, gC), num_warmups=10, num_iterations=100)
    bytes_per_sec = (3 * A.numel() * 2) / (avg_ms / 1e3)
    gb_per_sec = bytes_per_sec / 1e9
    print(f"achieved memory bandwidth {gb_per_sec:.2f} GB/s")
    