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
    # cC: cute.Tensor,
    layout_tv: cute.Layout,
    block_layout: cute.Shape
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()
    gdim, _, _ = cute.arch.grid_dim()

    block_coord = ((None, None), bidx)
    thread_coord = (tidx, (None, None))

    blkA = gA[block_coord]
    blkB = gB[block_coord]
    blkC = gC[block_coord]
    blkCoords = gC[block_coord]

    copy_atom_load = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gA.element_type)
    copy_atom_store = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gC.element_type)

    tiled_copy_A = cute.make_tiled_copy(copy_atom_load, layout_tv, block_layout)
    tiled_copy_B = cute.make_tiled_copy(copy_atom_load, layout_tv, block_layout)
    tiled_copy_C = cute.make_tiled_copy(copy_atom_store, layout_tv, block_layout)

    thr_copy_A = tiled_copy_A.get_slice(tidx)
    thr_copy_B = tiled_copy_B.get_slice(tidx)
    thr_copy_C = tiled_copy_C.get_slice(tidx)

    thrA = thr_copy_A.partition_S(blkA)
    thrB = thr_copy_B.partition_S(blkB)
    thrC = thr_copy_C.partition_S(blkC)

    regA = cute.make_fragment_like(thrA)
    regB = cute.make_fragment_like(thrB)
    regC = cute.make_fragment_like(thrC)

    cute.copy(copy_atom_load, thrA, regA)
    cute.copy(copy_atom_load, thrB, regB)

    result = regA.load() + regB.load()
    regC.store(result)

    cute.copy(copy_atom_store, regC, thrC)

    
@cute.jit
def element_wise_add(
    A: cute.Tensor,
    B: cute.Tensor,
    C: cute.Tensor
):

    thread_layout = cute.make_layout((8, 32), stride=(32, 1))
    val_layout = cute.make_layout((4, 8), stride=(8, 1))

    block_layout, layout_tv = cute.make_layout_tv(thread_layout, val_layout)
    gA = cute.zipped_divide(A, block_layout)
    gB = cute.zipped_divide(B, block_layout)
    gC = cute.zipped_divide(C, block_layout)

    print(f"block layout: {block_layout}")
    print(f"gA shape: {gA.layout}")

    # idC = cute.make_identity_tensor(C.shape)
    # cC = cute.zipped_divide(idC, block_layout)

    num_blocks = cute.size(gA, mode=[1])
    num_threads = cute.size(thread_layout)

    print(f"num_blocks: {num_blocks}, num_threads: {num_threads}")
    print(f"block layout: {block_layout}")

    element_wise_add_kernel(gA, gB, gC, layout_tv, block_layout).launch(
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
    