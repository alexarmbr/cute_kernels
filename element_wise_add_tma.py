import torch
import os
import cutlass.cute as cute
import cutlass.cute.nvgpu.cpasync as cpasync
import cutlass.utils as utils
from cutlass.cute.runtime import from_dlpack
from functools import partial
from benchmark import benchmark

os.environ["CUTE_DSL_ARCH"] = "sm_90a"

@cute.kernel
def element_wise_add_tma_kernel(
    tma_atom_a: cute.CopyAtom,
    tma_tensor_a: cute.Tensor,
    tma_atom_b: cute.CopyAtom,
    tma_tensor_b: cute.Tensor,
    gC: cute.Tensor,
    smem_layout_a: cute.Layout,
    smem_layout_b: cute.Layout,
    tile_shape: cute.Shape,
):
    """
    Element-wise add kernel using TMA to load data to shared memory
    """
    # Get thread and block coordinates
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    
    # Allocate shared memory
    smem = utils.SmemAllocator()
    
    @cute.struct
    class SharedStorage:
        sA: cute.struct.MemRange[cute.BFloat16, cute.cosize(smem_layout_a)]
        sB: cute.struct.MemRange[cute.BFloat16, cute.cosize(smem_layout_b)]
        barrier: cute.struct.MemRange[cute.Int64, 1]
    
    storage = smem.allocate(SharedStorage)
    
    # Create shared memory tensors
    sA = storage.sA.get_tensor(smem_layout_a)
    sB = storage.sB.get_tensor(smem_layout_b)
    barrier_ptr = storage.barrier.data_ptr()
    
    # Initialize barrier for TMA synchronization
    if tidx == 0:
        cute.arch.mbarrier_init_arrive_cnt(barrier_ptr, 1)
    cute.arch.mbarrier_init_fence()
    
    # Compute block coordinates for this tile
    block_coord = (bidx, bidy)
    
    # TMA load A and B to shared memory
    if tidx == 0:
        # Load tile A
        cute.copy(
            tma_atom_a,
            tma_tensor_a[block_coord],
            sA,
            tma_bar_ptr=barrier_ptr
        )
        # Load tile B  
        cute.copy(
            tma_atom_b,
            tma_tensor_b[block_coord],
            sB,
            tma_bar_ptr=barrier_ptr
        )
    
    # Wait for TMA loads to complete
    cute.arch.mbarrier_wait(barrier_ptr, 0)
    
    # Now perform element-wise addition using data from shared memory
    # Create copy atoms for loading from shared memory and storing to global memory
    copy_atom_load = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cute.BFloat16)
    copy_atom_store = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cute.BFloat16)
    
    # Create thread layouts for processing the tile
    thread_layout = cute.make_layout((8, 32), stride=(32, 1))
    val_layout = cute.make_layout((4, 1), stride=(1, 1))
    
    tiled_copy_load = cute.make_tiled_copy(copy_atom_load, thread_layout, val_layout)
    tiled_copy_store = cute.make_tiled_copy(copy_atom_store, thread_layout, val_layout)
    
    thr_copy_load = tiled_copy_load.get_slice(tidx)
    thr_copy_store = tiled_copy_store.get_slice(tidx)
    
    # Partition shared memory tensors for this thread
    thrA = thr_copy_load.partition_S(sA)
    thrB = thr_copy_load.partition_S(sB)
    
    # Partition global output tensor for this thread
    gC_tile = gC[((None, None), *block_coord)]
    thrC = thr_copy_store.partition_D(gC_tile)
    
    # Create register fragments
    regA = cute.make_fragment_like(thrA)
    regB = cute.make_fragment_like(thrB)
    regC = cute.make_fragment_like(thrC)
    
    # Load from shared memory to registers
    cute.copy(copy_atom_load, thrA, regA)
    cute.copy(copy_atom_load, thrB, regB)
    
    # Perform element-wise addition
    result = regA.load() + regB.load()
    regC.store(result)
    
    # Store result to global memory
    cute.copy(copy_atom_store, regC, thrC)


@cute.jit
def element_wise_add_tma(
    A: cute.Tensor,
    B: cute.Tensor, 
    C: cute.Tensor
):
    """
    Element-wise addition using TMA for loading data to shared memory
    """
    # Define tile shape - each tile processes 256x256 elements
    tile_shape = (256, 256)
    
    # Create shared memory layouts for the tiles
    smem_layout_a = cute.make_layout(tile_shape, stride=(256, 1))
    smem_layout_b = cute.make_layout(tile_shape, stride=(256, 1))
    
    # Create TMA copy operations for loading A and B
    # https://docs.nvidia.com/cutlass/media/docs/pythonDSL/cute_dsl_api/cute_nvgpu_cpasync.html#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp
    tma_load_op = cpasync.CopyBulkTensorTileG2SOp()
    
    # Create identity layout for the tile shape
    # cta_v_layout = cute.composition(
    #     A.layout, tile_shape
    # )

    # cta_layout = cute.zipped_divide(A.layout, tile_shape)

    # print(f"composition: {A.layout} o {tile_shape} = {cta_v_layout}")
    
    # Create TMA atoms and tensors for A and B using the correct signature
    tma_atom_a, tma_tensor_a = cpasync.make_tma_tile_atom(
        tma_load_op,
        A,
        smem_layout_a,
        (128, 128)
    )
    
    # tma_atom_b, tma_tensor_b = cpasync.make_tma_tile_atom(
    #     tma_load_op,
    #     B,
    #     smem_layout_b,
    #     tile_shape
    # )
    
    # # Compute grid dimensions
    # grid_m = cute.ceil_div(A.shape[0], tile_shape[0])
    # grid_n = cute.ceil_div(A.shape[1], tile_shape[1])
    
    # # Number of threads per block (256 threads)
    # num_threads = 256
    
    # print(f"Grid: ({grid_m}, {grid_n}), Threads: {num_threads}")
    # print(f"Tile shape: {tile_shape}")
    # print(f"Shared memory layout A: {smem_layout_a}")
    # print(f"Shared memory layout B: {smem_layout_b}")
    
    # Launch kernel
    # element_wise_add_tma_kernel(
    #     tma_atom_a,
    #     tma_tensor_a,
    #     tma_atom_b, 
    #     tma_tensor_b,
    #     C,
    #     smem_layout_a,
    #     smem_layout_b,
    #     tile_shape
    # ).launch(
    #     grid=[grid_m, grid_n, 1],
    #     block=[num_threads, 1, 1],
    #     smem=cute.struct.size_in_bytes(
    #         sA=cute.struct.MemRange[cute.BFloat16, cute.cosize(smem_layout_a)],
    #         sB=cute.struct.MemRange[cute.BFloat16, cute.cosize(smem_layout_b)],
    #         barrier=cute.struct.MemRange[cute.Int64, 1]
    #     )
    # )


if __name__ == "__main__":
    M, N = 8192, 8192
    
    A = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    C = torch.zeros(M, N, device="cuda", dtype=torch.bfloat16)
    
    gA = from_dlpack(A, assumed_align=16)
    gB = from_dlpack(B, assumed_align=16)
    gC = from_dlpack(C, assumed_align=16)
    
    element_wise_add_tma_ = cute.compile(element_wise_add_tma, gA, gB, gC)
    element_wise_add_tma_(gA, gB, gC)
    
    # assert torch.allclose(C, A + B, rtol=1e-3, atol=1e-3)
    # print("Correctness check passed!")
    
    # avg_ms = benchmark(partial(element_wise_add_tma_, gA, gB, gC), num_warmups=10, num_iterations=100)
    # bytes_per_sec = (3 * A.numel() * 2) / (avg_ms / 1e3)
    # gb_per_sec = bytes_per_sec / 1e9
    # print(f"Achieved memory bandwidth: {gb_per_sec:.2f} GB/s") 