import torch
import os
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

os.environ["CUTE_DSL_ARCH"] = "sm_90a"

@cute.kernel
def local_tile_kernel(
    mA_mkl: cute.Tensor,
):
    # Define tile parameters as compile-time constants
    tile_shape_mnk = (64, 32, 32)
    tile_coord_mnkl = (0, 0, None, 0)
    
    # Print input information from within the kernel
    if cute.arch.thread_idx()[0] == 0 and cute.arch.block_idx()[0] == 0:
        cute.printf("=== Inside Kernel ===")
        cute.printf("mA_mkl: {}", mA_mkl.shape)
        cute.printf("tile_shape_mnk: {}", tile_shape_mnk)
        cute.printf("tile_coord_mnkl: {}", tile_coord_mnkl)
        cute.printf("proj: (1, None, 1)")
    
    # Call cute.local_tile with projection (1, None, 1)
    # This projection selects dimensions [0, 2] from tile_shape_mnk and tile_coord_mnkl
    # So effective tile_shape becomes (64, 32) and tile_coord becomes (0, None, 0)
    # this is confusing though because tile_coord_mnkl is (0, 0, None, 0) and proj is (1, None, 1)
    gA_mkl = cute.local_tile(
        mA_mkl, 
        (64, 32, 32), 
         (0, 0, None, 0), 
        proj=(1, None, 1)
    )


    # gA_mkl_alt = cute.local_tile(
    #     mA_mkl,
    #     cute.dice((1, None, 1), tile_shape_mnk),
    #     cute.dice((1, None, 1), tile_coord_mnkl),
    # )
    
    # alternatively
    # zipped_divide((256,128,2) , (64, 32)) -> ((64, 32), (4, 4, 2))
    # tiled_ga = cute.zipped_divide(mA_mkl, (64, 32))
    # gA_mkl_alt = tiled_ga[(None, None), (0, None, 0)]

    # alternatively
    gA_mkl_alt_2 = cute.local_tile(
        mA_mkl,
        (64, 32),
        (0, None, 0)
    )
    
    # Print output information from within the kernel
    if cute.arch.thread_idx()[0] == 0 and cute.arch.block_idx()[0] == 0:
        cute.printf("=== Output Information ===")
        cute.printf("gA_mkl: {}", gA_mkl.layout)
        cute.printf("gA_mkl_alt: {}", gA_mkl_alt.layout)
        cute.printf("gA_mkl_alt_2: {}", gA_mkl_alt_2.layout)

@cute.jit
def demonstrate_local_tile(
    mA_mkl: cute.Tensor,
):
    # Launch kernel with single thread to demonstrate local_tile
    local_tile_kernel(mA_mkl).launch(
        grid=(1, 1, 1),
        block=[1, 1, 1],
    )

if __name__ == "__main__":
    # Create tensor outside the function
    M, K, L = 256, 128, 2
    tensor_torch = torch.randn(M, K, L, device="cuda", dtype=torch.float16)
    mA_mkl = from_dlpack(tensor_torch)
    
    # Define tile parameters (for display purposes)
    tile_shape_mnk = (64, 32, 32)
    tile_coord_mnkl = (0, 0, None, 0)
    
    # Print host-side information
    print("=== Host-side Information ===")
    print(f"Input tensor shape (mA_mkl): {mA_mkl.shape}")
    print(f"Tile shape (tile_shape_mnk): {tile_shape_mnk}")
    print(f"Tile coordinates (tile_coord_mnkl): {tile_coord_mnkl}")
    print(f"Projection (proj): (1, None, 1)")
    print()
    
    # Compile and call the jitted function
    compiled_demo = cute.compile(demonstrate_local_tile, mA_mkl)
    compiled_demo(mA_mkl)
    
    print()
    print("=== Explanation ===")
    print("The proj=(1, None, 1) parameter:")
    print("- Selects elements [0, 2] from tile_shape_mnk: (64, 32) from (64, 32, 32)")
    print("- Selects elements [0, 2] from tile_coord_mnkl: (0, None) from (0, 0, None, 0)")
    print("- This effectively creates a 2D tiling operation on the M and K dimensions")
    print(f"- Result: extracts a ({tile_shape_mnk[0]}, {tile_shape_mnk[2]}) tile from the input") 