import os
import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from cutlass import Constexpr, Int32, range_dynamic

os.environ["CUTE_DSL_ARCH"] = "sm_90a"

# A minimal user-defined class intended for use within a kernel.
class MyObject:
    def __init__(self, value):
        self.value = value

    @cute.jit
    def print_value(self):
        for i in range_dynamic(self.value):
            cute.printf("MyObject value: %d\n", self.value)

    def __get_mlir_types__(self):
        return [self.value.type]
    
    def __extract_mlir_values__(self):
        return [self.value.ir_value()]

    def __new_from_mlir_values__(self,values):
        return MyObject(Int32(values[0]))


class ToyKernel:
    @cute.jit
    def __call__(self, x):
        self.kernel(x).launch(grid=(1,1,1), block=(32,1,1))

    @cute.kernel
    def kernel(self, x):
        my_obj = MyObject(Int32(123))
        tidx, _, _ = cute.arch.thread_idx()
        if tidx == 0:
            my_obj.print_value()

if __name__ == "__main__":
    # Dummy tensor to satisfy the kernel signature.
    x = torch.empty(1, device="cuda")
    x_tensor = from_dlpack(x)

    toy_kernel = ToyKernel()
    compiled_kernel = cute.compile(toy_kernel, x_tensor)
    compiled_kernel(x_tensor)
