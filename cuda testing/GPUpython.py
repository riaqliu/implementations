from timeit import default_timer as timer

from numba import cuda
import numpy as np

@cuda.jit
def compute_factorials(array):
    tid = cuda.grid(1)
    if tid < array.size:
        n = tid
        result = 1
        for i in range(1, n + 1):
            result *= i
        array[tid] = result

N = 64
block_size = 256
num_blocks = (N + block_size - 1) // block_size

d_array = cuda.device_array(N, dtype=np.uint32)

compute_factorials[num_blocks, block_size](d_array)

h_array = d_array.copy_to_host()
print(h_array)