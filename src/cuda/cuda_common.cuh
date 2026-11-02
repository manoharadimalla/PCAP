#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

#define CURAND_CHECK(call) do { \
    curandStatus_t status = call; \
    if (status != CURAND_STATUS_SUCCESS) { \
        fprintf(stderr, "cuRAND Error at %s:%d: status %d\n", __FILE__, __LINE__, (int)status); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

namespace runway {
namespace cuda {

// 6x6 separation matrix in GPU constant memory for K3
extern __constant__ float c_sep[36];

template <typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block reduction across warps using shared memory
template <typename T, int BLOCK_SIZE>
__device__ __forceinline__ T block_reduce_sum(T val, T* shared_warp_sums) {
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    val = warp_reduce_sum(val);
    if (lane == 0) {
        shared_warp_sums[wid] = val;
    }
    __syncthreads();

    // First warp reduces the per-warp sums
    T block_sum = 0;
    int num_warps = BLOCK_SIZE / 32;
    if (wid == 0) {
        block_sum = (lane < num_warps) ? shared_warp_sums[lane] : (T)0;
        block_sum = warp_reduce_sum(block_sum);
    }
    return block_sum;
}

} // namespace cuda
} // namespace runway
