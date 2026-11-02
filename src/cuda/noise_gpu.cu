#include "cuda_common.cuh"
#include "runway/types.hpp"
#include <curand.h>

namespace runway {
namespace cuda {

__global__ void transform_noise_kernel(
    float* delay,
    float* mult,
    float* closeA,
    float* closeB,
    const float* u_delay,
    const float* u_mult,
    const float* u_cclose,
    const float* u_cdur,
    const float* u_cstart,
    const int* op,
    int n,
    int S,
    float t_min,
    float t_max,
    float p_close
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    // Transform per-scenario variables
    if (idx < S) {
        float r_w = u_mult[idx];
        mult[idx] = (r_w < 0.70f) ? 1.0f : ((r_w < 0.90f) ? 1.15f : 1.30f);

        float r_c = u_cclose[idx];
        if (r_c < p_close) {
            float dur = 300.0f + u_cdur[idx] * 600.0f;
            float max_start = t_max - dur;
            float start = (max_start > t_min) ? (t_min + u_cstart[idx] * (max_start - t_min)) : t_min;
            closeA[idx] = start;
            closeB[idx] = start + dur;
        } else {
            closeA[idx] = 0.0f;
            closeB[idx] = 0.0f;
        }
    }

    // Transform delay array: layout [a * S + s]
    int total_delay_elements = n * S;
    for (int i = idx; i < total_delay_elements; i += blockDim.x * gridDim.x) {
        int a = i / S;
        float mean_d = (op[a] == OP_ARR) ? 30.0f : 60.0f;
        float u = u_delay[i];
        if (u < 1e-7f) u = 1e-7f;
        delay[i] = -mean_d * logf(1.0f - u);
    }
}

void generate_noise_gpu(
    float* d_delay,
    float* d_mult,
    float* d_closeA,
    float* d_closeB,
    const int* d_op,
    int n,
    int S,
    float t_min,
    float t_max,
    uint64_t seed,
    float p_close
) {
    curandGenerator_t gen;
    CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, seed));

    float *d_u_delay, *d_u_mult, *d_u_cclose, *d_u_cdur, *d_u_cstart;
    size_t delay_bytes = static_cast<size_t>(n) * S * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_u_delay, delay_bytes));
    CUDA_CHECK(cudaMalloc(&d_u_mult, S * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_u_cclose, S * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_u_cdur, S * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_u_cstart, S * sizeof(float)));

    CURAND_CHECK(curandGenerateUniform(gen, d_u_delay, n * S));
    CURAND_CHECK(curandGenerateUniform(gen, d_u_mult, S));
    CURAND_CHECK(curandGenerateUniform(gen, d_u_cclose, S));
    CURAND_CHECK(curandGenerateUniform(gen, d_u_cdur, S));
    CURAND_CHECK(curandGenerateUniform(gen, d_u_cstart, S));

    int block = 256;
    int grid = (std::max(S, n * S) + block - 1) / block;
    if (grid > 1024) grid = 1024;

    transform_noise_kernel<<<grid, block>>>(
        d_delay, d_mult, d_closeA, d_closeB,
        d_u_delay, d_u_mult, d_u_cclose, d_u_cdur, d_u_cstart,
        d_op, n, S, t_min, t_max, p_close
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFree(d_u_delay));
    CUDA_CHECK(cudaFree(d_u_mult));
    CUDA_CHECK(cudaFree(d_u_cclose));
    CUDA_CHECK(cudaFree(d_u_cdur));
    CUDA_CHECK(cudaFree(d_u_cstart));
    CURAND_CHECK(curandDestroyGenerator(gen));
}

void upload_noise_host_to_device(
    float* d_delay,
    float* d_mult,
    float* d_closeA,
    float* d_closeB,
    const NoiseData& h_noise
) {
    size_t delay_bytes = h_noise.delay.size() * sizeof(float);
    size_t s_bytes = h_noise.S * sizeof(float);
    CUDA_CHECK(cudaMemcpy(d_delay, h_noise.delay.data(), delay_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mult, h_noise.mult.data(), s_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_closeA, h_noise.closeA.data(), s_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_closeB, h_noise.closeB.data(), s_bytes, cudaMemcpyHostToDevice));
}

} // namespace cuda
} // namespace runway
