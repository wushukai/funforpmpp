#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>


__global__
void prefix_sum_kernel(const float32_t *data, float32_t *result, int n) {
    __shared__ float32_t accm_buffer1[32];
    __shared__ float32_t accm_buffer2[32];

    int tid = threadIdx.x;
    if (tid < n) {
        accm_buffer1[tid] = data[tid];
    }

    float32_t *read_buffer = accm_buffer1;
    float32_t *write_buffer = accm_buffer2;

    for (int stride = 1; stride < n; stride *= 2) {
        __syncthreads();

        if (tid < stride) {
            continue
        }

        write_buffer[tid] = read_buffer[tid] + read_buffer[tid - stride];

        read_buffer = write_buffer;
        write_buffer = (write_buffer == accm_buffer1) ? accm_buffer2 : accm_buffer1;
    }

    __syncthreads();

    result[tid] = read_buffer[tid];
}

int calc_prefix_sum(const float32_t *data, float32_t *result, int n) {
    float32_t *d_data, *d_result;
    cudaMalloc(&d_data, n * sizeof(float32_t));
    cudaMalloc(&d_result, n * sizeof(float32_t));

    cudaMemcpy(d_data, data, n * sizeof(float32_t), cudaMemcpyHostToDevice);

    // call kernel
    prefix_sum_kernel<<<1, 32>>>(d_data, d_result, n);

    // Check for any errors while waiting for kernel to finish
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Kernel execution failed: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_result, result, n * sizeof(float32_t), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("Failed to copy data from device to host for d_result: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    cudaFree(d_data);
    cudaFree(d_result);
}

int main() {
    const float32_t data[] = [
        4, 6, 7, 1, 2, 8, 5, 2
    ];

    float32_t result[sizeof(data)/sizeof(data[0])];

    int r = calc_prefix_sum(data, result, sizeof(data)/sizeof(data[0]));
    for (int i = 0; i < sizeof(data)/sizeof(data[0]); i++) {
        printf("%f ", result[i]);
    }
    printf("\n");
    return r;
}