#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include <cuda_runtime.h>


__global__
void prefix_sum_kernel(const float *data, float *result, int n) {
    __shared__ float accm_buffer1[32];
    __shared__ float accm_buffer2[32];

    int tid = threadIdx.x;
    if (tid < n) {
        accm_buffer1[tid] = data[tid];
    }

    float *read_buffer = accm_buffer1;
    float *write_buffer = accm_buffer2;

    for (int stride = 1; stride < n; stride *= 2) {
        __syncthreads();

        if (tid < stride) {
            continue;
        }

        write_buffer[tid] = read_buffer[tid] + read_buffer[tid - stride];

        read_buffer = write_buffer;
        write_buffer = (write_buffer == accm_buffer1) ? accm_buffer2 : accm_buffer1;
    }

    __syncthreads();

    result[tid] = read_buffer[tid];
}

int calc_prefix_sum(const float *data, float *result, int n) {
    float *d_data, *d_result;
    cudaMalloc(&d_data, n * sizeof(float));
    cudaMalloc(&d_result, n * sizeof(float));

    cudaMemcpy(d_data, data, n * sizeof(float), cudaMemcpyHostToDevice);

    // call kernel
    prefix_sum_kernel<<<1, 32>>>(d_data, d_result, n);

    // Check for any errors while waiting for kernel to finish
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Kernel execution failed: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_result, result, n * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("Failed to copy data from device to host for d_result: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    cudaFree(d_data);
    cudaFree(d_result);
}

int main() {
    const float data[] = {
        4, 6, 7, 1, 2, 8, 5, 2
    };

    float result[sizeof(data)/sizeof(data[0])];

    int r = calc_prefix_sum(data, result, sizeof(data)/sizeof(data[0]));
    for (int i = 0; i < sizeof(data)/sizeof(data[0]); i++) {
        printf("%f ", result[i]);
    }
    printf("\n");
    return r;
}