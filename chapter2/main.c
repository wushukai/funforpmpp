#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>

__global__
void vecAddKernel(float *a, float *b, float *c, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

void vectorAdd(float *a_h, float *b_h, float *c_h, int n) {
    float *a_d, *b_d, *c_d;
    int size = n * sizeof(float);

    // allocate device memory
    cudaError_t err = cudaMalloc((void **)&a_d, size);
    if (err != cudaSuccess) {
        printf("Failed to allocate device memory for a_d: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void **)&b_d, size);
    if (err != cudaSuccess) {
        printf("Failed to allocate device memory for b_d: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void **)&c_d, size);
    if (err != cudaSuccess) {
        printf("Failed to allocate device memory for c_d: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


    // copy data from host to device
    err = cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("Failed to copy data from host to device for a_d: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(b_d, b_h, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("Failed to copy data from host to device for b_d: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }   

    // launch kernel
    int threadsPerBlock = 256;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    vecAddKernel<<<blocks, threadsPerBlock>>>(a_d, b_d, c_d, n);


    err = cudaMemcpy(c_d, c_h, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("Failed to copy data from host to device for c_d: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // clean up
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
}

int main() {
    int n = 1024;
    float a_h[n], b_h[n], c_h[n];
    vector_add(a_h, b_h, c_h, n);
    return 0;
}
