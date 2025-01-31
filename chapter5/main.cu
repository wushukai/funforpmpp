#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>

#define TILE_WIDTH 4

__global__
void matmulKernel(float *x, float *y, float *p, int width) {
    // define the shared memory
    __shared__ float sx[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sy[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    int K = (width + TILE_WIDTH - 1) / TILE_WIDTH;
    float pValue = 0;

    for (int k = 0; k < K; k++) {
        sx[threadIdx.y][threadIdx.x] = x[row+threadIdx.y][k*TILE_WIDTH+threadIdx.x];
        sy[threadIdx.y][threadIdx.x] = y[k*TILE_WIDTH+threadIdx.y][col+threadIdx.x];
        __syncthreads();

        for (int i = 0; i < TILE_WIDTH; i++) {
            pValue += sx[threadIdx.y][i] * sy[i][threadIdx.x];
        }

        __syncthreads();
    }

    p[row * width + col] = pValue;
}

void matmul(float *a_h, float *b_h, float *c_h, int width) {
    float *a_d, *b_d, *c_d;
    int size = width * width * sizeof(float);

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
    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 blocks((width + TILE_WIDTH - 1) / TILE_WIDTH, (width + TILE_WIDTH - 1) / TILE_WIDTH);
    matmulKernel<<<blocks, threadsPerBlock>>>(a_d, b_d, c_d, width);


    err = cudaMemcpy(c_h, c_d, size, cudaMemcpyDeviceToHost);
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
    int n = 8 * 8;
    float a_h[n], b_h[n], c_h[n];
    for (int i = 0; i < n; i++) {
        a_h[i] = i;
        b_h[i] = i;
    }
    matmul(a_h, b_h, c_h, n);
    for (int i = 0; i < n; i++) {
        printf("%f ", c_h[i]);
    }
    return 0;
}
