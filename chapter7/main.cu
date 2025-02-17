#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>

#define TILE_WIDTH 4
#define FILTER_WIDTH 3

__constant__ float F[FILTER_WIDTH*FILTER_WIDTH];

__global__
void convKernel(float *X, float *Y, int R) {
    // define the shared memory
    __shared__ float sx[TILE_WIDTH+FILTER_WIDTH-1][TILE_WIDTH+FILTER_WIDTH-1][TILE_WIDTH+FILTER_WIDTH-1];

    int x = blockIdx.x * TILE_WIDTH + threadIdx.x - FILTER_WIDTH/2;
    int y = blockIdx.y * TILE_WIDTH + threadIdx.y - FILTER_WIDTH/2;
    int z = blockIdx.z * TILE_WIDTH + threadIdx.z - FILTER_WIDTH/2;

    float pValue = 0;

    if (x >= 0 && x < R && y >= 0 && y < R && z >= 0 && z < R) {
        sx[threadIdx.z][threadIdx.y][threadIdx.x] = X[z*R*R+y*R+x];
    } else {
        sx[threadIdx.z][threadIdx.y][threadIdx.x] = 0;
    }
    __syncthreads();

    if (threadIdx.x >= FILTER_WIDTH/2 && threadIdx.x < TILE_WIDTH+FILTER_WIDTH/2 && threadIdx.y >= FILTER_WIDTH/2 && threadIdx.y < TILE_WIDTH+FILTER_WIDTH/2 && threadIdx.z >= FILTER_WIDTH/2 && threadIdx.z < TILE_WIDTH+FILTER_WIDTH/2) {
        for (int i = 0; i < FILTER_WIDTH; i++) {
            for (int j = 0; j < FILTER_WIDTH; j++) {
                for (int k = 0; k < FILTER_WIDTH; k++) {
                    pValue += sx[threadIdx.z-FILTER_WIDTH/2+i][threadIdx.y-FILTER_WIDTH/2+j][threadIdx.x-FILTER_WIDTH/2+k] * F[i*FILTER_WIDTH*FILTER_WIDTH+j*FILTER_WIDTH+k];
                }
            }
        }
        Y[z*R*R+y*R+x] = pValue;
    }
    __syncthreads();

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

    // Check for any errors while waiting for kernel to finish
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Kernel execution failed: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


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
    matmul(a_h, b_h, c_h, 8);
    for (int i = 0; i < n; i++) {
        printf("%f ", c_h[i]);
    }
    return 0;
}
