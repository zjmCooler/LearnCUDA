#include <cstdio>

#include "common.h"

__global__ void sum_matrix(float *matA, float *matB, float *matC,
                           int nx, int ny) {
    int ix = threadIdx.x + blockDim.x * blockIdx.x;
    int iy = threadIdx.y + blockDim.y * blockIdx.y;
    int idx = ix + iy * ny;
    if (ix < nx && iy < ny) {
        matC[idx] = matA[idx] + matB[idx];
    }
}

int main(int argc, char **argv) {
    int nx = 1 << 13;
    int ny = 1 << 13;
    int nxy = nx * ny;
    int nBytes = nxy * sizeof(float);

    // malloc
    float *A_host = (float *)malloc(nBytes);
    float *B_host = (float *)malloc(nBytes);
    float *C_host = (float *)malloc(nBytes);
    float *C_from_gpu = (float *)malloc(nBytes);

    initialData(A_host, nxy);
    initialData(B_host, nxy);

    //    cuda malloc
    float *A_gpu = nullptr;
    float *B_gpu = nullptr;
    float *C_gpu = nullptr;

    CHECK(cudaMalloc((void **)&A_gpu, nBytes));
    CHECK(cudaMalloc((void **)&B_gpu, nBytes));
    CHECK(cudaMalloc((void **)&C_gpu, nBytes));

    CHECK(cudaMemcpy(A_gpu, A_host, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(B_gpu, B_host, nBytes, cudaMemcpyHostToDevice));

    int dimx = argc > 2 ? atoi(argv[1]) : 32;
    int dimy = argc > 2 ? atoi(argv[2]) : 32;

    double i_start, i_elaps;
    dim3 block(dimx, dimy);
    dim3 grid((nx - 1) / block.x + 1, (ny - 1) / block.y + 1);
    i_start = cpuSecond();
    sum_matrix<<<grid, block>>>(A_gpu, B_gpu, C_gpu, nx, ny);
    CHECK(cudaDeviceSynchronize());

    i_elaps = cpuSecond() - i_start;

    printf("GPU exec configuration<<<(%d, %d), (%d, %d) | %f seconds",
           grid.x, grid.y, block.x, block.y, i_elaps);
    CHECK(cudaMemcpy(C_from_gpu, C_gpu, nBytes, cudaMemcpyDeviceToHost));

    cudaFree(A_gpu);
    cudaFree(B_gpu);
    cudaFree(C_gpu);

    free(A_host);
    free(B_host);
    free(C_host);
    free(C_from_gpu);

    cudaDeviceReset();

    return 0;
}