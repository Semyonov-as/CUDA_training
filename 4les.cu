#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>


__global__ void test() {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    printf("I,m block (%d,%d), thread (%d,%d), my i=%d, my j=%d\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, i, j);
}

__global__ void pi_est(int *count) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    //printf("I,m block (%d,%d), thread (%d,%d), my i=%d, my j=%d, check %d <= %d \n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, i, j, i*i + j*j, (gridDim.x*blockDim.x - 1)*(gridDim.x*blockDim.x-1));
    if ( i*i + j*j <= (gridDim.x*blockDim.x - 1)*(gridDim.x*blockDim.x-1)){
        atomicAdd(count, 1);
        
    }
}

int main() {
    int N_th = 32;
    int N_bl = 1000;

    int* dev_res;
    int res = 0;   

    cudaMalloc((void**)&dev_res, sizeof(int));
    cudaMemcpy(dev_res, &res, sizeof(int), cudaMemcpyHostToDevice);

    pi_est<<<dim3(N_bl, N_bl, 1), dim3(N_th, N_th, 1)>>>(dev_res);

    cudaMemcpy(&res, dev_res, sizeof(int), cudaMemcpyDeviceToHost);

    printf("Result %f", 4*static_cast<float>(res)/(N_th*N_th*N_bl*N_bl));


    return 0;
}