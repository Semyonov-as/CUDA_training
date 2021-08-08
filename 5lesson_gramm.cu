#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"


__global__ void scalar(float* a, float* b, float* c) {
    //each thread compute its own component
    atomicAdd(c, a[threadIdx.x]*b[threadIdx.x]);
    //printf("\tI'm thread %d my result is %f\n", threadIdx.x, a[threadIdx.x]*b[threadIdx.x]);
}

__host__ void test_scalar() {
    //Let's begin with quite easy task: compute a scalar product of 2 vectors
    //    Vector a = (1 2 3)\n
    //    Vector b = (4 5 6)\n
    //We expect result answer 32

    //For each component we will create separate thread, which will return the product of two floats

    float host_a[3] = {1, 2, 3};
    float host_b[3] = {4, 5, 6};
    float host_c = 0;

    float * dev_a, * dev_b, * dev_c;

    cudaMalloc((void**)&dev_a, 3*sizeof(float));
    cudaMalloc((void**)&dev_b, 3*sizeof(float));
    cudaMalloc((void**)&dev_c, sizeof(float));

    cudaMemcpy(dev_a, &host_a[0], 3*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, &host_b[0], 3*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_c, &host_c, sizeof(float), cudaMemcpyHostToDevice);

    scalar<<<1, 3>>>(dev_a, dev_b, dev_c);

    cudaMemcpy(&host_c, dev_c, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    printf("result is %f\n", host_c);
}

__global__ void ortagonalize(float *a, float *b, float tmp1, float tmp2, int i, int j) {
    a[i*blockDim.x + threadIdx.x] -= tmp1/tmp2*b[j*blockDim.x + threadIdx.x];
    printf("\tdone too\n");
}

__global__ void new_scalar(float *a, float *b, float *res, int i){
    printf("doing\n");
    atomicAdd(res, a[i*blockDim.x + threadIdx.x] * b[i*blockDim.x + threadIdx.x]);
}

__host__ int main(int argc, char* argv[]) {
    //N defines basis size
    int N = atoi(argv[1]);

    float * host_a = (float*)malloc(N*N*sizeof(float));
    float * host_b = (float*)malloc(N*N*sizeof(float));
    float * dev_a, * dev_b;

    cudaMalloc((void **)&dev_a, N*N*sizeof(float*));
    cudaMalloc((void **)&dev_b, N*N*sizeof(float*));

    for(int i = 0; i < N; i++) {
        for(int k = 0; k < N; k++){
            if(k >= i)
                host_a[i*N + k] = 1;
            else
                host_a[i*N + k] = 0;
        }

    }
    cudaMemcpy(dev_a, host_a, N*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, host_a, N*N*sizeof(float), cudaMemcpyHostToDevice);

    printf("Vectors for ortagonalisation:\n");
    for(int i = 0; i < N; i++)
        printf("a%d\t", i);
    printf("\n");

    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++)
            printf("%.2f\t", host_a[j*N + i]);
        printf("\n");
    }

    //Begin ortagonalisation
    float tmp = 0;

    float *tmp1, *tmp2;
    cudaMalloc((void**)&tmp1, sizeof(float));
    cudaMalloc((void**)&tmp2, sizeof(float));

    for(int i = 1; i < N; i++) {
        printf("Computing %d vector\n", i);
        for(int j = 0; j < i; j++) {
            cudaMemcpy(tmp1, &tmp, sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(tmp2, &tmp, sizeof(float), cudaMemcpyHostToDevice);

            printf("\tComputing %d part\n", j);
            new_scalar<<<1, N>>>(dev_b, dev_a, tmp1, j);
            new_scalar<<<1, N>>>(dev_b, dev_b, tmp2, j);

            ortagonalize<<<1, N>>>(dev_b, dev_b, *tmp1, *tmp2, i, j);
        }
    }

    // Get results

    cudaMemcpy(host_b, dev_b, N*N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dev_a);
    cudaFree(dev_b);


    printf("Ortagonalisation results:\n");
    for(int i = 0; i < N; i++)
        printf("b%d\t", i);
    printf("\n");

    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++)
            printf("%.2f\t", host_b[j*N + i]);
        printf("\n");
    }

    free(host_a);
    free(host_b);

    return 0;
}