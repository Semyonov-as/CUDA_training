#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>


float write_speed() {
    float* a;
    float* dev_a;
    int length = 1024*1024*10;

    a = (float*)malloc(length*sizeof(float));

    for(int i = 0; i < length; i++){ 
        a[i] = i*0.1;
    }


    cudaMalloc((void**)&dev_a, length*sizeof(float));
    clock_t start, end;
    
    start = clock();
    
    cudaMemcpy(dev_a, a, length*sizeof(float), cudaMemcpyHostToDevice);

    end = clock();
    time_t elapsed = end - start;
  
    float speed = 1000*sizeof(float)*length/elapsed;  

    printf("Elapsed time: %zd ms for copying %zd MB gives us speed of writing from host to device %.1f GB/s", elapsed, length*sizeof(float)/1024/1024, speed/1024/1024/1024);

    cudaFree(dev_a);
    free(a);

    return speed;
}

__global__ void calculate_pi(float* data, int N){
    float x = static_cast<float>(blockIdx.x)/N;
    //printf("Idx.x = %d, x = %f", blockIdx.x, x);
    data[blockIdx.x] = sqrt(1 - x*x)/N;
}

void calculate_pi_GPU() {
    int N = 100000;
    float* a = (float*)malloc(N*sizeof(float));
    float* dev_a;

    if(cudaMalloc((void**)&dev_a, N*sizeof(float)) != cudaSuccess)
        printf("Oh shit!");
    
    cudaEvent_t start, stop;
    float elapsedTime;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
  
    calculate_pi<<<N, 1>>>(dev_a, N);

    cudaEventRecord(stop,0);

    cudaMemcpy(a, dev_a, N*sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    float pi = 0;
    for(int i = 0; i < N; i++)
        pi += a[i];
    pi *= 4;

    printf("Time spent calpulating pi by the GPU in %d blocks: %.2f millseconds\nResult: %f, accuracy: %f\n", N, elapsedTime, pi, 1 - (pi - 3.141592)/3.141592);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void calculate_pi_CPU() {
    float pi = 0;
    int N = 100000;
    clock_t start, end;
    start = clock();
    for(int i = 0; i < N; i++) {
        float x = static_cast<float>(i)/N; 
        pi += sqrt(1 - x*x)/N;
    }
    pi *= 4;    

    end = clock();
    time_t elapsed = end - start;

    printf("Time spent calpulating pi by the CPU in %d iterations: %.2f millseconds\nResult: %f, accuracy: %f\n", N, elapsed, pi, 1 - (pi - 3.141592)/3.141592);
}


__global__ void calculate_zeta(float* data, float s) {
    data[blockIdx.x] = 1.0/pow(static_cast<float>(blockIdx.x + 1), s); 
}

void calculate_zeta_GPU(float s) {
    if (s <= 1){
         printf("If s is equal of less than 1, value is NaN");
         exit(-1);
    }

    int N = 10000;
    float* a = (float*)malloc(N*sizeof(float));
    float* dev_a;

    if(cudaMalloc((void**)&dev_a, N*sizeof(float)) != cudaSuccess)
        printf("Oh shit!");
    
    cudaEvent_t start, stop;
    float elapsedTime;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
  
    calculate_zeta<<<N, 1>>>(dev_a, s);

    cudaEventRecord(stop,0);

    cudaMemcpy(a, dev_a, N*sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    float zeta = 0;
    for(int i = 0; i < N; i++)
        zeta += a[i];


    printf("Time spent calpulating Zeta(%f) by the GPU in %d blocks: %.2f millseconds\nResult: %f\n", s, N, elapsedTime, zeta);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    calculate_zeta_GPU(2.0);
    return 0;
}