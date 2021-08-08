#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"

__global__ void exp_calc(float* data) {
    data[threadIdx.x] = __expf((float)threadIdx.x/blockDim.x);
    //printf("I,m thread %d, my result:%f\n", threadIdx.x, data[threadIdx.x]); 
}

__host__ int main(int argc, char* argv[]) {
    printf("Hey cocksucker, I'm here to let you know how to parse command line arguments, listen me now stupid bitch:\n");
    for(int k = 0; k < argc; k++){
        printf("\t%d argument is %s\n", k, argv[k]);
    }

    int N = atoi(argv[1]);
    printf("Hey ho N is %d, argv1 is %s\n", N, argv[1]);

    float* dev_data;

    float* host_data = (float *)malloc(N*sizeof(float));
    cudaMalloc((void**)&dev_data, N*sizeof(float));

    exp_calc<<<1, N>>>(dev_data);

    cudaMemcpy(&host_data[0], dev_data, N*sizeof(float), cudaMemcpyDeviceToHost);

    float err = 0;

    for(int i = 0; i < N; i++)
        err += abs(host_data[i] - exp((float)i/N));

    printf("Error is %f", err);

    return 0;
}
