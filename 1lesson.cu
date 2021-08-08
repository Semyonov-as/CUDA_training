#include<stdio.h>

__global__ void hello_cuda() {
    printf("Hello, world! I am block %d, thread %d! \n", blockIdx.x, threadIdx.x);
}

__global__ void sum(int a, int b) {
    printf("%d + %d = %d\n", a, b, a+b);
}

int main() {
    hello_cuda<<<2, 2>>>();
    sum<<<1, 1>>>(1, 2);
	
    return 0;
}