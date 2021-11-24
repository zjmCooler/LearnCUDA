#include <cstdio>
#include <vector>

__global__ void hello_world() {
    printf("GPU: Hello World\n");
}

int main() {
    printf("CPU: Hello World\n");

    hello_world<<<1, 10>>>();
    cudaDeviceReset();

    return 0;
}
