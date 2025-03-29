#include <iostream>
#include <cuda_runtime.h>

__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}

int main() {
    const int N = 1000;
    size_t size = N * sizeof(float);

    float *h_A = new float[N];
    float *h_B = new float[N];
    float *h_C = new float[N];

    for (int i = 0; i < N; ++i) {
        h_A[i] = i * 1.0f;
        h_B[i] = i * 2.0f;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    std::cout << "Resultado: ";
    for (int i = 0; i < 5; ++i) std::cout << h_C[i] << " ";
    std::cout << "...\n";

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}

