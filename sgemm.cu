#include <iostream>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <iomanip>
#include <string>

#define CHECK(call) \
{ \
    const cudaError_t cuda_ret = call; \
    if (cuda_ret != cudaSuccess) { \
        printf("Error: %s:%d, " __FILE__, __LINE__); \
        printf("code: %d, reason: %s\n", cuda_ret, cudaGetErrorString(cuda_ret)); \
        exit(1); \
    } \
}

__global__ void matrixMulKernel_1thread1element(int m, int k, int n, const float* A_d, const float* B_d, float* C_d) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float tempSum = 0.0;

    if ((row < m) && (col < n)) {
        for (int i = 0; i < k; ++i) {
            tempSum += A_d[row * k + i] * B_d[i * n + col];
        }
        C_d[row * n + col] = tempSum;
    }
}

__global__ void matrixMulKernel_1thread1row(int m, int k, int n, const float* A_d, const float* B_d, float* C_d) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < m) {
        for (int col = 0; col < n; ++col) {
            float tempSum = 0.0;

            for (int i = 0; i < k; ++i) {
                tempSum += A_d[row * k + i] * B_d[i * n + col];
            }
            C_d[row * n + col] = tempSum;
        }
    }
}

__global__ void matrixMulKernel_1thread1column(int m, int k, int n, const float* A_d, const float* B_d, float* C_d) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < n) {
        for (int row = 0; row < m; ++row) {
            float tempSum = 0.0;

            for (int i = 0; i < k; ++i) {
                tempSum += A_d[row * k + i] * B_d[i * n + col];
            }
            C_d[row * n + col] = tempSum;
        }
    }
}

void basicSgemm_h(int m, int k, int n, const float* A_h, const float* B_h, float* C_h) {
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            float sum = 0.0f;
            for (int i = 0; i < k; ++i) {
                sum += A_h[row * k + i] * B_h[i * n + col];
            }  
            C_h[row * n + col] = sum;
        }
    }
}

void initMatrices(int m, int k, int n, float* A_h, float* B_h) {
    std::srand(std::time(nullptr));

    for (int i = 0; i < m * k; i++) {
        A_h[i] = std::rand() % 100/100.0;
        // std::cout << A_h[i] << " ";
    }
    // std::cout << std::endl;

    for (int j = 0; j < k * n; j++) {
        B_h[j] = std::rand() % 100/100.0;
        // std::cout << B_h[j] << " ";
    }
    // std::cout << std::endl;
}

void basicSgemm_d_1thread1element(int m, int k, int n, const float* A_h, const float* B_h, float* C_h) {
    size_t bytes_A = m * k * sizeof(float);                              // calculate byte size for allocation
    size_t bytes_B = k * n * sizeof(float);
    size_t bytes_C = n * m * sizeof(float);                                

    float* a_d, * b_d, * c_d;                                            // init device 

    CHECK(cudaMalloc((void**)&a_d, bytes_A));                            // allocate cuda mem
    CHECK(cudaMalloc((void**)&b_d, bytes_B));
    CHECK(cudaMalloc((void**)&c_d, bytes_C));

    CHECK(cudaMemcpy(a_d, A_h, bytes_A, cudaMemcpyHostToDevice));        // copy data to device
    CHECK(cudaMemcpy(b_d, B_h, bytes_B, cudaMemcpyHostToDevice));

    dim3 blockDim(16, 16);
    dim3 gridDim(ceil((float)n / blockDim.x), ceil((float)m / blockDim.y));

    const float* A_d = a_d;
    const float* B_d = b_d;

    matrixMulKernel_1thread1element <<< gridDim, blockDim >>> (m, k, n, A_d, B_d, c_d);

    CHECK(cudaMemcpy(C_h, c_d, bytes_C, cudaMemcpyDeviceToHost));        // copy result back to host

    CHECK(cudaFree(a_d));
    CHECK(cudaFree(b_d));
    CHECK(cudaFree(c_d));
}

void basicSgemm_d_1thread1row(int m, int k, int n, const float* A_h, const float* B_h, float* C_h) {
    size_t bytes_A = m * k * sizeof(float);
    size_t bytes_B = k * n * sizeof(float);
    size_t bytes_C = n * m * sizeof(float);                              // calculate byte size for allocation

    float* a_d, * b_d, * c_d;                                            // init device 

    CHECK(cudaMalloc((void**)&a_d, bytes_A));                            // allocate cuda mem
    CHECK(cudaMalloc((void**)&b_d, bytes_B));
    CHECK(cudaMalloc((void**)&c_d, bytes_C));

    CHECK(cudaMemcpy(a_d, A_h, bytes_A, cudaMemcpyHostToDevice));        // copy data to device
    CHECK(cudaMemcpy(b_d, B_h, bytes_B, cudaMemcpyHostToDevice));

    dim3 blockDim(1, 16);
    dim3 gridDim(1, ceil((float)m / blockDim.y));

    const float* A_d = a_d;
    const float* B_d = b_d;

    matrixMulKernel_1thread1row << < gridDim, blockDim >> > (m, k, n, A_d, B_d, c_d);

    CHECK(cudaMemcpy(C_h, c_d, bytes_C, cudaMemcpyDeviceToHost));        // copy result back to host

    CHECK(cudaFree(a_d));
    CHECK(cudaFree(b_d));
    CHECK(cudaFree(c_d));
}

void basicSgemm_d_1thread1column(int m, int k, int n, const float* A_h, const float* B_h, float* C_h) {
    size_t bytes_A = m * k * sizeof(float);                              // calculate byte size for allocation
    size_t bytes_B = k * n * sizeof(float);
    size_t bytes_C = n * m * sizeof(float);                              

    float* a_d, * b_d, * c_d;                                            // init device 

    CHECK(cudaMalloc((void**)&a_d, bytes_A));                            // allocate cuda mem
    CHECK(cudaMalloc((void**)&b_d, bytes_B));
    CHECK(cudaMalloc((void**)&c_d, bytes_C));

    CHECK(cudaMemcpy(a_d, A_h, bytes_A, cudaMemcpyHostToDevice));        // copy data to device
    CHECK(cudaMemcpy(b_d, B_h, bytes_B, cudaMemcpyHostToDevice));

    dim3 blockDim(16, 1);
    dim3 gridDim(ceil((float)n / blockDim.x), 1);

    const float* A_d = a_d;
    const float* B_d = b_d;

    matrixMulKernel_1thread1column << < gridDim, blockDim >> > (m, k, n, A_d, B_d, c_d);

    CHECK(cudaMemcpy(C_h, c_d, bytes_C, cudaMemcpyDeviceToHost));        // copy result back to host

    CHECK(cudaFree(a_d));
    CHECK(cudaFree(b_d));
    CHECK(cudaFree(c_d));
}

bool verify(float* CPU_Answer, float* GPU_Answer, unsigned int nRows, unsigned int nCols) {
    for (int i = 0; i < nRows * nCols; i++) {
        if (fabs(CPU_Answer[i] - GPU_Answer[i]) > 1e-2) {
            return false;
        }
    }

    return true;
}

auto myCPUTimer() {
    return std::chrono::high_resolution_clock::now();
}

int main(int argc, char** argv) {
    const float* A_h;
    const float* B_h;

    const int m = std::stoi(argv[1]);
    const int k = std::stoi(argv[2]);
    const int n = std::stoi(argv[3]);

    size_t bytes_A = m * k * sizeof(float);                     // calculate byte size for allocation
    size_t bytes_B = k * n * sizeof(float);
    size_t bytes_C = n * m * sizeof(float);                     

    float* a_h = (float*)malloc(bytes_A);                       // allocate host memory for matrices
    float* b_h = (float*)malloc(bytes_B);
    float* c_h_CPU = (float*)malloc(bytes_C);
    float* c_h_GPU1 = (float*)malloc(bytes_C);
    float* c_h_GPU2 = (float*)malloc(bytes_C);
    float* c_h_GPU3 = (float*)malloc(bytes_C);

    initMatrices(m, k, n, a_h, b_h);                            // init matrix

    A_h = a_h;                                                  // assign matrices to constants
    B_h = b_h;

    std::cout << "m: " << m << " k: " << k << " n: " << n << std::endl;

    auto start = myCPUTimer();
    basicSgemm_h(m, k, n, A_h, B_h, c_h_CPU);                   
    auto end = myCPUTimer();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    std::cout << "CPU matrix mul: " << duration.count()/1e9 << " s" << std::endl;

    start = myCPUTimer();
    basicSgemm_d_1thread1element(m, k, n, A_h, B_h, c_h_GPU1);
    end = myCPUTimer();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    std::cout << "GPU 1 thread per element: " << duration.count() / 1e9 << " s" << std::endl;

    start = myCPUTimer();
    basicSgemm_d_1thread1row(m, k, n, A_h, B_h, c_h_GPU2);
    end = myCPUTimer();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    std::cout << "GPU 1 thread per row: " << duration.count() / 1e9 << " s" << std::endl;

    start = myCPUTimer();
    basicSgemm_d_1thread1column(m, k, n, A_h, B_h, c_h_GPU3);
    end = myCPUTimer();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    std::cout << "GPU 1 thread per column: " << duration.count() / 1e9 << " s" << std::endl;

    std::cout << std::endl;
    std::cout << "GPU 1 thread per element calculation correct: " << std::boolalpha << verify(c_h_CPU, c_h_GPU1, m, n) << std::endl;
    std::cout << "GPU 1 thread per row calculation correct: " << std::boolalpha << verify(c_h_CPU, c_h_GPU2, m, n) << std::endl;
    std::cout << "GPU 1 thread per column calculation correct: " << std::boolalpha << verify(c_h_CPU, c_h_GPU3, m, n) << std::endl;

    return 0;
}
