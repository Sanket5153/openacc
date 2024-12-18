#include <iostream>
#include <cstdlib>
#include <ctime>
#include <openacc.h>

void randomInit(float* data, int size) {
    for (int i = 0; i < size; ++i)
        data[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
}

void matrixMulGPU(float* C, float* A, float* B, int N) {
    #pragma acc kernels loop independent present(A[0:N*N], B[0:N*N], C[0:N*N])
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0;
            for (int k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

int main() {
    const int N = 100000; // Matrix size
    float *A = new float[N * N];
    float *B = new float[N * N];
    float *C = new float[N * N];

    srand(static_cast<unsigned int>(time(nullptr)));
    randomInit(A, N * N);
    randomInit(B, N * N);

    #pragma acc enter data copyin(A[0:N*N], B[0:N*N]) create(C[0:N*N])
    matrixMulGPU(C, A, B, N);
    #pragma acc exit data copyout(C[0:N*N])

    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}
