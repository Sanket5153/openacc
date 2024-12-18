#include <iostream>
#include <cstdlib>
#include <ctime>

void randomInit(float* data, int size) {
    for (int i = 0; i < size; ++i)
        data[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
}

void matrixMulCPU(float* C, const float* A, const float* B, int N) {
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
    const int N = 10000; // Matrix size
    float *A = new float[N * N];
    float *B = new float[N * N];
    float *C = new float[N * N];

    srand(static_cast<unsigned int>(time(nullptr)));
    randomInit(A, N * N);
    randomInit(B, N * N);

    matrixMulCPU(C, A, B, N);

    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}
