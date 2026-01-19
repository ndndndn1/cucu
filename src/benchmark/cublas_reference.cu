/**
 * BareMetal-SGEMM: cuBLAS Reference Implementation
 *
 * This module provides a cuBLAS-based SGEMM implementation
 * for correctness verification and performance comparison.
 *
 * cuBLAS is NVIDIA's highly optimized BLAS library and represents
 * the gold standard for GPU matrix operations.
 */

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <stdexcept>

namespace baremetal {

// ============================================================================
// cuBLAS Error Checking
// ============================================================================
#define CHECK_CUBLAS(call)                                                     \
    do {                                                                       \
        cublasStatus_t status = (call);                                        \
        if (status != CUBLAS_STATUS_SUCCESS) {                                 \
            throw std::runtime_error(std::string("cuBLAS Error at ") +         \
                                   __FILE__ + ":" + std::to_string(__LINE__)); \
        }                                                                      \
    } while (0)

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            throw std::runtime_error(std::string("CUDA Error: ") +             \
                                   cudaGetErrorString(err) + " at " +          \
                                   __FILE__ + ":" + std::to_string(__LINE__)); \
        }                                                                      \
    } while (0)

// ============================================================================
// cuBLAS SGEMM Reference
// ============================================================================
class CublasReference {
public:
    CublasReference() {
        CHECK_CUBLAS(cublasCreate(&handle_));
        // Use tensor cores if available (for Volta+)
        CHECK_CUBLAS(cublasSetMathMode(handle_, CUBLAS_DEFAULT_MATH));
    }

    ~CublasReference() {
        if (handle_) {
            cublasDestroy(handle_);
        }
    }

    CublasReference(const CublasReference&) = delete;
    CublasReference& operator=(const CublasReference&) = delete;

    /**
     * Perform SGEMM: C = alpha * A * B + beta * C
     *
     * Note: cuBLAS uses column-major layout by default.
     * For row-major matrices (C-style), we use the identity:
     *   C = A * B  <=>  C^T = B^T * A^T
     *
     * So we swap A and B and transpose the operation.
     */
    void sgemm(
        int M, int N, int K,
        float alpha,
        const float* d_A, int lda,
        const float* d_B, int ldb,
        float beta,
        float* d_C, int ldc) {

        // For row-major: compute C = A * B as (in column-major terms):
        // C^T = B^T * A^T
        // Which translates to: swap A<->B, swap M<->N, and adjust dimensions

        CHECK_CUBLAS(cublasSgemm(
            handle_,
            CUBLAS_OP_N,      // B is not transposed (but it's "A" in col-major view)
            CUBLAS_OP_N,      // A is not transposed (but it's "B" in col-major view)
            N,                // Rows of op(B^T) = cols of B = N
            M,                // Cols of op(A^T) = rows of A = M
            K,                // Inner dimension
            &alpha,
            d_B, ldb,         // B in our notation (N x K in col-major = K x N row-major)
            d_A, lda,         // A in our notation (K x M in col-major = M x K row-major)
            &beta,
            d_C, ldc          // C (N x M in col-major = M x N row-major)
        ));
    }

    /**
     * Benchmark cuBLAS SGEMM and return GFLOP/s
     */
    double benchmark(
        int M, int N, int K,
        int warmup_iters = 5,
        int bench_iters = 20) {

        // Allocate device memory
        float *d_A, *d_B, *d_C;
        CHECK_CUDA(cudaMalloc(&d_A, M * K * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_B, K * N * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_C, M * N * sizeof(float)));

        // Initialize with random data
        std::vector<float> h_A(M * K), h_B(K * N), h_C(M * N);
        for (auto& v : h_A) v = static_cast<float>(rand()) / RAND_MAX;
        for (auto& v : h_B) v = static_cast<float>(rand()) / RAND_MAX;
        for (auto& v : h_C) v = static_cast<float>(rand()) / RAND_MAX;

        CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_C, h_C.data(), M * N * sizeof(float), cudaMemcpyHostToDevice));

        float alpha = 1.0f, beta = 0.0f;

        // Warm-up
        for (int i = 0; i < warmup_iters; ++i) {
            sgemm(M, N, K, alpha, d_A, K, d_B, N, beta, d_C, N);
        }
        CHECK_CUDA(cudaDeviceSynchronize());

        // Benchmark
        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        CHECK_CUDA(cudaEventRecord(start));
        for (int i = 0; i < bench_iters; ++i) {
            sgemm(M, N, K, alpha, d_A, K, d_B, N, beta, d_C, N);
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float total_ms;
        CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));

        // Cleanup
        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
        CHECK_CUDA(cudaFree(d_A));
        CHECK_CUDA(cudaFree(d_B));
        CHECK_CUDA(cudaFree(d_C));

        // Calculate GFLOP/s
        double avg_ms = total_ms / bench_iters;
        double flops = 2.0 * M * N * K;
        double gflops = flops / (avg_ms * 1e6);

        return gflops;
    }

    /**
     * Compute reference result for correctness checking
     */
    void compute_reference(
        int M, int N, int K,
        float alpha,
        const float* h_A,
        const float* h_B,
        float beta,
        float* h_C) {

        // Allocate device memory
        float *d_A, *d_B, *d_C;
        CHECK_CUDA(cudaMalloc(&d_A, M * K * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_B, K * N * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_C, M * N * sizeof(float)));

        CHECK_CUDA(cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_C, h_C, M * N * sizeof(float), cudaMemcpyHostToDevice));

        sgemm(M, N, K, alpha, d_A, K, d_B, N, beta, d_C, N);

        CHECK_CUDA(cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

        CHECK_CUDA(cudaFree(d_A));
        CHECK_CUDA(cudaFree(d_B));
        CHECK_CUDA(cudaFree(d_C));
    }

    cublasHandle_t handle() const { return handle_; }

private:
    cublasHandle_t handle_ = nullptr;
};

// ============================================================================
// CPU Reference (for small matrices / debugging)
// ============================================================================
void sgemm_cpu_reference(
    int M, int N, int K,
    float alpha,
    const float* A, int lda,
    const float* B, int ldb,
    float beta,
    float* C, int ldc) {

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * lda + k] * B[k * ldb + j];
            }
            C[i * ldc + j] = alpha * sum + beta * C[i * ldc + j];
        }
    }
}

// ============================================================================
// Correctness Verification
// ============================================================================
bool verify_sgemm_result(
    const float* result,
    const float* reference,
    int M, int N,
    float rtol = 1e-4f,
    float atol = 1e-5f) {

    int errors = 0;
    float max_diff = 0.0f;
    int max_diff_idx = -1;

    for (int i = 0; i < M * N; ++i) {
        float diff = std::abs(result[i] - reference[i]);
        float threshold = atol + rtol * std::abs(reference[i]);

        if (diff > threshold) {
            if (errors < 10) {  // Print first 10 errors
                std::cerr << "  Error at [" << (i / N) << ", " << (i % N) << "]: "
                          << result[i] << " vs " << reference[i]
                          << " (diff=" << diff << ")" << std::endl;
            }
            errors++;
        }

        if (diff > max_diff) {
            max_diff = diff;
            max_diff_idx = i;
        }
    }

    if (errors > 0) {
        std::cerr << "Total errors: " << errors << " / " << (M * N) << std::endl;
        std::cerr << "Max diff: " << max_diff << " at index " << max_diff_idx << std::endl;
        return false;
    }

    std::cout << "Verification PASSED (max diff: " << max_diff << ")" << std::endl;
    return true;
}

}  // namespace baremetal

// ============================================================================
// C-style interface for linking with Driver API code
// ============================================================================
extern "C" {

void* create_cublas_reference() {
    return new baremetal::CublasReference();
}

void destroy_cublas_reference(void* ref) {
    delete static_cast<baremetal::CublasReference*>(ref);
}

double benchmark_cublas(void* ref, int M, int N, int K) {
    auto* cublas = static_cast<baremetal::CublasReference*>(ref);
    return cublas->benchmark(M, N, K);
}

void compute_cublas_reference(
    void* ref,
    int M, int N, int K,
    float alpha,
    const float* h_A,
    const float* h_B,
    float beta,
    float* h_C) {
    auto* cublas = static_cast<baremetal::CublasReference*>(ref);
    cublas->compute_reference(M, N, K, alpha, h_A, h_B, beta, h_C);
}

int verify_result(
    const float* result,
    const float* reference,
    int M, int N,
    float rtol,
    float atol) {
    return baremetal::verify_sgemm_result(result, reference, M, N, rtol, atol) ? 1 : 0;
}

void cpu_sgemm_reference(
    int M, int N, int K,
    float alpha,
    const float* A, int lda,
    const float* B, int ldb,
    float beta,
    float* C, int ldc) {
    baremetal::sgemm_cpu_reference(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

}  // extern "C"
