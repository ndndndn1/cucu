/**
 * BareMetal-SGEMM: Correctness Test Suite
 *
 * This module provides comprehensive correctness testing for all
 * SGEMM kernel implementations against cuBLAS reference.
 *
 * Tests include:
 * - Small matrices (for debugging)
 * - Various matrix sizes (square and non-square)
 * - Edge cases (non-aligned sizes, single row/column)
 * - Numerical precision tests
 */

#include "cuda_driver_wrapper.hpp"
#include "sgemm_kernels.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <iomanip>

using namespace baremetal;

// Forward declarations
extern "C" {
    void* create_cublas_reference();
    void destroy_cublas_reference(void* ref);
    void compute_cublas_reference(void* ref, int M, int N, int K,
                                  float alpha, const float* A, const float* B,
                                  float beta, float* C);
    void cpu_sgemm_reference(int M, int N, int K,
                             float alpha, const float* A, int lda,
                             const float* B, int ldb,
                             float beta, float* C, int ldc);
}

// ============================================================================
// Test Configuration
// ============================================================================
struct TestConfig {
    float rtol = 1e-4f;    // Relative tolerance
    float atol = 1e-5f;    // Absolute tolerance
    bool verbose = false;
};

// ============================================================================
// Test Result
// ============================================================================
struct TestResult {
    std::string name;
    bool passed;
    float max_error;
    float avg_error;
    std::string message;
};

// ============================================================================
// Test Runner
// ============================================================================
class CorrectnessTest {
public:
    CorrectnessTest(CudaContext& ctx, const std::string& ptx_dir = "./ptx")
        : ctx_(ctx), ptx_dir_(ptx_dir) {
        cublas_ref_ = create_cublas_reference();
    }

    ~CorrectnessTest() {
        if (cublas_ref_) {
            destroy_cublas_reference(cublas_ref_);
        }
    }

    /**
     * Test a specific kernel against cuBLAS reference.
     */
    TestResult test_kernel(
        OptLevel level,
        int M, int N, int K,
        float alpha = 1.0f, float beta = 0.0f,
        const TestConfig& config = TestConfig{}) {

        TestResult result;
        result.name = opt_level_name(level) + " " +
                      std::to_string(M) + "x" + std::to_string(N) + "x" + std::to_string(K);

        try {
            // Allocate host memory
            std::vector<float> h_A(M * K);
            std::vector<float> h_B(K * N);
            std::vector<float> h_C(M * N);
            std::vector<float> h_C_ref(M * N);
            std::vector<float> h_C_result(M * N);

            // Initialize with random data
            std::mt19937 gen(42);
            std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
            for (auto& v : h_A) v = dist(gen);
            for (auto& v : h_B) v = dist(gen);
            for (auto& v : h_C) v = dist(gen);
            h_C_ref = h_C;
            h_C_result = h_C;

            // Compute reference using cuBLAS
            compute_cublas_reference(cublas_ref_, M, N, K, alpha,
                                     h_A.data(), h_B.data(), beta, h_C_ref.data());

            // Allocate device memory
            DeviceMemory d_A(M * K * sizeof(float));
            DeviceMemory d_B(K * N * sizeof(float));
            DeviceMemory d_C(M * N * sizeof(float));

            d_A.copy_from_host(h_A.data(), M * K * sizeof(float));
            d_B.copy_from_host(h_B.data(), K * N * sizeof(float));
            d_C.copy_from_host(h_C_result.data(), M * N * sizeof(float));

            // Run our kernel
            JitOptions jit_opts;
            SgemmLauncher launcher(ctx_, level, ptx_dir_, jit_opts);
            SgemmParams params(M, N, K, alpha, beta, d_A.get(), d_B.get(), d_C.get());

            launcher.execute(params);
            ctx_.synchronize();

            // Copy result back
            d_C.copy_to_host(h_C_result.data(), M * N * sizeof(float));

            // Compare results
            float max_err = 0.0f;
            float sum_err = 0.0f;
            int error_count = 0;
            int first_error_idx = -1;

            for (int i = 0; i < M * N; ++i) {
                float diff = std::abs(h_C_result[i] - h_C_ref[i]);
                float rel_err = diff / (std::abs(h_C_ref[i]) + 1e-10f);

                sum_err += diff;
                if (diff > max_err) max_err = diff;

                if (diff > config.atol && rel_err > config.rtol) {
                    if (first_error_idx < 0) first_error_idx = i;
                    error_count++;
                }
            }

            result.max_error = max_err;
            result.avg_error = sum_err / (M * N);

            if (error_count == 0) {
                result.passed = true;
                result.message = "PASS";
            } else {
                result.passed = false;
                result.message = "FAIL: " + std::to_string(error_count) +
                                " errors, first at idx " + std::to_string(first_error_idx);

                if (config.verbose && first_error_idx >= 0) {
                    int row = first_error_idx / N;
                    int col = first_error_idx % N;
                    result.message += " [" + std::to_string(row) + "," +
                                      std::to_string(col) + "] = " +
                                      std::to_string(h_C_result[first_error_idx]) +
                                      " vs " + std::to_string(h_C_ref[first_error_idx]);
                }
            }

        } catch (const std::exception& e) {
            result.passed = false;
            result.max_error = -1;
            result.avg_error = -1;
            result.message = std::string("EXCEPTION: ") + e.what();
        }

        return result;
    }

    /**
     * Run all tests for a specific optimization level.
     */
    std::vector<TestResult> test_level(OptLevel level, const TestConfig& config = TestConfig{}) {
        std::vector<TestResult> results;

        // Test matrix sizes
        std::vector<std::tuple<int, int, int>> sizes = {
            // Small (for debugging)
            {16, 16, 16},
            {32, 32, 32},
            {64, 64, 64},

            // Medium
            {128, 128, 128},
            {256, 256, 256},
            {512, 512, 512},

            // Large
            {1024, 1024, 1024},
            {2048, 2048, 2048},

            // Non-square
            {128, 256, 64},
            {256, 128, 512},
            {1000, 1000, 1000},  // Non-power-of-2

            // Edge cases
            {1, 1024, 1024},    // Single row
            {1024, 1, 1024},    // Single column
            {127, 255, 63},     // Non-aligned
        };

        for (const auto& [M, N, K] : sizes) {
            results.push_back(test_kernel(level, M, N, K, 1.0f, 0.0f, config));
        }

        // Test with beta != 0
        results.push_back(test_kernel(level, 256, 256, 256, 1.0f, 1.0f, config));
        results.push_back(test_kernel(level, 256, 256, 256, 0.5f, 0.5f, config));

        return results;
    }

    /**
     * Run all tests for all optimization levels.
     */
    void run_all_tests(const TestConfig& config = TestConfig{}) {
        std::cout << "╔═══════════════════════════════════════════════════════════════════════════╗" << std::endl;
        std::cout << "║                    BareMetal-SGEMM Correctness Tests                       ║" << std::endl;
        std::cout << "╚═══════════════════════════════════════════════════════════════════════════╝" << std::endl;

        int total_tests = 0;
        int passed_tests = 0;

        for (int lvl = static_cast<int>(OptLevel::Naive);
             lvl <= static_cast<int>(OptLevel::AsyncCopy);
             ++lvl) {

            OptLevel level = static_cast<OptLevel>(lvl);
            std::cout << "\n### Testing " << opt_level_name(level) << " ###" << std::endl;

            try {
                auto results = test_level(level, config);

                for (const auto& r : results) {
                    total_tests++;
                    if (r.passed) passed_tests++;

                    std::cout << std::setw(40) << std::left << r.name << " ";
                    if (r.passed) {
                        std::cout << "\033[32m" << r.message << "\033[0m";  // Green
                    } else {
                        std::cout << "\033[31m" << r.message << "\033[0m";  // Red
                    }
                    std::cout << " (max_err=" << std::scientific << std::setprecision(2)
                              << r.max_error << ")" << std::endl;
                }

            } catch (const std::exception& e) {
                std::cout << "\033[31mFailed to test level: " << e.what() << "\033[0m" << std::endl;
            }
        }

        // Summary
        std::cout << "\n╔═══════════════════════════════════════════════════════════════════════════╗" << std::endl;
        std::cout << "║                              Test Summary                                  ║" << std::endl;
        std::cout << "╠═══════════════════════════════════════════════════════════════════════════╣" << std::endl;
        std::cout << "║ Total:  " << std::setw(4) << total_tests
                  << " | Passed: " << std::setw(4) << passed_tests
                  << " | Failed: " << std::setw(4) << (total_tests - passed_tests)
                  << " | Rate: " << std::setw(6) << std::fixed << std::setprecision(1)
                  << (100.0 * passed_tests / total_tests) << "%"
                  << "     ║" << std::endl;
        std::cout << "╚═══════════════════════════════════════════════════════════════════════════╝" << std::endl;

        if (passed_tests == total_tests) {
            std::cout << "\033[32mAll tests PASSED!\033[0m" << std::endl;
        } else {
            std::cout << "\033[31mSome tests FAILED!\033[0m" << std::endl;
        }
    }

private:
    CudaContext& ctx_;
    std::string ptx_dir_;
    void* cublas_ref_ = nullptr;
};

// ============================================================================
// Main
// ============================================================================
int main(int argc, char** argv) {
    std::string ptx_path = "./ptx";
    bool verbose = false;

    // Simple argument parsing
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "-p" && i + 1 < argc) {
            ptx_path = argv[++i];
        } else if (std::string(argv[i]) == "-v") {
            verbose = true;
        } else if (std::string(argv[i]) == "-h") {
            std::cout << "Usage: " << argv[0] << " [-p <ptx_path>] [-v] [-h]" << std::endl;
            return 0;
        }
    }

    try {
        std::cout << "Initializing CUDA..." << std::endl;
        CudaContext ctx(0);
        ctx.info().print();

        TestConfig config;
        config.verbose = verbose;
        config.rtol = 1e-3f;  // Slightly relaxed for numerical stability
        config.atol = 1e-4f;

        CorrectnessTest tester(ctx, ptx_path);
        tester.run_all_tests(config);

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
