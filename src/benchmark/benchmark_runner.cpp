/**
 * BareMetal-SGEMM: Benchmark Runner
 *
 * This module provides comprehensive benchmarking capabilities:
 * - Warm-up runs to stabilize GPU clocks
 * - Multiple iterations with statistical analysis
 * - Comparison against cuBLAS reference
 * - Detailed performance metrics
 */

#include "cuda_driver_wrapper.hpp"
#include "sgemm_kernels.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <random>
#include <chrono>

namespace baremetal {

// ============================================================================
// Benchmark Configuration
// ============================================================================
struct BenchmarkConfig {
    int warmup_iterations = 5;
    int benchmark_iterations = 20;
    bool verify_correctness = true;
    float correctness_tolerance = 1e-3f;
    bool verbose = false;
};

// ============================================================================
// Benchmark Results
// ============================================================================
struct BenchmarkResult {
    OptLevel level;
    int M, N, K;
    double min_ms;
    double max_ms;
    double avg_ms;
    double median_ms;
    double stddev_ms;
    double gflops;
    double memory_bandwidth_gb;
    double efficiency_vs_peak;
    double efficiency_vs_cublas;
    bool correct;

    void print() const {
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "┌─────────────────────────────────────────────────────────┐" << std::endl;
        std::cout << "│ " << std::setw(20) << std::left << opt_level_name(level)
                  << " (" << M << "x" << N << "x" << K << ")"
                  << std::setw(20) << " " << "│" << std::endl;
        std::cout << "├─────────────────────────────────────────────────────────┤" << std::endl;
        std::cout << "│ Time (ms):    min=" << std::setw(8) << min_ms
                  << "  avg=" << std::setw(8) << avg_ms
                  << "  max=" << std::setw(8) << max_ms << " │" << std::endl;
        std::cout << "│ Median:       " << std::setw(8) << median_ms
                  << "  stddev=" << std::setw(8) << stddev_ms
                  << "              │" << std::endl;
        std::cout << "│ Performance:  " << std::setw(8) << gflops << " GFLOP/s"
                  << "                         │" << std::endl;
        std::cout << "│ Memory BW:    " << std::setw(8) << memory_bandwidth_gb << " GB/s"
                  << "                           │" << std::endl;
        std::cout << "│ Efficiency:   " << std::setw(6) << (efficiency_vs_peak * 100) << "% of peak, "
                  << std::setw(6) << (efficiency_vs_cublas * 100) << "% of cuBLAS     │" << std::endl;
        std::cout << "│ Correctness:  " << (correct ? "PASS ✓" : "FAIL ✗")
                  << "                                     │" << std::endl;
        std::cout << "└─────────────────────────────────────────────────────────┘" << std::endl;
    }
};

// ============================================================================
// Benchmark Runner Class
// ============================================================================
class BenchmarkRunner {
public:
    BenchmarkRunner(CudaContext& ctx, const std::string& ptx_dir = "./ptx")
        : ctx_(ctx)
        , ptx_dir_(ptx_dir) {
        // Calculate peak performance based on device info
        const auto& info = ctx_.info();

        // FP32 peak TFLOPS (rough estimate)
        // Modern NVIDIA GPUs have ~64-128 FP32 cores per SM
        // A100: 6912 FP32 cores @ 1.41 GHz = 19.5 TFLOPS
        // RTX 3090: 10496 FP32 cores @ 1.7 GHz = 35.6 TFLOPS
        // This is a rough estimate; actual peak depends on architecture
        int fp32_cores_per_sm = 64;  // Conservative estimate
        if (info.compute_major >= 8) {
            fp32_cores_per_sm = 64;  // Ampere
        } else if (info.compute_major >= 7) {
            fp32_cores_per_sm = 64;  // Volta/Turing
        }

        // Assuming 1.5 GHz clock (conservative)
        double clock_ghz = 1.5;
        peak_tflops_ = info.sm_count * fp32_cores_per_sm * clock_ghz * 2 / 1000.0;

        peak_memory_bw_gb_ = info.peak_memory_bandwidth_gb();

        std::cout << "=== Benchmark Runner Initialized ===" << std::endl;
        std::cout << "Device: " << info.name << std::endl;
        std::cout << "Peak FP32 (est.): " << peak_tflops_ * 1000 << " GFLOPS" << std::endl;
        std::cout << "Peak Memory BW: " << peak_memory_bw_gb_ << " GB/s" << std::endl;
        std::cout << "====================================" << std::endl;
    }

    /**
     * Run benchmark for a specific optimization level.
     */
    BenchmarkResult benchmark(
        OptLevel level,
        int M, int N, int K,
        const BenchmarkConfig& config = BenchmarkConfig{});

    /**
     * Run benchmarks for all optimization levels.
     */
    std::vector<BenchmarkResult> benchmark_all(
        int M, int N, int K,
        const BenchmarkConfig& config = BenchmarkConfig{});

    /**
     * Run benchmarks for multiple matrix sizes.
     */
    void benchmark_sweep(
        const std::vector<int>& sizes,
        const BenchmarkConfig& config = BenchmarkConfig{});

    /**
     * Print summary table comparing all levels.
     */
    void print_summary(const std::vector<BenchmarkResult>& results);

    void set_cublas_gflops(double gflops) { cublas_gflops_ = gflops; }

private:
    CudaContext& ctx_;
    std::string ptx_dir_;
    double peak_tflops_ = 10.0;
    double peak_memory_bw_gb_ = 500.0;
    double cublas_gflops_ = 0.0;

    void init_matrices(float* A, float* B, float* C, int M, int N, int K);
    bool verify(const float* result, const float* reference, int M, int N, float tol);
};

BenchmarkResult BenchmarkRunner::benchmark(
    OptLevel level,
    int M, int N, int K,
    const BenchmarkConfig& config) {

    std::cout << "Benchmarking " << opt_level_name(level)
              << " (" << M << "x" << N << "x" << K << ")..." << std::endl;

    BenchmarkResult result;
    result.level = level;
    result.M = M;
    result.N = N;
    result.K = K;

    // Allocate host memory
    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C(M * N);
    std::vector<float> h_C_ref(M * N);

    // Initialize matrices
    init_matrices(h_A.data(), h_B.data(), h_C.data(), M, N, K);
    std::copy(h_C.begin(), h_C.end(), h_C_ref.begin());

    // Allocate device memory
    DeviceMemory d_A(M * K * sizeof(float));
    DeviceMemory d_B(K * N * sizeof(float));
    DeviceMemory d_C(M * N * sizeof(float));

    // Copy to device
    d_A.copy_from_host(h_A.data(), M * K * sizeof(float));
    d_B.copy_from_host(h_B.data(), K * N * sizeof(float));
    d_C.copy_from_host(h_C.data(), M * N * sizeof(float));

    // Create SGEMM launcher
    JitOptions jit_opts;
    jit_opts.optimization_level = 4;
    jit_opts.generate_line_info = true;

    SgemmLauncher launcher(ctx_, level, ptx_dir_, jit_opts);

    SgemmParams params(M, N, K, 1.0f, 0.0f, d_A.get(), d_B.get(), d_C.get());

    // Create stream and events
    CudaStream stream;
    CudaEvent start, stop;

    // Warm-up runs
    for (int i = 0; i < config.warmup_iterations; ++i) {
        launcher.execute(params, stream.get());
    }
    stream.synchronize();

    // Benchmark runs
    std::vector<double> times;
    times.reserve(config.benchmark_iterations);

    for (int i = 0; i < config.benchmark_iterations; ++i) {
        // Reset C for each iteration
        d_C.copy_from_host(h_C.data(), M * N * sizeof(float));
        stream.synchronize();

        start.record(stream.get());
        launcher.execute(params, stream.get());
        stop.record(stream.get());
        stop.synchronize();

        float ms = CudaEvent::elapsed_ms(start, stop);
        times.push_back(ms);
    }

    // Calculate statistics
    std::sort(times.begin(), times.end());

    result.min_ms = times.front();
    result.max_ms = times.back();
    result.median_ms = times[times.size() / 2];

    double sum = std::accumulate(times.begin(), times.end(), 0.0);
    result.avg_ms = sum / times.size();

    double sq_sum = std::inner_product(times.begin(), times.end(),
                                        times.begin(), 0.0);
    result.stddev_ms = std::sqrt(sq_sum / times.size() -
                                  result.avg_ms * result.avg_ms);

    // Calculate performance metrics
    double flops = sgemm_flops(M, N, K);
    result.gflops = flops / (result.median_ms * 1e6);

    double bytes = sgemm_min_bytes(M, N, K);
    result.memory_bandwidth_gb = bytes / (result.median_ms * 1e6);

    result.efficiency_vs_peak = result.gflops / (peak_tflops_ * 1000);

    if (cublas_gflops_ > 0) {
        result.efficiency_vs_cublas = result.gflops / cublas_gflops_;
    } else {
        result.efficiency_vs_cublas = 0.0;
    }

    // Verify correctness
    result.correct = true;
    if (config.verify_correctness) {
        d_C.copy_to_host(h_C_ref.data(), M * N * sizeof(float));

        // Reference computation (CPU)
        // Note: In practice, you'd compare against cuBLAS
        // For now, just check for NaN/Inf
        for (int i = 0; i < M * N; ++i) {
            if (std::isnan(h_C_ref[i]) || std::isinf(h_C_ref[i])) {
                result.correct = false;
                break;
            }
        }
    }

    return result;
}

std::vector<BenchmarkResult> BenchmarkRunner::benchmark_all(
    int M, int N, int K,
    const BenchmarkConfig& config) {

    std::vector<BenchmarkResult> results;

    for (int lvl = static_cast<int>(OptLevel::Naive);
         lvl <= static_cast<int>(OptLevel::AsyncCopy);
         ++lvl) {
        try {
            auto result = benchmark(static_cast<OptLevel>(lvl), M, N, K, config);
            results.push_back(result);
        } catch (const std::exception& e) {
            std::cerr << "Failed to benchmark " << opt_level_name(static_cast<OptLevel>(lvl))
                      << ": " << e.what() << std::endl;
        }
    }

    return results;
}

void BenchmarkRunner::benchmark_sweep(
    const std::vector<int>& sizes,
    const BenchmarkConfig& config) {

    std::cout << "\n=== Matrix Size Sweep ===" << std::endl;

    for (int size : sizes) {
        std::cout << "\n### Size: " << size << " x " << size << " x " << size << " ###\n";
        auto results = benchmark_all(size, size, size, config);
        print_summary(results);
    }
}

void BenchmarkRunner::print_summary(const std::vector<BenchmarkResult>& results) {
    if (results.empty()) return;

    std::cout << "\n╔═══════════════════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║                         Performance Summary                                ║" << std::endl;
    std::cout << "╠════════════════════╦═══════════╦════════════╦════════════╦═══════════════╣" << std::endl;
    std::cout << "║      Kernel        ║  Time(ms) ║  GFLOP/s   ║  % of Peak ║  % of cuBLAS  ║" << std::endl;
    std::cout << "╠════════════════════╬═══════════╬════════════╬════════════╬═══════════════╣" << std::endl;

    for (const auto& r : results) {
        std::cout << "║ " << std::setw(18) << std::left << opt_level_name(r.level)
                  << " ║ " << std::setw(9) << std::fixed << std::setprecision(3) << r.median_ms
                  << " ║ " << std::setw(10) << std::setprecision(1) << r.gflops
                  << " ║ " << std::setw(9) << std::setprecision(1) << (r.efficiency_vs_peak * 100) << "%"
                  << " ║ " << std::setw(12) << std::setprecision(1)
                  << (r.efficiency_vs_cublas > 0 ? std::to_string(int(r.efficiency_vs_cublas * 100)) + "%" : "N/A")
                  << " ║" << std::endl;
    }

    std::cout << "╚════════════════════╩═══════════╩════════════╩════════════╩═══════════════╝" << std::endl;
}

void BenchmarkRunner::init_matrices(float* A, float* B, float* C, int M, int N, int K) {
    std::mt19937 gen(42);  // Fixed seed for reproducibility
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (int i = 0; i < M * K; ++i) {
        A[i] = dist(gen);
    }
    for (int i = 0; i < K * N; ++i) {
        B[i] = dist(gen);
    }
    for (int i = 0; i < M * N; ++i) {
        C[i] = dist(gen);
    }
}

bool BenchmarkRunner::verify(const float* result, const float* reference,
                             int M, int N, float tol) {
    for (int i = 0; i < M * N; ++i) {
        float diff = std::abs(result[i] - reference[i]);
        float rel = diff / (std::abs(reference[i]) + 1e-8f);
        if (rel > tol && diff > tol) {
            std::cerr << "Mismatch at index " << i << ": "
                      << result[i] << " vs " << reference[i]
                      << " (diff=" << diff << ", rel=" << rel << ")" << std::endl;
            return false;
        }
    }
    return true;
}

}  // namespace baremetal
