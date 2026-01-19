/**
 * BareMetal-SGEMM: Main Benchmark Executable
 *
 * This is the main entry point for benchmarking the SGEMM implementations.
 *
 * Usage:
 *   ./sgemm_benchmark [options]
 *
 * Options:
 *   -m <size>   : Matrix M dimension (default: 4096)
 *   -n <size>   : Matrix N dimension (default: 4096)
 *   -k <size>   : Matrix K dimension (default: 4096)
 *   -l <level>  : Optimization level 0-5 (default: all)
 *   -i <iters>  : Benchmark iterations (default: 20)
 *   -w <iters>  : Warmup iterations (default: 5)
 *   -s          : Run size sweep (256, 512, 1024, 2048, 4096)
 *   -v          : Verbose output
 *   -p <path>   : Path to PTX files (default: ./ptx)
 *   -h          : Show help
 */

#include "cuda_driver_wrapper.hpp"
#include "sgemm_kernels.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <getopt.h>

using namespace baremetal;

// Forward declarations for cuBLAS reference (from cublas_reference.cu)
extern "C" {
    void* create_cublas_reference();
    void destroy_cublas_reference(void* ref);
    double benchmark_cublas(void* ref, int M, int N, int K);
    void compute_cublas_reference(void* ref, int M, int N, int K,
                                  float alpha, const float* A, const float* B,
                                  float beta, float* C);
    int verify_result(const float* result, const float* reference,
                      int M, int N, float rtol, float atol);
}

// Include benchmark runner
#include "benchmark_runner.cpp"

void print_usage(const char* prog_name) {
    std::cout << "BareMetal-SGEMM Benchmark" << std::endl;
    std::cout << "=========================" << std::endl;
    std::cout << std::endl;
    std::cout << "Usage: " << prog_name << " [options]" << std::endl;
    std::cout << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  -m <size>   Matrix M dimension (default: 4096)" << std::endl;
    std::cout << "  -n <size>   Matrix N dimension (default: 4096)" << std::endl;
    std::cout << "  -k <size>   Matrix K dimension (default: 4096)" << std::endl;
    std::cout << "  -l <level>  Optimization level 0-5 (default: all)" << std::endl;
    std::cout << "              0: Naive" << std::endl;
    std::cout << "              1: Coalesced" << std::endl;
    std::cout << "              2: Tiled" << std::endl;
    std::cout << "              3: Register Blocking" << std::endl;
    std::cout << "              4: Double Buffer" << std::endl;
    std::cout << "              5: Async Copy (Ampere+)" << std::endl;
    std::cout << "  -i <iters>  Benchmark iterations (default: 20)" << std::endl;
    std::cout << "  -w <iters>  Warmup iterations (default: 5)" << std::endl;
    std::cout << "  -s          Run size sweep" << std::endl;
    std::cout << "  -c          Compare with cuBLAS" << std::endl;
    std::cout << "  -V          Verify correctness against cuBLAS" << std::endl;
    std::cout << "  -v          Verbose output" << std::endl;
    std::cout << "  -p <path>   Path to PTX files (default: ./ptx)" << std::endl;
    std::cout << "  -h          Show this help" << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  " << prog_name << " -m 2048 -n 2048 -k 2048 -l 5 -c" << std::endl;
    std::cout << "  " << prog_name << " -s -c  # Run size sweep with cuBLAS comparison" << std::endl;
}

int main(int argc, char** argv) {
    // Default configuration
    int M = 4096, N = 4096, K = 4096;
    int opt_level = -1;  // -1 means run all
    int bench_iters = 20;
    int warmup_iters = 5;
    bool run_sweep = false;
    bool compare_cublas = false;
    bool verify_correctness = false;
    bool verbose = false;
    std::string ptx_path = "./ptx";

    // Parse command line arguments
    int opt;
    while ((opt = getopt(argc, argv, "m:n:k:l:i:w:scVvp:h")) != -1) {
        switch (opt) {
            case 'm': M = std::stoi(optarg); break;
            case 'n': N = std::stoi(optarg); break;
            case 'k': K = std::stoi(optarg); break;
            case 'l': opt_level = std::stoi(optarg); break;
            case 'i': bench_iters = std::stoi(optarg); break;
            case 'w': warmup_iters = std::stoi(optarg); break;
            case 's': run_sweep = true; break;
            case 'c': compare_cublas = true; break;
            case 'V': verify_correctness = true; break;
            case 'v': verbose = true; break;
            case 'p': ptx_path = optarg; break;
            case 'h':
                print_usage(argv[0]);
                return 0;
            default:
                print_usage(argv[0]);
                return 1;
        }
    }

    try {
        // Initialize CUDA context using Driver API
        std::cout << "Initializing CUDA Driver API..." << std::endl;
        CudaContext ctx(0);  // Use first GPU
        ctx.info().print();

        // Create benchmark runner
        BenchmarkRunner runner(ctx, ptx_path);

        // Get cuBLAS baseline if requested
        double cublas_gflops = 0.0;
        void* cublas_ref = nullptr;

        if (compare_cublas || verify_correctness) {
            std::cout << "\nBenchmarking cuBLAS reference..." << std::endl;
            cublas_ref = create_cublas_reference();
            cublas_gflops = benchmark_cublas(cublas_ref, M, N, K);
            std::cout << "cuBLAS GFLOP/s: " << cublas_gflops << std::endl;
            runner.set_cublas_gflops(cublas_gflops);
        }

        // Configure benchmark
        BenchmarkConfig config;
        config.warmup_iterations = warmup_iters;
        config.benchmark_iterations = bench_iters;
        config.verify_correctness = verify_correctness;
        config.verbose = verbose;

        // Run benchmarks
        if (run_sweep) {
            std::vector<int> sizes = {256, 512, 1024, 2048, 4096};
            runner.benchmark_sweep(sizes, config);
        } else if (opt_level >= 0 && opt_level <= 5) {
            // Single optimization level
            auto result = runner.benchmark(static_cast<OptLevel>(opt_level), M, N, K, config);
            result.print();
        } else {
            // All optimization levels
            auto results = runner.benchmark_all(M, N, K, config);
            runner.print_summary(results);

            // Print individual results
            for (const auto& r : results) {
                r.print();
            }
        }

        // Cleanup
        if (cublas_ref) {
            destroy_cublas_reference(cublas_ref);
        }

        std::cout << "\nBenchmark complete!" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
