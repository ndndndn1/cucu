/**
 * BareMetal-SGEMM: SGEMM Kernel Launcher
 *
 * This file handles the loading and launching of SGEMM kernels
 * at different optimization levels.
 */

#include "sgemm_kernels.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>

namespace baremetal {

// ============================================================================
// SgemmMetrics Implementation
// ============================================================================
void SgemmMetrics::print() const {
    std::cout << "=== SGEMM Performance Metrics ===" << std::endl;
    std::cout << "Elapsed Time:       " << elapsed_ms << " ms" << std::endl;
    std::cout << "GFLOP/s:           " << gflops << std::endl;
    std::cout << "Memory BW:          " << memory_bandwidth_gb << " GB/s" << std::endl;
    std::cout << "Arithmetic Int.:    " << arithmetic_intensity << " FLOP/byte" << std::endl;
    std::cout << "Efficiency (Peak):  " << (efficiency_vs_peak * 100.0) << "%" << std::endl;
    std::cout << "Efficiency (cuBLAS):" << (efficiency_vs_cublas * 100.0) << "%" << std::endl;
    std::cout << "=================================" << std::endl;
}

// ============================================================================
// SgemmLauncher Implementation
// ============================================================================
SgemmLauncher::SgemmLauncher(
    CudaContext& ctx,
    OptLevel level,
    const std::string& ptx_dir,
    const JitOptions& jit_opts)
    : level_(level)
    , ctx_(ctx) {
    load_kernel(ptx_dir, jit_opts);
}

void SgemmLauncher::load_kernel(
    const std::string& ptx_dir,
    const JitOptions& jit_opts) {

    // Construct PTX file path
    std::string ptx_file = ptx_dir + "/" + opt_level_ptx_file(level_);

    // Read PTX file
    std::ifstream file(ptx_file);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open PTX file: " + ptx_file);
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    ptx_source_ = buffer.str();

    // Load module from PTX (JIT compilation)
    module_ = CudaModule::from_ptx(ptx_source_, jit_opts);

    // Get kernel function
    kernel_func_ = module_->get_function(opt_level_kernel_name(level_));

    std::cout << "[SgemmLauncher] Loaded kernel: " << opt_level_name(level_)
              << " from " << ptx_file << std::endl;
}

LaunchConfig SgemmLauncher::get_launch_config(int M, int N, int K) const {
    dim3 grid, block;
    size_t smem_size = 0;

    switch (level_) {
        case OptLevel::Naive:
        case OptLevel::Coalesced:
            // Simple 16x16 thread block, each thread computes one element
            block = dim3(BLOCK_SIZE, BLOCK_SIZE);
            grid = dim3((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                       (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
            break;

        case OptLevel::Tiled:
            // 16x16 thread block with shared memory tiling
            block = dim3(BLOCK_SIZE, BLOCK_SIZE);
            grid = dim3((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                       (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
            // Two tiles for A and B
            smem_size = 2 * BLOCK_SIZE * BLOCK_SIZE * sizeof(float);
            break;

        case OptLevel::RegisterBlocking:
            // Each thread computes THREAD_TILE_M x THREAD_TILE_N elements
            // Block size: (TILE_N / THREAD_TILE_N) x (TILE_M / THREAD_TILE_M)
            block = dim3(TILE_N / THREAD_TILE_N, TILE_M / THREAD_TILE_M);
            grid = dim3((N + TILE_N - 1) / TILE_N,
                       (M + TILE_M - 1) / TILE_M);
            // Shared memory for A tile (TILE_M x TILE_K) and B tile (TILE_K x TILE_N)
            smem_size = (TILE_M * TILE_K + TILE_K * TILE_N) * sizeof(float);
            break;

        case OptLevel::DoubleBuffer:
        case OptLevel::AsyncCopy:
            // Same grid/block as register blocking, but double the shared memory
            block = dim3(TILE_N / THREAD_TILE_N, TILE_M / THREAD_TILE_M);
            grid = dim3((N + TILE_N - 1) / TILE_N,
                       (M + TILE_M - 1) / TILE_M);
            // Double buffer: 2x shared memory
            smem_size = 2 * (TILE_M * TILE_K + TILE_K * TILE_N) * sizeof(float);
            break;
    }

    return LaunchConfig(grid, block, smem_size);
}

size_t SgemmLauncher::get_shared_mem_size(int M, int N, int K) const {
    switch (level_) {
        case OptLevel::Naive:
        case OptLevel::Coalesced:
            return 0;

        case OptLevel::Tiled:
            return 2 * BLOCK_SIZE * BLOCK_SIZE * sizeof(float);

        case OptLevel::RegisterBlocking:
            return (TILE_M * TILE_K + TILE_K * TILE_N) * sizeof(float);

        case OptLevel::DoubleBuffer:
        case OptLevel::AsyncCopy:
            return 2 * (TILE_M * TILE_K + TILE_K * TILE_N) * sizeof(float);

        default:
            return 0;
    }
}

float SgemmLauncher::get_occupancy(int M, int N, int K) const {
    auto config = get_launch_config(M, N, K);
    int block_size = config.block.x * config.block.y * config.block.z;

    int blocks_per_sm = module_->get_max_active_blocks_per_sm(
        kernel_func_, block_size, config.shared_mem_bytes);

    // Calculate theoretical occupancy
    int max_threads_per_sm = ctx_.info().max_threads_per_sm;
    int active_threads = blocks_per_sm * block_size;

    return static_cast<float>(active_threads) / max_threads_per_sm;
}

void SgemmLauncher::execute(const SgemmParams& params, CUstream stream) {
    auto config = get_launch_config(params.M, params.N, params.K);
    config.stream = stream;

    // Pack kernel parameters
    // The kernel signature is:
    // void sgemm_xxx(float* A, float* B, float* C, int M, int N, int K,
    //                int lda, int ldb, int ldc, float alpha, float beta)
    void* kernel_args[] = {
        const_cast<CUdeviceptr*>(&params.A),
        const_cast<CUdeviceptr*>(&params.B),
        const_cast<CUdeviceptr*>(&params.C),
        const_cast<int*>(&params.M),
        const_cast<int*>(&params.N),
        const_cast<int*>(&params.K),
        const_cast<int*>(&params.lda),
        const_cast<int*>(&params.ldb),
        const_cast<int*>(&params.ldc),
        const_cast<float*>(&params.alpha),
        const_cast<float*>(&params.beta)
    };

    KernelLauncher::launch(kernel_func_, config, kernel_args);
}

}  // namespace baremetal
