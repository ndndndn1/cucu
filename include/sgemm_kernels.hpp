#pragma once

/**
 * BareMetal-SGEMM: Kernel Interface Definitions
 *
 * This header defines the interface for all SGEMM kernel implementations.
 * Each kernel represents a different optimization level:
 *
 * Level 0: Naive          - Direct global memory access
 * Level 1: Coalesced      - Memory coalescing + vectorization (float4)
 * Level 2: Tiled          - Shared memory tiling
 * Level 3: RegBlocking    - Register blocking (thread coarsening)
 * Level 4: DoubleBuffer   - Double buffering with software pipelining
 * Level 5: AsyncCopy      - Ampere async copy (cp.async) + Triple buffering
 *
 * All kernels compute: C = alpha * A * B + beta * C
 * Where A is MxK, B is KxN, C is MxN (row-major)
 */

#include "cuda_driver_wrapper.hpp"
#include <string>

namespace baremetal {

// ============================================================================
// Kernel Configuration Constants
// ============================================================================

// Tile sizes for shared memory tiling
constexpr int TILE_M = 128;  // Tile size in M dimension
constexpr int TILE_N = 128;  // Tile size in N dimension
constexpr int TILE_K = 8;    // Tile size in K dimension (for tiled kernel)

// Block sizes
constexpr int BLOCK_SIZE = 16;  // For naive/tiled kernels

// Register blocking parameters (for Level 3+)
constexpr int THREAD_TILE_M = 8;  // Each thread computes 8x8 results
constexpr int THREAD_TILE_N = 8;

// Warp-level parameters
constexpr int WARP_SIZE = 32;
constexpr int WARPS_PER_BLOCK_M = 4;  // 4 warps in M dimension
constexpr int WARPS_PER_BLOCK_N = 2;  // 2 warps in N dimension

// ============================================================================
// Kernel Names (for PTX loading)
// ============================================================================
namespace kernel_names {
    constexpr const char* NAIVE = "sgemm_naive";
    constexpr const char* COALESCED = "sgemm_coalesced";
    constexpr const char* TILED = "sgemm_tiled";
    constexpr const char* REGISTER_BLOCKING = "sgemm_register_blocking";
    constexpr const char* DOUBLE_BUFFER = "sgemm_double_buffer";
    constexpr const char* ASYNC_COPY = "sgemm_async_copy";
}

// ============================================================================
// Optimization Level Enum
// ============================================================================
enum class OptLevel {
    Naive = 0,
    Coalesced = 1,
    Tiled = 2,
    RegisterBlocking = 3,
    DoubleBuffer = 4,
    AsyncCopy = 5
};

inline std::string opt_level_name(OptLevel level) {
    switch (level) {
        case OptLevel::Naive: return "Naive";
        case OptLevel::Coalesced: return "Coalesced";
        case OptLevel::Tiled: return "Tiled";
        case OptLevel::RegisterBlocking: return "RegisterBlocking";
        case OptLevel::DoubleBuffer: return "DoubleBuffer";
        case OptLevel::AsyncCopy: return "AsyncCopy";
        default: return "Unknown";
    }
}

inline std::string opt_level_ptx_file(OptLevel level) {
    switch (level) {
        case OptLevel::Naive: return "sgemm_naive.ptx";
        case OptLevel::Coalesced: return "sgemm_coalesced.ptx";
        case OptLevel::Tiled: return "sgemm_tiled.ptx";
        case OptLevel::RegisterBlocking: return "sgemm_register_blocking.ptx";
        case OptLevel::DoubleBuffer: return "sgemm_double_buffer.ptx";
        case OptLevel::AsyncCopy: return "sgemm_async_copy.ptx";
        default: return "";
    }
}

inline const char* opt_level_kernel_name(OptLevel level) {
    switch (level) {
        case OptLevel::Naive: return kernel_names::NAIVE;
        case OptLevel::Coalesced: return kernel_names::COALESCED;
        case OptLevel::Tiled: return kernel_names::TILED;
        case OptLevel::RegisterBlocking: return kernel_names::REGISTER_BLOCKING;
        case OptLevel::DoubleBuffer: return kernel_names::DOUBLE_BUFFER;
        case OptLevel::AsyncCopy: return kernel_names::ASYNC_COPY;
        default: return "";
    }
}

// ============================================================================
// SGEMM Parameters
// ============================================================================
struct SgemmParams {
    int M;              // Rows of A and C
    int N;              // Cols of B and C
    int K;              // Cols of A, Rows of B
    float alpha;        // Scalar multiplier for A*B
    float beta;         // Scalar multiplier for C
    CUdeviceptr A;      // Device pointer to A (MxK, row-major)
    CUdeviceptr B;      // Device pointer to B (KxN, row-major)
    CUdeviceptr C;      // Device pointer to C (MxN, row-major)
    int lda;            // Leading dimension of A (usually K)
    int ldb;            // Leading dimension of B (usually N)
    int ldc;            // Leading dimension of C (usually N)

    SgemmParams(int m, int n, int k, float a, float b,
                CUdeviceptr pA, CUdeviceptr pB, CUdeviceptr pC)
        : M(m), N(n), K(k), alpha(a), beta(b),
          A(pA), B(pB), C(pC), lda(k), ldb(n), ldc(n) {}
};

// ============================================================================
// SGEMM Launcher
// ============================================================================
class SgemmLauncher {
public:
    /**
     * Create a launcher for a specific optimization level.
     *
     * @param ctx        CUDA context
     * @param level      Optimization level
     * @param ptx_dir    Directory containing PTX files
     * @param jit_opts   JIT compilation options
     */
    SgemmLauncher(
        CudaContext& ctx,
        OptLevel level,
        const std::string& ptx_dir = "./ptx",
        const JitOptions& jit_opts = JitOptions{});

    /**
     * Execute SGEMM: C = alpha * A * B + beta * C
     */
    void execute(const SgemmParams& params, CUstream stream = nullptr);

    /**
     * Get optimal launch configuration for given matrix dimensions.
     */
    LaunchConfig get_launch_config(int M, int N, int K) const;

    /**
     * Get shared memory requirement in bytes.
     */
    size_t get_shared_mem_size(int M, int N, int K) const;

    /**
     * Get theoretical occupancy.
     */
    float get_occupancy(int M, int N, int K) const;

    OptLevel level() const { return level_; }
    const std::string& ptx_source() const { return ptx_source_; }

private:
    OptLevel level_;
    std::unique_ptr<CudaModule> module_;
    CUfunction kernel_func_ = nullptr;
    CudaContext& ctx_;
    std::string ptx_source_;

    void load_kernel(const std::string& ptx_dir, const JitOptions& jit_opts);
};

// ============================================================================
// Performance Metrics
// ============================================================================
struct SgemmMetrics {
    double elapsed_ms;           // Kernel execution time
    double gflops;              // Achieved GFLOP/s
    double memory_bandwidth_gb; // Achieved memory bandwidth
    double arithmetic_intensity;// FLOP per byte
    double efficiency_vs_peak;  // Percentage of peak compute
    double efficiency_vs_cublas;// Percentage of cuBLAS performance

    void print() const;
};

// Calculate FLOP count for SGEMM
inline double sgemm_flops(int M, int N, int K) {
    // Each output element requires K FMAs (2 ops each) + beta scaling
    return 2.0 * M * N * K;
}

// Calculate minimum memory traffic (bytes)
inline double sgemm_min_bytes(int M, int N, int K) {
    // Read A (MxK), B (KxN), C (MxN), Write C (MxN)
    return sizeof(float) * (M * K + K * N + 2 * M * N);
}

}  // namespace baremetal
