/**
 * BareMetal-SGEMM: Shared Memory Tiled Kernel (Level 2)
 *
 * Optimization: Shared Memory Tiling with Bank Conflict Avoidance
 *
 * Key concepts:
 * 1. Divide matrices into tiles that fit in shared memory
 * 2. Load tile cooperatively, then compute from fast shared memory
 * 3. Data reuse: each element loaded once per tile, used BLOCK_SIZE times
 *
 * Bank Conflict Analysis:
 * - Shared memory has 32 banks (4 bytes each)
 * - Consecutive 4-byte words go to consecutive banks
 * - Bank conflict: multiple threads in a warp access different addresses
 *   in the same bank simultaneously
 *
 * Performance characteristics:
 * - Reduces global memory traffic by BLOCK_SIZE factor
 * - Must avoid bank conflicts for maximum shared memory bandwidth
 *
 * SASS Verification:
 * - LDS.128 for shared memory loads (if vectorized)
 * - Check "L1 Wavefronts Shared Excessive" for bank conflicts
 *   - Value > 1.0 indicates bank conflicts
 *   - Target: exactly 1.0
 *
 * Expected performance: ~20-40% of cuBLAS
 */

#define BLOCK_SIZE 16
#define BLOCK_SIZE_K 16

// Padding to avoid bank conflicts
// With 32 banks and 4-byte floats, conflict occurs when:
// (addr1 / 4) % 32 == (addr2 / 4) % 32 and addr1 != addr2
// Adding 1 float of padding shifts every row by 1 bank
#define SMEM_PADDING 1

extern "C" __global__ void sgemm_tiled(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int M,
    const int N,
    const int K,
    const int lda,
    const int ldb,
    const int ldc,
    const float alpha,
    const float beta)
{
    // Shared memory tiles with padding for bank conflict avoidance
    // A_tile: BLOCK_SIZE rows x BLOCK_SIZE_K cols
    // B_tile: BLOCK_SIZE_K rows x BLOCK_SIZE cols
    __shared__ float A_tile[BLOCK_SIZE][BLOCK_SIZE_K + SMEM_PADDING];
    __shared__ float B_tile[BLOCK_SIZE_K][BLOCK_SIZE + SMEM_PADDING];

    // Thread indices
    const int tx = threadIdx.x;  // Column within block
    const int ty = threadIdx.y;  // Row within block

    // Global indices for this thread's output element
    const int row = blockIdx.y * BLOCK_SIZE + ty;
    const int col = blockIdx.x * BLOCK_SIZE + tx;

    // Accumulator for the dot product
    float sum = 0.0f;

    // Number of tiles needed to cover K dimension
    const int num_tiles = (K + BLOCK_SIZE_K - 1) / BLOCK_SIZE_K;

    for (int t = 0; t < num_tiles; ++t) {
        // Collaborative loading of tiles into shared memory
        // Each thread loads one element of A_tile and one element of B_tile

        // Load A[row, t*BLOCK_SIZE_K + tx]
        const int a_col = t * BLOCK_SIZE_K + tx;
        if (row < M && a_col < K) {
            A_tile[ty][tx] = A[row * lda + a_col];
        } else {
            A_tile[ty][tx] = 0.0f;
        }

        // Load B[t*BLOCK_SIZE_K + ty, col]
        const int b_row = t * BLOCK_SIZE_K + ty;
        if (b_row < K && col < N) {
            B_tile[ty][tx] = B[b_row * ldb + col];
        } else {
            B_tile[ty][tx] = 0.0f;
        }

        // Synchronize to ensure tile is fully loaded
        __syncthreads();

        // Compute partial dot product using data in shared memory
        // Each element in shared memory is accessed BLOCK_SIZE times
        // -> Data reuse factor = BLOCK_SIZE
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE_K; ++k) {
            sum += A_tile[ty][k] * B_tile[k][tx];
        }

        // Synchronize before loading next tile
        // (to avoid overwriting data that other threads still need)
        __syncthreads();
    }

    // Write result to global memory
    if (row < M && col < N) {
        C[row * ldc + col] = alpha * sum + beta * C[row * ldc + col];
    }
}

/**
 * Version with explicit bank conflict analysis using PTX
 *
 * This version uses inline PTX to demonstrate the memory access pattern
 * and makes it easier to correlate with SASS output.
 */
extern "C" __global__ void sgemm_tiled_ptx(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int M,
    const int N,
    const int K,
    const int lda,
    const int ldb,
    const int ldc,
    const float alpha,
    const float beta)
{
    // Shared memory with padding
    __shared__ float A_tile[BLOCK_SIZE][BLOCK_SIZE_K + SMEM_PADDING];
    __shared__ float B_tile[BLOCK_SIZE_K][BLOCK_SIZE + SMEM_PADDING];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * BLOCK_SIZE + ty;
    const int col = blockIdx.x * BLOCK_SIZE + tx;

    float sum = 0.0f;
    const int num_tiles = (K + BLOCK_SIZE_K - 1) / BLOCK_SIZE_K;

    for (int t = 0; t < num_tiles; ++t) {
        // Load A tile
        const int a_col = t * BLOCK_SIZE_K + tx;
        if (row < M && a_col < K) {
            A_tile[ty][tx] = A[row * lda + a_col];
        } else {
            A_tile[ty][tx] = 0.0f;
        }

        // Load B tile
        const int b_row = t * BLOCK_SIZE_K + ty;
        if (b_row < K && col < N) {
            B_tile[ty][tx] = B[b_row * ldb + col];
        } else {
            B_tile[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Inner loop with PTX-level FMA
        // This helps ensure the compiler generates optimal FFMA instructions
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE_K; ++k) {
            float a = A_tile[ty][k];
            float b = B_tile[k][tx];

            // Use inline PTX for FMA to ensure it's not optimized away
            // FFMA d, a, b, c computes d = a * b + c
            asm volatile(
                "fma.rn.f32 %0, %1, %2, %0;"
                : "+f"(sum)
                : "f"(a), "f"(b)
            );
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        // Final scaling with PTX
        float result;
        float c_val = C[row * ldc + col];

        // result = alpha * sum + beta * c_val
        asm volatile(
            "fma.rn.f32 %0, %1, %2, %3;\n\t"  // tmp = alpha * sum
            "fma.rn.f32 %0, %4, %5, %0;"       // result = beta * c_val + tmp
            : "=f"(result)
            : "f"(alpha), "f"(sum), "f"(0.0f), "f"(beta), "f"(c_val)
        );

        C[row * ldc + col] = result;
    }
}

/**
 * Bank Conflict Analysis:
 *
 * Shared Memory Layout (without padding):
 *   A_tile[0][0], A_tile[0][1], ..., A_tile[0][15]  -> Banks 0-15
 *   A_tile[1][0], A_tile[1][1], ..., A_tile[1][15]  -> Banks 0-15 (conflict!)
 *
 * When threads ty=0..15 read A_tile[ty][k] for same k:
 *   Thread 0: A_tile[0][k] -> Bank k
 *   Thread 1: A_tile[1][k] -> Bank k  <- CONFLICT with Thread 0!
 *   ...
 *
 * With padding (+1):
 *   A_tile[0][0..15] -> Banks 0-15
 *   A_tile[1][0..15] -> Banks 1-16 (shifted by 1)
 *
 * Now when threads read A_tile[ty][k]:
 *   Thread 0: A_tile[0][k] -> Bank (0*17 + k) % 32
 *   Thread 1: A_tile[1][k] -> Bank (1*17 + k) % 32
 *   ...
 *   All different banks -> NO CONFLICT!
 *
 * Nsight Compute Verification:
 * 1. Open "Memory Workload Analysis" -> "L1/TEX Cache"
 * 2. Look for "Wavefronts Shared" metric
 * 3. Ideal value is 1.0 (no bank conflicts)
 * 4. Values > 1.0 indicate bank conflicts (replay cycles)
 */
