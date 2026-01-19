/**
 * BareMetal-SGEMM: Naive Kernel (Level 0)
 *
 * This is the baseline implementation with no optimizations.
 * Each thread computes exactly one element of C.
 *
 * Performance characteristics:
 * - No memory coalescing consideration
 * - No data reuse (each element loaded multiple times)
 * - Very low arithmetic intensity
 *
 * Expected performance: ~1-5% of cuBLAS
 *
 * SASS Analysis Points:
 * - Look for LDG.E (32-bit global loads) - should see many of these
 * - High "Long Scoreboard" stall cycles expected
 * - Memory bound (check SOL analysis)
 */

extern "C" __global__ void sgemm_naive(
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
    // Each thread computes one element of C
    // Thread (tx, ty) in block (bx, by) computes C[row, col]
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        // Compute dot product of row of A and column of B
        float sum = 0.0f;

        for (int k = 0; k < K; ++k) {
            // A[row, k] * B[k, col]
            sum += A[row * lda + k] * B[k * ldb + col];
        }

        // C = alpha * A * B + beta * C
        C[row * ldc + col] = alpha * sum + beta * C[row * ldc + col];
    }
}

/**
 * Analysis Notes for Nsight Compute:
 *
 * 1. Memory Access Pattern:
 *    - A: Threads in the same warp access different rows
 *          -> Non-coalesced reads (bad!)
 *    - B: Threads in the same warp access the same column but different rows
 *          -> Strided access pattern (bad!)
 *
 * 2. Data Reuse:
 *    - Each element of A is loaded M times (once per output row)
 *    - Each element of B is loaded N times (once per output column)
 *    - No reuse at all!
 *
 * 3. Arithmetic Intensity:
 *    - FLOP: 2*M*N*K (K FMAs per output, M*N outputs)
 *    - Bytes: M*K + K*N + M*N (loaded) + M*N (stored)
 *    - But actual bytes transferred is MUCH higher due to no reuse
 *
 * 4. Expected Stall Reasons:
 *    - "Long Scoreboard" > 80% (waiting for global memory)
 *    - "Memory Throttle" if bandwidth is saturated
 */
