/**
 * BareMetal-SGEMM: Coalesced + Vectorized Kernel (Level 1)
 *
 * Optimization: Memory Coalescing and Vectorization
 *
 * Key improvements over naive:
 * 1. Use float4 for 128-bit vectorized loads (LDG.E.128)
 * 2. Transpose B to enable coalesced access for both matrices
 * 3. Adjacent threads access adjacent memory locations
 *
 * Performance characteristics:
 * - Global Memory Load Efficiency: ~90%+ (target)
 * - Still no data reuse, but much better bandwidth utilization
 *
 * SASS Verification:
 * - Look for LDG.E.128 instead of LDG.E
 * - Global Load Efficiency metric should be >80%
 *
 * Expected performance: ~5-15% of cuBLAS
 */

// Helper to ensure 128-bit alignment hint for the compiler
#define FETCH_FLOAT4(pointer) (reinterpret_cast<const float4*>(&(pointer))[0])

extern "C" __global__ void sgemm_coalesced(
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
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;

        // Process 4 elements at a time using float4
        // This enables 128-bit loads (LDG.E.128)
        const int K_vec = K / 4;
        const int K_rem = K % 4;

        // Vectorized loop
        // Note: For this to generate LDG.E.128, pointers must be 16-byte aligned
        for (int k = 0; k < K_vec; ++k) {
            // Load 4 elements from A[row, k*4 : k*4+4]
            float4 a_vec = FETCH_FLOAT4(A[row * lda + k * 4]);

            // Load 4 elements from B[k*4 : k*4+4, col]
            // Note: B is still in row-major, so we need 4 separate loads
            // For truly coalesced access, B should be pre-transposed
            float b0 = B[(k * 4 + 0) * ldb + col];
            float b1 = B[(k * 4 + 1) * ldb + col];
            float b2 = B[(k * 4 + 2) * ldb + col];
            float b3 = B[(k * 4 + 3) * ldb + col];

            sum += a_vec.x * b0;
            sum += a_vec.y * b1;
            sum += a_vec.z * b2;
            sum += a_vec.w * b3;
        }

        // Handle remainder
        for (int k = K_vec * 4; k < K; ++k) {
            sum += A[row * lda + k] * B[k * ldb + col];
        }

        // C = alpha * A * B + beta * C
        C[row * ldc + col] = alpha * sum + beta * C[row * ldc + col];
    }
}

/**
 * Alternative: Truly coalesced version assuming B is transposed (B^T)
 *
 * If B is stored as B^T (column-major for original B),
 * then B^T[col, k] = B[k, col], and we can do coalesced loads for both.
 */
extern "C" __global__ void sgemm_coalesced_bt(
    const float* __restrict__ A,
    const float* __restrict__ BT,  // B transposed: BT[N][K]
    float* __restrict__ C,
    const int M,
    const int N,
    const int K,
    const int lda,
    const int ldbt,  // leading dimension of BT (= K)
    const int ldc,
    const float alpha,
    const float beta)
{
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        const int K_vec = K / 4;

        // Both A and BT can now be accessed with float4
        for (int k = 0; k < K_vec; ++k) {
            // A[row, k*4:k*4+4] - coalesced within warp if threads have consecutive rows
            float4 a_vec = FETCH_FLOAT4(A[row * lda + k * 4]);
            // BT[col, k*4:k*4+4] - coalesced within warp if threads have consecutive cols
            float4 b_vec = FETCH_FLOAT4(BT[col * ldbt + k * 4]);

            sum += a_vec.x * b_vec.x;
            sum += a_vec.y * b_vec.y;
            sum += a_vec.z * b_vec.z;
            sum += a_vec.w * b_vec.w;
        }

        // Remainder
        for (int k = K_vec * 4; k < K; ++k) {
            sum += A[row * lda + k] * BT[col * ldbt + k];
        }

        C[row * ldc + col] = alpha * sum + beta * C[row * ldc + col];
    }
}

/**
 * Analysis Notes for Nsight Compute:
 *
 * 1. Check "Memory Workload Analysis" section:
 *    - L1/TEX Hit Rate: Should see improvement
 *    - Sectors/Request for Global Loads: Should be close to 1.0 (perfect coalescing)
 *
 * 2. Source/SASS correlation:
 *    - FETCH_FLOAT4 should generate LDG.E.128
 *    - If you see LDG.E (32-bit), check alignment!
 *
 * 3. Why B access is still problematic:
 *    - B[k][col] where threads have different cols but same k
 *    - Adjacent threads access adjacent columns -> GOOD for coalescing
 *    - But we're reading from different rows of B -> cache thrashing
 *
 * 4. To fully optimize B access, you need:
 *    - Pre-transpose B, OR
 *    - Use shared memory tiling (next level)
 */
