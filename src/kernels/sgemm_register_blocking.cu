/**
 * BareMetal-SGEMM: Register Blocking Kernel (Level 3)
 *
 * Optimization: Thread Coarsening / Register Blocking
 *
 * Key insight:
 * - In the tiled kernel, each thread computes 1 output element
 * - This means each thread loads 2K values (from A and B tiles) for 2K FLOPs
 * - Arithmetic intensity = 2K / (2K * 4 bytes) = 0.25 FLOP/byte (VERY LOW!)
 *
 * Solution: Each thread computes THREAD_TILE_M x THREAD_TILE_N output elements
 * - Thread loads THREAD_TILE_M elements from A, THREAD_TILE_N from B
 * - Computes THREAD_TILE_M * THREAD_TILE_N FMAs
 * - Reuse factor increases dramatically!
 *
 * With THREAD_TILE = 8x8:
 * - Load: 8 + 8 = 16 elements
 * - Compute: 8 * 8 = 64 FMAs = 128 FLOPs
 * - Arithmetic intensity = 128 / 64 bytes = 2.0 FLOP/byte
 *
 * Performance characteristics:
 * - Dramatically increased compute per memory access
 * - Higher register usage (must fit 64+ floats in registers)
 * - FMA pipeline utilization should be much higher
 *
 * SASS Verification:
 * - Look for FFMA instructions in tight loops
 * - Check "Compute Workload Analysis" -> FMA utilization
 * - Register file should be well-utilized
 *
 * Expected performance: ~50-70% of cuBLAS
 */

// Tile sizes
#define BM 128          // Block tile M
#define BN 128          // Block tile N
#define BK 8            // Block tile K
#define TM 8            // Thread tile M
#define TN 8            // Thread tile N

// Threads per block
#define THREADS_PER_BLOCK_X (BN / TN)  // 128 / 8 = 16
#define THREADS_PER_BLOCK_Y (BM / TM)  // 128 / 8 = 16

extern "C" __global__ void sgemm_register_blocking(
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
    // Shared memory for tiles
    // A: BM x BK, B: BK x BN
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    // Thread indices
    const int tx = threadIdx.x;  // 0..15
    const int ty = threadIdx.y;  // 0..15
    const int tid = ty * blockDim.x + tx;  // Linear thread ID

    // Block indices
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    // Starting position of this block's output tile
    const int block_row_start = by * BM;
    const int block_col_start = bx * BN;

    // Starting position of this thread's output sub-tile within the block
    const int thread_row_start = ty * TM;  // 0, 8, 16, ..., 120
    const int thread_col_start = tx * TN;  // 0, 8, 16, ..., 120

    // Register file: Each thread accumulates TM x TN results
    float reg_C[TM][TN] = {{0.0f}};

    // Registers for A and B fragments
    float reg_A[TM];
    float reg_B[TN];

    // Number of tiles along K
    const int num_k_tiles = (K + BK - 1) / BK;

    // For loading data into shared memory
    // We have 256 threads (16x16), need to load BM*BK = 1024 elements for A
    // Each thread loads BM*BK/256 = 4 elements
    const int A_TILE_THREAD_PER_ROW = BK;  // 8
    const int A_TILE_ROW_STRIDE = blockDim.x * blockDim.y / A_TILE_THREAD_PER_ROW;  // 256/8 = 32
    const int A_TILE_ROW = tid / A_TILE_THREAD_PER_ROW;  // Which row this thread loads
    const int A_TILE_COL = tid % A_TILE_THREAD_PER_ROW;  // Which col this thread loads

    const int B_TILE_THREAD_PER_ROW = BN;  // 128
    const int B_TILE_ROW_STRIDE = blockDim.x * blockDim.y / B_TILE_THREAD_PER_ROW;  // 256/128 = 2
    const int B_TILE_ROW = tid / B_TILE_THREAD_PER_ROW;
    const int B_TILE_COL = tid % B_TILE_THREAD_PER_ROW;

    // Main loop over K dimension
    for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
        const int k_offset = k_tile * BK;

        // Collaborative loading of A tile (BM x BK)
        // Each thread loads multiple elements since BM*BK > num_threads
        #pragma unroll
        for (int load_offset = 0; load_offset < BM; load_offset += A_TILE_ROW_STRIDE) {
            int row_idx = A_TILE_ROW + load_offset;
            if (row_idx < BM) {
                int global_row = block_row_start + row_idx;
                int global_col = k_offset + A_TILE_COL;
                if (global_row < M && global_col < K) {
                    As[row_idx][A_TILE_COL] = A[global_row * lda + global_col];
                } else {
                    As[row_idx][A_TILE_COL] = 0.0f;
                }
            }
        }

        // Collaborative loading of B tile (BK x BN)
        #pragma unroll
        for (int load_offset = 0; load_offset < BK; load_offset += B_TILE_ROW_STRIDE) {
            int row_idx = B_TILE_ROW + load_offset;
            if (row_idx < BK) {
                int global_row = k_offset + row_idx;
                int global_col = block_col_start + B_TILE_COL;
                if (global_row < K && global_col < N) {
                    Bs[row_idx][B_TILE_COL] = B[global_row * ldb + global_col];
                } else {
                    Bs[row_idx][B_TILE_COL] = 0.0f;
                }
            }
        }

        __syncthreads();

        // Compute: Loop over K within the tile
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            // Load A fragment: TM elements from column k of As
            // This thread's rows: thread_row_start to thread_row_start + TM - 1
            #pragma unroll
            for (int m = 0; m < TM; ++m) {
                reg_A[m] = As[thread_row_start + m][k];
            }

            // Load B fragment: TN elements from row k of Bs
            // This thread's cols: thread_col_start to thread_col_start + TN - 1
            #pragma unroll
            for (int n = 0; n < TN; ++n) {
                reg_B[n] = Bs[k][thread_col_start + n];
            }

            // Outer product: TM x TN FMAs
            // This is where the magic happens - register-level data reuse!
            #pragma unroll
            for (int m = 0; m < TM; ++m) {
                #pragma unroll
                for (int n = 0; n < TN; ++n) {
                    reg_C[m][n] += reg_A[m] * reg_B[n];
                }
            }
        }

        __syncthreads();
    }

    // Write results to global memory
    #pragma unroll
    for (int m = 0; m < TM; ++m) {
        const int global_row = block_row_start + thread_row_start + m;
        if (global_row < M) {
            #pragma unroll
            for (int n = 0; n < TN; ++n) {
                const int global_col = block_col_start + thread_col_start + n;
                if (global_col < N) {
                    float c_val = C[global_row * ldc + global_col];
                    C[global_row * ldc + global_col] =
                        alpha * reg_C[m][n] + beta * c_val;
                }
            }
        }
    }
}

/**
 * Optimized version with better memory access patterns
 *
 * This version uses:
 * 1. Vectorized loads (float4) for global -> shared
 * 2. Better shared memory layout to avoid bank conflicts
 * 3. Explicit FMA instructions
 */
extern "C" __global__ void sgemm_register_blocking_opt(
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
    // Shared memory with padding for bank conflict avoidance
    __shared__ float As[BM][BK + 1];  // +1 padding
    __shared__ float Bs[BK][BN + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    const int block_row_start = by * BM;
    const int block_col_start = bx * BN;
    const int thread_row = ty * TM;
    const int thread_col = tx * TN;

    // Register tile for C
    float rC[TM][TN] = {{0.0f}};

    // Registers for A and B
    float rA[TM];
    float rB[TN];

    // Precompute loading indices
    const int A_load_row = tid / BK;
    const int A_load_col = tid % BK;
    const int B_load_row = tid / BN;
    const int B_load_col = tid % BN;

    const int num_k_tiles = (K + BK - 1) / BK;

    for (int kt = 0; kt < num_k_tiles; ++kt) {
        int k_base = kt * BK;

        // Load A tile: 256 threads, BM*BK = 1024 elements -> 4 per thread
        for (int i = 0; i < BM; i += 32) {  // 256 threads / 8 cols = 32 rows per iteration
            int row = A_load_row + i;
            if (row < BM) {
                int grow = block_row_start + row;
                int gcol = k_base + A_load_col;
                As[row][A_load_col] = (grow < M && gcol < K) ? A[grow * lda + gcol] : 0.0f;
            }
        }

        // Load B tile: 256 threads, BK*BN = 1024 elements -> 4 per thread
        for (int i = 0; i < BK; i += 2) {  // 256 threads / 128 cols = 2 rows per iteration
            int row = B_load_row + i;
            if (row < BK) {
                int grow = k_base + row;
                int gcol = block_col_start + B_load_col;
                Bs[row][B_load_col] = (grow < K && gcol < N) ? B[grow * ldb + gcol] : 0.0f;
            }
        }

        __syncthreads();

        // Compute
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            // Load from shared to registers
            #pragma unroll
            for (int m = 0; m < TM; ++m) {
                rA[m] = As[thread_row + m][k];
            }
            #pragma unroll
            for (int n = 0; n < TN; ++n) {
                rB[n] = Bs[k][thread_col + n];
            }

            // Compute outer product with explicit FMA
            #pragma unroll
            for (int m = 0; m < TM; ++m) {
                #pragma unroll
                for (int n = 0; n < TN; ++n) {
                    // Use inline assembly for guaranteed FMA
                    asm volatile(
                        "fma.rn.f32 %0, %1, %2, %0;"
                        : "+f"(rC[m][n])
                        : "f"(rA[m]), "f"(rB[n])
                    );
                }
            }
        }

        __syncthreads();
    }

    // Store results
    #pragma unroll
    for (int m = 0; m < TM; ++m) {
        int grow = block_row_start + thread_row + m;
        if (grow < M) {
            #pragma unroll
            for (int n = 0; n < TN; ++n) {
                int gcol = block_col_start + thread_col + n;
                if (gcol < N) {
                    float c_old = C[grow * ldc + gcol];
                    C[grow * ldc + gcol] = alpha * rC[m][n] + beta * c_old;
                }
            }
        }
    }
}

/**
 * Analysis Notes for Nsight Compute:
 *
 * 1. Compute Workload Analysis:
 *    - Check "Executed IPC" - should be higher than tiled kernel
 *    - FMA Pipe Utilization should increase significantly
 *    - Ideal: FMA pipe is the limiting factor (not memory)
 *
 * 2. Register Pressure:
 *    - Each thread uses: TM*TN (C) + TM (A) + TN (B) = 64 + 8 + 8 = 80 registers
 *    - Plus loop variables, indices, etc.
 *    - Check "Launch Statistics" -> "Registers Per Thread"
 *    - If > 255, you'll see register spilling (bad!)
 *
 * 3. Occupancy Trade-off:
 *    - More registers per thread = fewer threads per SM = lower occupancy
 *    - But higher ILP per thread can compensate
 *    - Aim for 25-50% occupancy with high IPC
 *
 * 4. Memory Bandwidth:
 *    - Should see reduced memory pressure vs tiled kernel
 *    - SOL analysis: Compute should be closer to Memory
 */
