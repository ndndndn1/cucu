/**
 * BareMetal-SGEMM: Double Buffering Kernel (Level 4)
 *
 * Optimization: Software Pipelining with Double Buffering
 *
 * Key insight:
 * - Global memory loads have high latency (hundreds of cycles)
 * - While waiting for data, compute units are idle
 * - Solution: Overlap data loading with computation
 *
 * Double Buffering Strategy:
 * 1. Use TWO sets of shared memory buffers
 * 2. While computing on buffer[i % 2], load next tile into buffer[(i+1) % 2]
 * 3. This hides memory latency behind computation
 *
 * Pipeline stages:
 *   Iteration 0: Load tile[0] -> buffer[0]
 *   Iteration 1: Compute buffer[0], Load tile[1] -> buffer[1]
 *   Iteration 2: Compute buffer[1], Load tile[2] -> buffer[0]
 *   ...
 *
 * Performance characteristics:
 * - Memory latency is hidden behind computation
 * - Requires 2x shared memory
 * - "Long Scoreboard" stalls should decrease significantly
 *
 * SASS Verification:
 * - Look for interleaved LDG and FFMA instructions
 * - Stall reason "Long Scoreboard" should be reduced
 *
 * Expected performance: ~70-85% of cuBLAS
 */

#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8

extern "C" __global__ void sgemm_double_buffer(
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
    // Double buffered shared memory
    // Buffer index alternates: 0 -> 1 -> 0 -> 1 -> ...
    __shared__ float As[2][BM][BK + 1];  // +1 for bank conflict avoidance
    __shared__ float Bs[2][BK][BN + 1];

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
    float rA[TM];
    float rB[TN];

    // Loading indices
    const int A_load_row = tid / BK;
    const int A_load_col = tid % BK;
    const int B_load_row = tid / BN;
    const int B_load_col = tid % BN;

    const int num_k_tiles = (K + BK - 1) / BK;

    // Helper to load A tile
    auto load_A_tile = [&](int buf_idx, int k_tile) {
        int k_base = k_tile * BK;
        for (int i = 0; i < BM; i += 32) {
            int row = A_load_row + i;
            if (row < BM) {
                int grow = block_row_start + row;
                int gcol = k_base + A_load_col;
                As[buf_idx][row][A_load_col] =
                    (grow < M && gcol < K) ? A[grow * lda + gcol] : 0.0f;
            }
        }
    };

    // Helper to load B tile
    auto load_B_tile = [&](int buf_idx, int k_tile) {
        int k_base = k_tile * BK;
        for (int i = 0; i < BK; i += 2) {
            int row = B_load_row + i;
            if (row < BK) {
                int grow = k_base + row;
                int gcol = block_col_start + B_load_col;
                Bs[buf_idx][row][B_load_col] =
                    (grow < K && gcol < N) ? B[grow * ldb + gcol] : 0.0f;
            }
        }
    };

    // Helper to compute using a buffer
    auto compute_tile = [&](int buf_idx) {
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            #pragma unroll
            for (int m = 0; m < TM; ++m) {
                rA[m] = As[buf_idx][thread_row + m][k];
            }
            #pragma unroll
            for (int n = 0; n < TN; ++n) {
                rB[n] = Bs[buf_idx][k][thread_col + n];
            }
            #pragma unroll
            for (int m = 0; m < TM; ++m) {
                #pragma unroll
                for (int n = 0; n < TN; ++n) {
                    rC[m][n] += rA[m] * rB[n];
                }
            }
        }
    };

    // Prologue: Load first tile
    if (num_k_tiles > 0) {
        load_A_tile(0, 0);
        load_B_tile(0, 0);
    }
    __syncthreads();

    // Main loop: Double buffered computation
    for (int kt = 0; kt < num_k_tiles; ++kt) {
        int curr_buf = kt % 2;
        int next_buf = (kt + 1) % 2;

        // Prefetch next tile (if exists) while computing current
        if (kt + 1 < num_k_tiles) {
            load_A_tile(next_buf, kt + 1);
            load_B_tile(next_buf, kt + 1);
        }

        // Compute on current buffer
        // Note: This happens concurrently with the loads above
        // (loads are async, computation uses registers)
        compute_tile(curr_buf);

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
 * Version with explicit software pipelining using inline PTX
 *
 * This version uses inline PTX to make the pipelining structure more explicit
 * and easier to verify in SASS output.
 */
extern "C" __global__ void sgemm_double_buffer_ptx(
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
    __shared__ float As[2][BM][BK + 1];
    __shared__ float Bs[2][BK][BN + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    const int block_row_start = by * BM;
    const int block_col_start = bx * BN;
    const int thread_row = ty * TM;
    const int thread_col = tx * TN;

    float rC[TM][TN] = {{0.0f}};
    float rA[TM];
    float rB[TN];

    // Precompute some offsets
    const float* A_base = A + block_row_start * lda;
    const float* B_base = B + block_col_start;

    const int A_stride = 32;  // 256 threads / 8 cols
    const int B_stride = 2;   // 256 threads / 128 cols

    const int A_load_row = tid / BK;
    const int A_load_col = tid % BK;
    const int B_load_row = tid / BN;
    const int B_load_col = tid % BN;

    const int num_k_tiles = (K + BK - 1) / BK;

    // Load first tile
    for (int i = 0; i < BM; i += A_stride) {
        int row = A_load_row + i;
        if (row < BM) {
            int grow = block_row_start + row;
            int gcol = A_load_col;
            if (grow < M && gcol < K) {
                // Use PTX load with cache hint
                float val;
                const float* ptr = A + grow * lda + gcol;
                asm volatile(
                    "ld.global.ca.f32 %0, [%1];"
                    : "=f"(val)
                    : "l"(ptr)
                );
                As[0][row][A_load_col] = val;
            } else {
                As[0][row][A_load_col] = 0.0f;
            }
        }
    }

    for (int i = 0; i < BK; i += B_stride) {
        int row = B_load_row + i;
        if (row < BK) {
            int grow = row;
            int gcol = block_col_start + B_load_col;
            if (grow < K && gcol < N) {
                float val;
                const float* ptr = B + grow * ldb + gcol;
                asm volatile(
                    "ld.global.ca.f32 %0, [%1];"
                    : "=f"(val)
                    : "l"(ptr)
                );
                Bs[0][row][B_load_col] = val;
            } else {
                Bs[0][row][B_load_col] = 0.0f;
            }
        }
    }

    __syncthreads();

    // Main loop
    for (int kt = 0; kt < num_k_tiles; ++kt) {
        int curr_buf = kt & 1;
        int next_buf = 1 - curr_buf;
        int k_base = kt * BK;
        int next_k_base = (kt + 1) * BK;

        // Prefetch next tile
        if (kt + 1 < num_k_tiles) {
            for (int i = 0; i < BM; i += A_stride) {
                int row = A_load_row + i;
                if (row < BM) {
                    int grow = block_row_start + row;
                    int gcol = next_k_base + A_load_col;
                    if (grow < M && gcol < K) {
                        float val;
                        const float* ptr = A + grow * lda + gcol;
                        asm volatile(
                            "ld.global.ca.f32 %0, [%1];"
                            : "=f"(val)
                            : "l"(ptr)
                        );
                        As[next_buf][row][A_load_col] = val;
                    } else {
                        As[next_buf][row][A_load_col] = 0.0f;
                    }
                }
            }

            for (int i = 0; i < BK; i += B_stride) {
                int row = B_load_row + i;
                if (row < BK) {
                    int grow = next_k_base + row;
                    int gcol = block_col_start + B_load_col;
                    if (grow < K && gcol < N) {
                        float val;
                        const float* ptr = B + grow * ldb + gcol;
                        asm volatile(
                            "ld.global.ca.f32 %0, [%1];"
                            : "=f"(val)
                            : "l"(ptr)
                        );
                        Bs[next_buf][row][B_load_col] = val;
                    } else {
                        Bs[next_buf][row][B_load_col] = 0.0f;
                    }
                }
            }
        }

        // Compute on current buffer
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            #pragma unroll
            for (int m = 0; m < TM; ++m) {
                rA[m] = As[curr_buf][thread_row + m][k];
            }
            #pragma unroll
            for (int n = 0; n < TN; ++n) {
                rB[n] = Bs[curr_buf][k][thread_col + n];
            }

            #pragma unroll
            for (int m = 0; m < TM; ++m) {
                #pragma unroll
                for (int n = 0; n < TN; ++n) {
                    // Explicit FMA using PTX
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

    // Store results with vectorized writes where possible
    #pragma unroll
    for (int m = 0; m < TM; ++m) {
        int grow = block_row_start + thread_row + m;
        if (grow < M) {
            #pragma unroll
            for (int n = 0; n < TN; n += 4) {
                int gcol = block_col_start + thread_col + n;
                if (gcol + 3 < N) {
                    // Vectorized store
                    float4 c_old;
                    float4 result;
                    float* c_ptr = C + grow * ldc + gcol;

                    c_old.x = c_ptr[0];
                    c_old.y = c_ptr[1];
                    c_old.z = c_ptr[2];
                    c_old.w = c_ptr[3];

                    result.x = alpha * rC[m][n+0] + beta * c_old.x;
                    result.y = alpha * rC[m][n+1] + beta * c_old.y;
                    result.z = alpha * rC[m][n+2] + beta * c_old.z;
                    result.w = alpha * rC[m][n+3] + beta * c_old.w;

                    *reinterpret_cast<float4*>(c_ptr) = result;
                } else {
                    // Scalar fallback for edge cases
                    for (int nn = n; nn < TN && gcol + (nn - n) < N; ++nn) {
                        float c_old = C[grow * ldc + gcol + (nn - n)];
                        C[grow * ldc + gcol + (nn - n)] =
                            alpha * rC[m][nn] + beta * c_old;
                    }
                }
            }
        }
    }
}

/**
 * Analysis Notes for Nsight Compute:
 *
 * 1. Pipeline Overlap Verification:
 *    - In SASS, look for LDG instructions followed by FFMA
 *    - The LDG results shouldn't be used immediately (latency hidden)
 *    - Check "Warp State Statistics" -> Stall reasons
 *    - "Long Scoreboard" should be lower than non-double-buffered version
 *
 * 2. Memory vs Compute Balance:
 *    - SOL analysis should show better balance
 *    - Both Memory and Compute closer to their peaks
 *
 * 3. Shared Memory Pressure:
 *    - 2x shared memory usage
 *    - Check if this limits occupancy
 *    - May need to reduce BM/BN if shared memory becomes bottleneck
 *
 * 4. Stall Analysis Comparison:
 *    Before (single buffer):
 *    - Long Scoreboard: ~40-50%
 *    - Math Pipe Throttle: ~10-20%
 *
 *    After (double buffer):
 *    - Long Scoreboard: ~20-30%
 *    - Math Pipe Throttle: ~30-40%
 *
 * 5. Common Issues:
 *    - If Long Scoreboard is still high: Try triple buffering
 *    - If Wait stalls are high: Check syncthreads placement
 *    - If Math Pipe Throttle is low: Check register pressure
 */
