# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BareMetal-SGEMM is a high-performance SGEMM (single-precision general matrix multiply) library targeting 90%+ of cuBLAS performance on NVIDIA GPUs. It uses the **CUDA Driver API** (not Runtime API) with JIT compilation from PTX. The project implements 6 progressive optimization levels, each building on the previous one.

## Build Commands

```bash
# Build (Release, SM 8.0 Ampere)
./scripts/build.sh Release 80

# Manual CMake build
mkdir build && cd build && cmake -DCMAKE_BUILD_TYPE=Release .. && make -j$(nproc)

# Generate PTX files only
./scripts/compile_ptx.sh
```

Requires: CUDA Toolkit 11.0+ (11.1+ for Ampere cp.async), CMake 3.18+, GCC 9+.

## Running

```bash
# Full benchmark (all levels, compare with cuBLAS)
./build/sgemm_benchmark -m 4096 -n 4096 -k 4096 -c

# Single optimization level (0-5)
./build/sgemm_benchmark -l 5 -c

# Correctness tests against cuBLAS
./build/sgemm_test

# Size sweep (256 to 4096)
./build/sgemm_benchmark -s -c
```

Key flags: `-l <level>` single level, `-c` compare with cuBLAS, `-V` verify correctness, `-v` verbose, `-p <path>` PTX directory.

## Architecture

The project has four layers:

**Benchmark Layer** (`src/benchmark/`) — CLI argument parsing, benchmark orchestration, cuBLAS reference, correctness verification.

**Kernel Interface** (`include/sgemm_kernels.hpp`) — `SgemmLauncher` loads PTX files and launches kernels. `OptLevel` enum defines 6 levels. `SgemmParams` carries matrix dimensions and device pointers. Tiling constants: `TILE_M=128`, `TILE_N=128`, `TILE_K=8`, `THREAD_TILE=8x8`.

**CUDA Driver Wrapper** (`include/cuda_driver_wrapper.hpp`, `src/driver/`) — RAII wrappers around the Driver API: `CudaContext`, `CudaModule` (PTX JIT loading), `DeviceMemory`, `KernelLauncher`. All GPU resources use move semantics and automatic cleanup.

**SGEMM Kernels** (`src/kernels/`) — 6 `.cu` files compiled to PTX, loaded at runtime via JIT:

| Level | File | Optimization | Key Technique |
|-------|------|-------------|---------------|
| 0 | `sgemm_naive.cu` | Baseline | One thread = one output element |
| 1 | `sgemm_coalesced.cu` | Memory coalescing | `float4` vectorized loads |
| 2 | `sgemm_tiled.cu` | Shared memory | Block-level data reuse, +1 padding for bank conflicts |
| 3 | `sgemm_register_blocking.cu` | Register blocking | 8x8 thread tile, 64 register accumulators |
| 4 | `sgemm_double_buffer.cu` | Double buffering | Software pipelining overlaps compute and loads |
| 5 | `sgemm_async_copy.cu` | `cp.async` (Ampere) | Triple-buffered, `LDGSTS` direct global→shared |

All kernels implement: `C = alpha * A*B + beta * C` with parameters `(A, B, C, M, N, K, lda, ldb, ldc, alpha, beta)`.

## Key Patterns

- **Error handling**: `CHECK_CU()` and `CHECK_NVRTC()` macros wrap all Driver API calls, throwing `std::runtime_error` with file:line info.
- **Memory alignment**: 256-byte alignment for optimal coalescing (`memory_manager.cpp`).
- **PTX workflow**: `.cu` → `.ptx` (nvcc -ptx) → JIT at runtime (cuModuleLoadData) → kernel function handle (cuModuleGetFunction).
- **Non-copyable, movable** resource wrappers (context, module, memory).

## Documentation

- `docs/ANALYSIS_GUIDE.md` — Nsight Compute profiling guide with SOL analysis and SASS verification.
- `docs/education/` — 11 comprehensive Korean-language educational modules covering GPU architecture through advanced optimization.
