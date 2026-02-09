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
- `docs/education/` — 14 comprehensive Korean-language educational modules covering GPU architecture through advanced optimization.
- `docs/gamification/` — 6 design documents for the gamification system (narrative, economy, mechanics, portfolio, motivation, tech guide).

## Education Game ("GPU Architect: Rise from Silicon")

`game/` contains a playable web-based gamification layer over the education content. It is a vanilla JS SPA with no build step.

### File Structure

| File | Role | Lines |
|------|------|-------|
| `game/index.html` | HTML + CSS (dark silicon theme) | ~167 |
| `game/data.js` | All content data (75 quizzes, code traces, reviews, bosses, shop, achievements) | ~420 |
| `game/app.js` | Game engine (state, navigation, 10 screens, combo/GFLOP/s systems) | ~633 |
| `game/gpu_architect.html` | Single-file version (all 3 inlined, offline playable) | ~1225 |

### Game Data Schema

- `MODULES[11]` — Module definitions with flags (`hasQuiz`, `hasSim`, `hasCode`, `hasReview`, `mastery`, `boss`)
- `RANKS[8]` — Rank progression (intern → chip architect) with GFLOP/s and cuBLAS% thresholds
- `QUIZZES{modId: [{q, o[4], a, b:bloomLevel, e}]}` — 75 questions across 10 modules
- `CODE_TRACES{modId: {title, code, steps[{prompt, answer, hint}]}}` — 6 modules (04-09), code from actual `src/kernels/`
- `REVIEWS{modId: [{q, o[4], a}]}` — 6 modules (04-09), 3 pre-review questions each
- `BOSSES{bossId: {name, module, phases[{title, desc, q, o, a, e}]}}` — 3 boss fights, 3 phases each

### Game State

Persisted in `localStorage` key `gpu_architect_save`. Key fields: `fc`, `sc`, `rankIdx`, `completed[]`, `masteryPassed[]`, `bossCleared[]`, `quizScores{}`, `codeTracesDone{}`, `maxCombo`, `gflops`, `streak`, `inventory{}`, `earnedBadges[]`.

### Development Direction

The game currently implements the core learning loop with quiz, code tracing, memory challenge, and boss fight mechanics. Future development areas:

1. **Simulation games (Game E)** — Interactive visualizations for 10 simulations from module 13 (thread-to-SM mapping, bank conflict detector, pipeline timeline, etc.)
2. **Concept matching (Game B)** — Drag-and-drop matching of analogies↔diagrams↔formulas
3. **Teach-back (Game D)** — Keyword-based explanation exercises for "why it works" sections
4. **Portfolio/export** — Learning timeline, radar chart (6-axis competency), PDF certificate, JSON export
5. **Leaderboard** — Opt-in percentile bands or competitive ranking
6. **Recovery mechanics** — Streak freeze tokens, comeback quest chains for returning learners

Design docs in `docs/gamification/` provide complete specifications for all planned features.
