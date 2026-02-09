# BareMetal-SGEMM

> CUDA Driver API와 Inline PTX를 활용한 고성능 SGEMM 라이브러리

**목표: cuBLAS 성능의 90% 이상 달성**

이 프로젝트는 단순한 코딩 연습이 아니라, NVIDIA의 **시스템 소프트웨어 엔지니어**나 **GPU 아키텍트**가 실제로 수행하는 "하드웨어 특성 분석 및 최적화(Characterization & Optimization)" 과정을 시뮬레이션합니다.

## 🏗️ 프로젝트 구조

```
baremetal-sgemm/
├── include/
│   ├── cuda_driver_wrapper.hpp   # Driver API 래퍼 클래스
│   └── sgemm_kernels.hpp         # SGEMM 커널 인터페이스
├── src/
│   ├── driver/
│   │   ├── cuda_driver.cpp       # Driver API 구현
│   │   ├── kernel_launcher.cpp   # 커널 런처
│   │   └── memory_manager.cpp    # 메모리 관리 유틸리티
│   ├── kernels/
│   │   ├── sgemm_naive.cu        # Level 0: 기본 구현
│   │   ├── sgemm_coalesced.cu    # Level 1: 메모리 코얼레싱
│   │   ├── sgemm_tiled.cu        # Level 2: 공유 메모리 타일링
│   │   ├── sgemm_register_blocking.cu  # Level 3: 레지스터 블로킹
│   │   ├── sgemm_double_buffer.cu      # Level 4: 더블 버퍼링
│   │   └── sgemm_async_copy.cu         # Level 5: cp.async (Ampere+)
│   └── benchmark/
│       ├── main.cpp              # 벤치마크 메인
│       ├── benchmark_runner.cpp  # 벤치마크 러너
│       ├── cublas_reference.cu   # cuBLAS 레퍼런스
│       └── test_correctness.cpp  # 정확성 테스트
├── ptx/                          # 생성된 PTX 파일
├── scripts/
│   ├── build.sh                  # 빌드 스크립트
│   └── compile_ptx.sh            # PTX 컴파일 스크립트
├── docs/
│   ├── ANALYSIS_GUIDE.md         # Nsight Compute 분석 가이드
│   ├── education/                # 한국어 교육 모듈 (14개)
│   └── gamification/             # 게임화 설계 문서 (6개)
├── game/
│   ├── index.html                # 게임 UI (HTML + CSS)
│   ├── data.js                   # 게임 데이터 (75 퀴즈, 코드 추적 등)
│   ├── app.js                    # 게임 엔진 (SPA)
│   └── gpu_architect.html        # 단일 파일 버전 (오프라인 플레이)
└── CMakeLists.txt
```

## 🚀 빠른 시작

### 요구사항
- CUDA Toolkit 11.0+ (Ampere의 cp.async 기능 사용시 11.1+)
- CMake 3.18+
- GCC 9+ 또는 Clang 10+
- NVIDIA GPU (Ampere 이상 권장)

### 빌드
```bash
# 빌드 스크립트 사용
chmod +x scripts/build.sh
./scripts/build.sh Release 80  # SM 8.0 (A100, RTX 30 series)

# 또는 직접 CMake 사용
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

### 실행
```bash
# 전체 벤치마크
./build/sgemm_benchmark -m 4096 -n 4096 -k 4096 -c

# 특정 최적화 레벨만 테스트
./build/sgemm_benchmark -l 5 -c  # AsyncCopy만

# 정확성 테스트
./build/sgemm_test

# 다양한 크기 테스트
./build/sgemm_benchmark -s -c
```

## 📊 최적화 단계별 설명

### Level 0: Naive (기본 구현)
```cuda
// 각 스레드가 C의 한 원소를 계산
C[row][col] = sum(A[row][k] * B[k][col])
```
- 예상 성능: cuBLAS의 ~1-5%
- 문제점: 메모리 코얼레싱 없음, 데이터 재사용 없음

### Level 1: Coalesced (메모리 코얼레싱)
```cuda
// float4 벡터 타입으로 128-bit 로드
float4 a_vec = FETCH_FLOAT4(A[row * lda + k * 4]);
```
- SASS 확인: `LDG.E.128` 명령어 생성 확인
- 예상 성능: cuBLAS의 ~5-15%

### Level 2: Tiled (공유 메모리 타일링)
```cuda
__shared__ float As[BLOCK_SIZE][BLOCK_SIZE + 1];  // +1은 뱅크 충돌 방지
// 타일 로드 후 공유 메모리에서 연산
```
- SASS 확인: `LDS.128`, Bank Conflict 메트릭
- 예상 성능: cuBLAS의 ~20-40%

### Level 3: Register Blocking (레지스터 블로킹)
```cuda
// 각 스레드가 8x8 = 64개 결과 계산
float rC[TM][TN] = {{0.0f}};
// 레지스터에서 데이터 재사용
```
- 핵심: Arithmetic Intensity 증가 (0.25 → 2.0 FLOP/byte)
- 예상 성능: cuBLAS의 ~50-70%

### Level 4: Double Buffering (더블 버퍼링)
```cuda
// 현재 타일 연산 중 다음 타일 로드
compute_tile(buffer[i % 2]);
load_tile(buffer[(i+1) % 2], next_tile);
```
- 효과: 메모리 레이턴시 숨기기
- 예상 성능: cuBLAS의 ~70-85%

### Level 5: Async Copy (비동기 복사, Ampere+)
```cuda
// cp.async로 레지스터 우회하여 직접 복사
asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
             :: "r"(smem_addr), "l"(gmem_ptr));
```
- SASS 확인: `LDGSTS` 명령어 생성 확인
- 예상 성능: cuBLAS의 ~85-95%

## 📈 예상 성능 비교

| Matrix Size | Naive | Coalesced | Tiled | RegBlock | DblBuffer | AsyncCopy | cuBLAS |
|-------------|-------|-----------|-------|----------|-----------|-----------|--------|
| 1024³       | 50    | 200       | 800   | 2000     | 4000      | 7000      | 8000   |
| 2048³       | 60    | 250       | 1000  | 2500     | 5000      | 8500      | 10000  |
| 4096³       | 70    | 300       | 1200  | 3000     | 6000      | 9500      | 11000  |

*(단위: GFLOP/s, A100 기준 예상치)*

## 🔬 Nsight Compute 분석 가이드

자세한 분석 방법은 [`docs/ANALYSIS_GUIDE.md`](docs/ANALYSIS_GUIDE.md)를 참조하세요.

### 핵심 메트릭

1. **Speed of Light (SOL)**
   - SM Throughput: 연산 유닛 활용도
   - Memory Throughput: 메모리 대역폭 활용도

2. **Stall 원인**
   - Long Scoreboard: 메모리 데이터 대기 → 버퍼링 필요
   - Math Pipe Throttle: 연산 포화 → 최적 상태!

3. **SASS 명령어 확인**
   - `LDG.E.128`: 벡터화된 글로벌 로드 ✓
   - `LDGSTS`: cp.async 사용 ✓
   - `LDL/STL`: 레지스터 스필링 ✗

## 🎯 학습 목표

이 프로젝트를 통해 습득할 수 있는 기술:

1. **Driver API 마스터리**
   - cuInit, cuCtxCreate, cuModuleLoadDataEx
   - 런타임 API의 "마법"을 이해하고 직접 제어

2. **메모리 계층 구조 이해**
   - 글로벌 메모리 코얼레싱
   - 공유 메모리 뱅크 충돌
   - 레지스터 vs 공유 메모리 트레이드오프

3. **파이프라이닝 기법**
   - Software prefetching
   - Double/Triple buffering
   - Ampere cp.async

4. **성능 분석 도구 사용**
   - Nsight Compute SASS 분석
   - SOL 분석 및 병목 지점 파악
   - 레지스터 압박 및 스필링 감지

## 🎮 교육 게임: GPU Architect — Rise from Silicon

BareMetal-SGEMM 교육 과정을 게임화한 웹 기반 학습 도구입니다. 브라우저에서 바로 플레이할 수 있습니다.

### 실행 방법

```bash
# 단일 파일 버전 (권장)
open game/gpu_architect.html    # macOS
xdg-open game/gpu_architect.html  # Linux

# 또는 로컬 서버
cd game && python3 -m http.server 8080
# http://localhost:8080 접속
```

### 콘텐츠 구성

| 항목 | 수량 | 설명 |
|------|------|------|
| 퀴즈 | 75문항 | 모듈별 5-10문항, Bloom's Taxonomy 6단계 |
| 코드 추적 | 6개 | 실제 `src/kernels/` 커널 코드 기반 스텝 실행 |
| 기억력 챌린지 | 6개 | 60초 타이머, 이전 모듈 복습 |
| 보스전 | 3개 | 뱅크 충돌 수호자, 레이턴시 드래곤, 최종 벤치마크 |
| 배지 | 22개 | 4티어 (모듈 완료 / 마스터리 / 스킬 / 히든) |

### 게임 시스템

- **직급 체계**: 수습 엔지니어 → 칩 아키텍트 (8단계)
- **이중 재화**: FLOP 코인 (FC, 학습 활동) + 실리콘 크리스탈 (SC, 탁월한 성취)
- **GFLOP/s 미터**: 학습 성과를 실제 커널 성능 수치에 매핑
- **콤보 시스템**: 연속 정답 시 x1 → x1.5 → x2 → x3 보상 배율
- **상점**: FC/SC로 힌트, 재시도 토큰, 코스메틱 구매

### 설계 문서

게임화 설계 문서는 `docs/gamification/`에 있습니다:

- `01_게임화_개요.md` — 내러티브, 직급 체계, 설계 원칙
- `02_재화_및_경제_시스템.md` — 이중 재화 모델, 상점, 경제 균형
- `03_게임_메커니즘.md` — 9-섹션 퀘스트 체인, 6가지 게임 유형, 보스전
- `04_이력_및_포트폴리오.md` — 타임라인, 레이더 차트, 업적, 내보내기
- `05_동기_설계.md` — 동기 루프, 회복 메커니즘, 리더보드
- `06_기술_구현_가이드.md` — 웹앱 아키텍처, 데이터 스키마, 파싱 전략

## 📝 참고 자료

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)
- [CUTLASS: CUDA Templates for Linear Algebra Subroutines](https://github.com/NVIDIA/cutlass)
- [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance](https://siboehm.com/articles/22/CUDA-MMM)

## 📜 라이선스

MIT License - 자유롭게 학습 및 수정에 활용하세요.

---

*이 프로젝트는 NVIDIA GPU 아키텍처와 CUDA 최적화에 대한 깊은 이해를 목표로 합니다.*
