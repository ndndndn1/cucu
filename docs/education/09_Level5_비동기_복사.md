# 모듈 9: Level 5 -- Ampere 비동기 복사 (cp.async) 와 트리플 버퍼링

---

## 1. 학습 목표

이 모듈을 완료하면 다음을 할 수 있다.

| 목표 | Bloom 수준 |
|------|-----------|
| cp.async의 3가지 PTX 명령을 설명하고 사용할 수 있다 | 이해(Understand) / 적용(Apply) |
| 트리플 버퍼링의 파이프라인 구조를 설계할 수 있다 | 적용(Apply) |
| 레지스터 우회의 점유율 이점을 분석할 수 있다 | 분석(Analyze) |
| SASS에서 LDGSTS 명령어를 식별할 수 있다 | 적용(Apply) |
| NUM_STAGES 튜닝의 트레이드오프를 평가할 수 있다 | 평가(Evaluate) |

---

## 2. 사전 복습

다음 3개 문항을 **보지 않고** 답한 뒤 모듈 7-8의 내용과 대조해 본다.

1. 더블 버퍼링에서 프롤로그(prologue)의 역할은 무엇인가?
2. 레지스터 블로킹 커널의 레지스터 사용량은 대략 몇 개인가?
3. Long Scoreboard 스톨의 의미와 해결 방법은?

<details>
<summary>정답 확인</summary>

1. 프롤로그는 메인 루프 진입 전에 첫 번째 타일을 미리 로드하여 파이프라인을 채우는 역할을 한다. 이를 통해 메인 루프의 첫 반복에서 이미 데이터가 준비되어 있으므로 연산과 로드를 중첩시킬 수 있다.
2. TM=8, TN=8 레지스터 블로킹 커널에서 누적 레지스터(rC) 64개, A/B 프래그먼트(rA, rB) 16개, 인덱싱/제어 변수 등을 포함하여 대략 100-128개의 레지스터를 사용한다.
3. Long Scoreboard 스톨은 글로벌 메모리 로드(LDG) 결과가 아직 도착하지 않아 해당 레지스터를 읽으려는 명령이 대기하는 상태이다. 해결 방법으로는 (1) 프리페치를 통해 로드와 연산을 중첩시키거나, (2) 점유율을 높여 다른 워프로 전환하거나, (3) 비동기 복사(cp.async)로 레지스터를 거치지 않는 방법이 있다.

</details>

---

## 3. 개념 설명

### 3.1 cp.async 하드웨어 기능

**Concrete (구체적 비유)**

택배 기사(DMA 엔진)가 물건을 직접 창고(공유 메모리)에 넣는다고 생각하자. 이전 방식에서는 택배 기사가 물건을 반드시 사무실 책상(레지스터)에 먼저 올려놓고, 직원이 그것을 다시 창고로 옮겨야 했다. 이 과정에서 책상 위의 공간이 점유되고, 직원의 시간도 소모된다. cp.async를 사용하면 택배 기사가 사무실을 거치지 않고 창고에 직접 배달하므로, 책상(레지스터) 공간이 확보되고 직원(연산 유닛)은 다른 일을 할 수 있다.

**Representational (표상적 -- 데이터 흐름 다이어그램)**

```
기존 방식 (cp.async 없음):
+----------+     LDG      +----------+     STS      +----------+
|  Global  | -----------> | Register | -----------> |  Shared  |
|  Memory  |   (400cyc)   |   File   |   (20cyc)    |  Memory  |
+----------+              +----------+              +----------+
                           ^^^^^^^^^
                           레지스터 점유!
                           2단계 과정

SASS: LDG Rx, [addr_global]    (글로벌 -> 레지스터)
      STS [addr_shared], Rx    (레지스터 -> 공유메모리)

cp.async 방식 (Ampere SM 8.0+):
+----------+    LDGSTS     +----------+
|  Global  | ------------> |  Shared  |
|  Memory  |  (직접 전송)   |  Memory  |
+----------+              +----------+
  레지스터 우회! 1단계 과정

SASS: LDGSTS [addr_shared], [addr_global]    (글로벌 -> 공유메모리 직접)
```

**Abstract (추상적 원리)**

cp.async는 Ampere 아키텍처(Compute Capability 8.0)에서 도입된 하드웨어 기능으로, 글로벌 메모리에서 공유 메모리로의 복사를 레지스터 파일을 거치지 않고 직접 수행한다. 이는 다음을 의미한다.

1. 레지스터 파일 압력이 감소하여 점유율이 향상될 수 있다.
2. 복사가 진정한 의미에서 비동기적이다. 발행 후 즉시 다음 명령으로 진행할 수 있다.
3. 하드웨어 DMA 엔진이 전송을 담당하므로, SM의 연산 파이프라인이 자유롭다.

### 3.2 세 가지 PTX 명령

cp.async를 사용하려면 세 가지 PTX 명령을 조합해야 한다.

**명령 1: `cp.async.cg.shared.global [dst], [src], size`**

16바이트(또는 4바이트) 단위의 비동기 복사를 발행한다.

```cuda
// 16바이트(float4) 비동기 복사
__device__ __forceinline__ void cp_async_cg_shared_global_16(
    float* __restrict__ dst,
    const float* __restrict__ src)
{
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;"
        :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(dst))),
           "l"(src)
    );
}
```

- `.cg` (cache global): L2 캐시에만 캐싱한다 (L1 우회).
- `.shared.global`: 소스가 글로벌 메모리, 목적지가 공유 메모리임을 지정한다.
- `16`: 복사할 바이트 수. 4(float 1개) 또는 16(float4)을 사용할 수 있다.
- `__cvta_generic_to_shared(dst)`: 제네릭 주소를 공유 메모리 주소 공간으로 변환한다. PTX에서 공유 메모리 주소는 32비트이므로 `uint32_t`로 캐스팅한다.

코드 참조: `src/kernels/sgemm_async_copy.cu:50-62`

**명령 2: `cp.async.commit_group`**

지금까지 발행된 모든 cp.async 복사를 하나의 그룹으로 묶는다.

```cuda
__device__ __forceinline__ void cp_async_commit_group()
{
    asm volatile("cp.async.commit_group;");
}
```

이 명령은 "여기까지 발행한 비동기 복사들을 하나의 단위로 묶어라"라는 의미이다. 이후 `wait_group`으로 이 그룹 단위의 완료를 추적할 수 있다.

코드 참조: `src/kernels/sgemm_async_copy.cu:77-80`

**명령 3: `cp.async.wait_group N`**

미완료 그룹이 N개 이하가 될 때까지 대기한다.

```cuda
__device__ __forceinline__ void cp_async_wait_group(int N)
{
    switch (N) {
        case 0: asm volatile("cp.async.wait_group 0;"); break;
        case 1: asm volatile("cp.async.wait_group 1;"); break;
        case 2: asm volatile("cp.async.wait_group 2;"); break;
        case 3: asm volatile("cp.async.wait_group 3;"); break;
        default: asm volatile("cp.async.wait_all;"); break;
    }
}
```

- `N=0`: 모든 미완료 그룹이 완료될 때까지 대기한다. 가장 보수적이다.
- `N=NUM_STAGES-1` (예: N=2): 미완료 그룹이 2개 이하가 될 때까지 대기한다. 즉, 가장 오래된 그룹의 완료만 보장한다. 최신 그룹은 아직 진행 중일 수 있다.

이 세 명령의 관계를 정리하면 다음과 같다.

```
cp.async (발행)  -->  commit_group (그룹화)  -->  wait_group (동기화)

[복사1]                                         wait_group(2):
[복사2]  -- commit_group -> [그룹 A]             "미완료 그룹 2개까지는
[복사3]                                          허용. 그보다 오래된 건
[복사4]  -- commit_group -> [그룹 B]              완료를 보장해라"
[복사5]
[복사6]  -- commit_group -> [그룹 C]
```

코드 참조: `src/kernels/sgemm_async_copy.cu:83-94`

### 3.3 트리플 버퍼링

더블 버퍼링(NUM_STAGES=2)에서는 2개의 공유 메모리 버퍼를 번갈아 사용했다. 하나를 연산에 사용하는 동안 다른 하나에 다음 데이터를 로드하는 방식이다.

트리플 버퍼링(NUM_STAGES=3)은 여기서 한 단계 더 나아간다.

```cuda
#define NUM_STAGES 3
__shared__ float As[NUM_STAGES][BM][BK + 1];  // As[3][128][9]
__shared__ float Bs[NUM_STAGES][BK][BN + 1];  // Bs[3][8][129]
```

3개의 버퍼를 링 버퍼(ring buffer) 방식으로 순환하며 사용한다.

```
스테이지 인덱스: curr_stage = kt % NUM_STAGES

kt:  0  1  2  3  4  5  6  7  8  ...
%3:  0  1  2  0  1  2  0  1  2  ...
```

트리플 버퍼링의 핵심 트레이드오프는 다음과 같다.

| 항목 | 더블 버퍼링 (NUM_STAGES=2) | 트리플 버퍼링 (NUM_STAGES=3) |
|------|---------------------------|----------------------------|
| 공유 메모리 사용량 | 2배 | 3배 |
| 파이프라인 깊이 | 1 (로드 1개 중첩) | 2 (로드 2개 중첩) |
| 레이턴시 은닉 능력 | 보통 | 높음 |
| 최대 점유율 | 높음 | 약간 낮음 (공유 메모리 소모) |

공유 메모리 사용량을 구체적으로 계산하면 다음과 같다.

```
As[3][128][9]:  3 * 128 * 9 * 4 = 13,824 바이트
Bs[3][8][129]:  3 * 8 * 129 * 4 = 12,384 바이트
합계:                              26,208 바이트 (약 25.6 KB)

비교 - 더블 버퍼링:
As[2][128][9]:  2 * 128 * 9 * 4 =  9,216 바이트
Bs[2][8][129]:  2 * 8 * 129 * 4 =  8,256 바이트
합계:                              17,472 바이트 (약 17.1 KB)
```

A100의 공유 메모리가 SM당 최대 164 KB이므로, 26 KB는 충분히 수용 가능하다.

### 3.4 파이프라인 구조

전체 파이프라인은 프롤로그(prologue)와 메인 루프로 구성된다.

**프롤로그: 파이프라인 채우기**

```cuda
// 처음 NUM_STAGES-1 = 2개의 타일을 미리 로드
#pragma unroll
for (int s = 0; s < NUM_STAGES - 1 && s < num_k_tiles; ++s) {
    async_load_tile(s, s);  // stage s에 타일 s를 비동기 로드 + commit_group
}
```

프롤로그에서 타일 0과 타일 1을 비동기 로드하고 각각 commit_group으로 그룹화한다. 이 시점에서 미완료 그룹이 2개(그룹 0, 그룹 1)가 존재한다.

**메인 루프: 연산과 프리페치의 중첩**

```cuda
for (int kt = 0; kt < num_k_tiles; ++kt) {
    int curr_stage = kt % NUM_STAGES;          // 현재 연산할 스테이지
    int prefetch_tile = kt + NUM_STAGES - 1;   // 미래 타일 프리페치

    // (1) 프리페치: 아직 로드할 타일이 남아있으면 비동기 로드 발행
    if (prefetch_tile < num_k_tiles) {
        int prefetch_stage = prefetch_tile % NUM_STAGES;
        async_load_tile(prefetch_stage, prefetch_tile);
    }

    // (2) 동기화: 현재 타일의 데이터가 준비될 때까지 대기
    cp_async_wait_group(NUM_STAGES - 1);   // 미완료 그룹 2개 이하 보장
    __syncthreads();

    // (3) 연산: 현재 스테이지의 데이터로 FMA 수행
    for (int k = 0; k < BK; ++k) {
        // rA, rB 로드 후 외적(outer product) 계산
        // ...FMA 연산...
    }

    __syncthreads();
}
```

각 반복에서 벌어지는 일을 구체적으로 추적하면 다음과 같다.

```
프롤로그 후 상태: 그룹0(타일0->Buf0), 그룹1(타일1->Buf1) 진행 중

kt=0: curr_stage = 0
      prefetch_tile = 2 -> async_load(stage=2, tile=2) + commit -> 그룹2 생성
      wait_group(2): 미완료 3개 -> 가장 오래된 그룹0 완료 대기 -> Buf0 준비 보장
      syncthreads
      Compute on Buf0 (타일 0)
      syncthreads

kt=1: curr_stage = 1
      prefetch_tile = 3 -> async_load(stage=0, tile=3) + commit -> 그룹3 생성
      wait_group(2): 미완료 3개 -> 가장 오래된 그룹1 완료 대기 -> Buf1 준비 보장
      syncthreads
      Compute on Buf1 (타일 1)
      syncthreads

kt=2: curr_stage = 2
      prefetch_tile = 4 -> async_load(stage=1, tile=4) + commit -> 그룹4 생성
      wait_group(2): 미완료 3개 -> 가장 오래된 그룹2 완료 대기 -> Buf2 준비 보장
      syncthreads
      Compute on Buf2 (타일 2)
      syncthreads

kt=3: curr_stage = 0
      prefetch_tile = 5 -> async_load(stage=2, tile=5) + commit -> 그룹5 생성
      wait_group(2): 가장 오래된 그룹3 완료 대기 -> Buf0 준비 보장
      syncthreads
      Compute on Buf0 (타일 3)
      syncthreads

...패턴 반복...
```

핵심 관찰: `wait_group(NUM_STAGES-1)`은 미완료 그룹이 `NUM_STAGES-1`개 이하가 될 때까지 대기한다. 프리페치까지 포함하면 미완료 그룹이 최대 `NUM_STAGES`개가 되므로, 가장 오래된 1개 그룹의 완료가 보장된다. 이 가장 오래된 그룹이 바로 `curr_stage`에 해당하는 데이터이다.

### 3.5 레지스터 우회의 이점

기존 방식과 cp.async 방식의 레지스터 사용량 차이를 분석한다.

**기존 방식 (LDG + STS)**

```
LDG R10, [addr_global]    // R10에 글로벌 데이터 로드 (400+ 사이클 소요)
// ... R10을 사용할 수 없는 400 사이클 동안 R10은 점유된 상태 ...
STS [addr_shared], R10    // R10에서 공유 메모리로 저장
// 이제야 R10 해제
```

이 패턴에서 R10은 LDG 발행 시점부터 STS 완료 시점까지 점유된다. BM x BK / 스레드 수만큼의 원소를 전송하려면, 그만큼의 레지스터가 로드-스토어 전환 과정에서 점유된다.

**cp.async 방식 (LDGSTS)**

```
LDGSTS [addr_shared], [addr_global], 16    // 레지스터를 거치지 않고 직접 전송
// 레지스터 소모 없음!
```

레지스터를 사용하지 않으므로 다음의 이점이 발생한다.

1. **레지스터 압력 감소**: 전송 과정에서 레지스터가 점유되지 않으므로, 같은 커널에서 더 많은 레지스터를 연산(rC, rA, rB)에 할당할 수 있다.
2. **점유율 향상 가능**: 레지스터 사용량이 줄면, SM당 더 많은 블록을 동시 실행할 수 있어 점유율이 높아진다. 이는 레이턴시 은닉 능력을 더욱 강화한다.
3. **Long Scoreboard 스톨 감소**: 기존 방식에서는 LDG의 목적지 레지스터를 STS가 읽어야 하므로 Long Scoreboard 스톨이 발생한다. cp.async는 레지스터를 거치지 않으므로 이 스톨이 원천적으로 제거된다.

### 3.6 벡터화 비동기 복사 변형

`sgemm_async_copy.cu`에는 두 가지 커널이 있다. 기본 버전(`sgemm_async_copy`)은 4바이트 단위로 복사하고, 벡터화 버전(`sgemm_async_copy_vec`)은 16바이트 단위로 복사한다.

```cuda
// 4바이트 복사: float 1개씩
cp_async_cg_shared_global_4(dst, src);   // 대역폭 활용 낮음

// 16바이트 복사: float4 (float 4개)를 한 번에
cp_async_cg_shared_global_16(dst, src);  // 대역폭 활용 높음
```

16바이트 복사의 이점은 다음과 같다.

- **대역폭 효율**: 글로벌 메모리 트랜잭션 수가 1/4로 줄어든다.
- **SASS 효율**: 하나의 LDGSTS 명령이 16바이트를 전송하므로, 명령 발행 오버헤드가 감소한다.
- **정렬 요구사항**: 소스 주소와 목적지 주소 모두 16바이트 정렬(alignment)이 필요하다. 정렬되지 않으면 하드웨어 예외가 발생할 수 있다.

벡터화 버전의 로드 패턴(코드 참조: `src/kernels/sgemm_async_copy.cu:334-376`):

```
B 타일 로드 (BK x BN = 8 x 128):
  256 스레드, 각 스레드가 float4(16바이트) 1개 로드
  -> 256 * 4 = 1024 float = BK * BN = 8 * 128 원소
  -> B_load_row = tid / 32, B_load_col = (tid % 32) * 4

A 타일 로드 (BM x BK = 128 x 8):
  256 스레드, 각 스레드가 float4(16바이트) 1개 로드
  -> 256 * 4 = 1024 float = BM * BK = 128 * 8 원소
  -> A_load_row = tid / 2, A_load_col = (tid % 2) * 4
```

### 3.7 왜 100%가 아닌 85-95%인가?

cp.async와 트리플 버퍼링까지 적용한 이 커널은 cuBLAS 대비 약 85-95%의 성능을 달성한다. 100%에 도달하지 못하는 이유는 다음과 같다.

1. **Tensor Core 미사용**: cuBLAS는 FP32 SGEMM에서도 TF32(TensorFloat-32) 모드의 Tensor Core를 활용할 수 있다. Tensor Core는 FMA 유닛보다 행렬 곱셈 처리량이 수 배 높다. 이 커널은 순수 FP32 FMA만 사용한다.

2. **아키텍처별 타일 크기 최적화 부재**: cuBLAS는 실행 중인 GPU 아키텍처에 따라 최적의 BM, BN, BK, TM, TN 조합을 자동으로 선택한다. 이 커널은 단일 설정(BM=BN=128, BK=8, TM=TN=8)으로 고정되어 있다.

3. **워프 스케줄링 최적화 부재**: cuBLAS는 워프 간의 명령 인터리빙(interleaving)을 더 정교하게 제어하여 파이프라인 효율을 극대화한다.

4. **소프트웨어 파이프라이닝 깊이**: cuBLAS는 메인 루프 내의 명령 수준 파이프라이닝까지 수동으로 조율한다.

그럼에도 불구하고, 순수 FP32 FMA 기반으로 85-95%를 달성한 것은 이 커널이 메모리 계층과 비동기 파이프라인을 효과적으로 활용하고 있음을 의미한다.

---

## 4. 코드 분석 (완전 워크스루)

소스 파일: `src/kernels/sgemm_async_copy.cu`

### 라인 1-41: 헤더 및 매크로 정의

```cuda
#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8
#define NUM_STAGES 3

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
```

타일 크기와 레지스터 블로킹 매개변수는 이전 모듈과 동일하다. `NUM_STAGES = 3`이 트리플 버퍼링의 핵심 설정이며, `__CUDA_ARCH__ >= 800` 가드로 Ampere 이상의 GPU에서만 cp.async 코드가 컴파일되도록 한다.

### 라인 48-100: cp.async 헬퍼 함수들

```cuda
// 16바이트 비동기 복사
__device__ __forceinline__ void cp_async_cg_shared_global_16(
    float* __restrict__ dst, const float* __restrict__ src)
{
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;"
        :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(dst))),
           "l"(src)
    );
}

// 4바이트 비동기 복사
__device__ __forceinline__ void cp_async_cg_shared_global_4(
    float* __restrict__ dst, const float* __restrict__ src)
{
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 4;"
        :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(dst))),
           "l"(src)
    );
}

// 그룹 커밋
__device__ __forceinline__ void cp_async_commit_group()
{
    asm volatile("cp.async.commit_group;");
}

// 그룹 대기 (미완료 N개 이하)
__device__ __forceinline__ void cp_async_wait_group(int N)
{
    switch (N) {
        case 0: asm volatile("cp.async.wait_group 0;"); break;
        case 1: asm volatile("cp.async.wait_group 1;"); break;
        case 2: asm volatile("cp.async.wait_group 2;"); break;
        case 3: asm volatile("cp.async.wait_group 3;"); break;
        default: asm volatile("cp.async.wait_all;"); break;
    }
}
```

인라인 PTX 어셈블리의 핵심 요소를 분석한다.

| 구문 | 의미 |
|------|------|
| `asm volatile(...)` | 컴파일러가 이 어셈블리를 재배치하거나 제거하지 못하게 한다 |
| `"r"(...)` | 32비트 레지스터 피연산자 (공유 메모리 주소용) |
| `"l"(...)` | 64비트 레지스터 피연산자 (글로벌 메모리 주소용) |
| `__cvta_generic_to_shared` | 제네릭 포인터를 공유 메모리 주소 공간으로 변환하는 내장 함수 |

`wait_group`이 `switch`문으로 구현된 이유: PTX의 `cp.async.wait_group` 명령은 즉시값(immediate)만 인자로 받을 수 있다. 런타임 변수를 직접 전달할 수 없으므로, 가능한 값에 대해 분기하는 방식을 사용한다.

### 라인 117-120: 트리플 버퍼 공유 메모리

```cuda
__shared__ float As[NUM_STAGES][BM][BK + 1];  // As[3][128][9]
__shared__ float Bs[NUM_STAGES][BK][BN + 1];  // Bs[3][8][129]
```

첫 번째 차원 `NUM_STAGES`가 3개의 버퍼를 형성한다. `BK + 1`과 `BN + 1`은 이전 모듈에서 설명한 뱅크 충돌 방지 패딩이다.

### 라인 148-189: async_load_tile 람다

```cuda
auto async_load_tile = [&](int stage, int k_tile) {
    const int k_base = k_tile * BK;

    // A 타일 비동기 로드
    for (int i = 0; i < BM; i += 32) {
        int row = A_load_row + i;
        if (row < BM) {
            int grow = block_row_start + row;
            int gcol = k_base + A_load_col;
            float* dst = &As[stage][row][A_load_col];

            if (grow < M && gcol < K) {
                const float* src = &A[grow * lda + gcol];
                cp_async_cg_shared_global_4(dst, src);
            } else {
                *dst = 0.0f;  // 경계 밖은 0으로 직접 저장
            }
        }
    }

    // B 타일 비동기 로드 (유사한 구조)
    // ...

    cp_async_commit_group();  // 이 타일의 모든 복사를 하나의 그룹으로 묶음
};
```

중요한 세부사항을 정리한다.

- **경계 처리**: `grow >= M` 또는 `gcol >= K`인 경우 cp.async를 사용할 수 없다(유효하지 않은 글로벌 주소를 읽을 수 없음). 따라서 직접 0을 저장한다(`*dst = 0.0f`).
- **commit_group 위치**: 타일 하나의 모든 비동기 복사가 발행된 직후에 호출한다. 이로써 타일 단위로 완료를 추적할 수 있다.
- **A 타일 로드 패턴**: `BM / 32 = 4`회 반복. 256개 스레드가 각각 4개의 원소를 담당하여 BM x BK = 128 x 8 = 1024개 원소를 로드한다.

### 라인 193-196: 프롤로그

```cuda
#pragma unroll
for (int s = 0; s < NUM_STAGES - 1 && s < num_k_tiles; ++s) {
    async_load_tile(s, s);
}
```

`NUM_STAGES - 1 = 2`개의 타일(타일 0, 타일 1)을 미리 로드한다. 이 시점에서 미완료 그룹 2개가 파이프라인에 존재한다. `s < num_k_tiles` 조건은 K가 매우 작아 타일이 2개 미만인 경우의 안전 장치이다.

### 라인 199-247: 메인 루프

```cuda
for (int kt = 0; kt < num_k_tiles; ++kt) {
    int curr_stage = kt % NUM_STAGES;

    // 프리페치
    int prefetch_tile = kt + NUM_STAGES - 1;
    if (prefetch_tile < num_k_tiles) {
        int prefetch_stage = prefetch_tile % NUM_STAGES;
        async_load_tile(prefetch_stage, prefetch_tile);
    }

    // 현재 타일 데이터 대기
    cp_async_wait_group(NUM_STAGES - 1);
    __syncthreads();

    // 연산: BK 내부 루프
    #pragma unroll
    for (int k = 0; k < BK; ++k) {
        #pragma unroll
        for (int m = 0; m < TM; ++m)
            rA[m] = As[curr_stage][thread_row + m][k];
        #pragma unroll
        for (int n = 0; n < TN; ++n)
            rB[n] = Bs[curr_stage][k][thread_col + n];

        #pragma unroll
        for (int m = 0; m < TM; ++m) {
            #pragma unroll
            for (int n = 0; n < TN; ++n) {
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
```

연산 부분은 레지스터 블로킹 커널과 동일하다. 달라진 것은 데이터 로드 메커니즘뿐이다.

| 기존 (LDG + STS) | cp.async |
|-------------------|----------|
| LDG로 레지스터에 로드 | cp.async 발행 |
| 동기화 후 STS로 공유 메모리에 저장 | commit_group으로 그룹화 |
| 다시 동기화 | wait_group + syncthreads |

### 라인 250-263: 에필로그

```cuda
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
```

에필로그는 이전 모듈과 동일하다. 누적된 rC 레지스터의 값에 alpha/beta 스케일링을 적용하여 C에 저장한다.

### SASS 검증: LDGSTS 확인

`cuobjdump`나 Nsight Compute의 Source-SASS 뷰에서 확인할 핵심 명령어는 다음과 같다.

```
기대하는 SASS:
  LDGSTS [R_shared], [R_global], 0x10    (16바이트 직접 전송)
  또는
  LDGSTS [R_shared], [R_global], 0x4     (4바이트 직접 전송)

기대하지 않는 SASS:
  LDG.E R10, [R_global]     (글로벌 -> 레지스터)
  STS [R_shared], R10        (레지스터 -> 공유)
```

LDGSTS가 나타나면 cp.async가 정상적으로 하드웨어 명령으로 변환된 것이다. LDG + STS 쌍이 나타나면 컴파일러가 cp.async를 올바르게 인식하지 못한 것이므로, 컴파일 옵션(`-arch=sm_80` 이상)을 확인해야 한다.

---

## 5. 왜 이것이 작동하는가?

### 질문 1: cp_async_commit_group()을 호출하지 않으면 어떻게 되는가?

<details>
<summary>정답</summary>

commit_group을 호출하지 않으면, 발행된 모든 cp.async 복사가 하나의 암묵적 그룹으로 누적된다. wait_group(N)은 그룹 단위로 완료를 추적하므로, 그룹이 제대로 분리되지 않으면 다음 문제가 발생한다.

1. wait_group(2)가 의미를 잃는다. 타일별로 그룹이 분리되지 않았으므로, 특정 타일의 완료를 보장할 수 없다.
2. wait_group(0)을 사용하면 모든 복사가 완료될 때까지 기다려야 하므로, 파이프라인의 비동기성이 완전히 사라진다.
3. 결과적으로 정확성 문제(아직 로드되지 않은 데이터로 연산) 또는 성능 문제(불필요한 전체 대기)가 발생한다.

</details>

### 질문 2: cp_async_wait_group(NUM_STAGES-1)이 정확히 현재 타일의 완료를 보장하는 이유는?

<details>
<summary>정답</summary>

파이프라인의 논리를 단계별로 추적하면 다음과 같다.

프롤로그에서 `NUM_STAGES-1 = 2`개의 그룹을 발행한다. 메인 루프의 kt번째 반복에서 프리페치로 1개의 새 그룹을 추가하면, 미완료 그룹의 최대 개수는 `NUM_STAGES = 3`개이다.

wait_group(2)는 "미완료 그룹이 2개 이하가 될 때까지 대기"를 의미한다. 미완료 그룹이 3개일 때 wait_group(2)를 호출하면, 가장 오래된 1개 그룹이 완료되어야 2개 이하가 된다. 이 가장 오래된 그룹이 바로 `curr_stage`에 해당하는 타일의 로드 그룹이다.

일반화하면, 항상 `발행된 그룹 수 - wait_group 인자 = 완료 보장되는 그룹 수`이다. `NUM_STAGES - (NUM_STAGES - 1) = 1`이므로, 가장 오래된 1개 그룹의 완료가 보장된다.

</details>

### 질문 3: 트리플 버퍼링이 더블 버퍼링보다 레이턴시를 더 잘 숨기는 이유는?

<details>
<summary>정답</summary>

더블 버퍼링에서는 파이프라인에 1개의 미완료 로드만 존재한다. 현재 타일을 연산하는 동안 다음 타일 1개만 로드할 수 있다. 연산 시간이 로드 레이턴시보다 짧으면, 로드 완료를 기다리는 유휴 시간이 발생한다.

트리플 버퍼링에서는 파이프라인에 2개의 미완료 로드가 존재한다. 현재 타일을 연산하는 동안 다음 타일과 그다음 타일까지 로드가 진행된다. 로드 레이턴시가 연산 시간의 최대 2배까지라도 유휴 시간 없이 숨길 수 있다.

```
더블 버퍼링 (깊이 1):
  연산[0]  |  연산[1]  | 대기... | 연산[2]  |
  로드[1]      로드[2]             로드[3]
                      ^
                      로드가 연산보다 느려 유휴 발생

트리플 버퍼링 (깊이 2):
  연산[0]  |  연산[1]  |  연산[2]  |  연산[3]  |
  로드[1]      로드[2]     로드[3]     로드[4]
  로드[2]      로드[3]     로드[4]     로드[5]
              ^
              2개 로드가 중첩되어 유휴 없음
```

</details>

### 질문 4: LDGSTS가 LDG+STS보다 효율적인 하드웨어적 이유는?

<details>
<summary>정답</summary>

1. **레지스터 파일 우회**: LDG+STS는 데이터가 레지스터 파일을 경유하므로, 레지스터 파일의 읽기/쓰기 포트를 사용한다. LDGSTS는 전용 DMA 경로를 사용하여 레지스터 파일 대역폭을 소비하지 않는다.

2. **명령어 수 감소**: LDG+STS는 2개의 명령어를 발행해야 하지만, LDGSTS는 1개의 명령어로 동일한 작업을 수행한다. 명령어 발행 대역폭(issue bandwidth)이 절약된다.

3. **의존성 사슬 제거**: LDG -> STS 사이에는 진정한 데이터 의존성(true dependency)이 있다. LDG가 완료되어야 STS를 실행할 수 있다. LDGSTS는 하드웨어가 이 전체 전송을 하나의 원자적 동작으로 처리하므로, 소프트웨어 수준의 의존성 관리가 불필요하다.

4. **에너지 효율**: 레지스터 파일 읽기/쓰기는 상당한 에너지를 소모한다. LDGSTS는 이 에너지를 절약한다.

</details>

---

## 6. 시뮬레이션 9: 비동기 파이프라인 상태 기계

3개 버퍼의 상태를 추적하는 상태 기계를 그려본다.

### 버퍼 상태 정의

- **L (Loading)**: 비동기 로드가 진행 중인 상태. 데이터가 아직 도착하지 않았을 수 있다.
- **R (Ready)**: 로드가 완료되어 데이터가 준비된 상태. 연산에 사용할 수 있다.
- **C (Computing)**: 현재 이 버퍼의 데이터로 연산이 진행 중인 상태.
- **_ (Idle)**: 아직 사용되지 않았거나, 연산이 끝나 비어있는 상태.

### 프롤로그 추적

```
초기 상태:     Buf0=_  Buf1=_  Buf2=_

프롤로그 s=0:
  async_load_tile(stage=0, tile=0)     -> Buf0에 타일0 로드 발행
  cp_async_commit_group()              -> 그룹G0 생성
  상태:        Buf0=L  Buf1=_  Buf2=_    미완료 그룹: {G0}

프롤로그 s=1:
  async_load_tile(stage=1, tile=1)     -> Buf1에 타일1 로드 발행
  cp_async_commit_group()              -> 그룹G1 생성
  상태:        Buf0=L  Buf1=L  Buf2=_    미완료 그룹: {G0, G1}
```

### 메인 루프 6회 반복 추적

```
=== kt=0 ===
  curr_stage = 0 % 3 = 0
  prefetch_tile = 0 + 2 = 2 -> async_load(stage=2, tile=2) -> G2 생성
  cp_async_wait_group(2): 미완료 {G0,G1,G2} = 3개 -> G0 완료 대기
  __syncthreads()
  -> Buf0 = R (G0 완료 보장)
  Compute on Buf0
  __syncthreads()

  상태 정리:   Buf0=C->_ Buf1=L/R  Buf2=L    미완료 그룹: {G1, G2}

=== kt=1 ===
  curr_stage = 1 % 3 = 1
  prefetch_tile = 1 + 2 = 3 -> async_load(stage=0, tile=3) -> G3 생성
  cp_async_wait_group(2): 미완료 {G1,G2,G3} = 3개 -> G1 완료 대기
  __syncthreads()
  -> Buf1 = R (G1 완료 보장)
  Compute on Buf1
  __syncthreads()

  상태 정리:   Buf0=L    Buf1=C->_ Buf2=L/R  미완료 그룹: {G2, G3}

=== kt=2 ===
  curr_stage = 2 % 3 = 2
  prefetch_tile = 2 + 2 = 4 -> async_load(stage=1, tile=4) -> G4 생성
  cp_async_wait_group(2): 미완료 {G2,G3,G4} = 3개 -> G2 완료 대기
  __syncthreads()
  -> Buf2 = R (G2 완료 보장)
  Compute on Buf2
  __syncthreads()

  상태 정리:   Buf0=L/R  Buf1=L    Buf2=C->_ 미완료 그룹: {G3, G4}

=== kt=3 ===
  curr_stage = 3 % 3 = 0
  prefetch_tile = 3 + 2 = 5 -> async_load(stage=2, tile=5) -> G5 생성
  cp_async_wait_group(2): 미완료 {G3,G4,G5} = 3개 -> G3 완료 대기
  __syncthreads()
  -> Buf0 = R (G3 완료 보장, Buf0에 타일3 데이터)
  Compute on Buf0
  __syncthreads()

  상태 정리:   Buf0=C->_ Buf1=L/R  Buf2=L    미완료 그룹: {G4, G5}

=== kt=4 ===
  curr_stage = 4 % 3 = 1
  prefetch_tile = 4 + 2 = 6 -> async_load(stage=0, tile=6) -> G6 생성
  cp_async_wait_group(2): 미완료 {G4,G5,G6} = 3개 -> G4 완료 대기
  __syncthreads()
  -> Buf1 = R (G4 완료 보장, Buf1에 타일4 데이터)
  Compute on Buf1
  __syncthreads()

  상태 정리:   Buf0=L    Buf1=C->_ Buf2=L/R  미완료 그룹: {G5, G6}

=== kt=5 ===
  curr_stage = 5 % 3 = 2
  prefetch_tile = 5 + 2 = 7 -> async_load(stage=1, tile=7) -> G7 생성
  cp_async_wait_group(2): 미완료 {G5,G6,G7} = 3개 -> G5 완료 대기
  __syncthreads()
  -> Buf2 = R (G5 완료 보장, Buf2에 타일5 데이터)
  Compute on Buf2
  __syncthreads()

  상태 정리:   Buf0=L/R  Buf1=L    Buf2=C->_ 미완료 그룹: {G6, G7}
```

### 링 버퍼 순환 패턴 요약

```
kt:      0    1    2    3    4    5    6    7    8  ...
연산:  Buf0 Buf1 Buf2 Buf0 Buf1 Buf2 Buf0 Buf1 Buf2 ...
프리: tile2 tile3 tile4 tile5 tile6 tile7 tile8 tile9 tile10...
목적: Buf2 Buf0 Buf1 Buf2 Buf0 Buf1 Buf2 Buf0 Buf1 ...
```

이 패턴에서 다음이 관찰된다.

1. 연산 버퍼는 `0 -> 1 -> 2 -> 0 -> 1 -> 2 -> ...` 순환이다.
2. 프리페치 목적지는 연산 버퍼보다 항상 2단계 앞선 버퍼이다.
3. 각 버퍼는 "로드 -> 대기 -> 연산 -> 다음 로드에 재사용"의 3단계 생명주기를 거친다.

### commit_group과 wait_group 호출 시점 추적

```
시간선 (타일 0~7 기준):

프롤로그:
  commit(G0) -- 타일0 로드 완료 표시
  commit(G1) -- 타일1 로드 완료 표시

kt=0: commit(G2) -- 타일2 로드 표시   -> wait(2) -- G0 완료 보장
kt=1: commit(G3) -- 타일3 로드 표시   -> wait(2) -- G1 완료 보장
kt=2: commit(G4) -- 타일4 로드 표시   -> wait(2) -- G2 완료 보장
kt=3: commit(G5) -- 타일5 로드 표시   -> wait(2) -- G3 완료 보장
kt=4: commit(G6) -- 타일6 로드 표시   -> wait(2) -- G4 완료 보장
kt=5: commit(G7) -- 타일7 로드 표시   -> wait(2) -- G5 완료 보장
kt=6: (prefetch 없음, 타일 소진)      -> wait(2) -- G6 완료 보장
kt=7: (prefetch 없음)                 -> wait(2) -- G7 완료 보장
```

마지막 몇 반복에서는 프리페치할 타일이 없으므로 새 그룹이 생성되지 않는다. 이 경우 wait_group(2)는 이미 미완료 그룹이 2개 이하이므로 즉시 통과하거나, 1개가 남아 있으면 그것의 완료를 보장한다.

---

## 7. 핵심 정리

- **cp.async**: Ampere(SM 8.0+)에서 도입된 글로벌 메모리 -> 공유 메모리 직접 복사 기능. 레지스터 파일을 우회하여 전송한다.
- **3개의 PTX 명령**:
  1. `cp.async.cg.shared.global [dst], [src], size` -- 비동기 복사 발행
  2. `cp.async.commit_group` -- 발행된 복사들을 그룹으로 묶음
  3. `cp.async.wait_group N` -- 미완료 그룹이 N개 이하가 될 때까지 대기
- **트리플 버퍼링**: `NUM_STAGES=3`으로 3개의 공유 메모리 버퍼를 링 버퍼 방식으로 순환하며, 파이프라인 깊이를 2로 증가시켜 더블 버퍼링보다 더 많은 레이턴시를 숨긴다.
- **SASS 확인**: 올바르게 컴파일되면 `LDGSTS` 명령어가 생성된다. `LDG` + `STS` 쌍이 보이면 cp.async가 적용되지 않은 것이다.
- **레지스터 우회**: 전송 과정에서 레지스터를 소비하지 않으므로 레지스터 압력이 감소하고, 이는 점유율 향상이나 더 공격적인 레지스터 블로킹을 가능하게 한다.
- **최종 성능**: 순수 FP32 FMA 기반으로 cuBLAS의 약 85-95%를 달성한다. 나머지 차이는 Tensor Core, 아키텍처별 튜닝, 워프 스케줄링 최적화에 기인한다.

---

## 8. 최종 마스터리 체크포인트

이 퀴즈는 전체 교육 과정의 최종 평가이다. 모듈 4-9에서 다룬 6단계 최적화(나이브 -> 코얼레싱 -> 타일링 -> 레지스터 블로킹 -> 더블 버퍼링 -> 비동기 복사)의 핵심 개념을 종합적으로 확인한다.

### Q1. (Remember)

cp.async 명령어를 사용하기 위해 필요한 최소 Compute Capability는 무엇인가?

<details>
<summary>정답</summary>

8.0 (Ampere 아키텍처). 코드에서 `__CUDA_ARCH__ >= 800`으로 가드되어 있다. Volta(7.0)나 Turing(7.5)에서는 cp.async를 사용할 수 없다.

코드 참조: `src/kernels/sgemm_async_copy.cu:43`

</details>

### Q2. (Remember)

SASS 어셈블리에서 cp.async가 성공적으로 하드웨어 명령으로 변환되었을 때 나타나는 명령어 이름은 무엇인가?

<details>
<summary>정답</summary>

LDGSTS (Load Global Store Shared). 이 명령어는 글로벌 메모리에서 공유 메모리로의 직접 전송을 하나의 하드웨어 명령으로 수행한다. 기존의 LDG(Load Global) + STS(Store Shared) 2개 명령어 조합과 대조된다.

</details>

### Q3. (Understand)

cp.async의 레지스터 우회가 가져오는 구체적인 이점 2가지를 설명하시오.

<details>
<summary>정답</summary>

1. **레지스터 압력 감소로 인한 점유율 향상**: 기존 LDG+STS 방식에서는 전송 과정에서 레지스터가 점유된다. cp.async는 레지스터를 사용하지 않으므로, 같은 커널에서 전체 레지스터 사용량이 줄어든다. SM당 허용되는 레지스터 총량이 고정되어 있으므로, 블록당 레지스터 수가 줄면 더 많은 블록을 동시에 실행할 수 있다(점유율 향상).

2. **Long Scoreboard 스톨 제거**: 기존 방식에서 LDG의 목적지 레지스터는 글로벌 메모리 로드가 완료될 때까지(400+ 사이클) 사용할 수 없다. STS가 이 레지스터를 읽으려 하면 Long Scoreboard 스톨이 발생한다. cp.async는 레지스터를 거치지 않으므로, 이 의존성 사슬 자체가 존재하지 않는다.

</details>

### Q4. (Apply)

`cp_async_wait_group(2)`를 호출했을 때 하드웨어가 보장하는 것은 정확히 무엇인가?

<details>
<summary>정답</summary>

현재 미완료(pending) 상태인 비동기 복사 그룹 중에서, 가장 오래된 것들부터 완료시켜 미완료 그룹의 수가 2개 이하가 될 때까지 대기한다.

예를 들어, 미완료 그룹이 {G3, G4, G5}로 3개라면, G3(가장 오래된 그룹)의 완료를 보장한다. 이후 미완료 그룹은 {G4, G5}로 2개가 된다.

미완료 그룹이 이미 2개 이하라면, wait_group(2)는 아무런 대기 없이 즉시 반환된다.

</details>

### Q5. (Apply)

`NUM_STAGES=3`, `kt=5`일 때, `curr_stage`와 `prefetch_stage`의 값을 각각 계산하시오.

<details>
<summary>정답</summary>

```
curr_stage = kt % NUM_STAGES = 5 % 3 = 2

prefetch_tile = kt + NUM_STAGES - 1 = 5 + 3 - 1 = 7
prefetch_stage = prefetch_tile % NUM_STAGES = 7 % 3 = 1
```

따라서 kt=5에서는 Buf2의 데이터로 연산하면서, 타일 7을 Buf1에 비동기 로드한다.

검증: kt=5에서 연산하는 것은 5번째 K-타일(타일 인덱스 5)이고, 이 데이터는 이전에 Buf2(= 5 이전의 어떤 반복에서 프리페치된 것)에 로드되어 있어야 한다. 실제로 kt=3에서 prefetch_tile=5가 stage=5%3=2에 로드되었다.

</details>

### Q6. (Analyze)

`cp_async_commit_group()`을 호출하지 않은 채 `cp_async_wait_group(2)`를 호출하면 어떤 문제가 발생하는가?

<details>
<summary>정답</summary>

commit_group을 호출하지 않으면 발행된 cp.async 복사들이 그룹으로 분리되지 않는다. 모든 복사가 하나의 암묵적 그룹(또는 그룹 없음 상태)에 속하게 되어 다음 문제가 발생한다.

1. **세밀한 동기화 불가**: wait_group(2)는 그룹 단위로 동기화한다. 그룹이 형성되지 않았으므로, 특정 타일의 로드 완료를 보장할 수 없다. 결과적으로 아직 도착하지 않은 데이터로 연산을 시도하여 잘못된 결과가 나올 수 있다.

2. **파이프라인 무효화**: 트리플 버퍼링의 핵심은 "타일 단위로 그룹을 만들고, 가장 오래된 그룹의 완료만 보장하면서 나머지는 비동기로 진행"하는 것이다. 그룹이 없으면 이 메커니즘이 작동하지 않으며, 안전을 위해 wait_group(0)(전체 대기)을 사용해야 한다. 이는 비동기성을 완전히 제거하여 성능이 크게 저하된다.

</details>

### Q7. (Analyze)

트리플 버퍼링이 더블 버퍼링보다 레이턴시 은닉에 유리한 이유를 파이프라인 깊이의 관점에서 설명하시오.

<details>
<summary>정답</summary>

파이프라인 깊이란 동시에 진행 중인 비동기 로드의 수를 의미한다.

- **더블 버퍼링**: 파이프라인 깊이 = `NUM_STAGES - 1 = 1`. 현재 타일을 연산하는 동안 다음 타일 1개만 로드할 수 있다. 연산 시간을 T_compute, 로드 레이턴시를 T_load라 하면, `T_load > T_compute`일 때 `T_load - T_compute`만큼의 유휴 시간이 발생한다.

- **트리플 버퍼링**: 파이프라인 깊이 = `NUM_STAGES - 1 = 2`. 현재 타일을 연산하는 동안 2개의 로드가 동시에 진행된다. `T_load > 2 * T_compute`가 아닌 한, 유휴 시간 없이 연산과 로드를 완전히 중첩시킬 수 있다.

BK=8로 타일이 작아 T_compute가 비교적 짧은 이 커널에서는, T_load가 T_compute보다 상당히 클 수 있으므로 파이프라인 깊이 2(트리플 버퍼링)가 깊이 1(더블 버퍼링)보다 유의미한 성능 향상을 가져온다.

</details>

### Q8. (Evaluate)

`NUM_STAGES=4` (쿼드 버퍼링)으로 변경할 경우의 장단점을 공유 메모리 사용량과 점유율 관점에서 평가하시오.

<details>
<summary>정답</summary>

**장점:**
- 파이프라인 깊이가 3으로 증가하여, 더 긴 메모리 레이턴시도 숨길 수 있다.
- 메모리 접근 패턴이 불규칙하거나 L2 캐시 미스가 많은 경우에 유리하다.

**단점:**
- 공유 메모리 사용량이 증가한다.
  - NUM_STAGES=3: As + Bs = 약 26.2 KB
  - NUM_STAGES=4: As[4][128][9] + Bs[4][8][129] = 약 34.9 KB
  - 증가분: 약 8.7 KB
- SM당 공유 메모리가 한정되어 있으므로(A100 기준 164 KB), 블록당 공유 메모리 사용량이 증가하면 SM당 동시 실행 가능한 블록 수가 줄어들어 점유율이 낮아질 수 있다.
- 대부분의 실제 시나리오에서 NUM_STAGES=3이면 충분한 레이턴시 은닉을 제공하므로, NUM_STAGES=4의 추가적인 파이프라인 깊이가 공유 메모리 증가를 정당화하지 못하는 경우가 많다.

**결론**: NUM_STAGES 튜닝은 "파이프라인 깊이 vs 공유 메모리 소비"의 트레이드오프이다. Nsight Compute에서 Wait 스톨이 높게 나타나면 NUM_STAGES를 늘리는 것을 고려하고, 공유 메모리가 점유율 병목이라면 NUM_STAGES를 줄이는 것이 바람직하다.

</details>

### Q9. (Create)

Ampere 미만의 GPU(예: Turing, Compute Capability 7.5)에서도 동작하는 폴백(fallback) 전략을 설계하시오. cp.async를 사용할 수 없을 때 어떻게 유사한 파이프라인 효과를 달성할 수 있는가?

<details>
<summary>정답</summary>

다음과 같은 폴백 전략을 설계할 수 있다.

1. **컴파일 타임 분기**: 이미 커널에 구현된 `#if __CUDA_ARCH__ >= 800` 가드를 활용하여, Ampere 이상에서는 cp.async 경로를, 그 미만에서는 기존 경로를 선택한다.

2. **더블 버퍼링 + LDG/STS 폴백**: Ampere 미만에서는 `NUM_STAGES=2`의 더블 버퍼링을 사용하되, 데이터 전송을 전통적인 LDG(글로벌->레지스터) + STS(레지스터->공유) 방식으로 수행한다.

3. **레지스터 프리페치**: cp.async의 비동기성을 소프트웨어적으로 모방하기 위해, K 루프의 연산 시작 전에 다음 타일의 LDG를 미리 발행한다. 글로벌 로드는 하드웨어적으로 비동기이므로, LDG 발행 후 연산을 먼저 수행하고 STS를 나중에 하면 일부 레이턴시를 숨길 수 있다.

4. **구조**:
```cuda
#if __CUDA_ARCH__ >= 800
    // cp.async + 트리플 버퍼링 경로
    // NUM_STAGES=3, LDGSTS 사용
#else
    // LDG/STS + 더블 버퍼링 경로
    // NUM_STAGES=2, 레지스터를 통한 전송
    // 프롤로그: LDG로 첫 타일을 레지스터에 로드, STS로 공유에 저장
    // 메인 루프: LDG(다음 타일) -> 연산(현재 타일) -> STS(다음 타일) -> 동기화
#endif
```

이 폴백은 cp.async만큼 효율적이지는 않지만(레지스터를 소비하고, Long Scoreboard 스톨이 일부 발생), 파이프라인의 기본 구조는 유지된다.

</details>

### Q10. (Create)

BK를 현재의 8에서 16으로 변경할 경우, 연쇄적으로 조정해야 할 매개변수들과 그 영향을 분석하시오.

<details>
<summary>정답</summary>

BK=16으로 변경하면 다음의 연쇄적 영향이 발생한다.

1. **공유 메모리 사용량 증가**:
   - As[3][128][17]: 3 * 128 * 17 * 4 = 26,112 바이트 (기존 13,824에서 약 2배)
   - Bs[3][16][129]: 3 * 16 * 129 * 4 = 24,768 바이트 (기존 12,384에서 약 2배)
   - 합계: 약 50.9 KB (기존 26.2 KB에서 약 2배)
   - 여전히 A100의 164 KB 한도 내이지만, 점유율에 영향을 줄 수 있다.

2. **K 내부 루프 반복 횟수 증가**: BK 내부 루프가 8회에서 16회로 증가하여, 타일당 연산량이 2배가 된다. 이는 산술 강도를 높여 메모리 레이턴시 은닉에 유리하다.

3. **K 타일 수 감소**: num_k_tiles = K / BK가 절반으로 줄어든다. 메인 루프 반복 횟수가 줄어들어 루프 오버헤드가 감소한다.

4. **로드 패턴 조정**: BM * BK = 128 * 16 = 2048 원소를 256 스레드가 로드하려면 스레드당 8개 원소를 담당해야 한다(기존 4개에서 2배). 이는 async_load_tile 내의 루프 반복 횟수에 영향을 준다.

5. **뱅크 충돌 패딩**: `BK + 1 = 17`로 변경되어 패딩이 유지된다. 다만 패딩으로 인한 추가 공유 메모리 오버헤드가 증가한다.

6. **NUM_STAGES 재검토**: 공유 메모리 사용량이 2배로 증가하므로, NUM_STAGES=3이 점유율을 과도하게 낮추는지 확인해야 한다. 필요하면 NUM_STAGES=2로 줄여 공유 메모리를 절약할 수 있다.

7. **성능 트레이드오프**: BK 증가는 산술 강도를 높이지만(연산/타일 증가), 공유 메모리 소모 증가로 점유율이 낮아질 수 있다. 최적의 BK는 프로파일링을 통해 결정해야 한다.

</details>

---

## 9. 다음 단계 미리보기

축하한다. 6단계 최적화를 모두 완료했다. 나이브 커널(cuBLAS 대비 1-5%)에서 출발하여, 메모리 코얼레싱, 공유 메모리 타일링, 레지스터 블로킹, 더블 버퍼링, 그리고 비동기 복사까지 적용하여 cuBLAS의 85-95%에 도달하는 과정을 단계별로 이해하고 구현했다.

다음 모듈에서는 Nsight Compute를 사용하여 전체 커널을 체계적으로 프로파일링하는 방법을 배운다. 지금까지 정성적으로 이해한 성능 병목(메모리 대역폭, 레이턴시 은닉, 점유율, 명령어 처리량)을 정량적으로 측정하고, 루프라인 모델 위에서 각 최적화 단계의 위치를 확인하게 된다.

---

*이 문서의 소스 코드 참조: `src/kernels/sgemm_async_copy.cu`*
