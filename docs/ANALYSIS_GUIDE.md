# BareMetal-SGEMM: Nsight Compute Analysis Guide

이 가이드는 SGEMM 커널의 성능을 Nsight Compute로 분석하는 방법을 설명합니다.

## 1. Nsight Compute 기본 사용법

### 프로파일링 실행
```bash
# 기본 프로파일링
ncu --set full -o sgemm_profile ./sgemm_benchmark -m 4096 -n 4096 -k 4096 -l 5

# 특정 메트릭만 수집
ncu --metrics sm__throughput.avg_pct_of_peak_sustained_elapsed,\
dram__throughput.avg_pct_of_peak_sustained_elapsed \
-o sgemm_profile ./sgemm_benchmark

# SASS 코드 분석 포함
ncu --set full --section SourceCounters -o sgemm_sass ./sgemm_benchmark
```

### 결과 보기
```bash
ncu-ui sgemm_profile.ncu-rep
```

## 2. Speed of Light (SOL) 분석

### 핵심 메트릭
- **SM Throughput (%)**: 연산 유닛 활용도
- **Memory Throughput (%)**: 메모리 대역폭 활용도

### 해석
| SM | Memory | 상태 |
|----|--------|------|
| High | Low | Compute bound - 최적 상태 |
| Low | High | Memory bound - 데이터 재사용 필요 |
| Low | Low | Latency bound - 파이프라인 최적화 필요 |
| High | High | 두 리소스 모두 효율적 사용 - 최적 상태 |

### 커널별 예상 SOL
```
Naive:           SM ~5%,  Memory ~20%  (latency bound)
Coalesced:       SM ~10%, Memory ~50%  (memory bound)
Tiled:           SM ~30%, Memory ~60%  (memory bound)
RegisterBlocking: SM ~60%, Memory ~70%  (balanced)
DoubleBuffer:    SM ~75%, Memory ~80%  (good balance)
AsyncCopy:       SM ~85%, Memory ~85%  (optimal)
```

## 3. Stall 원인 분석

### Warp State Statistics에서 확인

#### Long Scoreboard
- **의미**: 글로벌 메모리 데이터 대기
- **해결책**:
  - Double/Triple buffering 적용
  - cp.async 사용 (Ampere+)
  - 프리페치 타이밍 조정

#### Wait
- **의미**: 동기화 대기 (barrier, cp.async.wait 등)
- **해결책**:
  - 파이프라인 깊이 증가
  - __syncthreads() 위치 최적화

#### Math Pipe Throttle
- **의미**: 연산 유닛이 포화 상태
- **해석**: 최적의 상태! 더 이상 최적화 불필요

#### LG Throttle
- **의미**: L1/공유 메모리 대역폭 제한
- **해결책**: 뱅크 충돌 해결, 벡터화 적용

## 4. 메모리 분석

### Global Memory Coalescing
```
확인 위치: Memory Workload Analysis > L1/TEX Cache > Sectors/Request

이상적인 값: 1.0 (완벽한 coalescing)
문제가 있는 값: >4.0 (많은 트랜잭션 필요)
```

### SASS 명령어 확인
```
좋은 패턴:
- LDG.E.128: 128-bit 글로벌 로드 (float4)
- STG.E.128: 128-bit 글로벌 스토어

나쁜 패턴:
- 많은 LDG.E (32-bit 로드)
- 비정렬 접근으로 인한 다수의 트랜잭션
```

### Bank Conflict 확인
```
확인 위치: Memory Workload Analysis > L1/TEX Cache >
           Wavefronts Shared Excessive

이상적인 값: 1.0
문제가 있는 값: >1.0 (bank conflict 발생)
```

### cp.async 확인 (Ampere+)
```
SASS에서 확인할 명령어:
- LDGSTS: Load Global Store Shared (cp.async의 하드웨어 구현)

보이면 좋은 것:
- LDGSTS 명령어가 연산 명령어와 인터리브되어 있음

보이면 안 좋은 것:
- LDG 후 STS 패턴 (레지스터를 거치는 비효율적 경로)
```

## 5. Register Pressure 분석

### 레지스터 사용량 확인
```
확인 위치: Launch Statistics > Registers Per Thread

권장 값: 64-128 (아키텍처에 따라 다름)
경고: 255 초과시 spilling 발생
```

### Register Spilling 확인
```
SASS에서 확인:
- LDL: Load Local (레지스터 -> 로컬 메모리)
- STL: Store Local (로컬 메모리 -> 레지스터)

이 명령어가 보이면 성능 저하!

해결책:
1. __launch_bounds__ 사용
2. 타일 크기 줄이기
3. 변수 범위 최소화
```

## 6. Occupancy 분석

### 점유율 계산
```
확인 위치: Occupancy > Achieved Occupancy

계산: (활성 워프 수) / (SM당 최대 워프 수)

영향 요소:
- 레지스터 사용량
- 공유 메모리 사용량
- 블록 크기
```

### 최적 점유율
```
일반적인 오해: 높은 점유율 = 높은 성능

실제:
- 레지스터 blocking은 낮은 점유율로도 높은 성능 달성 가능
- ILP(Instruction-Level Parallelism)가 높으면 25-50% 점유율로 충분
- 메모리 바운드 커널은 높은 점유율이 유리
```

## 7. 실전 분석 체크리스트

### Phase 1 (Naive → Coalesced)
- [ ] LDG.E가 LDG.E.128로 변경되었는가?
- [ ] Global Load Efficiency > 80%?
- [ ] Sectors/Request ≈ 1.0?

### Phase 2 (Coalesced → Tiled)
- [ ] 공유 메모리 사용량이 예상대로인가?
- [ ] Wavefronts Shared Excessive ≈ 1.0?
- [ ] Long Scoreboard가 감소했는가?

### Phase 3 (Tiled → Register Blocking)
- [ ] FMA 파이프라인 활용도가 증가했는가?
- [ ] 레지스터 사용량이 255 이하인가?
- [ ] LDL/STL이 없는가 (no spilling)?

### Phase 4 (Register Blocking → Double Buffer)
- [ ] Long Scoreboard가 감소했는가?
- [ ] 로드와 연산이 오버랩되는가?
- [ ] 공유 메모리 사용량이 2배가 되었는가?

### Phase 5 (Double Buffer → Async Copy)
- [ ] LDGSTS 명령어가 생성되었는가?
- [ ] LDG + STS 패턴이 제거되었는가?
- [ ] Wait stall이 적절한가 (너무 높지 않은가)?

## 8. 성능 튜닝 결정 트리

```
SOL에서 SM Throughput이 낮은가?
├─ Yes → Memory가 높은가?
│        ├─ Yes → 데이터 재사용 증가 (Register Blocking)
│        └─ No → Latency 숨기기 (Double Buffering, cp.async)
└─ No → 최적 상태, micro-optimization 검토

Stall 원인이 Long Scoreboard인가?
├─ Yes → 프리페치/파이프라이닝 개선
└─ No → Wait인가?
         ├─ Yes → 동기화 최적화
         └─ No → Math Pipe Throttle이면 최적!
```

## 9. cuBLAS 비교 분석

cuBLAS와 비교할 때 확인할 점:
1. **텐서 코어 사용 여부**: cuBLAS는 FP32도 텐서코어 활용 가능 (TF32)
2. **타일 크기**: cuBLAS는 아키텍처별 최적 타일 크기 사용
3. **스케줄링**: 더 정교한 워프 스케줄링

90% 이상 달성 못하는 경우:
- 텐서 코어 미사용 (우리 구현)
- 타일 크기 최적화 부족
- 마이크로 아키텍처 특화 최적화 부족
