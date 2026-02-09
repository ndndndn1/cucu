// ============================================================
// GPU Architect: Rise from Silicon — Game Data
// All 75 quiz questions, code traces, reviews, modules, ranks
// ============================================================

// --- Module Definitions ---
const MODULES = [
  {id:'00',title:'서론 및 학습 안내',sub:'학습 로드맵',level:null,hasQuiz:false,hasSim:false,hasCode:false,hasReview:false},
  {id:'01',title:'GPU 아키텍처',sub:'SM, 워프, 메모리 계층',level:null,hasQuiz:true,hasSim:true,hasCode:false,hasReview:false},
  {id:'02',title:'CUDA 프로그래밍 기초',sub:'Driver API, PTX JIT',level:null,hasQuiz:true,hasSim:true,hasCode:false,hasReview:false},
  {id:'03',title:'행렬곱셈 수학적 배경',sub:'FLOP, 산술 강도, 루프라인',level:null,hasQuiz:true,hasSim:true,hasCode:false,hasReview:false},
  {id:'04',title:'Level 0 — 나이브 구현',sub:'1 thread = 1 element',level:0,hasQuiz:true,hasSim:true,hasCode:true,hasReview:true},
  {id:'05',title:'Level 1 — 메모리 코얼레싱',sub:'float4 벡터화 로드',level:1,hasQuiz:true,hasSim:true,hasCode:true,hasReview:true},
  {id:'06',title:'Level 2 — 공유 메모리 타일링',sub:'__shared__ + 뱅크 충돌 패딩',level:2,hasQuiz:true,hasSim:true,hasCode:true,hasReview:true,mastery:true,boss:'boss1'},
  {id:'07',title:'Level 3 — 레지스터 블로킹',sub:'8×8 외적 마이크로커널',level:3,hasQuiz:true,hasSim:true,hasCode:true,hasReview:true},
  {id:'08',title:'Level 4 — 더블 버퍼링',sub:'소프트웨어 파이프라이닝',level:4,hasQuiz:true,hasSim:true,hasCode:true,hasReview:true,mastery:true,boss:'boss2'},
  {id:'09',title:'Level 5 — 비동기 복사',sub:'cp.async, 트리플 버퍼',level:5,hasQuiz:true,hasSim:true,hasCode:true,hasReview:true,mastery:true,boss:'boss3'},
  {id:'10',title:'종합 성능분석',sub:'Nsight Compute, SOL 분석',level:null,hasQuiz:true,hasSim:true,hasCode:false,hasReview:false},
];

// --- Rank Definitions ---
const RANKS = [
  {id:'intern',name:'수습 엔지니어',en:'Intern Engineer',icon:'🔰',modules:[],gflops:0,cubl:0},
  {id:'junior',name:'주니어 엔지니어',en:'Junior Engineer',icon:'⚙️',modules:['00','01','02','03'],gflops:60,cubl:3},
  {id:'engineer',name:'엔지니어',en:'Engineer',icon:'🔧',modules:['04'],gflops:250,cubl:10},
  {id:'senior',name:'시니어 엔지니어',en:'Senior Engineer',icon:'🛠️',modules:['05'],masteryReq:'06',gflops:1000,cubl:30},
  {id:'staff',name:'스태프 엔지니어',en:'Staff Engineer',icon:'⚡',modules:['06'],gflops:2500,cubl:60},
  {id:'principal',name:'프린시펄 엔지니어',en:'Principal Engineer',icon:'💎',modules:['07'],masteryReq:'08',gflops:5000,cubl:78},
  {id:'distinguished',name:'디스팅귀시드 엔지니어',en:'Distinguished Engineer',icon:'🌟',modules:['08'],masteryReq:'09',gflops:8000,cubl:90},
  {id:'chip_architect',name:'칩 아키텍트',en:'Chip Architect',icon:'👑',modules:['09','10'],gflops:9500,cubl:95},
];

// --- Quiz Data: All 75 Questions ---
// Format: {q, o:[4 options], a:correctIndex, b:bloomLevel, e:explanation}
const QUIZZES = {
'01': [
  {q:'GPU에서 하나의 워프(Warp)를 구성하는 스레드 수는?',o:['16개','32개','64개','128개'],a:1,b:'R',e:'WARP_SIZE = 32. NVIDIA GPU의 기본 실행 단위인 워프는 항상 32개 스레드로 구성됩니다.'},
  {q:'글로벌 메모리(HBM) 접근의 지연시간은 약 몇 사이클인가?',o:['20-30 사이클','100-200 사이클','400-600 사이클','2000-3000 사이클'],a:2,b:'R',e:'글로벌 메모리 접근은 약 400-600 사이클의 레이턴시를 가집니다. 레지스터(1 사이클), 공유 메모리(20-30 사이클)와 비교됩니다.'},
  {q:'레지스터와 공유 메모리의 범위(scope) 차이로 옳은 것은?',o:['둘 다 블록 단위로 공유','레지스터=스레드 전용, 공유 메모리=블록 공유','레지스터=블록 공유, 공유 메모리=그리드 공유','둘 다 스레드 전용'],a:1,b:'U',e:'레지스터는 thread-private, 공유 메모리는 block-shared입니다.'},
  {q:'GPU가 CPU보다 행렬곱에 유리한 이유로 적절하지 않은 것은?',o:['높은 데이터 병렬성','높은 단일 스레드 성능','높은 연산/메모리 비율','예측 가능한 메모리 접근 패턴'],a:1,b:'U',e:'GPU의 단일 스레드 성능은 CPU보다 낮습니다. GPU의 강점은 대규모 병렬성과 높은 처리량입니다.'},
  {q:'4096×4096 행렬에 16×16 스레드 블록일 때, 그리드의 블록 수는?',o:['256 블록','4,096 블록','65,536 블록','16,777,216 블록'],a:2,b:'Ap',e:'dim3(4096/16, 4096/16) = dim3(256, 256), 총 65,536 블록입니다.'},
  {q:'SM당 최대 레지스터 65,536개, 커널이 스레드당 128개 사용 시 SM당 최대 스레드 수는?',o:['256 (점유율 12.5%)','512 (점유율 25%)','1024 (점유율 50%)','2048 (점유율 100%)'],a:1,b:'Ap',e:'65,536 / 128 = 512 스레드 = 16 워프. 최대 2048 대비 점유율 25%입니다.'},
  {q:'점유율이 25%인데 높은 성능을 보이는 경우의 주된 원인은?',o:['L2 캐시 적중률이 높아서','레지스터 블로킹에 의한 높은 ILP','공유 메모리 대역폭이 충분해서','워프 스케줄러의 최적화 덕분'],a:1,b:'An',e:'64개 독립 FMA 명령어가 높은 ILP(명령어 수준 병렬성)를 제공하여, TLP가 낮아도 파이프라인을 가득 채울 수 있습니다.'},
  {q:'supports_async_copy가 compute_major >= 8일 때 true인 이유는?',o:['Turing에서 cp.async 도입','Ampere에서 cp.async 도입','Hopper에서 cp.async 도입','모든 세대에서 지원'],a:1,b:'U',e:'cp.async 명령어는 Compute Capability 8.0 (Ampere) 이상에서만 지원됩니다.'},
],
'02': [
  {q:'Driver API에서 컨텍스트를 생성하는 함수 이름은?',o:['cudaCreateContext','cuCtxCreate','cuContextInit','cuDeviceCreateContext'],a:1,b:'R',e:'cuCtxCreate는 CUDA Driver API에서 컨텍스트를 생성하는 함수입니다.'},
  {q:'PTX JIT 컴파일의 장점이 아닌 것은?',o:['아키텍처 이식성','런타임 최적화','컴파일 시간 단축','동적 로딩'],a:2,b:'U',e:'JIT 컴파일은 런타임에 수행되므로 오히려 초기 로딩 시간이 증가합니다. 장점은 이식성, 최적화, 동적 로딩입니다.'},
  {q:'커널 파라미터 배열 void** args에서 각 원소가 담는 것은?',o:['변수의 값 자체','변수의 주소(포인터)','변수의 타입 정보','변수의 크기(bytes)'],a:1,b:'Ap',e:'void* args[] = { &d_A, &d_B, &d_C, &N }; 각 원소는 값의 주소입니다.'},
  {q:'JIT 컴파일 실패 시 에러 로그를 얻는 데 사용하는 옵션은?',o:['CU_JIT_INFO_LOG_BUFFER','CU_JIT_ERROR_LOG_BUFFER','CU_JIT_COMPILE_LOG','CU_JIT_FAILURE_REASON'],a:1,b:'An',e:'CU_JIT_ERROR_LOG_BUFFER와 CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES 옵션으로 에러 로그를 받을 수 있습니다.'},
  {q:'DeviceMemory에서 RAII 패턴이 중요한 이유는?',o:['성능이 향상되므로','GPU 메모리 누수를 방지하므로','코드가 짧아지므로','멀티스레드 안전성을 위해'],a:1,b:'U',e:'GPU 메모리는 제한적이고 누수 시 자동 회수되지 않습니다. RAII로 예외 발생 시에도 안전하게 해제합니다.'},
  {q:'4096×4096 float 행렬 3개의 GPU 메모리 할당량은?',o:['48 MB','96 MB','192 MB','384 MB'],a:2,b:'Ap',e:'4096×4096×4 = 67,108,864 바이트 = 64 MB. 3개 행렬이면 192 MB입니다.'},
],
'03': [
  {q:'SGEMM 총 FLOP 수 공식은?',o:['M × N × K','2 × M × N × K','M × N × K × 3','M × N + N × K'],a:1,b:'R',e:'C의 각 원소는 K번의 곱셈과 K번의 덧셈 = 2K FLOPs. 총 M×N 원소이므로 2×M×N×K입니다.'},
  {q:'M=N=K=2048일 때 총 GFLOP 수는?',o:['약 4.29','약 8.59','약 17.18','약 34.36'],a:2,b:'Ap',e:'2 × 2048³ = 2 × 8,589,934,592 ≈ 17.18 GFLOP입니다.'},
  {q:'"산술 강도가 높다"는 것의 의미는?',o:['연산 속도가 빠르다','전송 바이트 대비 많은 연산 수행','메모리 대역폭이 크다','코드가 효율적이다'],a:1,b:'U',e:'산술 강도 = FLOP/byte. 높으면 데이터를 한 번 로드 후 여러 번 재사용한다는 의미입니다.'},
  {q:'M=N=K=4096, 실행 시간 2.0ms일 때 GFLOP/s는?',o:['약 17,180','약 34,360','약 68,720','약 137,440'],a:2,b:'Ap',e:'2×4096³/2.0ms = 137.44 GFLOP / 0.002s ≈ 68,720... 아, 계산: 2×4096³ = 137,438,953,472 FLOP ÷ 0.002 = 68,719 GFLOP/s. 이 값은 비현실적이며 피크를 초과합니다.'},
  {q:'루프라인 모델에서 릿지 포인트를 결정하는 요소 2가지는?',o:['코어 수와 클럭','피크 GFLOP/s와 피크 메모리 대역폭','캐시 크기와 레이턴시','SM 수와 워프 크기'],a:1,b:'An',e:'릿지 포인트 = 피크 GFLOP/s ÷ 피크 GB/s. 이 두 하드웨어 요소가 릿지 포인트를 결정합니다.'},
],
'04': [
  {q:'나이브 커널에서 한 스레드가 계산하는 C 원소의 수는?',o:['1개','4개','8개','16개'],a:0,b:'R',e:'나이브 구현에서는 1 thread = 1 output element입니다.'},
  {q:'blockIdx.y의 역할은?',o:['블록 내 스레드 y축 위치','그리드 내 블록 y축 위치','워프 내 스레드 인덱스','SM 내 블록 인덱스'],a:1,b:'U',e:'blockIdx.y는 그리드 내에서 해당 블록의 y축 좌표를 나타냅니다.'},
  {q:'M=1024, N=1024, 블록 16×16일 때 총 스레드 수는?',o:['65,536','262,144','1,048,576','4,194,304'],a:2,b:'Ap',e:'그리드: 64×64 = 4,096 블록. 블록당 256 스레드. 총 1,048,576 스레드입니다.'},
  {q:'같은 워프에서 ty가 다른 두 스레드의 A 접근 주소 차이(lda=4096)는?',o:['4 바이트','1,024 바이트','16,384 바이트','65,536 바이트'],a:2,b:'An',e:'lda × sizeof(float) = 4096 × 4 = 16,384 바이트. 이는 비코얼레싱 접근입니다.'},
  {q:'나이브 커널의 산술 강도가 낮은 근본 원인은?',o:['루프 반복이 많아서','공유 메모리를 사용하지 않아서','레지스터가 부족해서','블록 크기가 작아서'],a:1,b:'An',e:'데이터 재사용이 없어 산술 강도가 0.25 FLOP/byte로 매우 낮습니다. 공유 메모리 캐싱이 핵심 해결책입니다.'},
  {q:'__restrict__ 키워드의 목적은?',o:['메모리를 정렬하기 위해','포인터 간 메모리 겹침이 없음을 보장','스레드를 동기화하기 위해','레지스터 사용을 줄이기 위해'],a:1,b:'U',e:'__restrict__는 컴파일러에게 포인터 에일리어싱이 없음을 알려 최적화를 가능하게 합니다.'},
  {q:'K=4096일 때 한 스레드의 총 글로벌 메모리 로드 횟수는?',o:['약 2,048회','약 4,096회','약 8,194회','약 16,384회'],a:2,b:'Ap',e:'A에서 K번 + B에서 K번 + 경계 조건 = 약 8,194회 글로벌 메모리 접근입니다.'},
  {q:'cuBLAS의 1-5%에 불과한 근본 원인 3가지 중 해당하지 않는 것은?',o:['메모리 코얼레싱 부재','데이터 재사용 부재','블록 크기가 16×16이라서','낮은 산술 강도 0.25 FLOP/byte'],a:2,b:'Ev',e:'블록 크기 16×16은 적절합니다. 근본 원인은 비코얼레싱, 재사용 부재, 낮은 산술 강도입니다.'},
],
'05': [
  {q:'128-bit 벡터화 로드에 대응하는 SASS 명령어는?',o:['LDG.E','LDG.E.64','LDG.E.128','LDS.128'],a:2,b:'R',e:'float4 (128-bit) 벡터화 로드는 SASS에서 LDG.E.128 명령어로 변환됩니다.'},
  {q:'메모리 코얼레싱이 중요한 이유는?',o:['캐시 적중률을 높이므로','코얼레싱 없으면 최대 32배 대역폭 낭비','레지스터 사용을 줄이므로','컴파일 시간이 단축되므로'],a:1,b:'U',e:'워프의 32 스레드가 비코얼레싱 접근하면 최대 32개 트랜잭션이 필요해 32배 대역폭 낭비입니다.'},
  {q:'K=1024일 때 벡터화 루프의 반복 횟수는?',o:['128회','256회','512회','1024회'],a:1,b:'Ap',e:'K_vec = K / 4 = 1024 / 4 = 256회입니다.'},
  {q:'B 행렬 접근이 float4 벡터화가 불가능한 이유는?',o:['B가 정수형이라서','k 증가 시 ldb×4 바이트 점프하여 불연속','B가 읽기 전용이라서','정렬이 안 되어 있어서'],a:1,b:'An',e:'k를 바꾸면 ldb×4 바이트만큼 점프하여 연속적인 16바이트 접근이 불가능합니다.'},
  {q:'FETCH_FLOAT4(A[row * lda + k*4])를 전개하면?',o:['*(float4*)(&A[row*lda+k*4])','reinterpret_cast<const float4*>(&(A[row*lda+k*4]))[0]','__ldg(&A[row*lda+k*4])','atomicAdd(&A[row*lda+k*4], 0)'],a:1,b:'Ap',e:'FETCH_FLOAT4 매크로는 주소를 float4 포인터로 캐스팅 후 역참조합니다.'},
  {q:'전치된 B를 사용하면 나은 이유는?',o:['행렬 크기가 줄어서','B도 float4 연속 접근이 가능해져서','연산량이 줄어서','동기화가 불필요해서'],a:1,b:'U',e:'BT[col][k]에서 k가 열 방향으로 연속 배치되어 A와 B 모두 float4 벡터화가 가능합니다.'},
  {q:'Level 0 대비 Level 1의 3-5배 향상의 한계 원인은?',o:['레지스터가 부족해서','산술 강도가 여전히 0.25로 낮아서','블록 크기가 작아서','PTX가 비효율적이라서'],a:1,b:'Ev',e:'실효 대역폭은 60-80%로 향상되었으나, 산술 강도는 여전히 0.25 FLOP/byte로 메모리 바운드입니다.'},
],
'06': [
  {q:'GPU 공유 메모리의 뱅크 수는?',o:['16개','32개','64개','128개'],a:1,b:'R',e:'GPU 공유 메모리는 32개 뱅크로 구성되어 있습니다.'},
  {q:'공유 메모리 타일링의 핵심 원리를 가장 잘 표현한 것은?',o:['스레드 수를 늘려 병렬성 증가','데이터를 글로벌에서 1번 로드, 공유에서 BLOCK_SIZE번 재사용','캐시 적중률을 최적화','루프를 언롤하여 분기 제거'],a:1,b:'U',e:'타일링의 핵심은 데이터 재사용입니다. 글로벌 메모리에서 한 번 로드한 데이터를 공유 메모리에서 BLOCK_SIZE번 재사용합니다.'},
  {q:'BLOCK_SIZE=32일 때 공유 메모리 총 사용량(패딩 무시)은?',o:['4 KB','8 KB','16 KB','32 KB'],a:1,b:'Ap',e:'A_tile(32×32) + B_tile(32×32) = 2 × 32 × 32 × 4 = 8,192 바이트 = 8 KB입니다.'},
  {q:'협력적 로딩에서 A_tile[ty][tx]에 로드하는 글로벌 메모리 주소는?',o:['A[row * lda + t * BLOCK_SIZE + tx]','A[ty * lda + tx]','A[blockIdx.y * lda + tx]','A[row * N + col]'],a:0,b:'Ap',e:'a_col = t * BLOCK_SIZE_K + tx이고 A[row * lda + a_col]입니다.'},
  {q:'두 번째 __syncthreads()를 제거하면 발생하는 문제는?',o:['컴파일 에러','빠른 스레드가 다음 타일을 덮어써 잘못된 값 사용','성능이 살짝 저하','아무 문제 없음'],a:1,b:'An',e:'빠른 스레드가 다음 타일의 데이터를 로드하여 기존 값을 덮어쓰면, 느린 스레드가 잘못된 값을 사용합니다.'},
  {q:'패딩 없이 ty=0과 ty=2가 A_tile[ty][0] 접근 시 둘 다 같은 뱅크에 매핑되는 이유는?',o:['배열 폭 16의 배수가 32의 약수이므로','뱅크 할당이 랜덤이므로','ty가 홀수/짝수이므로','패딩이 있으면 달라지지 않으므로'],a:0,b:'An',e:'폭=16일 때 bank=(ty*16+0)%32. ty=0→bank 0, ty=2→bank 0 (32%32=0). 16이 32의 약수이므로 충돌이 발생합니다.'},
  {q:'fma.rn.f32 %0, %1, %2, %0에서 수행하는 연산은?',o:['%0 = %1 + %2','%0 = %1 * %2','%0 = %1 * %2 + %0','%0 = %0 * %1 + %2'],a:2,b:'Ap',e:'FMA: d = a * b + c. 여기서 %0(sum) = %1(A값) × %2(B값) + %0(누적합)입니다.'},
  {q:'타일링의 메모리 트래픽 감소 배율은 16배인데 cuBLAS의 20-40%인 이유로 적절하지 않은 것은?',o:['산술 강도 4.0은 여전히 부족','레이턴시 은닉 메커니즘 부재','블록 크기 16×16이 너무 작아서','스레드당 1원소 계산으로 레지스터 비효율'],a:2,b:'Ev',e:'블록 크기는 문제가 아닙니다. 진짜 이유는 산술 강도 부족, 레이턴시 은닉 부재, 레지스터 비효율입니다.'},
],
'07': [
  {q:'TM=8, TN=8일 때 한 스레드가 계산하는 C 원소 수는?',o:['8개','16개','32개','64개'],a:3,b:'R',e:'TM × TN = 8 × 8 = 64개의 C 원소를 한 스레드가 계산합니다.'},
  {q:'레지스터 블로킹이 산술 강도를 높이는 공식은?',o:['AI = TM + TN','AI = TM × TN / (TM + TN)','AI = TM × TN / (2 × (TM + TN))','AI = 2 × TM × TN'],a:2,b:'U',e:'AI = TM×TN / (2×(TM+TN)). TM=TN=8일 때 64/32 = 2.0 FLOP/byte입니다.'},
  {q:'BM=128, BN=128, BK=8일 때 공유 메모리 사용량(패딩 포함)은?',o:['4 KB','8 KB','약 8.5 KB','16 KB'],a:2,b:'Ap',e:'A: 128×(8+1)×4 + B: 8×(128+1)×4 = 4,608 + 4,128 = 8,736 ≈ 8.5 KB입니다.'},
  {q:'256개 스레드가 A 타일(128×8=1024 원소) 로드 시 스레드당 로드 횟수는?',o:['1회','2회','4회','8회'],a:2,b:'Ap',e:'1024 원소 / 256 스레드 = 4 원소/스레드. A_TILE_ROW_STRIDE=32, 128/32=4번 반복입니다.'},
  {q:'A_TILE_ROW_STRIDE = 256/8 = 32의 의미는?',o:['32개 스레드가 한 행을 로드','256 스레드를 8열로 나누면 32행 한 번에 로드','32바이트 정렬 단위','BK의 4배'],a:1,b:'An',e:'256 스레드를 BK=8 열로 나누면 한 번에 32행을 로드할 수 있습니다. 128/32=4번 반복하면 전체 타일 완성.'},
  {q:'레지스터 255개 초과 시 발생하는 현상은?',o:['컴파일 에러','레지스터 스필링 (LDL/STL 명령어)','자동으로 공유 메모리 사용','성능에 영향 없음'],a:1,b:'An',e:'레지스터가 부족하면 로컬 메모리(DRAM)로 스필링됩니다. SASS에서 LDL/STL 명령어로 확인 가능합니다.'},
  {q:'점유율 25%인 Level 3가 점유율 50%인 Level 2보다 빠른 이유는?',o:['캐시가 더 커서','산술 강도 8배 + 64개 독립 FMA의 높은 ILP','워프가 더 많아서','메모리 대역폭이 더 높아서'],a:1,b:'Ev',e:'AI 8배 향상 + 64개 독립 FMA에 의한 ILP가 낮은 점유율(TLP)을 완전히 보상합니다.'},
  {q:'TM=4, TN=16으로 변경 시 산술 강도 변화는?',o:['AI = 2.0 (변화 없음)','AI = 1.6 (20% 감소)','AI = 2.5 (25% 증가)','AI = 4.0 (2배 증가)'],a:1,b:'Cr',e:'AI = 4×16/(2×(4+16)) = 64/40 = 1.6 FLOP/byte. 정사각형(8×8)이 최적입니다.'},
],
'08': [
  {q:'더블 버퍼링에서 사용하는 공유 메모리 버퍼 세트 수는?',o:['1','2','3','4'],a:1,b:'R',e:'더블 버퍼링은 이름 그대로 2개의 버퍼 세트를 사용합니다.'},
  {q:'소프트웨어 파이프라이닝의 핵심 원리는?',o:['메모리를 더 많이 사용하여 접근 횟수를 줄인다','루프를 언롤하여 분기 오버헤드를 제거한다','메모리 로드와 연산을 시간적으로 중첩시켜 레이턴시를 숨긴다','공유 메모리를 레지스터로 대체한다'],a:2,b:'U',e:'핵심은 로드(다음 타일)와 연산(현재 타일)을 동시에 수행하여 메모리 레이턴시를 숨기는 것입니다.'},
  {q:'kt=0,1,2,3에서 curr_buf 값의 패턴은?',o:['0,0,0,0','0,1,0,1','0,1,2,0','1,0,1,0'],a:1,b:'Ap',e:'curr_buf = kt % 2. kt=0→0, kt=1→1, kt=2→0, kt=3→1. 0과 1이 교대합니다.'},
  {q:'프롤로그를 생략하면 발생하는 문제는?',o:['컴파일 에러','초기화되지 않은 공유 메모리로 연산하여 잘못된 결과','성능이 약간 저하','다음 타일 로드가 실패'],a:1,b:'An',e:'프롤로그가 없으면 첫 번째 연산 시 버퍼에 유효한 데이터가 없어 잘못된 결과가 나옵니다.'},
  {q:'Long Scoreboard 감소 + Math Pipe Throttle 증가의 의미는?',o:['성능 저하 신호','메모리 오류 발생','GPU가 기다림에서 연산으로 시간 재분배 — 긍정적 신호','열 제한에 도달'],a:2,b:'An',e:'메모리 대기(Long Scoreboard)가 줄고 연산(Math Pipe)이 늘었으므로 파이프라이닝이 효과적입니다.'},
  {q:'공유 메모리 2배 증가가 점유율에 미치는 영향은?',o:['점유율이 절반으로 감소','A100에서 17KB/164KB=10.4%로 병목 가능성 낮음','점유율에 영향 없음','점유율이 두 배로 증가'],a:1,b:'Ev',e:'17 KB는 A100 SM당 164 KB의 10.4%로, 레지스터가 주된 점유율 제한 요인이라면 영향이 적습니다.'},
  {q:'K%BK≠0일 때 에필로그에서 해야 할 처리는?',o:['나머지를 무시','메인 루프에서 완전 타일 처리 후 나머지를 별도 처리','K를 BK의 배수로 패딩','에러를 반환'],a:1,b:'Cr',e:'에필로그에서 나머지 K%BK개 열을 별도로 로드하고 연산하여 정확성을 보장합니다.'},
],
'09': [
  {q:'cp.async의 최소 Compute Capability는?',o:['7.0 (Volta)','7.5 (Turing)','8.0 (Ampere)','9.0 (Hopper)'],a:2,b:'R',e:'cp.async 명령어는 Compute Capability 8.0 (Ampere) 이상에서만 사용 가능합니다.'},
  {q:'SASS에서 cp.async가 변환되는 명령어 이름은?',o:['LDG.E.128','LDGSTS','STS.128','MEMBAR'],a:1,b:'R',e:'cp.async는 SASS에서 LDGSTS (Load Global Store Shared) 명령어로 변환됩니다.'},
  {q:'cp.async가 레지스터를 우회하는 이점 2가지 중 해당하지 않는 것은?',o:['레지스터 압력 감소','Long Scoreboard 스톨 제거','연산 처리량 증가','공유 메모리 용량 감소'],a:3,b:'U',e:'레지스터 우회의 이점은 (1) 레지스터 압력 감소→점유율 향상 (2) 메모리 의존성 제거→스톨 감소입니다.'},
  {q:'cp_async_wait_group(2) 호출 시 보장하는 것은?',o:['모든 그룹 완료','미완료 그룹이 2개 이하','정확히 2개 그룹 완료','2번째 그룹만 완료'],a:1,b:'Ap',e:'wait_group(N)은 미완료 그룹이 최대 N개가 될 때까지 대기합니다. 가장 오래된 그룹의 완료가 보장됩니다.'},
  {q:'NUM_STAGES=3, kt=5일 때 curr_stage는?',o:['0','1','2','5'],a:2,b:'Ap',e:'curr_stage = kt % NUM_STAGES = 5 % 3 = 2입니다.'},
  {q:'commit_group 없이 wait_group(2) 호출 시 문제는?',o:['컴파일 에러','그룹이 형성되지 않아 특정 타일 완료를 보장할 수 없음','성능만 약간 저하','아무 문제 없음'],a:1,b:'An',e:'commit_group 없이는 그룹 경계가 미정의되어 wait_group의 의미가 상실됩니다.'},
  {q:'트리플 버퍼링이 더블보다 레이턴시 은닉에 유리한 이유는?',o:['메모리가 더 커서','파이프라인 깊이가 2로 증가하여 더 긴 로드를 숨길 수 있음','연산이 빨라져서','동기화가 줄어서'],a:1,b:'An',e:'파이프라인 깊이가 1(더블)→2(트리플)로 증가. T_load > 2×T_compute가 아닌 한 유휴 시간이 없습니다.'},
  {q:'NUM_STAGES=4의 단점은?',o:['연산 속도 저하','공유 메모리 ~34.9KB 사용으로 점유율 감소 가능','코드 복잡도만 증가','단점 없음'],a:1,b:'Ev',e:'장점: 파이프라인 깊이 3. 단점: 공유 메모리 사용량 증가. 보통 NUM_STAGES=3이면 충분합니다.'},
  {q:'Ampere 미만 GPU를 위한 폴백 전략은?',o:['cp.async를 무시','#if __CUDA_ARCH__ >= 800 분기로 더블 버퍼링+LDG/STS 폴백','에러를 반환','cuBLAS를 호출'],a:1,b:'Cr',e:'조건부 컴파일로 Ampere 이상은 cp.async, 미만은 더블 버퍼링+일반 로드를 사용합니다.'},
  {q:'BK를 8→16으로 변경 시 공유 메모리 사용량은?',o:['약 17 KB','약 26 KB','약 35 KB','약 51 KB'],a:3,b:'Cr',e:'3 × (128×(16+1) + 16×(128+1)) × 4 ≈ 3 × (2176+2064) × 4 ≈ 50,880 바이트 ≈ 약 51 KB입니다.'},
],
'10': [
  {q:'SOL 분석의 두 가지 핵심 축은?',o:['점유율과 IPC','SM Throughput(%)과 Memory Throughput(%)','레이턴시와 대역폭','레지스터와 공유 메모리'],a:1,b:'R',e:'Speed Of Light 분석은 SM 처리량과 메모리 처리량의 두 축으로 성능 병목을 진단합니다.'},
  {q:'SM과 Memory Throughput이 모두 낮은 Latency Bound의 원인은?',o:['하드웨어 고장','요청→대기→연산이 직렬 실행되어 대역폭 미포화','캐시가 너무 작아서','블록 수가 부족해서'],a:1,b:'U',e:'레이턴시 바운드는 메모리 요청 후 대기하는 직렬 패턴으로, 대역폭도 연산 유닛도 충분히 활용되지 않습니다.'},
  {q:'피크 19,500 GFLOP/s, 2,039 GB/s에서 릿지 포인트 산술 강도는?',o:['약 4.78','약 9.56','약 19.12','약 38.24'],a:1,b:'Ap',e:'릿지 포인트 = 19,500 / 2,039 ≈ 9.56 FLOP/byte입니다.'},
  {q:'M=N=K=4096 SGEMM이 3.2ms에 완료 시 GFLOP/s는?',o:['약 21,475','약 42,950','약 85,900','약 137,440'],a:1,b:'Ap',e:'2×4096³ / 0.0032 = 137,438,953,472 / 0.0032 ≈ 42,950 GFLOP/s입니다.'},
  {q:'SASS에서 LDGSTS 대신 LDG+STS가 보이면?',o:['정상 동작','cp.async 미작동 — 레지스터 경유, 비동기 이점 상실','컴파일러 최적화의 결과','더 빠른 코드'],a:1,b:'An',e:'LDGSTS가 아닌 LDG+STS는 cp.async가 적용되지 않았음을 의미합니다.'},
  {q:'Long Scoreboard 60% + Wait 25%일 때 상태와 개선 방향은?',o:['정상 — 개선 불필요','합쳐 85% 유휴 — 더블 버퍼링/cp.async 필요','메모리 오류 — 디버깅 필요','연산 바운드 — 알고리즘 변경 필요'],a:1,b:'An',e:'85%가 유휴 상태이므로 심각한 레이턴시 바운드입니다. 프리페치, 더블 버퍼링, cp.async가 필요합니다.'},
  {q:'점유율 25% 레지스터 블로킹이 점유율 62% 타일링보다 3배 빠른 이유는?',o:['점유율이 높을수록 항상 좋은 것은 아니므로, 산술 강도 16배 + ILP','캐시 적중률이 더 높아서','메모리 대역폭을 더 쓰므로','워프 스케줄링이 더 효율적이므로'],a:0,b:'Ev',e:'산술 강도 16배 향상 + 128개 독립 FMA의 ILP + 메모리 트래픽 감소가 핵심입니다.'},
  {q:'cuBLAS 90% 이후 수익 체감이 심한 이유는?',o:['하드웨어 한계에 도달해서','암달의 법칙 + cuBLAS 비공개 최적화 + 트레이드오프 복잡화','최적화 기법이 더 없어서','메모리가 부족해서'],a:1,b:'Ev',e:'90% 이후에는 암달의 법칙, HW 한계, cuBLAS 비공개 최적화, 복잡한 트레이드오프로 수익 체감이 심합니다.'},
],
};

// --- Code Tracing Data ---
// Each: {code, steps:[{prompt, answer(number), hint}]}
const CODE_TRACES = {
'04': {
  title:'나이브 SGEMM — 스레드 인덱스 계산',
  code:`extern "C" __global__ void sgemm_naive(
    const float* A, const float* B, float* C,
    int M, int N, int K, int lda, int ldb, int ldc,
    float alpha, float beta)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k)
            sum += A[row * lda + k] * B[k * ldb + col];
        C[row * ldc + col] = alpha * sum + beta * C[row*ldc+col];
    }
}`,
  steps:[
    {prompt:'blockIdx.y=2, blockDim.y=16, threadIdx.y=5 일 때 row = ?',answer:37,hint:'row = blockIdx.y × blockDim.y + threadIdx.y'},
    {prompt:'blockIdx.x=1, blockDim.x=16, threadIdx.x=10 일 때 col = ?',answer:26,hint:'col = blockIdx.x × blockDim.x + threadIdx.x'},
    {prompt:'row=37, lda=64, k=3 일 때 A의 인덱스 (row*lda+k) = ?',answer:2371,hint:'37 × 64 + 3'},
    {prompt:'k=3, ldb=128, col=26 일 때 B의 인덱스 (k*ldb+col) = ?',answer:410,hint:'3 × 128 + 26'},
    {prompt:'M=4096, 블록=16×16일 때 y축 그리드 크기 ((M+15)/16) = ?',answer:256,hint:'(4096 + 15) / 16 = 256'},
  ]
},
'05': {
  title:'코얼레싱 SGEMM — 벡터화 루프',
  code:`#define FETCH_FLOAT4(p) (reinterpret_cast<const float4*>(&(p))[0])

int K_vec = K / 4;
float sum = 0.0f;
for (int k = 0; k < K_vec; ++k) {
    float4 a_vec = FETCH_FLOAT4(A[row * lda + k * 4]);
    float b0 = B[(k*4 + 0) * ldb + col];
    float b1 = B[(k*4 + 1) * ldb + col];
    sum += a_vec.x * b0 + a_vec.y * b1
         + a_vec.z * B[(k*4+2)*ldb+col]
         + a_vec.w * B[(k*4+3)*ldb+col];
}`,
  steps:[
    {prompt:'K=1024일 때 K_vec = K/4 = ?',answer:256,hint:'1024 / 4'},
    {prompt:'k=10, lda=1024일 때 A의 float4 시작 인덱스 (row=0 가정): k*4 = ?',answer:40,hint:'10 × 4 = 40'},
    {prompt:'k=5일 때 B 접근의 첫 번째 행 인덱스 (k*4+0) = ?',answer:20,hint:'5 × 4 + 0 = 20'},
    {prompt:'float4 로드 하나가 전송하는 바이트 수는?',answer:16,hint:'float 4개 × 4바이트 = 16바이트'},
  ]
},
'06': {
  title:'타일링 SGEMM — 뱅크 충돌 분석',
  code:`#define BLOCK_SIZE 16
#define SMEM_PADDING 1   // 뱅크 충돌 방지

__shared__ float A_tile[BLOCK_SIZE][BLOCK_SIZE + SMEM_PADDING]; // [16][17]

// 뱅크 = (바이트주소 / 4) % 32
// A_tile[ty][k]의 바이트 오프셋 = (ty * 17 + k) * 4
// 따라서 뱅크 = (ty * 17 + k) % 32`,
  steps:[
    {prompt:'패딩 없이 A_tile[16][16]일 때, A_tile[0][5]의 뱅크 = (0×16+5)%32 = ?',answer:5,hint:'(0 × 16 + 5) % 32'},
    {prompt:'패딩 없이 A_tile[2][5]의 뱅크 = (2×16+5)%32 = ?',answer:5,hint:'(32 + 5) % 32 = 37 % 32 = 5. 충돌!'},
    {prompt:'패딩 있음: A_tile[0][5]의 뱅크 = (0×17+5)%32 = ?',answer:5,hint:'(0 × 17 + 5) % 32'},
    {prompt:'패딩 있음: A_tile[2][5]의 뱅크 = (2×17+5)%32 = ?',answer:7,hint:'(34 + 5) % 32 = 39 % 32 = 7. 충돌 없음!'},
    {prompt:'K=48, BLOCK_SIZE=16일 때 타일 반복 횟수 ceil(48/16) = ?',answer:3,hint:'(48 + 15) / 16 = 3'},
  ]
},
'07': {
  title:'레지스터 블로킹 — 외적 연산',
  code:`#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8

float reg_C[TM][TN] = {{0.0f}};  // 64 accumulators
float reg_A[TM], reg_B[TN];       // 8 + 8 registers

// thread_row_start = ty * TM
// thread_col_start = tx * TN
// A_TILE_ROW_STRIDE = blockDim.x * blockDim.y / BK`,
  steps:[
    {prompt:'TM × TN = 한 스레드의 누적 레지스터 수 = ?',answer:64,hint:'8 × 8 = 64'},
    {prompt:'ty=5일 때 thread_row_start = ty × TM = ?',answer:40,hint:'5 × 8 = 40'},
    {prompt:'tx=3일 때 thread_col_start = tx × TN = ?',answer:24,hint:'3 × 8 = 24'},
    {prompt:'blockDim = 16×16 = 256. A_TILE_ROW_STRIDE = 256/BK = 256/8 = ?',answer:32,hint:'256 / 8'},
    {prompt:'한 BK 스텝에서의 총 FMA 수: TM × TN = ?',answer:64,hint:'8 × 8 = 64'},
    {prompt:'외적에서 reg_A 로드 + reg_B 로드 = 공유 메모리 접근 총 횟수 = ?',answer:16,hint:'TM + TN = 8 + 8 = 16'},
  ]
},
'08': {
  title:'더블 버퍼링 — 파이프라인 추적',
  code:`// Prologue: Load tile 0 into buffer[0]
load_tile(0, 0);
__syncthreads();

for (int kt = 0; kt < num_k_tiles; ++kt) {
    int curr_buf = kt % 2;
    int next_buf = (kt + 1) % 2;
    if (kt + 1 < num_k_tiles)
        load_tile(next_buf, kt + 1);  // prefetch
    compute_tile(curr_buf);            // compute
    __syncthreads();
}`,
  steps:[
    {prompt:'kt=0일 때 curr_buf = 0%2 = ?',answer:0,hint:'0 % 2 = 0'},
    {prompt:'kt=0일 때 next_buf = (0+1)%2 = ?',answer:1,hint:'1 % 2 = 1'},
    {prompt:'kt=3일 때 curr_buf = 3%2 = ?',answer:1,hint:'3 % 2 = 1'},
    {prompt:'K=32, BK=8일 때 num_k_tiles = K/BK = ?',answer:4,hint:'32 / 8 = 4'},
    {prompt:'T_load=400, T_comp=200, N=4일 때 싱글버퍼 총 사이클 = N×(T_load+T_comp) = ?',answer:2400,hint:'4 × (400+200) = 2400'},
  ]
},
'09': {
  title:'비동기 복사 — 트리플 버퍼 스테이지',
  code:`#define NUM_STAGES 3

for (int kt = 0; kt < num_k_tiles; ++kt) {
    int curr_stage = kt % NUM_STAGES;
    int prefetch_tile = kt + NUM_STAGES - 1;
    int prefetch_stage = prefetch_tile % NUM_STAGES;

    if (prefetch_tile < num_k_tiles)
        async_load(prefetch_stage, prefetch_tile);

    cp_async_wait_group(NUM_STAGES - 1);  // wait_group(2)
    __syncthreads();
    compute(curr_stage);
}`,
  steps:[
    {prompt:'kt=0일 때 curr_stage = 0%3 = ?',answer:0,hint:'0 % 3 = 0'},
    {prompt:'kt=4일 때 curr_stage = 4%3 = ?',answer:1,hint:'4 % 3 = 1'},
    {prompt:'kt=2일 때 prefetch_tile = 2+3-1 = ?',answer:4,hint:'2 + 2 = 4'},
    {prompt:'prefetch_tile=7일 때 prefetch_stage = 7%3 = ?',answer:1,hint:'7 % 3 = 1'},
    {prompt:'NUM_STAGES=3일 때 wait_group의 인수 = NUM_STAGES-1 = ?',answer:2,hint:'3 - 1 = 2'},
  ]
},
};

// --- Pre-Review Questions (Game A: Memory Challenge) ---
// Each module has 3 questions from previous modules
const REVIEWS = {
'04': [
  {q:'SGEMM 총 FLOP 수 공식은?',o:['M×N×K','2×M×N×K','M×N+K','2×(M+N)×K'],a:1},
  {q:'Row-major에서 A[i][j]의 1차원 인덱스는?',o:['i+j×lda','i×lda+j','j×lda+i','i×j+lda'],a:1},
  {q:'글로벌 메모리 접근 지연시간은 대략?',o:['1-5 사이클','20-30 사이클','400-800 사이클','10,000+ 사이클'],a:2},
],
'05': [
  {q:'워프(Warp)의 크기는?',o:['16 스레드','32 스레드','64 스레드','128 스레드'],a:1},
  {q:'나이브 커널이 느린 주된 이유가 아닌 것은?',o:['비코얼레싱','데이터 재사용 없음','블록 크기가 큼','낮은 산술 강도'],a:2},
  {q:'A100 글로벌 메모리 이론적 대역폭은?',o:['약 500 GB/s','약 1 TB/s','약 2 TB/s','약 4 TB/s'],a:2},
],
'06': [
  {q:'float4 벡터화 로드의 SASS 명령어는?',o:['LDG.E','LDG.E.128','STS.128','LDS.E'],a:1},
  {q:'공유 메모리 접근 지연시간은 대략?',o:['1 사이클','20-30 사이클','200 사이클','400 사이클'],a:1},
  {q:'나이브 커널의 산술 강도가 낮은 근본 원인은?',o:['레지스터 부족','데이터 재사용 없음','블록 크기 문제','코드 비효율'],a:1},
],
'07': [
  {q:'타일링에서 데이터 재사용 팩터는?',o:['2','BK','TILE_SIZE','TM×TN'],a:2},
  {q:'뱅크 충돌 방지 패딩 +1의 원리는?',o:['행 폭을 32와 서로소로 만듦','메모리 정렬','캐시 적중률 향상','레지스터 절약'],a:0},
  {q:'산술 강도의 단위는?',o:['GFLOP/s','FLOP/byte','byte/s','사이클'],a:1},
],
'08': [
  {q:'외적 마이크로커널에서 reg_A[m]의 재사용 횟수는?',o:['TM=8번','TN=8번','BK=8번','1번'],a:1},
  {q:'뱅크 충돌 방지 패딩 크기는?',o:['+0','+1','+2','+4'],a:1},
  {q:'레지스터 블로킹의 AI 공식에서 TM=TN=8일 때 AI는?',o:['0.25','1.0','2.0','4.0'],a:2},
],
'09': [
  {q:'더블 버퍼링 프롤로그의 역할은?',o:['루프 초기화','첫 타일을 미리 로드하여 파이프라인 채움','에러 체크','메모리 할당'],a:1},
  {q:'레지스터 블로킹 커널의 레지스터 사용량은 대략?',o:['약 30개','약 60개','약 100-128개','약 256개'],a:2},
  {q:'Long Scoreboard 스톨의 해결 방법이 아닌 것은?',o:['프리페치','점유율 높이기','cp.async','블록 크기 축소'],a:3},
],
};

// --- Boss Fight Data ---
const BOSSES = {
boss1: {
  name:'뱅크 충돌 수호자',module:'06',
  phases:[
    {title:'Phase 1: 충돌 식별',
     desc:'아래 공유 메모리 레이아웃에서 뱅크 충돌이 발생하는 경우를 찾으세요.\nfloat smem[8][16] (패딩 없음), 8개 스레드가 smem[tid][4]를 동시 접근',
     q:'tid=0과 tid=2가 접근하는 뱅크가 같은가? bank=(tid×16+4)%32',
     o:['다르다 (충돌 없음)','같다 (뱅크 충돌!)'],a:1,
     e:'bank(0)=(0×16+4)%32=4, bank(2)=(2×16+4)%32=36%32=4. 같은 뱅크 4에 매핑되어 충돌!'},
    {title:'Phase 2: 패딩 적용',
     desc:'뱅크 충돌을 해결하기 위한 패딩 값은?\nfloat smem[8][16+?]로 변경하여 충돌을 제거하려면?',
     q:'16+?의 값이 32와 서로소(coprime)가 되는 패딩 값은?',
     o:['+0 (16→32의 약수)','+ 1 (17→32와 서로소)','+ 2 (18→32와 서로소 아님)','+ 4 (20→32의 약수)'],a:1,
     e:'+1 패딩으로 폭이 17이 되면 32와 서로소(GCD(17,32)=1)이므로 모든 스레드가 다른 뱅크에 매핑됩니다.'},
    {title:'Phase 3: 검증',
     desc:'smem[8][17]에서 tid=0과 tid=2가 smem[tid][4] 접근 시',
     q:'두 뱅크 값은? bank(0)=(0×17+4)%32=?, bank(2)=(2×17+4)%32=?',
     o:['4, 4 (여전히 충돌)','4, 6 (충돌 해결!)','4, 38 (계산 불가)','17, 32 (충돌 해결!)'],a:1,
     e:'bank(0)=4, bank(2)=(34+4)%32=38%32=6. 서로 다른 뱅크! 충돌 해결 확인.'},
  ]
},
boss2: {
  name:'레이턴시 드래곤',module:'08',
  phases:[
    {title:'Phase 1: 스톨 구간 식별',
     desc:'싱글 버퍼 타임라인: [Load₀][Wait][Comp₀][Load₁][Wait][Comp₁]...\nT_load=100, T_comp=10, 타일 4개',
     q:'싱글 버퍼의 총 사이클 수는?',
     o:['400 사이클','410 사이클','440 사이클','480 사이클'],a:2,
     e:'4 × (100+10) = 440 사이클. 각 타일마다 로드 후 연산이 직렬 실행됩니다.'},
    {title:'Phase 2: 더블 버퍼 변환',
     desc:'더블 버퍼: 프롤로그에서 첫 타일 로드 → 메인 루프에서 로드와 연산 중첩',
     q:'더블 버퍼 총 사이클 수 = T_load + (N-1)×max(T_load, T_comp) + T_comp는?',
     o:['310 사이클','400 사이클','410 사이클','440 사이클'],a:2,
     e:'T_load + (N-1)×max(T_load,T_comp) + T_comp = 100 + 3×100 + 10 = 410 사이클입니다. 프롤로그(100) + 메인루프 3반복(300) + 에필로그 연산(10).'},
    {title:'Phase 3: Long Scoreboard 분석',
     desc:'프로파일에서 Long Scoreboard 60%, Math Pipe 10%가 보인다.',
     q:'이 커널의 상태와 해결 방법은?',
     o:['연산 바운드 — 알고리즘 개선 필요','레이턴시 바운드 — 더블 버퍼링으로 로드/연산 중첩 필요','메모리 바운드 — 캐시 최적화 필요','정상 — 개선 불필요'],a:1,
     e:'Long Scoreboard 60%는 메모리 로드 대기 상태. 더블 버퍼링으로 연산과 로드를 중첩하면 개선됩니다.'},
  ]
},
boss3: {
  name:'최종 벤치마크',module:'09',
  phases:[
    {title:'Phase 1: 프로파일 해석',
     desc:'커널 프로파일:\nSM Throughput: 78%, Memory Throughput: 45%\nLong Scoreboard: 8%, Math Pipe: 65%\n레지스터: 128개/스레드, 점유율: 25%',
     q:'이 커널의 가장 적절한 상태는?',
     o:['Latency Bound','Memory Bound','Compute Bound (연산 위주)','균형 (Balanced)'],a:2,
     e:'SM 78%로 높고 Memory 45%로 중간, Math Pipe가 주요 스톨. 연산 위주 상태입니다.'},
    {title:'Phase 2: 성능 갭 분석',
     desc:'Level 5 커널: 8,500 GFLOP/s (cuBLAS의 ~88%)\ncuBLAS: 9,650 GFLOP/s',
     q:'잔여 12% 갭의 주된 원인은?',
     o:['메모리 코얼레싱 부족','cuBLAS의 비공개 최적화(분할-K, 워프 특화 등)','레지스터 부족','CUDA 버전 차이'],a:1,
     e:'cuBLAS는 분할-K 전략, 워프 레벨 특화, 자동 튜닝 등 공개되지 않은 최적화를 사용합니다.'},
    {title:'Phase 3: 개선안 설계',
     desc:'추가 성능 향상을 위한 개선안을 선택하세요.',
     q:'가장 효과적인 다음 최적화는?',
     o:['블록 크기를 32×32로 변경','분할-K (Split-K) 전략으로 K 차원 병렬화','공유 메모리를 제거하고 레지스터만 사용','float16으로 정밀도 변경'],a:1,
     e:'분할-K는 K 차원을 여러 블록에 분배하여 병렬성을 높입니다. 큰 K에서 효과적입니다.'},
  ]
},
};

// --- Shop Items ---
const SHOP_FC = [
  {id:'hint',name:'힌트 토큰',desc:'퀴즈 1문항에 힌트 표시',price:50,max:null,icon:'💡'},
  {id:'sim_help',name:'시뮬레이션 도우미',desc:'코드 추적 시 단계별 힌트',price:75,max:null,icon:'🔍'},
  {id:'retry',name:'재시도 토큰',desc:'퀴즈 1회 재도전을 1차 시도로 리셋',price:100,max:10,icon:'🔄'},
  {id:'tooltip',name:'용어 툴팁 팩',desc:'학습 중 용어 마우스 오버 정의',price:30,max:1,icon:'📖'},
  {id:'freeze',name:'프리즈 토큰',desc:'연속 접속 기록 1일 보존',price:100,max:null,icon:'❄️'},
  {id:'frame1',name:'프로필 프레임: 실리콘',desc:'프로필 테두리 실리콘 웨이퍼 스킨',price:200,max:1,icon:'🖼️'},
  {id:'frame2',name:'프로필 프레임: 네온',desc:'프로필 테두리 네온 글로우 스킨',price:200,max:1,icon:'🌈'},
  {id:'title1',name:'칭호: 메모리 마스터',desc:'프로필에 "메모리 마스터" 칭호 표시',price:300,max:1,icon:'🏷️'},
  {id:'title2',name:'칭호: 파이프라인 장인',desc:'프로필에 "파이프라인 장인" 칭호 표시',price:300,max:1,icon:'🏷️'},
];
const SHOP_SC = [
  {id:'bonus_ch',name:'보너스 챌린지',desc:'추가 심화 문제 세트',price:5,max:10,icon:'🎯'},
  {id:'mentor',name:'멘토 모드',desc:'AI 대화형 설명 (미래 기능)',price:10,max:10,icon:'🧑‍🏫'},
  {id:'advanced',name:'심화 콘텐츠 팩',desc:'SASS 분석 심화 가이드',price:8,max:1,icon:'📚'},
  {id:'theme_premium',name:'프리미엄 포트폴리오 테마',desc:'PDF 인증서 프리미엄 디자인',price:15,max:3,icon:'✨'},
];

// --- Achievements ---
const ACHIEVEMENTS = [
  // Tier 1: Module completion (14)
  {id:'m00',name:'입문자의 첫걸음',tier:1,icon:'🚀',cond:s=>s.completed.includes('00')},
  {id:'m01',name:'하드웨어 탐험가',tier:1,icon:'🔌',cond:s=>s.completed.includes('01')},
  {id:'m02',name:'CUDA 드라이버',tier:1,icon:'💻',cond:s=>s.completed.includes('02')},
  {id:'m03',name:'행렬의 수학자',tier:1,icon:'📐',cond:s=>s.completed.includes('03')},
  {id:'m04',name:'첫 번째 커널',tier:1,icon:'⚡',cond:s=>s.completed.includes('04')},
  {id:'m05',name:'메모리 정렬자',tier:1,icon:'📏',cond:s=>s.completed.includes('05')},
  {id:'m06',name:'타일 건축가',tier:1,icon:'🧱',cond:s=>s.completed.includes('06')},
  {id:'m07',name:'레지스터 마에스트로',tier:1,icon:'🎹',cond:s=>s.completed.includes('07')},
  {id:'m08',name:'파이프라인 엔지니어',tier:1,icon:'🔀',cond:s=>s.completed.includes('08')},
  {id:'m09',name:'비동기 마스터',tier:1,icon:'⚡',cond:s=>s.completed.includes('09')},
  {id:'m10',name:'프로파일러',tier:1,icon:'📊',cond:s=>s.completed.includes('10')},
  // Tier 2: Mastery (3)
  {id:'boss1',name:'뱅크 충돌 슬레이어',tier:2,icon:'🛡️',sc:2,cond:s=>s.bossCleared.includes('boss1')},
  {id:'boss2',name:'레이턴시 슬레이어',tier:2,icon:'🐉',sc:2,cond:s=>s.bossCleared.includes('boss2')},
  {id:'boss3',name:'비동기 아키텍트',tier:2,icon:'🏛️',sc:3,cond:s=>s.bossCleared.includes('boss3')},
  // Tier 3: Skill (8)
  {id:'combo10',name:'Combo King',tier:3,icon:'🔥',sc:2,cond:s=>s.maxCombo>=10},
  {id:'gflops5k',name:'GFLOP/s Champion',tier:3,icon:'🏆',sc:3,cond:s=>s.gflops>=5000},
  {id:'fc5k',name:'Economy Wizard',tier:3,icon:'💰',sc:2,cond:s=>s.fcTotal>=5000},
  {id:'perfect3',name:'Diamond Mastery',tier:3,icon:'💎',sc:3,cond:s=>s.perfectModules>=3},
  {id:'full',name:'Full Architect',tier:3,icon:'👑',sc:3,cond:s=>s.completed.length>=11},
  // Tier 4: Hidden (3)
  {id:'humble',name:'The Humble Beginning',tier:4,icon:'🌱',sc:2,hidden:true,cond:s=>s.completed.includes('04')&&s.quizScores['04']>=80},
  {id:'100x',name:'100x',tier:4,icon:'💯',sc:2,hidden:true,cond:s=>s.completed.includes('09')&&s.quizScores['09']>=80},
  {id:'why_all',name:'Why Not How',tier:4,icon:'🤔',sc:3,hidden:true,cond:s=>Object.keys(s.codeTracesDone||{}).length>=5},
];

// --- Bloom Level Labels ---
const BLOOM_LABELS = {R:'기억',U:'이해',Ap:'적용',An:'분석',Ev:'평가',Cr:'창조'};
const BLOOM_ICONS = {R:'💭',U:'📖',Ap:'🔧',An:'🔍',Ev:'⚖️',Cr:'🏗️'};
