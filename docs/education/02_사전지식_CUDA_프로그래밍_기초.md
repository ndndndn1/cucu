# 모듈 2: CUDA 프로그래밍 기초 -- Driver API와 JIT 컴파일

> **소요 시간**: 90분
> **난이도**: 중급
> **선수 모듈**: 모듈 1 (GPU 아키텍처 기초)
> **코드 참조**: `include/cuda_driver_wrapper.hpp`, `src/driver/cuda_driver.cpp`, `src/driver/kernel_launcher.cpp`

---

## 1. 학습 목표

이 모듈을 완료하면 다음을 수행할 수 있다:

1. **Runtime API와 Driver API의 차이**를 설명할 수 있다 -- 각각 어떤 상황에서 적합한지, 추상화 수준의 차이가 무엇인지 이해한다.
2. **CUDA 컨텍스트의 역할과 생명주기**를 이해한다 -- 컨텍스트가 GPU 리소스의 소유권 경계임을 설명할 수 있다.
3. **PTX 중간 표현과 JIT 컴파일 과정**을 설명할 수 있다 -- CUDA 소스에서 PTX, PTX에서 SASS로의 변환 단계를 추적할 수 있다.
4. **CUDA Driver API로 메모리 할당, 전송, 커널 실행을 수행**할 수 있다 -- 프로젝트의 래퍼 코드를 읽고 각 Driver API 호출의 의미를 파악할 수 있다.

---

## 2. 사전 복습 (Retrieval Practice)

모듈 1에서 배운 내용을 떠올려 보자. 아래 질문에 답한 뒤 다음으로 넘어간다.

**질문 1**: GPU 메모리 계층에서 가장 빠른 메모리와 가장 느린 메모리는 각각 무엇인가?

<details>
<summary>정답 확인</summary>

- **가장 빠른 메모리**: 레지스터 (Register) -- 접근 지연시간 약 1 사이클, 스레드 전용
- **가장 느린 메모리**: 글로벌 메모리 (Global Memory / DRAM) -- 접근 지연시간 약 400~600 사이클

그 사이에 공유 메모리(약 20~30 사이클)와 L1/L2 캐시가 위치한다.
</details>

**질문 2**: 워프(Warp)란 무엇이며 크기는 얼마인가?

<details>
<summary>정답 확인</summary>

워프는 GPU에서 동시에 같은 명령어를 실행하는 **32개 스레드의 그룹**이다. SIMT(Single Instruction Multiple Thread) 실행 모델의 기본 스케줄링 단위이며, 워프 내 모든 스레드는 같은 프로그램 카운터를 공유한다.
</details>

**질문 3**: SM(Streaming Multiprocessor)이 하는 역할은 무엇인가?

<details>
<summary>정답 확인</summary>

SM은 GPU의 핵심 연산 유닛으로, 여러 워프를 병렬 실행한다. SM은 자체 레지스터 파일, 공유 메모리, L1 캐시, 워프 스케줄러를 포함하며, 하나 이상의 스레드 블록이 SM에 배정되어 실행된다. 우리 코드에서 SM 수는 `DeviceInfo::sm_count`로 조회할 수 있다 (`include/cuda_driver_wrapper.hpp:60`).
</details>

---

## 3. 개념 설명

### 3.1 Runtime API vs Driver API

**핵심 비유(CRA)**: Runtime API는 **자동변속기**, Driver API는 **수동변속기**이다.

자동변속기(Runtime API)는 운전자가 기어를 직접 바꿀 필요 없이 편리하게 주행할 수 있다. 수동변속기(Driver API)는 기어를 직접 조작해야 하지만, 엔진 회전수와 변속 타이밍을 세밀하게 제어할 수 있어 레이싱과 같은 극한 상황에서 더 높은 성능을 끌어낼 수 있다.

우리 프로젝트 BareMetal-SGEMM은 Driver API를 선택했다. 이유는 PTX JIT 컴파일, 컨텍스트 관리, 모듈 로딩 등을 **명시적으로 제어**해야 최적화 과정을 투명하게 이해할 수 있기 때문이다.

#### 비교표

| 항목 | Runtime API | Driver API |
|------|------------|------------|
| **헤더** | `<cuda_runtime.h>` | `<cuda.h>` |
| **컨텍스트** | 암시적 (자동 생성) | 명시적 (`cuCtxCreate`) |
| **커널 실행** | `<<<grid, block>>>` 구문 | `cuLaunchKernel` 함수 호출 |
| **모듈 로딩** | 자동 (링크 시 포함) | `cuModuleLoadDataEx` 명시 호출 |
| **파라미터 전달** | 자동 패킹 | `void**` 배열로 수동 패킹 |
| **초기화** | 암시적 (첫 API 호출 시) | `cuInit(0)` 명시 호출 |
| **에러 타입** | `cudaError_t` | `CUresult` |
| **함수 접두사** | `cuda*` | `cu*` |

#### 프로젝트 코드: 전체 Driver API 래퍼

우리 프로젝트의 Driver API 래퍼는 `include/cuda_driver_wrapper.hpp`에 정의되어 있다. 이 파일의 구조를 살펴보자:

```cpp
// include/cuda_driver_wrapper.hpp (전체 구조)

#include <cuda.h>        // Driver API -- cudart가 아닌 cuda.h
#include <nvrtc.h>       // Runtime Compilation

namespace baremetal {

// 에러 검사 매크로
#define CHECK_CU(call)                                                         \
    do {                                                                       \
        CUresult err = (call);                                                 \
        if (err != CUDA_SUCCESS) {                                             \
            const char* errStr;                                                \
            cuGetErrorString(err, &errStr);                                    \
            throw std::runtime_error(std::string("CUDA Driver Error: ") +      \
                                   errStr + " at " + __FILE__ + ":" +          \
                                   std::to_string(__LINE__));                  \
        }                                                                      \
    } while (0)

// 주요 클래스 (각각 아래 절에서 상세히 다룬다)
struct DeviceInfo { ... };        // GPU 정보 조회
class CudaContext { ... };        // 컨텍스트 관리 (RAII)
struct JitOptions { ... };        // JIT 컴파일 옵션
class CudaModule { ... };         // PTX/CUBIN 모듈 로더
class DeviceMemory { ... };       // 디바이스 메모리 (RAII)
class PinnedMemory { ... };       // 고정 호스트 메모리 (RAII)
class CudaStream { ... };         // 스트림 (RAII)
class CudaEvent { ... };          // 이벤트 -- 타이밍 (RAII)
struct LaunchConfig { ... };      // 커널 실행 구성
class KernelLauncher { ... };     // 커널 실행기

}  // namespace baremetal
```

핵심 관찰: 모든 클래스가 **RAII 패턴**을 따른다. 생성자에서 리소스를 획득하고 소멸자에서 자동 해제한다. 이것이 Driver API의 복잡성을 관리하는 열쇠이다.

---

### 3.2 컨텍스트 관리

CUDA 컨텍스트는 **GPU와의 연결**을 나타낸다. CPU의 프로세스가 운영체제 리소스를 소유하듯이, 컨텍스트는 GPU의 메모리 할당, 모듈, 스트림 등 모든 리소스를 소유한다.

`src/driver/cuda_driver.cpp`의 `CudaContext` 생성자를 한 줄씩 따라가 보자 (라인 54-83):

```cpp
// src/driver/cuda_driver.cpp:54-83

CudaContext::CudaContext(int device_id, unsigned int flags) {
    // [1단계] 드라이버 초기화
    // 모든 Driver API 호출 전에 반드시 한 번 호출해야 한다.
    // Runtime API에서는 이 과정이 첫 cuda* 호출 시 자동으로 일어난다.
    if (!s_cuda_initialized) {
        CHECK_CU(cuInit(0));       // 0 = 플래그 (현재 0만 유효)
        s_cuda_initialized = true;
    }

    // [2단계] GPU 열거
    // 시스템에 몇 개의 GPU가 있는지 확인한다.
    int device_count = 0;
    CHECK_CU(cuDeviceGetCount(&device_count));
    if (device_count == 0) {
        throw std::runtime_error("No CUDA-capable devices found");
    }
    if (device_id >= device_count) {
        throw std::runtime_error("Invalid device ID: " + std::to_string(device_id) +
                                " (only " + std::to_string(device_count) + " devices available)");
    }

    // [3단계] 장치 핸들 획득
    // device_id(정수)를 CUdevice 핸들로 변환한다.
    CHECK_CU(cuDeviceGet(&device_, device_id));

    // [4단계] 컨텍스트 생성
    // 이 시점에서 GPU와의 연결이 수립된다.
    // CU_CTX_SCHED_AUTO: 스케줄링 정책을 드라이버가 자동 결정
    CHECK_CU(cuCtxCreate(&context_, flags, device_));

    // [5단계] 장치 속성 조회
    device_info_.device_id = device_id;
    query_device_info();
}
```

#### RAII 패턴: 소멸자에서의 자동 정리

```cpp
// src/driver/cuda_driver.cpp:85-91

CudaContext::~CudaContext() {
    if (owns_context_ && context_ != nullptr) {
        // 컨텍스트를 파괴하면 이 컨텍스트에 속한 모든 리소스
        // (메모리, 모듈, 스트림 등)도 함께 해제된다.
        cuCtxDestroy(context_);
        context_ = nullptr;
    }
}
```

이 패턴의 장점은 명확하다: 예외가 발생하더라도 소멸자가 호출되므로 **GPU 리소스 누수가 원천적으로 방지**된다.

#### 컨텍스트 생명주기 다이어그램

```
cuInit(0)
    |
    v
cuDeviceGetCount(&count)
    |
    v
cuDeviceGet(&device, 0)
    |
    v
cuCtxCreate(&ctx, flags, device)    <-- 컨텍스트 활성화
    |                                    이 시점부터 GPU 리소스 사용 가능
    v
[메모리 할당, 모듈 로딩, 커널 실행, ...]
    |
    v
cuCtxDestroy(ctx)                   <-- 모든 리소스 자동 해제
```

---

### 3.3 PTX와 JIT 컴파일

#### PTX란 무엇인가

PTX(Parallel Thread Execution)는 GPU의 **"어셈블리"**에 해당하는 중간 표현(Intermediate Representation)이다. CPU의 x86 어셈블리와 유사하지만, 특정 GPU 아키텍처에 종속되지 않는 가상 명령어 집합이다.

컴파일 파이프라인은 다음과 같다:

```
CUDA 소스 (.cu)
    |  nvcc -ptx (빌드 시점, AOT)
    v
PTX 코드 (.ptx)              <-- 아키텍처 독립적 중간 표현
    |  cuModuleLoadDataEx (실행 시점, JIT)
    v
SASS 기계어                   <-- 특정 GPU 아키텍처의 기계어
    |
    v
GPU 실행
```

#### 빌드 시스템: PTX 생성 (CMakeLists.txt:54-66)

프로젝트의 CMakeLists.txt에서 PTX 파일을 생성하는 과정을 살펴보자:

```cmake
# CMakeLists.txt:50-68 (PTX 커널 생성)

foreach(KERNEL_SRC ${KERNEL_SOURCES})
    get_filename_component(KERNEL_NAME ${KERNEL_SRC} NAME_WE)
    set(PTX_OUTPUT ${CMAKE_SOURCE_DIR}/ptx/${KERNEL_NAME}.ptx)

    add_custom_command(
        OUTPUT ${PTX_OUTPUT}
        COMMAND ${CMAKE_CUDA_COMPILER}
                -ptx                              # PTX 출력 지정
                -o ${PTX_OUTPUT}
                -arch=sm_80                       # 대상 아키텍처 (가상)
                -lineinfo                         # 프로파일링용 라인 정보
                --use_fast_math                   # 빠른 수학 함수
                -I${CMAKE_SOURCE_DIR}/include
                ${CMAKE_SOURCE_DIR}/${KERNEL_SRC}
        DEPENDS ${CMAKE_SOURCE_DIR}/${KERNEL_SRC}
        COMMENT "Generating PTX for ${KERNEL_NAME}"
    )
    list(APPEND PTX_OUTPUTS ${PTX_OUTPUT})
endforeach()
```

핵심: `nvcc -ptx` 명령이 각 `.cu` 커널 파일을 `.ptx` 파일로 변환한다. 이 PTX 파일은 빌드 결과물로 `ptx/` 디렉터리에 저장된다.

#### JIT 컴파일: PTX에서 SASS로 (src/driver/kernel_launcher.cpp:43-68)

실행 시점에 PTX를 로드하고 JIT 컴파일하는 과정은 `SgemmLauncher::load_kernel()` 함수에 구현되어 있다:

```cpp
// src/driver/kernel_launcher.cpp:43-68

void SgemmLauncher::load_kernel(
    const std::string& ptx_dir,
    const JitOptions& jit_opts) {

    // [1] PTX 파일 경로 구성
    std::string ptx_file = ptx_dir + "/" + opt_level_ptx_file(level_);

    // [2] PTX 파일 읽기
    std::ifstream file(ptx_file);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open PTX file: " + ptx_file);
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    ptx_source_ = buffer.str();

    // [3] PTX를 JIT 컴파일하여 모듈 생성
    //     내부에서 cuModuleLoadDataEx가 호출된다
    module_ = CudaModule::from_ptx(ptx_source_, jit_opts);

    // [4] 모듈에서 커널 함수 핸들 획득
    kernel_func_ = module_->get_function(opt_level_kernel_name(level_));

    std::cout << "[SgemmLauncher] Loaded kernel: " << opt_level_name(level_)
              << " from " << ptx_file << std::endl;
}
```

`CudaModule::from_ptx()` 내부에서 실제 JIT 컴파일이 일어난다 (`src/driver/cuda_driver.cpp:217-248`):

```cpp
// src/driver/cuda_driver.cpp:217-248

std::unique_ptr<CudaModule> CudaModule::from_ptx(
    const std::string& ptx_source,
    const JitOptions& options) {

    auto module = std::unique_ptr<CudaModule>(new CudaModule());

    // JIT 컴파일 옵션 준비
    auto cu_options = options.to_cu_options();
    auto cu_values = options.to_cu_values();

    // 핵심: cuModuleLoadDataEx로 PTX를 SASS(기계어)로 JIT 컴파일
    CUresult result = cuModuleLoadDataEx(
        &module->module_,                              // 출력: 모듈 핸들
        ptx_source.c_str(),                            // 입력: PTX 소스 문자열
        static_cast<unsigned int>(cu_options.size()),  // 옵션 개수
        cu_options.data(),                             // 옵션 키 배열
        cu_values.data());                             // 옵션 값 배열

    if (result != CUDA_SUCCESS) {
        // JIT 컴파일 실패 시 에러 로그 추출
        char* error_log = static_cast<char*>(cu_values[3]);
        std::string error_msg = "PTX JIT compilation failed: "
                                + std::string(error_log);
        throw std::runtime_error(error_msg);
    }

    // 정보 로그 저장 (디버깅용)
    char* info_log = static_cast<char*>(cu_values[1]);
    module->jit_log_ = info_log;

    return module;
}
```

#### JIT 컴파일의 이점

왜 빌드 시점에 바로 SASS로 컴파일하지 않고 PTX를 거치는가?

1. **아키텍처 이식성**: PTX는 가상 ISA이므로 sm_80(Ampere)에서 빌드한 PTX를 sm_90(Hopper)에서도 JIT 컴파일하여 실행할 수 있다.
2. **최적화 기회**: JIT 컴파일러가 실행 시점의 정확한 GPU 모델을 알기 때문에, 해당 모델에 특화된 최적화를 적용할 수 있다.
3. **배포 편의성**: 하나의 PTX 바이너리로 여러 세대의 GPU를 지원할 수 있다.

---

### 3.4 메모리 관리

Driver API에서 GPU 메모리 관리는 `cuMemAlloc`/`cuMemFree`로 수행한다. 우리 프로젝트는 이를 RAII로 감싼 `DeviceMemory` 클래스를 제공한다.

#### DeviceMemory 클래스 (include/cuda_driver_wrapper.hpp:196-221)

```cpp
// include/cuda_driver_wrapper.hpp:196-221

class DeviceMemory {
public:
    DeviceMemory() = default;
    explicit DeviceMemory(size_t size);   // cuMemAlloc 호출
    ~DeviceMemory();                      // cuMemFree 호출

    // 복사 금지, 이동만 허용 (소유권 이전)
    DeviceMemory(const DeviceMemory&) = delete;
    DeviceMemory& operator=(const DeviceMemory&) = delete;
    DeviceMemory(DeviceMemory&& other) noexcept;
    DeviceMemory& operator=(DeviceMemory&& other) noexcept;

    CUdeviceptr get() const { return ptr_; }       // 디바이스 포인터 반환
    CUdeviceptr* ptr_to_ptr() { return &ptr_; }    // 포인터의 포인터
    size_t size() const { return size_; }

    void copy_from_host(const void* host_ptr, size_t bytes);     // cuMemcpyHtoD
    void copy_to_host(void* host_ptr, size_t bytes) const;       // cuMemcpyDtoH
    void copy_from_host_async(const void* host_ptr, size_t bytes,
                              CUstream stream);                   // 비동기 HtoD
    void copy_to_host_async(void* host_ptr, size_t bytes,
                            CUstream stream) const;               // 비동기 DtoH
    void memset(unsigned char value, size_t bytes);               // cuMemsetD8
    void memset_async(unsigned char value, size_t bytes,
                      CUstream stream);

private:
    CUdeviceptr ptr_ = 0;
    size_t size_ = 0;
};
```

#### 구현 상세 (src/driver/cuda_driver.cpp:374-428)

```cpp
// src/driver/cuda_driver.cpp:374-378
DeviceMemory::DeviceMemory(size_t size) : size_(size) {
    if (size > 0) {
        CHECK_CU(cuMemAlloc(&ptr_, size));    // GPU 메모리 할당
    }
}

// src/driver/cuda_driver.cpp:380-385
DeviceMemory::~DeviceMemory() {
    if (ptr_ != 0) {
        cuMemFree(ptr_);   // GPU 메모리 해제 (CHECK_CU 없음 -- 소멸자에서 예외 방지)
        ptr_ = 0;
    }
}

// src/driver/cuda_driver.cpp:406-408
void DeviceMemory::copy_from_host(const void* host_ptr, size_t bytes) {
    CHECK_CU(cuMemcpyHtoD(ptr_, host_ptr, bytes));   // Host -> Device 복사
}

// src/driver/cuda_driver.cpp:410-412
void DeviceMemory::copy_to_host(void* host_ptr, size_t bytes) const {
    CHECK_CU(cuMemcpyDtoH(host_ptr, ptr_, bytes));   // Device -> Host 복사
}
```

#### 정렬(Alignment) 상수

`src/driver/memory_manager.cpp`에서 정의된 정렬 상수에 주목하자:

```cpp
// src/driver/memory_manager.cpp:21-22
constexpr size_t ALIGNMENT_BYTES = 256;  // 최적 코얼레싱을 위한 정렬
constexpr size_t CACHE_LINE_SIZE = 128;  // L2 캐시 라인 크기
```

GPU 메모리 접근이 256바이트 경계에 정렬되어 있으면 메모리 트랜잭션 효율이 최대화된다. `AlignedMatrix` 클래스는 행렬의 리딩 디멘션을 이 경계에 맞추어 할당한다.

---

### 3.5 커널 실행

Driver API에서 커널 실행은 `cuLaunchKernel` 함수를 통해 이루어진다. Runtime API의 `<<<grid, block>>>` 구문과 달리, 모든 파라미터를 `void**` 배열로 명시적으로 패킹해야 한다.

#### KernelLauncher::launch() (include/cuda_driver_wrapper.hpp:326-341)

```cpp
// include/cuda_driver_wrapper.hpp:312-341

class KernelLauncher {
public:
    // 방법 1: void** 배열을 직접 전달
    static void launch(
        CUfunction func,
        const LaunchConfig& config,
        void** kernel_params);

    // 방법 2: 가변 인자 템플릿으로 편리하게 사용
    template<typename... Args>
    static void launch(CUfunction func, const LaunchConfig& config,
                       Args&&... args) {
        void* params[] = {const_cast<void*>(
            static_cast<const void*>(&args))...};
        launch(func, config, params);
    }
};
```

내부 구현 (`src/driver/cuda_driver.cpp:556-570`):

```cpp
void KernelLauncher::launch(
    CUfunction func,
    const LaunchConfig& config,
    void** kernel_params) {

    CHECK_CU(cuLaunchKernel(
        func,                                    // 커널 함수 핸들
        config.grid.x, config.grid.y, config.grid.z,    // 그리드 크기
        config.block.x, config.block.y, config.block.z,  // 블록 크기
        static_cast<unsigned int>(config.shared_mem_bytes), // 동적 공유 메모리
        config.stream,                           // 실행 스트림
        kernel_params,                           // 커널 파라미터 배열
        nullptr                                  // 추가 옵션 (미사용)
    ));
}
```

#### 파라미터 패킹: SGEMM 커널 실행 (src/driver/kernel_launcher.cpp:151-174)

```cpp
// src/driver/kernel_launcher.cpp:151-174

void SgemmLauncher::execute(const SgemmParams& params, CUstream stream) {
    auto config = get_launch_config(params.M, params.N, params.K);
    config.stream = stream;

    // 파라미터 패킹: 커널 시그니처의 각 인자에 대한 포인터 배열
    // 커널 시그니처:
    //   void sgemm_xxx(float* A, float* B, float* C,
    //                  int M, int N, int K,
    //                  int lda, int ldb, int ldc,
    //                  float alpha, float beta)
    void* kernel_args[] = {
        const_cast<CUdeviceptr*>(&params.A),     // float* A
        const_cast<CUdeviceptr*>(&params.B),     // float* B
        const_cast<CUdeviceptr*>(&params.C),     // float* C
        const_cast<int*>(&params.M),             // int M
        const_cast<int*>(&params.N),             // int N
        const_cast<int*>(&params.K),             // int K
        const_cast<int*>(&params.lda),           // int lda
        const_cast<int*>(&params.ldb),           // int ldb
        const_cast<int*>(&params.ldc),           // int ldc
        const_cast<float*>(&params.alpha),       // float alpha
        const_cast<float*>(&params.beta)         // float beta
    };

    KernelLauncher::launch(kernel_func_, config, kernel_args);
}
```

**주의**: `void*` 배열의 각 원소는 실제 값이 아니라 **값의 주소**를 가리켜야 한다. 예를 들어 `int M = 4096`이면 배열에는 `&M`이 들어간다. 이 레이아웃이 틀리면 커널이 잘못된 값을 읽어 정확성 오류나 세그멘테이션 폴트가 발생한다.

---

### 3.6 이벤트 타이밍

GPU 커널의 실행 시간을 정확히 측정하려면 GPU 이벤트를 사용해야 한다. CPU 타이머(예: `std::chrono`)는 커널 실행이 비동기적이므로 정확하지 않다.

#### CudaEvent 클래스 (include/cuda_driver_wrapper.hpp:271-290)

```cpp
// include/cuda_driver_wrapper.hpp:271-290

class CudaEvent {
public:
    CudaEvent(unsigned int flags = CU_EVENT_DEFAULT);
    ~CudaEvent();

    // 복사 금지, 이동만 허용
    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;
    CudaEvent(CudaEvent&& other) noexcept;
    CudaEvent& operator=(CudaEvent&& other) noexcept;

    CUevent get() const { return event_; }
    void record(CUstream stream = nullptr);    // 스트림에 타임스탬프 기록
    void synchronize();                         // 이벤트 완료까지 대기

    // 두 이벤트 사이의 경과 시간(밀리초) 반환
    static float elapsed_ms(const CudaEvent& start, const CudaEvent& end);

private:
    CUevent event_ = nullptr;
};
```

#### 타이밍 측정 패턴

```cpp
// 사용 예시 (개념 코드)
CudaEvent start, end;

start.record(stream);           // cuEventRecord: 시작 타임스탬프
KernelLauncher::launch(func, config, args);   // 커널 실행
end.record(stream);             // cuEventRecord: 종료 타임스탬프

end.synchronize();              // cuEventSynchronize: GPU 작업 완료 대기

float ms = CudaEvent::elapsed_ms(start, end);  // cuEventElapsedTime
std::cout << "커널 실행 시간: " << ms << " ms" << std::endl;
```

내부 구현 (`src/driver/cuda_driver.cpp:539-551`):

```cpp
void CudaEvent::record(CUstream stream) {
    CHECK_CU(cuEventRecord(event_, stream));
}

void CudaEvent::synchronize() {
    CHECK_CU(cuEventSynchronize(event_));
}

float CudaEvent::elapsed_ms(const CudaEvent& start, const CudaEvent& end) {
    float ms;
    CHECK_CU(cuEventElapsedTime(&ms, start.event_, end.event_));
    return ms;
}
```

---

## 4. 코드 분석: 커널 실행의 전체 흐름

이제 프로젝트에서 SGEMM 커널이 로드되고 실행되기까지의 전체 시퀀스를 추적해 보자.

### 4.1 1단계: 커널 로딩 (`SgemmLauncher` 생성자)

```
SgemmLauncher 생성자
    |
    v
load_kernel(ptx_dir, jit_opts)
    |
    +---> [1] PTX 파일 경로 구성
    |         ptx_dir + "/" + opt_level_ptx_file(level_)
    |         예: "./ptx/sgemm_naive.ptx"
    |
    +---> [2] PTX 파일을 문자열로 읽기
    |         std::ifstream -> std::stringstream -> ptx_source_
    |
    +---> [3] CudaModule::from_ptx(ptx_source_, jit_opts)
    |         |
    |         +---> JitOptions -> CUjit_option/void* 배열 변환
    |         |     (최적화 레벨, 로그 버퍼, 레지스터 제한 등)
    |         |
    |         +---> cuModuleLoadDataEx(&module, ptx_str, ...)
    |         |     PTX -> SASS JIT 컴파일 실행
    |         |
    |         +---> 실패 시: 에러 로그(cu_values[3])에서 원인 추출
    |         |     성공 시: 정보 로그(cu_values[1]) 저장
    |         |
    |         +---> return unique_ptr<CudaModule>
    |
    +---> [4] module_->get_function(kernel_name)
              |
              +---> cuModuleGetFunction(&func, module, "sgemm_naive")
              +---> 함수 캐시에 저장 (재호출 시 캐시 히트)
              +---> return CUfunction
```

### 4.2 2단계: 커널 실행 (`SgemmLauncher::execute()`)

```
execute(params, stream)
    |
    +---> [1] get_launch_config(M, N, K)
    |         |
    |         +---> 최적화 레벨에 따라 grid, block, smem_size 결정
    |               Naive: block(16,16), grid(ceil(N/16), ceil(M/16)), smem=0
    |               RegBlocking: block(16,16), grid(ceil(N/128), ceil(M/128)),
    |                            smem=(128*8+8*128)*4 bytes
    |
    +---> [2] 파라미터 패킹
    |         void* kernel_args[] = {
    |             &params.A, &params.B, &params.C,
    |             &params.M, &params.N, &params.K,
    |             &params.lda, &params.ldb, &params.ldc,
    |             &params.alpha, &params.beta
    |         };
    |
    +---> [3] KernelLauncher::launch(kernel_func_, config, kernel_args)
              |
              +---> cuLaunchKernel(func,
                        grid.x, grid.y, grid.z,
                        block.x, block.y, block.z,
                        shared_mem_bytes,
                        stream,
                        kernel_args,
                        nullptr)
```

### 4.3 코드 흐름 요약

| 단계 | 함수 | Driver API 호출 | 파일:라인 |
|------|------|-----------------|----------|
| 드라이버 초기화 | `CudaContext()` | `cuInit(0)` | `cuda_driver.cpp:58` |
| 장치 선택 | `CudaContext()` | `cuDeviceGet` | `cuda_driver.cpp:74` |
| 컨텍스트 생성 | `CudaContext()` | `cuCtxCreate` | `cuda_driver.cpp:78` |
| PTX 로드+JIT | `CudaModule::from_ptx()` | `cuModuleLoadDataEx` | `cuda_driver.cpp:229-234` |
| 함수 핸들 획득 | `CudaModule::get_function()` | `cuModuleGetFunction` | `cuda_driver.cpp:350` |
| 메모리 할당 | `DeviceMemory()` | `cuMemAlloc` | `cuda_driver.cpp:376` |
| 데이터 전송 (H->D) | `copy_from_host()` | `cuMemcpyHtoD` | `cuda_driver.cpp:407` |
| 커널 실행 | `KernelLauncher::launch()` | `cuLaunchKernel` | `cuda_driver.cpp:561-569` |
| 데이터 전송 (D->H) | `copy_to_host()` | `cuMemcpyDtoH` | `cuda_driver.cpp:411` |
| 메모리 해제 | `~DeviceMemory()` | `cuMemFree` | `cuda_driver.cpp:382` |
| 컨텍스트 파괴 | `~CudaContext()` | `cuCtxDestroy` | `cuda_driver.cpp:88` |

---

## 5. 왜 이것이 작동하는가? (Why Does This Work?)

아래 질문에 대해 스스로 생각해 본 뒤 답을 확인하라.

### 질문 1: Driver API가 Runtime API보다 유리한 구체적 상황은?

<details>
<summary>답변 확인</summary>

1. **런타임 PTX/CUBIN 로딩**: 커널을 파일로부터 동적으로 로드해야 할 때. Runtime API는 커널이 빌드 시 링크되어야 하지만, Driver API는 `cuModuleLoadDataEx`로 실행 시점에 아무 PTX/CUBIN이나 로드할 수 있다.

2. **다중 컨텍스트 관리**: 여러 GPU를 세밀하게 제어해야 할 때. Runtime API는 컨텍스트를 암시적으로 관리하므로, 스레드별 컨텍스트 바인딩을 직접 제어하기 어렵다.

3. **JIT 컴파일 옵션 제어**: 커널별로 최대 레지스터 수(`CU_JIT_MAX_REGISTERS`)나 최적화 수준을 다르게 지정해야 할 때.

4. **우리 프로젝트의 경우**: 6개의 서로 다른 최적화 레벨의 PTX 파일을 런타임에 선택적으로 로드해야 하므로 Driver API가 적합하다.
</details>

### 질문 2: JIT 컴파일이 AOT(Ahead-of-Time) 컴파일보다 유리한 경우는?

<details>
<summary>답변 확인</summary>

1. **아키텍처 이식성**: PTX는 가상 ISA이므로 빌드 시 존재하지 않았던 미래의 GPU 아키텍처에서도 JIT 컴파일을 통해 실행 가능하다. AOT로 sm_80용 SASS를 생성하면 sm_90에서는 호환 모드로 실행되거나 아예 실행되지 않는다.

2. **배포 크기 최소화**: 여러 아키텍처를 지원하려면 AOT는 각 아키텍처별 SASS를 모두 포함해야 한다(fat binary). JIT는 하나의 PTX만 배포하면 된다.

3. **런타임 최적화**: JIT 컴파일러는 실행 시점의 정확한 GPU 모델과 드라이버 버전을 알기 때문에, 더 공격적인 최적화를 적용할 수 있다.

4. **단점**: 첫 실행 시 JIT 컴파일 오버헤드가 발생한다. 이는 컴파일 캐시(`~/.nv/ComputeCache`)로 완화된다.
</details>

### 질문 3: RAII 패턴이 CUDA 리소스 관리에서 특히 중요한 이유는?

<details>
<summary>답변 확인</summary>

1. **GPU 리소스 누수 방지**: GPU 메모리는 CPU 메모리보다 훨씬 제한적이다 (보통 8~48 GB). `cuMemAlloc` 후 `cuMemFree`를 잊으면 금방 메모리 부족이 발생한다.

2. **예외 안전성**: CUDA Driver API 호출이 실패하면 `CHECK_CU` 매크로가 예외를 던진다. RAII가 없으면 이전에 할당한 리소스를 수동으로 정리하는 코드가 복잡해진다. RAII는 스택 해제(stack unwinding) 과정에서 소멸자가 자동 호출되므로 이 문제를 해결한다.

3. **소유권 명확화**: `DeviceMemory`의 복사 금지 + 이동 허용 정책은 GPU 메모리의 소유권이 항상 하나의 객체에만 있음을 컴파일 타임에 보장한다.

4. **의존성 순서**: 컨텍스트가 파괴되면 그에 속한 모든 리소스도 무효화된다. RAII의 역순 소멸 보장(스택에서 마지막에 생성된 것이 먼저 파괴됨)이 올바른 정리 순서를 자동으로 달성한다.
</details>

### 질문 4: void** 파라미터 배열의 메모리 레이아웃이 정확해야 하는 이유는?

<details>
<summary>답변 확인</summary>

`cuLaunchKernel`의 `void** kernelParams`는 커널 함수의 인자 목록과 **정확히 같은 순서와 타입**으로 대응해야 한다. Driver API는 Runtime API와 달리 타입 검사를 수행하지 않는다.

```
커널 시그니처: void sgemm(float* A, float* B, float* C, int M, int N, int K, ...)
                          [0]      [1]      [2]     [3]  [4]  [5]

void* args[]:             &A       &B       &C      &M   &N   &K   ...
```

각 `args[i]`는 i번째 인자의 **값이 저장된 메모리 위치**를 가리켜야 한다. 순서가 뒤바뀌거나 타입 크기가 맞지 않으면:
- `int`(4바이트)와 `CUdeviceptr`(8바이트)가 뒤섞이면 이후 모든 인자가 잘못된 오프셋에서 읽힌다.
- 포인터 값 자체가 커널에 전달되어 잘못된 메모리 접근이 발생한다.
- GPU 페이지 폴트 또는 잘못된 계산 결과를 초래하며, 디버깅이 매우 어렵다.
</details>

---

## 6. 시뮬레이션 2: Driver API 호출 시퀀스

### 과제

간단한 벡터 덧셈(`C[i] = A[i] + B[i]`, N개 원소)을 수행하기 위해 필요한 Driver API 호출 순서를 나열하시오. 종이에 직접 적어본 뒤 아래 정답을 확인하라.

### 정답: Driver API 호출 시퀀스

```
[초기화]
  1. cuInit(0)                         -- 드라이버 초기화
  2. cuDeviceGet(&device, 0)           -- 장치 핸들 획득
  3. cuCtxCreate(&ctx, 0, device)      -- 컨텍스트 생성

[모듈 로딩]
  4. cuModuleLoad(&module, "vecadd.ptx")       -- PTX 로드 + JIT 컴파일
  5. cuModuleGetFunction(&func, module, "vecadd") -- 함수 핸들 획득

[메모리 할당]
  6. cuMemAlloc(&d_A, N * sizeof(float))       -- 디바이스 메모리 A
  7. cuMemAlloc(&d_B, N * sizeof(float))       -- 디바이스 메모리 B
  8. cuMemAlloc(&d_C, N * sizeof(float))       -- 디바이스 메모리 C

[데이터 전송: Host -> Device]
  9. cuMemcpyHtoD(d_A, h_A, N * sizeof(float)) -- A 복사
 10. cuMemcpyHtoD(d_B, h_B, N * sizeof(float)) -- B 복사

[커널 실행]
 11. cuLaunchKernel(func, gridDim, 1, 1,       -- 커널 실행
                    blockDim, 1, 1,
                    0, NULL, args, NULL)

[데이터 전송: Device -> Host]
 12. cuMemcpyDtoH(h_C, d_C, N * sizeof(float)) -- 결과 복사

[정리]
 13. cuMemFree(d_A)                    -- 메모리 해제
 14. cuMemFree(d_B)
 15. cuMemFree(d_C)
 16. cuModuleUnload(module)            -- 모듈 언로드
 17. cuCtxDestroy(ctx)                 -- 컨텍스트 파괴
```

**총 17회 API 호출**

### 비교: Runtime API 동등 코드

```
[초기화]
  (없음 -- 암시적)

[모듈 로딩]
  (없음 -- 빌드 시 링크)

[메모리 할당]
  1. cudaMalloc(&d_A, N * sizeof(float))
  2. cudaMalloc(&d_B, N * sizeof(float))
  3. cudaMalloc(&d_C, N * sizeof(float))

[데이터 전송: Host -> Device]
  4. cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice)
  5. cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice)

[커널 실행]
  6. vecadd<<<gridDim, blockDim>>>(d_A, d_B, d_C, N)

[데이터 전송: Device -> Host]
  7. cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost)

[정리]
  8. cudaFree(d_A)
  9. cudaFree(d_B)
 10. cudaFree(d_C)
```

**총 10회 API 호출** -- Driver API의 절반 수준

핵심 차이: Runtime API는 초기화, 컨텍스트 관리, 모듈 로딩, 파라미터 패킹을 자동으로 처리한다. 그러나 이 "자동" 과정에서 어떤 일이 일어나는지 개발자가 제어할 수 없다. 우리 프로젝트에서는 PTX JIT 컴파일 옵션, 컨텍스트 수명, 에러 로그 접근 등이 중요하므로 Driver API를 선택했다.

---

## 7. 핵심 정리

이 모듈에서 배운 내용을 한 문단으로 요약한다:

> **CUDA Driver API**는 Runtime API의 암시적 동작을 모두 명시적으로 드러낸다. `cuInit`으로 드라이버를 초기화하고, `cuCtxCreate`로 컨텍스트(GPU 연결)를 생성하며, `cuModuleLoadDataEx`로 PTX를 JIT 컴파일하고, `cuLaunchKernel`로 커널을 실행한다. 우리 프로젝트는 이 과정을 RAII 패턴의 C++ 클래스(`CudaContext`, `CudaModule`, `DeviceMemory`, `CudaEvent`)로 감싸 안전성과 편의성을 확보한다. PTX JIT 컴파일은 아키텍처 이식성을 제공하며, `void**` 파라미터 패킹은 커널 인자의 순서와 타입을 정확히 맞추어야 한다.

### 핵심 개념 체크리스트

| 개념 | 설명 | 코드 참조 |
|------|------|----------|
| Driver API 초기화 | `cuInit(0)` 필수 호출 | `cuda_driver.cpp:58` |
| 컨텍스트 | GPU 리소스의 소유권 경계 | `cuda_driver.cpp:54-83` |
| PTX | 아키텍처 독립적 GPU 중간 표현 | `CMakeLists.txt:54-66` |
| JIT 컴파일 | PTX를 실행 시점에 SASS로 변환 | `cuda_driver.cpp:229-234` |
| RAII | 생성자에서 획득, 소멸자에서 해제 | 모든 래퍼 클래스 |
| void** 패킹 | 커널 인자를 포인터 배열로 전달 | `kernel_launcher.cpp:159-171` |
| 이벤트 타이밍 | GPU 시간 측정의 정확한 방법 | `cuda_driver.cpp:539-551` |

---

## 8. 퀴즈

### 문제 1 (Remember)

Driver API에서 컨텍스트를 생성하는 함수의 이름은 무엇인가?

<details>
<summary>정답</summary>

**`cuCtxCreate`**

시그니처: `CUresult cuCtxCreate(CUcontext* pctx, unsigned int flags, CUdevice dev)`

우리 코드에서의 사용:
```cpp
// src/driver/cuda_driver.cpp:78
CHECK_CU(cuCtxCreate(&context_, flags, device_));
```
</details>

---

### 문제 2 (Understand)

PTX JIT 컴파일의 장점을 두 가지 이상 설명하시오.

<details>
<summary>정답</summary>

1. **아키텍처 이식성**: PTX는 가상 ISA이므로 sm_80에서 빌드한 PTX를 sm_90 이상의 미래 GPU에서도 JIT 컴파일하여 실행할 수 있다. SASS는 특정 아키텍처에 종속된다.

2. **런타임 최적화**: JIT 컴파일러는 실행 시점의 정확한 GPU 모델, 드라이버 버전, 사용 가능한 리소스를 알고 있으므로, 해당 환경에 최적화된 기계어를 생성할 수 있다.

3. **배포 편의성**: 여러 GPU 아키텍처를 지원할 때, AOT 방식은 각 아키텍처별 SASS를 모두 포함한 fat binary를 생성해야 한다. JIT 방식은 하나의 PTX 파일만 배포하면 된다.

4. **동적 로딩**: 커널을 빌드 시 링크하지 않고 실행 시점에 파일로부터 로드할 수 있다. 우리 프로젝트처럼 6개의 최적화 레벨을 선택적으로 로드하는 데 적합하다.
</details>

---

### 문제 3 (Apply)

다음 커널 시그니처에 대해 `cuLaunchKernel`에 전달할 `void**` 파라미터 배열을 작성하시오.

```cuda
__global__ void vecadd(float* A, float* B, float* C, int N)
```

변수: `CUdeviceptr d_A, d_B, d_C; int N = 1024;`

<details>
<summary>정답</summary>

```cpp
void* args[] = {
    &d_A,    // float* A  (CUdeviceptr는 포인터 크기)
    &d_B,    // float* B
    &d_C,    // float* C
    &N       // int N
};

// cuLaunchKernel 호출
cuLaunchKernel(func,
    gridDim, 1, 1,      // 그리드 크기
    blockDim, 1, 1,      // 블록 크기
    0, nullptr,          // 공유 메모리, 스트림
    args, nullptr);      // 파라미터 배열, 추가 옵션
```

핵심: 각 원소는 **값의 주소**이다. `d_A` 자체(값)가 아니라 `&d_A`(주소)를 넣어야 한다.
</details>

---

### 문제 4 (Analyze)

JIT 컴파일이 실패했을 때, 에러 원인을 담은 로그를 어떻게 얻을 수 있는가? 프로젝트 코드를 참조하여 설명하시오.

<details>
<summary>정답</summary>

`JitOptions::to_cu_options()`에서 `CU_JIT_ERROR_LOG_BUFFER`와 `CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES` 옵션을 설정한다:

```cpp
// src/driver/cuda_driver.cpp:171-189 (JitOptions::to_cu_options)
options.push_back(CU_JIT_ERROR_LOG_BUFFER);
options.push_back(CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES);

// src/driver/cuda_driver.cpp:191-212 (JitOptions::to_cu_values)
static char error_log[8192];
values.push_back(error_log);
values.push_back(reinterpret_cast<void*>(sizeof(error_log)));
```

`cuModuleLoadDataEx`가 실패하면 에러 로그 버퍼에 컴파일 에러 메시지가 기록된다. 이를 `cu_values[3]`에서 추출한다:

```cpp
// src/driver/cuda_driver.cpp:236-241
if (result != CUDA_SUCCESS) {
    char* error_log = static_cast<char*>(cu_values[3]);
    std::string error_msg = "PTX JIT compilation failed: "
                            + std::string(error_log);
    throw std::runtime_error(error_msg);
}
```

마찬가지로, 성공 시에도 `cu_values[1]`의 정보 로그(`CU_JIT_INFO_LOG_BUFFER`)에 경고나 최적화 정보가 기록된다.
</details>

---

### 문제 5 (Understand)

CUDA 프로그래밍에서 RAII 패턴이 특히 중요한 이유를 설명하시오. `DeviceMemory` 클래스를 예로 들어 답하시오.

<details>
<summary>정답</summary>

GPU 메모리는 CPU 메모리보다 훨씬 제한적(일반적으로 8~48GB)이며, 누수된 GPU 메모리는 프로그램이 종료될 때까지 회수되지 않는다.

`DeviceMemory`의 RAII 패턴:

```cpp
// 생성자: 리소스 획득
DeviceMemory::DeviceMemory(size_t size) : size_(size) {
    CHECK_CU(cuMemAlloc(&ptr_, size));    // GPU 메모리 할당
}

// 소멸자: 리소스 해제
DeviceMemory::~DeviceMemory() {
    if (ptr_ != 0) {
        cuMemFree(ptr_);    // GPU 메모리 자동 해제
    }
}
```

이 패턴이 없다면:
1. `cuMemAlloc` 이후의 코드에서 예외가 발생하면 `cuMemFree`가 호출되지 않아 메모리가 누수된다.
2. 복잡한 `try-catch-finally` 블록이 필요하며, 여러 리소스를 다룰 때 정리 순서를 수동으로 관리해야 한다.
3. 복사 금지(`delete`)와 이동 허용을 통해 소유권이 항상 하나의 객체에만 존재하므로, 이중 해제(double-free) 문제도 방지된다.
</details>

---

### 문제 6 (Apply)

4096 x 4096 크기의 `float` 행렬을 GPU 디바이스 메모리에 할당하려면 몇 바이트가 필요한가? 이를 수행하는 코드를 `DeviceMemory` 클래스를 사용하여 작성하시오.

<details>
<summary>정답</summary>

**필요 바이트 수**:

```
4096 * 4096 * sizeof(float) = 4096 * 4096 * 4 = 67,108,864 바이트 = 64 MB
```

계산 과정:
- 행 수: 4096
- 열 수: 4096
- 원소당 크기: `sizeof(float)` = 4 바이트
- 총 바이트: 4096 * 4096 * 4 = 16,777,216 * 4 = 67,108,864 바이트 = 64 MiB

**코드**:

```cpp
// DeviceMemory 클래스 사용
constexpr int N = 4096;
size_t matrix_bytes = N * N * sizeof(float);   // 67,108,864 = 64 MB

baremetal::DeviceMemory d_matrix(matrix_bytes); // cuMemAlloc 호출

// 호스트 데이터를 디바이스로 복사
std::vector<float> h_matrix(N * N);
d_matrix.copy_from_host(h_matrix.data(), matrix_bytes);

// d_matrix가 스코프를 벗어나면 자동으로 cuMemFree 호출
```

참고: SGEMM에서 A(MxK), B(KxN), C(MxN) 세 행렬 모두 4096 x 4096이면 총 64 MB * 3 = 192 MB의 디바이스 메모리가 필요하다.
</details>

---

## 9. 다음 단계 미리보기: 모듈 3 -- 행렬곱셈의 수학적 배경

이 모듈에서는 **도구**(Driver API)를 배웠다. 다음 모듈에서는 그 도구로 해결할 **문제**(행렬곱셈)의 수학을 깊이 파고든다.

다음 모듈에서 다룰 핵심 질문:

- **SGEMM의 연산량은 얼마인가?** 4096 x 4096 행렬곱의 총 FLOP는 `2 * 4096 * 4096 * 4096 = 약 1,370억 FLOP`이다. 이것을 1밀리초 안에 끝내려면 137 TFLOP/s가 필요하다.
- **메모리 전송량 대비 연산량(산술 강도)은?** 이 비율이 커널이 메모리 바운드인지 연산 바운드인지를 결정한다.
- **루프라인 모델(Roofline Model)**이란 무엇이며, 우리 커널의 성능 상한을 어떻게 예측하는가?
- **행렬을 타일로 분할하면 왜 성능이 올라가는가?** 데이터 재사용(data reuse)의 수학적 근거를 분석한다.

행렬곱셈의 수학을 이해해야 이후 모듈(Level 0~Level 5)에서 각 최적화가 **왜** 효과적인지 납득할 수 있다.
