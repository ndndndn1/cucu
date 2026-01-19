/**
 * BareMetal-SGEMM: CUDA Driver API Implementation
 *
 * This file implements the core Driver API wrapper functionality.
 *
 * Key concepts demonstrated:
 * 1. Manual CUDA initialization (cuInit)
 * 2. Explicit context management (cuCtxCreate/cuCtxDestroy)
 * 3. JIT compilation from PTX (cuModuleLoadDataEx)
 * 4. Device memory management without Runtime API
 */

#include "cuda_driver_wrapper.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cstring>

namespace baremetal {

// ============================================================================
// DeviceInfo Implementation
// ============================================================================
void DeviceInfo::print() const {
    std::cout << "=== GPU Device Information ===" << std::endl;
    std::cout << "Device ID:        " << device_id << std::endl;
    std::cout << "Name:             " << name << std::endl;
    std::cout << "Compute:          " << compute_major << "." << compute_minor << std::endl;
    std::cout << "Total Memory:     " << (total_memory / (1024.0 * 1024.0 * 1024.0))
              << " GB" << std::endl;
    std::cout << "SM Count:         " << sm_count << std::endl;
    std::cout << "Max Threads/Block:" << max_threads_per_block << std::endl;
    std::cout << "Max Threads/SM:   " << max_threads_per_sm << std::endl;
    std::cout << "Max Regs/Block:   " << max_registers_per_block << std::endl;
    std::cout << "Max Regs/SM:      " << max_registers_per_sm << std::endl;
    std::cout << "Shared Mem/Block: " << (max_shared_memory_per_block / 1024.0) << " KB" << std::endl;
    std::cout << "Shared Mem/SM:    " << (max_shared_memory_per_sm / 1024.0) << " KB" << std::endl;
    std::cout << "Warp Size:        " << warp_size << std::endl;
    std::cout << "Memory Bus Width: " << memory_bus_width << " bit" << std::endl;
    std::cout << "L2 Cache Size:    " << (l2_cache_size / 1024.0) << " KB" << std::endl;
    std::cout << "Peak Bandwidth:   " << peak_memory_bandwidth_gb() << " GB/s" << std::endl;
    std::cout << "Async Copy:       " << (supports_async_copy ? "Yes" : "No") << std::endl;
    std::cout << "===============================" << std::endl;
}

// ============================================================================
// CudaContext Implementation
// ============================================================================

// Static initialization flag
static bool s_cuda_initialized = false;

CudaContext::CudaContext(int device_id, unsigned int flags) {
    // Initialize CUDA Driver API (must be called before any other Driver API call)
    // This is the equivalent of the hidden initialization in Runtime API
    if (!s_cuda_initialized) {
        CHECK_CU(cuInit(0));
        s_cuda_initialized = true;
    }

    // Check device count
    int device_count = 0;
    CHECK_CU(cuDeviceGetCount(&device_count));
    if (device_count == 0) {
        throw std::runtime_error("No CUDA-capable devices found");
    }
    if (device_id >= device_count) {
        throw std::runtime_error("Invalid device ID: " + std::to_string(device_id) +
                                " (only " + std::to_string(device_count) + " devices available)");
    }

    // Get device handle
    CHECK_CU(cuDeviceGet(&device_, device_id));

    // Create context for this device
    // The context is the GPU equivalent of a process - it owns all resources
    CHECK_CU(cuCtxCreate(&context_, flags, device_));

    // Query device properties
    device_info_.device_id = device_id;
    query_device_info();
}

CudaContext::~CudaContext() {
    if (owns_context_ && context_ != nullptr) {
        // Destroy context and free all resources
        cuCtxDestroy(context_);
        context_ = nullptr;
    }
}

CudaContext::CudaContext(CudaContext&& other) noexcept
    : device_(other.device_)
    , context_(other.context_)
    , device_info_(std::move(other.device_info_))
    , owns_context_(other.owns_context_) {
    other.context_ = nullptr;
    other.owns_context_ = false;
}

CudaContext& CudaContext::operator=(CudaContext&& other) noexcept {
    if (this != &other) {
        if (owns_context_ && context_ != nullptr) {
            cuCtxDestroy(context_);
        }
        device_ = other.device_;
        context_ = other.context_;
        device_info_ = std::move(other.device_info_);
        owns_context_ = other.owns_context_;
        other.context_ = nullptr;
        other.owns_context_ = false;
    }
    return *this;
}

void CudaContext::synchronize() {
    CHECK_CU(cuCtxSynchronize());
}

void CudaContext::set_current() {
    CHECK_CU(cuCtxSetCurrent(context_));
}

void CudaContext::query_device_info() {
    char name[256];
    CHECK_CU(cuDeviceGetName(name, sizeof(name), device_));
    device_info_.name = name;

    CHECK_CU(cuDeviceGetAttribute(&device_info_.compute_major,
        CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device_));
    CHECK_CU(cuDeviceGetAttribute(&device_info_.compute_minor,
        CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device_));

    size_t total_mem;
    CHECK_CU(cuDeviceTotalMem(&total_mem, device_));
    device_info_.total_memory = total_mem;

    CHECK_CU(cuDeviceGetAttribute(&device_info_.sm_count,
        CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device_));
    CHECK_CU(cuDeviceGetAttribute(&device_info_.max_threads_per_block,
        CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, device_));
    CHECK_CU(cuDeviceGetAttribute(&device_info_.max_threads_per_sm,
        CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, device_));
    CHECK_CU(cuDeviceGetAttribute(&device_info_.max_registers_per_block,
        CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK, device_));
    CHECK_CU(cuDeviceGetAttribute(&device_info_.max_registers_per_sm,
        CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR, device_));
    CHECK_CU(cuDeviceGetAttribute(&device_info_.max_shared_memory_per_block,
        CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, device_));
    CHECK_CU(cuDeviceGetAttribute(&device_info_.max_shared_memory_per_sm,
        CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR, device_));
    CHECK_CU(cuDeviceGetAttribute(&device_info_.warp_size,
        CU_DEVICE_ATTRIBUTE_WARP_SIZE, device_));
    CHECK_CU(cuDeviceGetAttribute(&device_info_.memory_bus_width,
        CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, device_));
    CHECK_CU(cuDeviceGetAttribute(&device_info_.memory_clock_rate,
        CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, device_));
    CHECK_CU(cuDeviceGetAttribute(&device_info_.l2_cache_size,
        CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, device_));

    // Ampere (SM 8.0+) supports async memory copy
    device_info_.supports_async_copy =
        (device_info_.compute_major > 8) ||
        (device_info_.compute_major == 8 && device_info_.compute_minor >= 0);
}

// ============================================================================
// JitOptions Implementation
// ============================================================================
std::vector<CUjit_option> JitOptions::to_cu_options() const {
    std::vector<CUjit_option> options;
    options.push_back(CU_JIT_OPTIMIZATION_LEVEL);
    options.push_back(CU_JIT_INFO_LOG_BUFFER);
    options.push_back(CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES);
    options.push_back(CU_JIT_ERROR_LOG_BUFFER);
    options.push_back(CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES);

    if (max_registers > 0) {
        options.push_back(CU_JIT_MAX_REGISTERS);
    }
    if (generate_debug_info) {
        options.push_back(CU_JIT_GENERATE_DEBUG_INFO);
    }
    if (generate_line_info) {
        options.push_back(CU_JIT_GENERATE_LINE_INFO);
    }
    return options;
}

std::vector<void*> JitOptions::to_cu_values() const {
    static char info_log[8192];
    static char error_log[8192];

    std::vector<void*> values;
    values.push_back(reinterpret_cast<void*>(static_cast<size_t>(optimization_level)));
    values.push_back(info_log);
    values.push_back(reinterpret_cast<void*>(sizeof(info_log)));
    values.push_back(error_log);
    values.push_back(reinterpret_cast<void*>(sizeof(error_log)));

    if (max_registers > 0) {
        values.push_back(reinterpret_cast<void*>(static_cast<size_t>(max_registers)));
    }
    if (generate_debug_info) {
        values.push_back(reinterpret_cast<void*>(1));
    }
    if (generate_line_info) {
        values.push_back(reinterpret_cast<void*>(1));
    }
    return values;
}

// ============================================================================
// CudaModule Implementation
// ============================================================================
std::unique_ptr<CudaModule> CudaModule::from_ptx(
    const std::string& ptx_source,
    const JitOptions& options) {

    auto module = std::unique_ptr<CudaModule>(new CudaModule());

    // Prepare JIT compilation options
    auto cu_options = options.to_cu_options();
    auto cu_values = options.to_cu_values();

    // JIT compile PTX to machine code
    // This is the key technique for architecture-independent deployment
    CUresult result = cuModuleLoadDataEx(
        &module->module_,
        ptx_source.c_str(),
        static_cast<unsigned int>(cu_options.size()),
        cu_options.data(),
        cu_values.data());

    if (result != CUDA_SUCCESS) {
        // Get error log from JIT compiler
        char* error_log = static_cast<char*>(cu_values[3]);
        std::string error_msg = "PTX JIT compilation failed: " + std::string(error_log);
        throw std::runtime_error(error_msg);
    }

    // Store info log for debugging
    char* info_log = static_cast<char*>(cu_values[1]);
    module->jit_log_ = info_log;

    return module;
}

std::unique_ptr<CudaModule> CudaModule::from_ptx_file(
    const std::string& filepath,
    const JitOptions& options) {

    // Read PTX file
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open PTX file: " + filepath);
    }

    std::stringstream buffer;
    buffer << file.rdbuf();

    return from_ptx(buffer.str(), options);
}

std::unique_ptr<CudaModule> CudaModule::from_cuda_source(
    const std::string& cuda_source,
    const std::string& kernel_name,
    const std::vector<std::string>& compile_options) {

    // Create NVRTC program
    nvrtcProgram prog;
    CHECK_NVRTC(nvrtcCreateProgram(
        &prog,
        cuda_source.c_str(),
        (kernel_name + ".cu").c_str(),
        0, nullptr, nullptr));

    // Prepare compile options
    std::vector<const char*> options;
    for (const auto& opt : compile_options) {
        options.push_back(opt.c_str());
    }
    // Add default options
    options.push_back("--use_fast_math");
    options.push_back("-default-device");

    // Compile to PTX
    nvrtcResult compile_result = nvrtcCompileProgram(prog,
        static_cast<int>(options.size()), options.data());

    if (compile_result != NVRTC_SUCCESS) {
        // Get compilation log
        size_t log_size;
        nvrtcGetProgramLogSize(prog, &log_size);
        std::string log(log_size, '\0');
        nvrtcGetProgramLog(prog, &log[0]);
        nvrtcDestroyProgram(&prog);
        throw std::runtime_error("NVRTC compilation failed:\n" + log);
    }

    // Get PTX
    size_t ptx_size;
    CHECK_NVRTC(nvrtcGetPTXSize(prog, &ptx_size));
    std::string ptx(ptx_size, '\0');
    CHECK_NVRTC(nvrtcGetPTX(prog, &ptx[0]));

    nvrtcDestroyProgram(&prog);

    // Load PTX as module
    return from_ptx(ptx);
}

CudaModule::~CudaModule() {
    if (module_ != nullptr) {
        cuModuleUnload(module_);
        module_ = nullptr;
    }
}

CudaModule::CudaModule(CudaModule&& other) noexcept
    : module_(other.module_)
    , function_cache_(std::move(other.function_cache_))
    , jit_log_(std::move(other.jit_log_)) {
    other.module_ = nullptr;
}

CudaModule& CudaModule::operator=(CudaModule&& other) noexcept {
    if (this != &other) {
        if (module_ != nullptr) {
            cuModuleUnload(module_);
        }
        module_ = other.module_;
        function_cache_ = std::move(other.function_cache_);
        jit_log_ = std::move(other.jit_log_);
        other.module_ = nullptr;
    }
    return *this;
}

CUfunction CudaModule::get_function(const std::string& name) {
    // Check cache first
    auto it = function_cache_.find(name);
    if (it != function_cache_.end()) {
        return it->second;
    }

    // Get function from module
    CUfunction func;
    CUresult result = cuModuleGetFunction(&func, module_, name.c_str());
    if (result != CUDA_SUCCESS) {
        throw std::runtime_error("Failed to get function '" + name +
                                "' from module");
    }

    function_cache_[name] = func;
    return func;
}

int CudaModule::get_max_active_blocks_per_sm(
    CUfunction func,
    int block_size,
    size_t dynamic_smem_size) {

    int num_blocks;
    CHECK_CU(cuOccupancyMaxActiveBlocksPerMultiprocessor(
        &num_blocks, func, block_size, dynamic_smem_size));
    return num_blocks;
}

// ============================================================================
// DeviceMemory Implementation
// ============================================================================
DeviceMemory::DeviceMemory(size_t size) : size_(size) {
    if (size > 0) {
        CHECK_CU(cuMemAlloc(&ptr_, size));
    }
}

DeviceMemory::~DeviceMemory() {
    if (ptr_ != 0) {
        cuMemFree(ptr_);
        ptr_ = 0;
    }
}

DeviceMemory::DeviceMemory(DeviceMemory&& other) noexcept
    : ptr_(other.ptr_), size_(other.size_) {
    other.ptr_ = 0;
    other.size_ = 0;
}

DeviceMemory& DeviceMemory::operator=(DeviceMemory&& other) noexcept {
    if (this != &other) {
        if (ptr_ != 0) {
            cuMemFree(ptr_);
        }
        ptr_ = other.ptr_;
        size_ = other.size_;
        other.ptr_ = 0;
        other.size_ = 0;
    }
    return *this;
}

void DeviceMemory::copy_from_host(const void* host_ptr, size_t bytes) {
    CHECK_CU(cuMemcpyHtoD(ptr_, host_ptr, bytes));
}

void DeviceMemory::copy_to_host(void* host_ptr, size_t bytes) const {
    CHECK_CU(cuMemcpyDtoH(host_ptr, ptr_, bytes));
}

void DeviceMemory::copy_from_host_async(const void* host_ptr, size_t bytes, CUstream stream) {
    CHECK_CU(cuMemcpyHtoDAsync(ptr_, host_ptr, bytes, stream));
}

void DeviceMemory::copy_to_host_async(void* host_ptr, size_t bytes, CUstream stream) const {
    CHECK_CU(cuMemcpyDtoHAsync(host_ptr, ptr_, bytes, stream));
}

void DeviceMemory::memset(unsigned char value, size_t bytes) {
    CHECK_CU(cuMemsetD8(ptr_, value, bytes));
}

void DeviceMemory::memset_async(unsigned char value, size_t bytes, CUstream stream) {
    CHECK_CU(cuMemsetD8Async(ptr_, value, bytes, stream));
}

// ============================================================================
// PinnedMemory Implementation
// ============================================================================
PinnedMemory::PinnedMemory(size_t size, unsigned int flags) : size_(size) {
    if (size > 0) {
        CHECK_CU(cuMemAllocHost(&ptr_, size));
    }
}

PinnedMemory::~PinnedMemory() {
    if (ptr_ != nullptr) {
        cuMemFreeHost(ptr_);
        ptr_ = nullptr;
    }
}

PinnedMemory::PinnedMemory(PinnedMemory&& other) noexcept
    : ptr_(other.ptr_), size_(other.size_) {
    other.ptr_ = nullptr;
    other.size_ = 0;
}

PinnedMemory& PinnedMemory::operator=(PinnedMemory&& other) noexcept {
    if (this != &other) {
        if (ptr_ != nullptr) {
            cuMemFreeHost(ptr_);
        }
        ptr_ = other.ptr_;
        size_ = other.size_;
        other.ptr_ = nullptr;
        other.size_ = 0;
    }
    return *this;
}

// ============================================================================
// CudaStream Implementation
// ============================================================================
CudaStream::CudaStream(unsigned int flags) {
    CHECK_CU(cuStreamCreate(&stream_, flags));
}

CudaStream::~CudaStream() {
    if (stream_ != nullptr) {
        cuStreamDestroy(stream_);
        stream_ = nullptr;
    }
}

CudaStream::CudaStream(CudaStream&& other) noexcept : stream_(other.stream_) {
    other.stream_ = nullptr;
}

CudaStream& CudaStream::operator=(CudaStream&& other) noexcept {
    if (this != &other) {
        if (stream_ != nullptr) {
            cuStreamDestroy(stream_);
        }
        stream_ = other.stream_;
        other.stream_ = nullptr;
    }
    return *this;
}

void CudaStream::synchronize() {
    CHECK_CU(cuStreamSynchronize(stream_));
}

bool CudaStream::query() {
    CUresult result = cuStreamQuery(stream_);
    if (result == CUDA_SUCCESS) {
        return true;
    } else if (result == CUDA_ERROR_NOT_READY) {
        return false;
    } else {
        CHECK_CU(result);
        return false;
    }
}

// ============================================================================
// CudaEvent Implementation
// ============================================================================
CudaEvent::CudaEvent(unsigned int flags) {
    CHECK_CU(cuEventCreate(&event_, flags));
}

CudaEvent::~CudaEvent() {
    if (event_ != nullptr) {
        cuEventDestroy(event_);
        event_ = nullptr;
    }
}

CudaEvent::CudaEvent(CudaEvent&& other) noexcept : event_(other.event_) {
    other.event_ = nullptr;
}

CudaEvent& CudaEvent::operator=(CudaEvent&& other) noexcept {
    if (this != &other) {
        if (event_ != nullptr) {
            cuEventDestroy(event_);
        }
        event_ = other.event_;
        other.event_ = nullptr;
    }
    return *this;
}

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

// ============================================================================
// KernelLauncher Implementation
// ============================================================================
void KernelLauncher::launch(
    CUfunction func,
    const LaunchConfig& config,
    void** kernel_params) {

    CHECK_CU(cuLaunchKernel(
        func,
        config.grid.x, config.grid.y, config.grid.z,
        config.block.x, config.block.y, config.block.z,
        static_cast<unsigned int>(config.shared_mem_bytes),
        config.stream,
        kernel_params,
        nullptr  // extra options (not used)
    ));
}

}  // namespace baremetal
