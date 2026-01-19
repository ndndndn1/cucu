#pragma once

/**
 * BareMetal-SGEMM: CUDA Driver API Wrapper
 *
 * This module provides a clean C++ interface to the CUDA Driver API,
 * replacing the convenience of Runtime API with explicit control.
 *
 * Key differences from Runtime API:
 * - Manual context management (cuCtxCreate vs automatic)
 * - JIT compilation from PTX (cuModuleLoadDataEx)
 * - Explicit kernel parameter packing (void**)
 * - No hidden memory allocations or synchronizations
 */

#include <cuda.h>
#include <nvrtc.h>
#include <string>
#include <vector>
#include <memory>
#include <stdexcept>
#include <unordered_map>

namespace baremetal {

// ============================================================================
// Error Checking Macros
// ============================================================================
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

#define CHECK_NVRTC(call)                                                      \
    do {                                                                       \
        nvrtcResult err = (call);                                              \
        if (err != NVRTC_SUCCESS) {                                            \
            throw std::runtime_error(std::string("NVRTC Error: ") +            \
                                   nvrtcGetErrorString(err) + " at " +         \
                                   __FILE__ + ":" + std::to_string(__LINE__)); \
        }                                                                      \
    } while (0)

// ============================================================================
// Device Information
// ============================================================================
struct DeviceInfo {
    int device_id;
    std::string name;
    int compute_major;
    int compute_minor;
    size_t total_memory;
    int sm_count;
    int max_threads_per_block;
    int max_threads_per_sm;
    int max_registers_per_block;
    int max_registers_per_sm;
    int max_shared_memory_per_block;
    int max_shared_memory_per_sm;
    int warp_size;
    int memory_bus_width;
    int memory_clock_rate;  // kHz
    int l2_cache_size;
    bool supports_async_copy;  // Ampere+

    double peak_memory_bandwidth_gb() const {
        // Memory bandwidth = memory_clock_rate * bus_width * 2 (DDR) / 8 (bits to bytes)
        return (memory_clock_rate * 1e3 * memory_bus_width * 2.0) / 8.0 / 1e9;
    }

    void print() const;
};

// ============================================================================
// CUDA Context Manager (RAII)
// ============================================================================
class CudaContext {
public:
    /**
     * Initialize CUDA and create a context for the specified device.
     * This replaces the implicit initialization of Runtime API.
     *
     * @param device_id GPU device ordinal (0 for first GPU)
     * @param flags Context creation flags (e.g., CU_CTX_SCHED_AUTO)
     */
    explicit CudaContext(int device_id = 0, unsigned int flags = CU_CTX_SCHED_AUTO);
    ~CudaContext();

    // Non-copyable, movable
    CudaContext(const CudaContext&) = delete;
    CudaContext& operator=(const CudaContext&) = delete;
    CudaContext(CudaContext&& other) noexcept;
    CudaContext& operator=(CudaContext&& other) noexcept;

    CUcontext get() const { return context_; }
    CUdevice device() const { return device_; }
    const DeviceInfo& info() const { return device_info_; }

    void synchronize();
    void set_current();

private:
    CUdevice device_ = 0;
    CUcontext context_ = nullptr;
    DeviceInfo device_info_;
    bool owns_context_ = true;

    void query_device_info();
};

// ============================================================================
// JIT Compilation Options
// ============================================================================
struct JitOptions {
    int max_registers = 0;           // 0 = no limit, e.g., 64, 128, 255
    int optimization_level = 4;      // 0-4
    bool generate_debug_info = false;
    bool generate_line_info = true;  // For profiling
    std::string arch;                // e.g., "sm_80"

    std::vector<CUjit_option> to_cu_options() const;
    std::vector<void*> to_cu_values() const;
};

// ============================================================================
// Module (PTX/CUBIN) Loader
// ============================================================================
class CudaModule {
public:
    /**
     * Load a module from PTX source code (JIT compilation).
     * This is the key technique used by NVIDIA's driver team for
     * architecture-agnostic deployment.
     */
    static std::unique_ptr<CudaModule> from_ptx(
        const std::string& ptx_source,
        const JitOptions& options = JitOptions{});

    /**
     * Load a module from PTX file.
     */
    static std::unique_ptr<CudaModule> from_ptx_file(
        const std::string& filepath,
        const JitOptions& options = JitOptions{});

    /**
     * Compile CUDA source to PTX at runtime using NVRTC,
     * then load as module. Maximum flexibility.
     */
    static std::unique_ptr<CudaModule> from_cuda_source(
        const std::string& cuda_source,
        const std::string& kernel_name,
        const std::vector<std::string>& compile_options = {});

    ~CudaModule();

    CudaModule(const CudaModule&) = delete;
    CudaModule& operator=(const CudaModule&) = delete;
    CudaModule(CudaModule&& other) noexcept;
    CudaModule& operator=(CudaModule&& other) noexcept;

    CUmodule get() const { return module_; }

    /**
     * Get a function handle from the module.
     * Caches the result for repeated calls.
     */
    CUfunction get_function(const std::string& name);

    /**
     * Get kernel occupancy information.
     */
    int get_max_active_blocks_per_sm(
        CUfunction func,
        int block_size,
        size_t dynamic_smem_size = 0);

private:
    CudaModule() = default;

    CUmodule module_ = nullptr;
    std::unordered_map<std::string, CUfunction> function_cache_;
    std::string jit_log_;
};

// ============================================================================
// Device Memory (RAII)
// ============================================================================
class DeviceMemory {
public:
    DeviceMemory() = default;
    explicit DeviceMemory(size_t size);
    ~DeviceMemory();

    DeviceMemory(const DeviceMemory&) = delete;
    DeviceMemory& operator=(const DeviceMemory&) = delete;
    DeviceMemory(DeviceMemory&& other) noexcept;
    DeviceMemory& operator=(DeviceMemory&& other) noexcept;

    CUdeviceptr get() const { return ptr_; }
    CUdeviceptr* ptr_to_ptr() { return &ptr_; }
    size_t size() const { return size_; }

    void copy_from_host(const void* host_ptr, size_t bytes);
    void copy_to_host(void* host_ptr, size_t bytes) const;
    void copy_from_host_async(const void* host_ptr, size_t bytes, CUstream stream);
    void copy_to_host_async(void* host_ptr, size_t bytes, CUstream stream) const;
    void memset(unsigned char value, size_t bytes);
    void memset_async(unsigned char value, size_t bytes, CUstream stream);

private:
    CUdeviceptr ptr_ = 0;
    size_t size_ = 0;
};

// ============================================================================
// Pinned Host Memory (for async transfers)
// ============================================================================
class PinnedMemory {
public:
    PinnedMemory() = default;
    explicit PinnedMemory(size_t size, unsigned int flags = 0);
    ~PinnedMemory();

    PinnedMemory(const PinnedMemory&) = delete;
    PinnedMemory& operator=(const PinnedMemory&) = delete;
    PinnedMemory(PinnedMemory&& other) noexcept;
    PinnedMemory& operator=(PinnedMemory&& other) noexcept;

    void* get() const { return ptr_; }
    template<typename T> T* as() { return static_cast<T*>(ptr_); }
    template<typename T> const T* as() const { return static_cast<const T*>(ptr_); }
    size_t size() const { return size_; }

private:
    void* ptr_ = nullptr;
    size_t size_ = 0;
};

// ============================================================================
// Stream (RAII)
// ============================================================================
class CudaStream {
public:
    CudaStream(unsigned int flags = CU_STREAM_DEFAULT);
    ~CudaStream();

    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;
    CudaStream(CudaStream&& other) noexcept;
    CudaStream& operator=(CudaStream&& other) noexcept;

    CUstream get() const { return stream_; }
    void synchronize();
    bool query();  // Returns true if all operations completed

private:
    CUstream stream_ = nullptr;
};

// ============================================================================
// Event (for timing)
// ============================================================================
class CudaEvent {
public:
    CudaEvent(unsigned int flags = CU_EVENT_DEFAULT);
    ~CudaEvent();

    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;
    CudaEvent(CudaEvent&& other) noexcept;
    CudaEvent& operator=(CudaEvent&& other) noexcept;

    CUevent get() const { return event_; }
    void record(CUstream stream = nullptr);
    void synchronize();

    // Returns elapsed time in milliseconds
    static float elapsed_ms(const CudaEvent& start, const CudaEvent& end);

private:
    CUevent event_ = nullptr;
};

// ============================================================================
// Kernel Launch Configuration
// ============================================================================
struct LaunchConfig {
    dim3 grid;
    dim3 block;
    size_t shared_mem_bytes = 0;
    CUstream stream = nullptr;

    LaunchConfig(dim3 g, dim3 b, size_t smem = 0, CUstream s = nullptr)
        : grid(g), block(b), shared_mem_bytes(smem), stream(s) {}

    int total_threads() const {
        return grid.x * grid.y * grid.z * block.x * block.y * block.z;
    }
};

// ============================================================================
// Kernel Launcher
// ============================================================================
class KernelLauncher {
public:
    /**
     * Launch a kernel using Driver API.
     *
     * The key difference from Runtime API:
     * - Parameters are packed into void** array
     * - Explicit grid/block dimensions
     * - Explicit shared memory size
     *
     * Example:
     *   void* args[] = {&d_A, &d_B, &d_C, &M, &N, &K};
     *   launcher.launch(func, config, args);
     */
    static void launch(
        CUfunction func,
        const LaunchConfig& config,
        void** kernel_params);

    /**
     * Variadic template for convenient parameter packing.
     *
     * Example:
     *   launcher.launch(func, config, d_A, d_B, d_C, M, N, K);
     */
    template<typename... Args>
    static void launch(CUfunction func, const LaunchConfig& config, Args&&... args) {
        void* params[] = {const_cast<void*>(static_cast<const void*>(&args))...};
        launch(func, config, params);
    }
};

}  // namespace baremetal
