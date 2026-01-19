/**
 * BareMetal-SGEMM: Memory Manager
 *
 * This file provides utilities for efficient memory management
 * in SGEMM operations, including:
 * - Matrix allocation with proper alignment for vectorized loads
 * - Padding for bank conflict avoidance
 * - Memory pool for reducing allocation overhead
 */

#include "cuda_driver_wrapper.hpp"
#include <vector>
#include <algorithm>
#include <cstring>

namespace baremetal {

// ============================================================================
// Alignment Constants
// ============================================================================
constexpr size_t ALIGNMENT_BYTES = 256;  // For optimal coalescing
constexpr size_t CACHE_LINE_SIZE = 128;  // L2 cache line

// ============================================================================
// Aligned Matrix Allocation
// ============================================================================
class AlignedMatrix {
public:
    /**
     * Allocate a matrix with proper alignment for vectorized loads.
     *
     * @param rows      Number of rows
     * @param cols      Number of columns
     * @param elem_size Size of each element (default: sizeof(float))
     * @param padding   Extra padding per row for bank conflict avoidance
     */
    AlignedMatrix(int rows, int cols, size_t elem_size = sizeof(float),
                  int padding = 0)
        : rows_(rows)
        , cols_(cols)
        , elem_size_(elem_size)
        , padding_(padding) {

        // Calculate padded leading dimension
        ld_ = cols + padding;

        // Ensure leading dimension is aligned
        size_t row_bytes = ld_ * elem_size_;
        size_t aligned_row_bytes = (row_bytes + ALIGNMENT_BYTES - 1) /
                                   ALIGNMENT_BYTES * ALIGNMENT_BYTES;
        ld_ = aligned_row_bytes / elem_size_;

        // Total size with alignment
        size_t total_bytes = rows_ * ld_ * elem_size_;

        // Allocate device memory
        device_mem_ = std::make_unique<DeviceMemory>(total_bytes);
    }

    int rows() const { return rows_; }
    int cols() const { return cols_; }
    int ld() const { return ld_; }  // Leading dimension
    size_t elem_size() const { return elem_size_; }
    size_t total_bytes() const { return rows_ * ld_ * elem_size_; }
    CUdeviceptr ptr() const { return device_mem_->get(); }

    void copy_from_host(const void* host_ptr) {
        // Copy row by row if there's padding
        if (ld_ != cols_) {
            const char* src = static_cast<const char*>(host_ptr);
            for (int i = 0; i < rows_; ++i) {
                CHECK_CU(cuMemcpyHtoD(
                    device_mem_->get() + i * ld_ * elem_size_,
                    src + i * cols_ * elem_size_,
                    cols_ * elem_size_));
            }
        } else {
            device_mem_->copy_from_host(host_ptr, rows_ * cols_ * elem_size_);
        }
    }

    void copy_to_host(void* host_ptr) const {
        if (ld_ != cols_) {
            char* dst = static_cast<char*>(host_ptr);
            for (int i = 0; i < rows_; ++i) {
                CHECK_CU(cuMemcpyDtoH(
                    dst + i * cols_ * elem_size_,
                    device_mem_->get() + i * ld_ * elem_size_,
                    cols_ * elem_size_));
            }
        } else {
            device_mem_->copy_to_host(host_ptr, rows_ * cols_ * elem_size_);
        }
    }

private:
    int rows_;
    int cols_;
    int ld_;
    size_t elem_size_;
    int padding_;
    std::unique_ptr<DeviceMemory> device_mem_;
};

// ============================================================================
// Memory Pool for Reducing Allocation Overhead
// ============================================================================
class MemoryPool {
public:
    explicit MemoryPool(size_t initial_size = 256 * 1024 * 1024)  // 256 MB default
        : total_size_(initial_size)
        , used_size_(0) {
        pool_ = std::make_unique<DeviceMemory>(initial_size);
    }

    /**
     * Allocate memory from the pool.
     * Returns 0 if not enough space (does not throw).
     */
    CUdeviceptr allocate(size_t size, size_t alignment = ALIGNMENT_BYTES) {
        // Align the current offset
        size_t aligned_offset = (used_size_ + alignment - 1) / alignment * alignment;

        if (aligned_offset + size > total_size_) {
            return 0;  // Not enough space
        }

        CUdeviceptr ptr = pool_->get() + aligned_offset;
        used_size_ = aligned_offset + size;
        return ptr;
    }

    /**
     * Reset the pool (does not free memory, just resets offset).
     */
    void reset() {
        used_size_ = 0;
    }

    size_t available() const { return total_size_ - used_size_; }
    size_t used() const { return used_size_; }
    size_t total() const { return total_size_; }

private:
    std::unique_ptr<DeviceMemory> pool_;
    size_t total_size_;
    size_t used_size_;
};

// ============================================================================
// Matrix Utilities
// ============================================================================
namespace matrix_utils {

/**
 * Initialize a matrix with random values on the host.
 */
void random_init(float* data, int rows, int cols, float min_val = -1.0f,
                 float max_val = 1.0f) {
    for (int i = 0; i < rows * cols; ++i) {
        data[i] = min_val + static_cast<float>(rand()) /
                  RAND_MAX * (max_val - min_val);
    }
}

/**
 * Initialize a matrix with zeros.
 */
void zero_init(float* data, int rows, int cols) {
    std::memset(data, 0, rows * cols * sizeof(float));
}

/**
 * Initialize a matrix as identity (for square matrices).
 */
void identity_init(float* data, int n) {
    std::memset(data, 0, n * n * sizeof(float));
    for (int i = 0; i < n; ++i) {
        data[i * n + i] = 1.0f;
    }
}

/**
 * Compare two matrices and return max absolute difference.
 */
float max_diff(const float* a, const float* b, int rows, int cols) {
    float max_d = 0.0f;
    for (int i = 0; i < rows * cols; ++i) {
        float d = std::abs(a[i] - b[i]);
        max_d = std::max(max_d, d);
    }
    return max_d;
}

/**
 * Check if two matrices are approximately equal.
 */
bool approximately_equal(const float* a, const float* b, int rows, int cols,
                         float rtol = 1e-4f, float atol = 1e-5f) {
    for (int i = 0; i < rows * cols; ++i) {
        float diff = std::abs(a[i] - b[i]);
        float threshold = atol + rtol * std::abs(b[i]);
        if (diff > threshold) {
            return false;
        }
    }
    return true;
}

/**
 * Transpose a matrix (host operation).
 */
void transpose(const float* src, float* dst, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            dst[j * rows + i] = src[i * cols + j];
        }
    }
}

/**
 * Print a small matrix (for debugging).
 */
void print_matrix(const float* data, int rows, int cols,
                  const char* name = nullptr, int max_rows = 8, int max_cols = 8) {
    if (name) {
        std::cout << name << " (" << rows << "x" << cols << "):" << std::endl;
    }

    int print_rows = std::min(rows, max_rows);
    int print_cols = std::min(cols, max_cols);

    for (int i = 0; i < print_rows; ++i) {
        std::cout << "  [";
        for (int j = 0; j < print_cols; ++j) {
            printf("%8.4f", data[i * cols + j]);
            if (j < print_cols - 1) std::cout << ", ";
        }
        if (cols > max_cols) std::cout << ", ...";
        std::cout << "]" << std::endl;
    }
    if (rows > max_rows) {
        std::cout << "  ..." << std::endl;
    }
}

}  // namespace matrix_utils

}  // namespace baremetal
