// include/llm/tensor.hpp
#pragma once
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <vector>

#include "llm/common.hpp"
#include "llm/dtype.hpp"
#include "llm/memory.hpp"

namespace llm {

using Shape   = std::vector<int64_t>;
using Strides = std::vector<int64_t>;

int64_t numel(const Shape& shape);               // product of dims; 1 for a scalar
Strides contiguous_strides(const Shape& shape);  // row-major strides

// Raw, reference-counted, aligned byte buffer. Tensors hold a shared_ptr to one,
// so views share storage and the buffer is freed when the last view dies.
class Storage {
public:
    explicit Storage(size_t nbytes, size_t alignment = 64);
    ~Storage();
    Storage(const Storage&)            = delete;
    Storage& operator=(const Storage&) = delete;

    uint8_t*       data()       noexcept { return data_; }
    const uint8_t* data() const noexcept { return data_; }
    size_t         nbytes() const noexcept { return nbytes_; }

    // Instrumentation (single-threaded): how many Storages have ever been built.
    // The boss challenge uses this to prove view ops allocate nothing.
    static uint64_t alloc_count() noexcept { return s_alloc_count; }

private:
    uint8_t* data_   = nullptr;
    size_t   nbytes_ = 0;
    inline static uint64_t s_alloc_count = 0;
};

class Tensor {
public:
    Tensor() = default;
    static Tensor empty(Shape shape, DType dtype, size_t alignment = 64);

    // ---- layout queries ----
    const Shape&   shape()   const noexcept { return shape_; }
    const Strides& strides() const noexcept { return strides_; }
    DType          dtype()   const noexcept { return dtype_; }
    size_t         ndim()    const noexcept { return shape_.size(); }
    int64_t        numel()   const noexcept { return llm::numel(shape_); }
    size_t  nbytes() const noexcept {
        return static_cast<size_t>(numel()) * dtype_size(dtype_);
    }
    bool is_contiguous() const noexcept;
    bool defined()       const noexcept { return storage_ != nullptr; }

    // ---- data access ----
    // Pointer to *this tensor's* first element (already includes the view offset).
    template <typename T> T* data_ptr() {
        return reinterpret_cast<T*>(storage_->data()) + offset_;
    }
    template <typename T> const T* data_ptr() const {
        return reinterpret_cast<const T*>(storage_->data()) + offset_;
    }

    // Element access for tests/debugging only (strided, branchy -- not hot paths).
    template <typename T> T& at(std::initializer_list<int64_t> idx) {
        return data_ptr<T>()[rel_offset(idx)];
    }
    template <typename T> const T& at(std::initializer_list<int64_t> idx) const {
        return data_ptr<T>()[rel_offset(idx)];
    }

    // ---- zero-copy views (Challenge 1.3) ----
    Tensor reshape(Shape new_shape) const;   // requires contiguous; shares storage
    Tensor view(Shape new_shape) const { return reshape(std::move(new_shape)); }
    Tensor permute(const std::vector<int>& dims) const;
    Tensor transpose(int dim0, int dim1) const;
    Tensor slice(int dim, int64_t start, int64_t stop) const;

    // Materialize a contiguous copy (allocates iff not already contiguous).
    Tensor contiguous() const;

    // Identity helpers for tests: tensors sharing storage return the same pointer.
    const void* storage_id() const noexcept { return storage_.get(); }
    int64_t     offset()     const noexcept { return offset_; }

private:
    int64_t rel_offset(std::initializer_list<int64_t> idx) const;

    std::shared_ptr<Storage> storage_;
    int64_t  offset_ = 0;            // in elements, from start of storage
    Shape    shape_;
    Strides  strides_;
    DType    dtype_  = DType::F32;
};

}  // namespace llm
