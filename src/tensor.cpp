// src/tensor.cpp
#include "llm/tensor.hpp"
#include <cstring>
#include <utility>

namespace llm {

int64_t numel(const Shape& shape) {
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    return n;   // empty shape -> 1 (a scalar)
}

Strides contiguous_strides(const Shape& shape) {
    Strides s(shape.size());
    int64_t acc = 1;
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
        s[i] = acc;
        acc *= shape[i];
    }
    return s;
}

// ---- Storage ----
Storage::Storage(size_t nbytes, size_t alignment) : nbytes_(nbytes) {
    data_ = static_cast<uint8_t*>(aligned_malloc(nbytes, alignment));
    LLM_CHECK(data_ != nullptr, "Storage: allocation failed");
    ++s_alloc_count;
}
Storage::~Storage() { aligned_free(data_); }

// ---- Tensor ----
Tensor Tensor::empty(Shape shape, DType dtype, size_t alignment) {
    Tensor t;
    t.dtype_   = dtype;
    t.shape_   = std::move(shape);
    t.strides_ = contiguous_strides(t.shape_);
    t.offset_  = 0;
    const size_t bytes =
        static_cast<size_t>(llm::numel(t.shape_)) * dtype_size(dtype);
    t.storage_ = std::make_shared<Storage>(bytes, alignment);
    return t;
}

bool Tensor::is_contiguous() const noexcept {
    int64_t expected = 1;
    for (int i = static_cast<int>(ndim()) - 1; i >= 0; --i) {
        if (shape_[i] == 1) continue;          // size-1 dims impose no constraint
        if (strides_[i] != expected) return false;
        expected *= shape_[i];
    }
    return true;
}

int64_t Tensor::rel_offset(std::initializer_list<int64_t> idx) const {
    LLM_ASSERT(idx.size() == ndim(), "at(): wrong number of indices");
    int64_t off = 0;
    size_t d = 0;
    for (int64_t i : idx) {
        LLM_ASSERT(i >= 0 && i < shape_[d], "at(): index out of range");
        off += i * strides_[d];
        ++d;
    }
    return off;   // relative to data_ptr(), which already includes offset_
}


Tensor Tensor::reshape(Shape new_shape) const {
    LLM_CHECK(llm::numel(new_shape) == numel(),
              "reshape: element count must be preserved");
    LLM_CHECK(is_contiguous(),
              "reshape: tensor must be contiguous; call contiguous() first");
    Tensor out   = *this;                 // shares storage_ (shared_ptr), same offset_
    out.shape_   = std::move(new_shape);
    out.strides_ = contiguous_strides(out.shape_);
    return out;
}

Tensor Tensor::permute(const std::vector<int>& dims) const {
    LLM_CHECK(dims.size() == ndim(), "permute: need one entry per dimension");
    Shape   ns(ndim());
    Strides nst(ndim());
    std::vector<bool> seen(ndim(), false);
    for (size_t i = 0; i < dims.size(); ++i) {
        const int d = dims[i];
        LLM_CHECK(d >= 0 && d < static_cast<int>(ndim()) && !seen[d],
                  "permute: invalid permutation");
        seen[d] = true;
        ns[i]   = shape_[d];
        nst[i]  = strides_[d];
    }
    Tensor out   = *this;
    out.shape_   = std::move(ns);
    out.strides_ = std::move(nst);
    return out;
}

Tensor Tensor::transpose(int dim0, int dim1) const {
    LLM_CHECK(dim0 >= 0 && dim0 < static_cast<int>(ndim()) &&
              dim1 >= 0 && dim1 < static_cast<int>(ndim()),
              "transpose: dim out of range");
    Tensor out = *this;
    std::swap(out.shape_[dim0],   out.shape_[dim1]);
    std::swap(out.strides_[dim0], out.strides_[dim1]);
    return out;
}

Tensor Tensor::slice(int dim, int64_t start, int64_t stop) const {
    LLM_CHECK(dim >= 0 && dim < static_cast<int>(ndim()), "slice: dim out of range");
    LLM_CHECK(0 <= start && start <= stop && stop <= shape_[dim],
              "slice: range out of bounds");
    Tensor out      = *this;
    out.offset_     = offset_ + start * strides_[dim];   // shift into the buffer
    out.shape_[dim] = stop - start;                      // shrink that dimension
    // strides unchanged
    return out;
}

Tensor Tensor::contiguous() const {
    if (is_contiguous()) return *this;                   // no copy needed
    Tensor out      = Tensor::empty(shape_, dtype_);
    const size_t es = dtype_size(dtype_);
    const uint8_t* src = storage_->data();
    uint8_t*       dst = out.storage_->data();
    const int64_t  n   = numel();
    std::vector<int64_t> coord(ndim(), 0);
    for (int64_t linear = 0; linear < n; ++linear) {
        int64_t src_elem = offset_;
        for (size_t d = 0; d < ndim(); ++d) src_elem += coord[d] * strides_[d];
        std::memcpy(dst + static_cast<size_t>(linear)   * es,
                    src + static_cast<size_t>(src_elem) * es, es);
        // odometer increment: last axis fastest
        for (int d = static_cast<int>(ndim()) - 1; d >= 0; --d) {
            if (++coord[d] < shape_[d]) break;
            coord[d] = 0;
        }
    }
    return out;
}

}  // namespace llm
