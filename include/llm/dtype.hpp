// include/llm/dtype.hpp
#pragma once
#include <cstddef>
#include <cstdint>
#include "llm/common.hpp"

namespace llm {

// Dense element types for now. Block-quantized types (Q8_0, Q4_0, ...) arrive in
// Unit 9 and need special handling: their bytes-per-element is per block, not
// per scalar, so they don't fit this simple size model.
enum class DType : uint8_t { F32, F16, I8 };

inline size_t dtype_size(DType dt) {
    switch (dt) {
        case DType::F32: return 4;
        case DType::F16: return 2;
        case DType::I8:  return 1;
    }
    LLM_UNREACHABLE();
}

inline const char* dtype_name(DType dt) {
    switch (dt) {
        case DType::F32: return "f32";
        case DType::F16: return "f16";
        case DType::I8:  return "i8";
    }
    LLM_UNREACHABLE();
}

}  // namespace llm
