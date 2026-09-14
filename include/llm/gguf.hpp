// include/llm/gguf.hpp
#pragma once
#include <unordered_map>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include "llm/tensor.hpp"
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <unordered_map>
#include "llm/common.hpp"

namespace llm {

enum GGMLType : uint32_t { GGML_F32 = 0, GGML_F16 = 1, GGML_Q4_0 = 2, GGML_Q8_0 = 8 };

struct TensorInfo { std::vector<int64_t> dims; GGMLType type; uint64_t offset; };


enum GGUFType : uint32_t {
    GGUF_U8=0, GGUF_I8=1, GGUF_U16=2, GGUF_I16=3, GGUF_U32=4, GGUF_I32=5,
    GGUF_F32=6, GGUF_BOOL=7, GGUF_STRING=8, GGUF_ARRAY=9, GGUF_U64=10,
    GGUF_I64=11, GGUF_F64=12
};

struct MetaValue {
    GGUFType type = GGUF_U32;
    uint64_t u = 0;                 // any integer/bool (signed sign-extended)
    double   f = 0.0;               // any float
    std::string str;
    GGUFType arr_type = GGUF_U32;
    std::vector<uint64_t>    arr_u;
    std::vector<double>      arr_f;
    std::vector<std::string> arr_s;
};

struct ByteCursor {
    const uint8_t* p;
    const uint8_t* end;
    template <class T> T read() { T v; std::memcpy(&v, p, sizeof(T)); p += sizeof(T); return v; }
    std::string read_string() {
        const uint64_t n = read<uint64_t>();
        std::string s(reinterpret_cast<const char*>(p), reinterpret_cast<const char*>(p) + n);
        p += n; return s;
    }
};

inline void read_scalar(ByteCursor& c, GGUFType t, MetaValue& v) {
    switch (t) {
        case GGUF_U8:  v.u = c.read<uint8_t>();  break;
        case GGUF_I8:  v.u = (uint64_t)(int64_t)c.read<int8_t>();  break;
        case GGUF_U16: v.u = c.read<uint16_t>(); break;
        case GGUF_I16: v.u = (uint64_t)(int64_t)c.read<int16_t>(); break;
        case GGUF_U32: v.u = c.read<uint32_t>(); break;
        case GGUF_I32: v.u = (uint64_t)(int64_t)c.read<int32_t>(); break;
        case GGUF_U64: v.u = c.read<uint64_t>(); break;
        case GGUF_I64: v.u = (uint64_t)c.read<int64_t>(); break;
        case GGUF_F32: v.f = c.read<float>();    break;
        case GGUF_F64: v.f = c.read<double>();   break;
        case GGUF_BOOL:v.u = c.read<uint8_t>();  break;
        case GGUF_STRING: v.str = c.read_string(); break;
        default: break;
    }
}

inline MetaValue read_value(ByteCursor& c) {
    MetaValue v; v.type = static_cast<GGUFType>(c.read<uint32_t>());
    if (v.type == GGUF_ARRAY) {
        v.arr_type = static_cast<GGUFType>(c.read<uint32_t>());
        const uint64_t n = c.read<uint64_t>();
        for (uint64_t i = 0; i < n; ++i) {
            MetaValue e; read_scalar(c, v.arr_type, e);
            if (v.arr_type == GGUF_STRING)                       v.arr_s.push_back(e.str);
            else if (v.arr_type == GGUF_F32 || v.arr_type == GGUF_F64) v.arr_f.push_back(e.f);
            else                                                 v.arr_u.push_back(e.u);
        }
    } else {
        read_scalar(c, v.type, v);
    }
    return v;
}

class GGUFReader {
public:
    bool parse(const uint8_t* data, size_t size) {       // parse an in-memory blob
        ByteCursor c{data, data + size};
        if (c.read<uint32_t>() != 0x46554747u) return false;   // magic
        version_   = c.read<uint32_t>();
        n_tensors_ = c.read<uint64_t>();
        const uint64_t n_kv = c.read<uint64_t>();
        for (uint64_t i = 0; i < n_kv; ++i) {
            std::string key = c.read_string();
            meta_[key] = read_value(c);
        }
        cursor_after_meta_ = c.p;
        base_ = data;
        parse_tensors(c);                          // tensors continue here (Unit 10.2)
        return true;
    }

    uint32_t meta_u32(const std::string& k, uint32_t def = 0) const {
        auto it = meta_.find(k); return it == meta_.end() ? def : (uint32_t)it->second.u;
    }
    float meta_f32(const std::string& k, float def = 0.0f) const {
        auto it = meta_.find(k); return it == meta_.end() ? def : (float)it->second.f;
    }
    std::string meta_str(const std::string& k) const {
        auto it = meta_.find(k); return it == meta_.end() ? std::string() : it->second.str;
    }
    const std::vector<std::string>& meta_str_array(const std::string& k) const {
        static const std::vector<std::string> empty;
        auto it = meta_.find(k); return it == meta_.end() ? empty : it->second.arr_s;
    }

protected:
    std::unordered_map<std::string, MetaValue> meta_;
    const uint8_t* base_ = nullptr;
    const uint8_t* cursor_after_meta_ = nullptr;
    uint32_t version_ = 0;
    uint64_t n_tensors_ = 0;

public:
    void parse_tensors(ByteCursor& c) {
    for (uint64_t i = 0; i < n_tensors_; ++i) {
        std::string name = c.read_string();
        TensorInfo ti;
        const uint32_t nd = c.read<uint32_t>();
        for (uint32_t d = 0; d < nd; ++d) ti.dims.push_back((int64_t)c.read<uint64_t>());
        ti.type   = static_cast<GGMLType>(c.read<uint32_t>());
        ti.offset = c.read<uint64_t>();
        tensors_[name] = ti;
    }
    const uint64_t align = meta_u32("general.alignment", 32);
    const uint64_t pos   = (uint64_t)(c.p - base_);
    data_ = base_ + (pos + align - 1) / align * align;       // aligned data section
}

const TensorInfo* tensor(const std::string& n) const {
    auto it = tensors_.find(n); return it == tensors_.end() ? nullptr : &it->second;
}
const uint8_t* tensor_data(const std::string& n) const {
    auto it = tensors_.find(n); return it == tensors_.end() ? nullptr : data_ + it->second.offset;
}

bool load(const std::string& path) {
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) return false;
    struct stat st;
    if (::fstat(fd, &st) != 0) { ::close(fd); return false; }
    mapped_size_ = (size_t)st.st_size;
    void* m = ::mmap(nullptr, mapped_size_, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (m == MAP_FAILED) return false;
    mapped_ = static_cast<const uint8_t*>(m);
    return parse(mapped_, mapped_size_);             // weights now page in on demand
}
~GGUFReader() { if (mapped_) ::munmap(const_cast<uint8_t*>(mapped_), mapped_size_); }
private:
    std::unordered_map<std::string, TensorInfo> tensors_;
    const uint8_t* data_ = nullptr;
    const uint8_t* mapped_ = nullptr;
    size_t mapped_size_ = 0;

};


inline float f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1F, mant = h & 0x3FF, bits;
    if (exp == 0) {
        if (mant == 0) bits = sign;                          // +/- 0
        else { exp = 1; while ((mant & 0x400) == 0) { mant <<= 1; --exp; }
               mant &= 0x3FF; bits = sign | ((exp + 112) << 23) | (mant << 13); }
    } else if (exp == 0x1F) {
        bits = sign | 0x7F800000u | (mant << 13);            // inf / nan
    } else {
        bits = sign | ((exp + 112) << 23) | (mant << 13);    // 112 = 127 - 15
    }
    float f; std::memcpy(&f, &bits, sizeof(f)); return f;
}

inline void dequant_gguf_q8_0(const uint8_t* p, int64_t n, float* out) {
    for (int64_t b = 0; b < n / 32; ++b) {
        uint16_t dh; std::memcpy(&dh, p, 2); p += 2;
        const float d = f16_to_f32(dh);
        const int8_t* q = reinterpret_cast<const int8_t*>(p); p += 32;
        for (int i = 0; i < 32; ++i) out[b * 32 + i] = q[i] * d;
    }
}

inline void dequant_gguf_q4_0(const uint8_t* p, int64_t n, float* out) {
    for (int64_t b = 0; b < n / 32; ++b) {
        uint16_t dh; std::memcpy(&dh, p, 2); p += 2;
        const float d = f16_to_f32(dh);
        const uint8_t* q = p; p += 16;
        for (int i = 0; i < 16; ++i) {
            out[b * 32 + i]      = ((int)(q[i] & 0x0F) - 8) * d;
            out[b * 32 + i + 16] = ((int)(q[i] >> 4)   - 8) * d;
        }
    }
}

inline Tensor load_tensor_f32(const GGUFReader& r, const std::string& name) {
    const TensorInfo* ti = r.tensor(name);
    LLM_CHECK(ti != nullptr, "GGUF tensor not found: " + name);
    int64_t n = 1; for (int64_t d : ti->dims) n *= d;
    std::vector<int64_t> shape(ti->dims.rbegin(), ti->dims.rend());   // reverse to row-major
    Tensor t = Tensor::empty(shape, DType::F32);
    const uint8_t* data = r.tensor_data(name);
    float* out = t.data_ptr<float>();
    switch (ti->type) {
        case GGML_F32:  std::memcpy(out, data, sizeof(float) * n); break;
        case GGML_F16:  { auto* h = reinterpret_cast<const uint16_t*>(data);
                          for (int64_t i = 0; i < n; ++i) out[i] = f16_to_f32(h[i]); } break;
        case GGML_Q8_0: dequant_gguf_q8_0(data, n, out); break;
        case GGML_Q4_0: dequant_gguf_q4_0(data, n, out); break;
        default: LLM_CHECK(false, "unsupported ggml type");
    }
    return t;
}


// loader
ModelConfig load_config(const GGUFReader& r) {
    const std::string arch = r.meta_str("general.architecture");
    auto k = [&](const std::string& s){ return arch + "." + s; };
    ModelConfig c;
    c.hidden_size       = r.meta_u32(k("embedding_length"));
    c.n_layers          = r.meta_u32(k("block_count"));
    c.n_heads           = r.meta_u32(k("attention.head_count"));
    c.n_kv_heads        = r.meta_u32(k("attention.head_count_kv"), c.n_heads);
    c.intermediate_size = r.meta_u32(k("feed_forward_length"));
    c.rms_norm_eps      = r.meta_f32(k("attention.layer_norm_rms_epsilon"), 1e-5f);
    c.rope_theta        = r.meta_f32(k("rope.freq_base"), 10000.0f);
    c.head_dim          = (c.n_heads > 0) ? c.hidden_size / c.n_heads : 0;
    c.vocab_size        = (int)r.meta_str_array("tokenizer.ggml.tokens").size();
    return c;
}

ModelWeights load_model(const GGUFReader& r, const ModelConfig& cfg) {
    ModelWeights m;
    m.token_embedding = load_tensor_f32(r, "token_embd.weight");
    m.final_norm      = load_tensor_f32(r, "output_norm.weight");
    m.lm_head = r.tensor("output.weight") ? load_tensor_f32(r, "output.weight")
                                          : m.token_embedding;        // tied if absent
    m.layers.resize(cfg.n_layers);
    for (int i = 0; i < cfg.n_layers; ++i) {
        const std::string p = "blk." + std::to_string(i) + ".";
        LayerWeights& w = m.layers[i];
        w.attn_norm = load_tensor_f32(r, p + "attn_norm.weight");
        w.wq        = load_tensor_f32(r, p + "attn_q.weight");
        w.wk        = load_tensor_f32(r, p + "attn_k.weight");
        w.wv        = load_tensor_f32(r, p + "attn_v.weight");
        w.wo        = load_tensor_f32(r, p + "attn_output.weight");
        w.ffn_norm  = load_tensor_f32(r, p + "ffn_norm.weight");
        w.w_gate    = load_tensor_f32(r, p + "ffn_gate.weight");
        w.w_up      = load_tensor_f32(r, p + "ffn_up.weight");
        w.w_down    = load_tensor_f32(r, p + "ffn_down.weight");
    }
    return m;
}

}  // namespace llm