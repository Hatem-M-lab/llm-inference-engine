// include/llm/flash.hpp
#pragma once
#include <cmath>
#include <vector>
#include "llm/common.hpp"
#include "llm/model.hpp"   // [fix] LayerWeights, ModelConfig

namespace llm {

struct OnlineSoftmax {
    float m;                       // running max
    float l;                       // running denominator
    std::vector<float> acc;        // running weighted output [head_dim]
    explicit OnlineSoftmax(int head_dim)
        : m(-INFINITY), l(0.0f), acc(head_dim, 0.0f) {}
};

// Fold a block of `block` scores (and their [block, head_dim] values) into the state.
inline void online_update(OnlineSoftmax& st, const float* scores, int block,
                          const float* values, int head_dim) {
    float bmax = -INFINITY;
    for (int j = 0; j < block; ++j) bmax = std::max(bmax, scores[j]);
    const float new_m = std::max(st.m, bmax);
    const float corr  = std::exp(st.m - new_m);     // 0 on the first block (st.m = -inf)

    st.l *= corr;                                   // rescale old state to new_m
    for (int d = 0; d < head_dim; ++d) st.acc[d] *= corr;

    for (int j = 0; j < block; ++j) {               // add this block
        const float p = std::exp(scores[j] - new_m);
        st.l += p;
        const float* vj = values + (int64_t)j * head_dim;
        for (int d = 0; d < head_dim; ++d) st.acc[d] += p * vj[d];
    }
    st.m = new_m;
}

inline void online_finalize(const OnlineSoftmax& st, float* out, int head_dim) {
    const float inv = 1.0f / st.l;
    for (int d = 0; d < head_dim; ++d) out[d] = st.acc[d] * inv;
}


// flash.hpp / model.hpp
void flash_attention(float* O, const float* Q, const float* K, const float* V,
                     int seq, int n_heads, int n_kv_heads, int head_dim,
                     int pos0, float scale, int block);
void attention_flash(Tensor& out, const Tensor& x, const LayerWeights& w,
                     const ModelConfig& cfg, int pos0 = 0, int block = 64);

}  // namespace llm
