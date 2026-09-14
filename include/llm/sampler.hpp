// include/llm/sampler.hpp
#pragma once
#include <cstdint>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include "llm/model.hpp"

namespace llm {

struct RNG {
    uint64_t s;
    explicit RNG(uint64_t seed = 0x2545F4914F6CDD1DULL) : s(seed ? seed : 1) {}
    uint32_t next_u32() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return (uint32_t)(s >> 32); }
    float    next_float() { return (next_u32() >> 8) * (1.0f / 16777216.0f); }   // [0, 1)
};

inline int argmax(const float* logits, int n) {
    int best = 0;
    for (int i = 1; i < n; ++i) if (logits[i] > logits[best]) best = i;
    return best;
}

inline void softmax_temperature(const float* logits, float* probs, int n, float temp) {
    const float inv_t = 1.0f / temp;
    float m = -INFINITY;
    for (int i = 0; i < n; ++i) m = std::max(m, logits[i] * inv_t);
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) { float e = std::exp(logits[i] * inv_t - m); probs[i] = e; sum += e; }
    const float is = 1.0f / sum;
    for (int i = 0; i < n; ++i) probs[i] *= is;
}

inline int sample_categorical(const float* probs, int n, RNG& rng) {
    const float r = rng.next_float();
    float cum = 0.0f;
    for (int i = 0; i < n; ++i) { cum += probs[i]; if (r < cum) return i; }
    return n - 1;                                    // rounding safety net
}


// sampler.hpp

struct SamplingParams {
    float temperature = 1.0f;
    int   top_k       = 0;        // 0 = disabled
    float top_p       = 1.0f;     // 1 = disabled
    bool  greedy      = false;
};

inline int sample_token(const float* logits, int n, const SamplingParams& sp, RNG& rng) {
    if (sp.greedy || sp.temperature <= 0.0f) return argmax(logits, n);

    std::vector<float> probs(n);
    softmax_temperature(logits, probs.data(), n, sp.temperature);

    const bool truncate = (sp.top_k > 0 && sp.top_k < n) || (sp.top_p < 1.0f);
    if (!truncate) return sample_categorical(probs.data(), n, rng);

    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int a, int b){ return probs[a] > probs[b]; });

    int keep = n;
    if (sp.top_k > 0) keep = std::min(keep, sp.top_k);          // top-k cap
    if (sp.top_p < 1.0f) {                                      // top-p nucleus
        float cum = 0.0f; int np = 0;
        for (int i = 0; i < keep; ++i) { cum += probs[idx[i]]; ++np; if (cum >= sp.top_p) break; }
        keep = np;
    }

    float sum = 0.0f;
    for (int i = 0; i < keep; ++i) sum += probs[idx[i]];        // mass of the kept set
    float r = rng.next_float() * sum, c = 0.0f;
    for (int i = 0; i < keep; ++i) { c += probs[idx[i]]; if (r < c) return idx[i]; }
    return idx[keep - 1];
}


// generation

inline std::vector<int> generate(const ModelWeights& model, const ModelConfig& cfg,
                                 const std::vector<int>& prompt, int max_new,
                                 int eos_id, const SamplingParams& sp, RNG& rng) {
    const int V = cfg.vocab_size;
    KVCache cache = KVCache::allocate(cfg, (int)prompt.size() + max_new + 8);

    Tensor logits = forward_cached(model, cfg, prompt, cache);          // prefill
    int next = sample_token(logits.data_ptr<float>() + (int64_t)(prompt.size() - 1) * V,
                            V, sp, rng);                                 // from the LAST position

    std::vector<int> out;
    for (int step = 0; step < max_new; ++step) {
        if (next == eos_id) break;                                      // stop, do not emit EOS
        out.push_back(next);
        Tensor l = forward_cached(model, cfg, {next}, cache);           // one cached decode step
        next = sample_token(l.data_ptr<float>(), V, sp, rng);
    }
    return out;
}

}  // namespace llm
