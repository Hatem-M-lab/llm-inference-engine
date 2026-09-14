// include/llm/speculative.hpp
#pragma once
#include <vector>
#include "llm/model.hpp"
#include "llm/sampler.hpp"   // argmax
#include "llm/sampler.hpp"   // RNG
#include <algorithm>

namespace llm {

// One greedy speculative round: returns the tokens to emit (accepted drafts + one correction).
inline std::vector<int> speculative_round_greedy(
        const ModelWeights& target, const ModelConfig& tc,
        const ModelWeights& draft,  const ModelConfig& dc,
        const std::vector<int>& seq, int K) {
    // 1. draft K tokens greedily from the draft model
    std::vector<int> d, dseq = seq;
    for (int j = 0; j < K; ++j) {
        Tensor dl = forward(draft, dc, dseq);
        int dt = argmax(dl.data_ptr<float>() + (int64_t)(dseq.size() - 1) * dc.vocab_size,
                        dc.vocab_size);
        d.push_back(dt); dseq.push_back(dt);
    }
    // 2. verify all K in ONE target forward pass
    std::vector<int> cand = seq;
    cand.insert(cand.end(), d.begin(), d.end());
    Tensor tl = forward(target, tc, cand);
    const int V = tc.vocab_size, L = (int)seq.size();

    // 3. accept the matching prefix; one correction (or bonus) at the end
    int a = 0, correction = 0;
    for (a = 0; a < K; ++a) {
        int ti = argmax(tl.data_ptr<float>() + (int64_t)(L - 1 + a) * V, V);
        if (d[a] != ti) { correction = ti; break; }              // first mismatch
    }
    if (a == K)                                                  // all accepted -> bonus token
        correction = argmax(tl.data_ptr<float>() + (int64_t)(L - 1 + K) * V, V);

    std::vector<int> emit(d.begin(), d.begin() + a);
    emit.push_back(correction);
    return emit;
}


// speculative.hpp

// Accept draft token x (drawn from q) against target p, or resample from the residual.
// The returned token is distributed exactly as p.
inline int speculative_accept(const float* p, const float* q, int n, int x, RNG& rng) {
    const float a = (q[x] > 0.0f) ? std::min(1.0f, p[x] / q[x]) : 1.0f;
    if (rng.next_float() < a) return x;                          // accept

    float sum = 0.0f;                                            // residual (p - q)_+
    for (int i = 0; i < n; ++i) sum += std::max(0.0f, p[i] - q[i]);
    if (sum <= 0.0f) return x;                                   // degenerate guard
    float r = rng.next_float() * sum, c = 0.0f;
    for (int i = 0; i < n; ++i) {
        c += std::max(0.0f, p[i] - q[i]);
        if (r < c) return i;
    }
    return n - 1;
}


// speculative.hpp
inline std::vector<int> spec_generate_greedy(
        const ModelWeights& target, const ModelConfig& tc,
        const ModelWeights& draft,  const ModelConfig& dc,
        const std::vector<int>& prompt, int max_new, int K, int eos_id,
        int* target_passes) {
    std::vector<int> seq = prompt, out;
    int passes = 0; bool done = false;

    while (!done && (int)out.size() < max_new) {
        std::vector<int> emit = speculative_round_greedy(target, tc, draft, dc, seq, K);
        ++passes;                                     // one target forward pass per round
        for (int tok : emit) {
            if (tok == eos_id) { done = true; break; }
            out.push_back(tok); seq.push_back(tok);
            if ((int)out.size() >= max_new) { done = true; break; }
        }
    }
    if (target_passes) *target_passes = passes;
    return out;
}

}  // namespace llm
