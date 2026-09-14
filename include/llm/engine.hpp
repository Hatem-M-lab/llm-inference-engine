// include/llm/engine.hpp
#pragma once
#include <string>
#include <vector>
#include "llm/model.hpp"
#include "llm/tokenizer.hpp"
#include "llm/sampler.hpp"      // generate, SamplingParams, RNG
#include "llm/threadpool.hpp"   // linear_mt is used inside the forward pass

namespace llm {

// The whole engine: prompt text in, generated text out.
inline std::string run_engine(const ModelWeights& model, const ModelConfig& cfg, const Tokenizer& tok,
                              const std::string& prompt, int max_new,
                              const SamplingParams& sp, RNG& rng) {
    std::vector<int> ids = tok.encode(prompt, /*add_bos=*/true);          // Unit 4
    std::vector<int> out = generate(model, cfg, ids, max_new, tok.eos_id(), sp, rng);  // Units 5-14
    return tok.decode(out);                                              // Unit 4
}

}  // namespace llm
