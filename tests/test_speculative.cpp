#include "test_framework.hpp"
#include "test_helpers.hpp"
#include "llm/speculative.hpp"
#include "llm/sampler.hpp"
#include "llm/model.hpp"
#include <vector>

// tests/test_speculative.cpp
// (reuses tiny_model/tiny_cfg from model tests, forward, argmax)
#include "test_framework.hpp"
#include "llm/speculative.hpp"
#include "llm/model.hpp"
#include "llm/sampler.hpp"
using namespace llm;

// the target's greedy continuation of a sequence
static int target_greedy(const ModelWeights& m, const ModelConfig& c, const std::vector<int>& s) {
    Tensor l = forward(m, c, s);
    return argmax(l.data_ptr<float>() + (int64_t)(s.size() - 1) * c.vocab_size, c.vocab_size);
}

TEST(spec_round_same_model_accepts_all) {
    const ModelConfig c = tiny_cfg();
    const ModelWeights m = tiny_model(c, 4);
    const std::vector<int> seq = {3, 7, 1};
    auto emit = speculative_round_greedy(m, c, m, c, seq, /*K=*/4);   // draft == target
    CHECK((int)emit.size() == 5);                                     // K accepted + bonus
}

TEST(spec_round_emits_target_greedy_regardless_of_draft) {
    const ModelConfig c = tiny_cfg();
    const ModelWeights target = tiny_model(c, 4);
    const ModelWeights draft  = tiny_model(c, 99);                    // a poor, different draft
    const std::vector<int> seq = {5, 2, 8};
    auto emit = speculative_round_greedy(target, c, draft, c, seq, 4);
    REQUIRE(!emit.empty());
    std::vector<int> s = seq;                                        // every emit == target greedy
    for (int tok : emit) { CHECK(tok == target_greedy(target, c, s)); s.push_back(tok); }
}

// tests/test_speculative.cpp  (continued; uses sampler.hpp)
#include "llm/sampler.hpp"
#include <cmath>

TEST(speculative_accept_emits_target_distribution) {
    const int n = 4;
    float p[4] = {0.4f, 0.3f, 0.2f, 0.1f};       // target
    float q[4] = {0.1f, 0.2f, 0.3f, 0.4f};       // draft -- deliberately very different
    int count[4] = {0, 0, 0, 0};
    RNG rng(1);
    const int N = 200000;
    for (int t = 0; t < N; ++t) {
        int x = sample_categorical(q, n, rng);   // draft draws from q
        int emit = speculative_accept(p, q, n, x, rng);
        count[emit]++;
    }
    for (int i = 0; i < n; ++i)
        CHECK(std::fabs(count[i] / (float)N - p[i]) < 0.02f);   // emitted ~ p, not q
}

TEST(speculative_accept_certain_when_p_equals_q) {
    const int n = 3;
    float p[3] = {0.5f, 0.3f, 0.2f}, q[3] = {0.5f, 0.3f, 0.2f};   // identical
    RNG rng(4);
    for (int t = 0; t < 1000; ++t) {
        int x = t % n;
        CHECK(speculative_accept(p, q, n, x, rng) == x);          // p==q -> always accept
    }
}

TEST(speculative_accept_probability_matches_min_ratio) {
    const int n = 3;
    float p[3] = {0.5f, 0.3f, 0.2f}, q[3] = {0.25f, 0.5f, 0.25f};
    // token 1: p/q = 0.6; residual mass at 1 is max(0,0.3-0.5)=0, so emit==1 only if accepted
    int got1 = 0; RNG rng(2); const int N = 50000;
    for (int t = 0; t < N; ++t) if (speculative_accept(p, q, n, 1, rng) == 1) got1++;
    CHECK(std::fabs(got1 / (float)N - 0.6f) < 0.02f);            // accept prob = min(1, 0.6)
}

// add to tests/test_speculative.cpp  (reuses tiny_model/tiny_cfg, generate)

TEST(boss_spec_greedy_matches_plain_greedy) {
    const ModelConfig c = tiny_cfg();
    const ModelWeights target = tiny_model(c, 4);
    const ModelWeights draft  = tiny_model(c, 99);    // a poor, unrelated draft
    const std::vector<int> prompt = {3, 7, 1};

    SamplingParams sp; sp.greedy = true; RNG rng(0);
    auto plain = generate(target, c, prompt, /*max_new=*/8, /*eos=*/-1, sp, rng);
    auto spec  = spec_generate_greedy(target, c, draft, c, prompt, 8, /*K=*/4, /*eos=*/-1, nullptr);
    CHECK(spec == plain);                             // identical, whatever the draft
}

TEST(boss_spec_perfect_draft_accelerates) {
    const ModelConfig c = tiny_cfg();
    const ModelWeights m = tiny_model(c, 4);
    const std::vector<int> prompt = {3, 7, 1};
    int passes = 0;
    auto out = spec_generate_greedy(m, c, m, c, prompt, /*max_new=*/12, /*K=*/4, /*eos=*/-1, &passes);
    REQUIRE(!out.empty());
    REQUIRE(passes > 0);
    CHECK((double)out.size() / passes > 1.5);         // > 1 token per target pass
}
