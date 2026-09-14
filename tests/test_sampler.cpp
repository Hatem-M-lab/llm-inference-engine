#include "test_framework.hpp"
#include "test_helpers.hpp"
#include "llm/sampler.hpp"
#include "llm/model.hpp"
#include <vector>
#include <cmath>

// tests/test_sampler.cpp
#include "test_framework.hpp"
#include "llm/sampler.hpp"
#include <vector>
#include <cmath>
using namespace llm;

TEST(argmax_picks_largest) {
    std::vector<float> l = {0.1f, 2.5f, -1.0f, 2.4f};
    CHECK(argmax(l.data(), (int)l.size()) == 1);
}

TEST(low_temperature_sharpens_toward_greedy) {
    std::vector<float> l = {1.0f, 2.0f, 3.0f};     // token 2 is the max
    std::vector<float> p(3);
    softmax_temperature(l.data(), p.data(), 3, 0.1f);   // very low T
    CHECK(p[2] > 0.99f);                           // nearly all mass on the argmax
    float sum = p[0] + p[1] + p[2];
    CHECK_CLOSE(sum, 1.0f, 1e-5);
}

TEST(sampling_is_seed_reproducible) {
    std::vector<float> l = {1.0f, 1.0f, 1.0f, 1.0f};
    std::vector<float> p(4); softmax_temperature(l.data(), p.data(), 4, 1.0f);
    RNG a(42), b(42);
    for (int i = 0; i < 10; ++i)
        CHECK(sample_categorical(p.data(), 4, a) == sample_categorical(p.data(), 4, b));
}

TEST(sampling_distribution_matches_probs) {
    std::vector<float> l = {std::log(0.6f), std::log(0.3f), std::log(0.1f)};  // probs 0.6/0.3/0.1
    std::vector<float> p(3); softmax_temperature(l.data(), p.data(), 3, 1.0f);
    int count[3] = {0, 0, 0}; RNG rng(7);
    const int N = 20000;
    for (int i = 0; i < N; ++i) count[sample_categorical(p.data(), 3, rng)]++;
    CHECK(std::fabs(count[0] / (float)N - 0.6f) < 0.03f);   // empirical ~ probabilities
    CHECK(std::fabs(count[1] / (float)N - 0.3f) < 0.03f);
}

// add to tests/test_sampler.cpp
#include <set>

TEST(top_k_restricts_to_k_tokens) {
    std::vector<float> l = {3.0f, 2.0f, 1.0f, 0.0f, -1.0f};   // tokens 0,1 are the top two
    SamplingParams p; p.temperature = 1.0f; p.top_k = 2;
    RNG rng(5); std::set<int> seen;
    for (int i = 0; i < 2000; ++i) seen.insert(sample_token(l.data(), 5, p, rng));
    for (int t : seen) CHECK(t == 0 || t == 1);               // never a token outside top-2
}

TEST(top_p_keeps_only_the_nucleus) {
    std::vector<float> l = {std::log(0.95f), std::log(0.03f), std::log(0.02f)};
    SamplingParams p; p.temperature = 1.0f; p.top_p = 0.9f;   // 0.95 >= 0.9 -> keep just token 0
    RNG rng(3);
    for (int i = 0; i < 1000; ++i) CHECK(sample_token(l.data(), 3, p, rng) == 0);
}

TEST(greedy_param_returns_argmax) {
    std::vector<float> l = {0.5f, 3.0f, 1.0f};
    SamplingParams p; p.greedy = true; RNG rng(0);
    CHECK(sample_token(l.data(), 3, p, rng) == 1);
}

// add to tests/test_sampler.cpp
// (reuses tiny_model/tiny_cfg from the model tests, forward/forward_cached, argmax)
#include "llm/model.hpp"

TEST(boss_generate_greedy_deterministic) {
    const ModelConfig c = tiny_cfg();
    const ModelWeights m = tiny_model(c, 4);
    SamplingParams sp; sp.greedy = true;
    const std::vector<int> prompt = {3, 7, 1};
    RNG r1(123), r2(123);
    auto a = generate(m, c, prompt, /*max_new=*/5, /*eos=*/-1, sp, r1);
    auto b = generate(m, c, prompt, 5, -1, sp, r2);
    CHECK(a == b);
    CHECK((int)a.size() == 5);                    // no EOS (eos=-1) -> runs to the cap
}

TEST(boss_generate_matches_full_recompute) {
    const ModelConfig c = tiny_cfg();
    const ModelWeights m = tiny_model(c, 7);
    SamplingParams sp; sp.greedy = true;
    const std::vector<int> prompt = {5, 2, 8};
    RNG rng(0);
    auto gen = generate(m, c, prompt, /*max_new=*/6, /*eos=*/-1, sp, rng);

    std::vector<int> seq = prompt, ref;           // reference: recompute + argmax each step
    const int V = c.vocab_size;
    for (int step = 0; step < 6; ++step) {
        Tensor lg = forward(m, c, seq);
        int nxt = argmax(lg.data_ptr<float>() + (int64_t)(seq.size() - 1) * V, V);
        ref.push_back(nxt); seq.push_back(nxt);
    }
    CHECK(gen == ref);                            // cache + loop == from-scratch reference
}

TEST(boss_generate_stops_at_eos) {
    const ModelConfig c = tiny_cfg();
    const ModelWeights m = tiny_model(c, 4);
    SamplingParams sp; sp.greedy = true;
    const std::vector<int> prompt = {3, 7, 1};
    RNG r0(0);
    auto full = generate(m, c, prompt, 5, /*eos=*/-1, sp, r0);
    REQUIRE(!full.empty());
    const int first = full[0];                    // make THAT token the EOS
    RNG r1(0);
    auto stopped = generate(m, c, prompt, 5, /*eos=*/first, sp, r1);
    CHECK(stopped.empty());                       // EOS first -> nothing emitted
}
