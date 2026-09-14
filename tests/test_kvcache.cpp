#include "test_framework.hpp"
#include "test_helpers.hpp"
#include "llm/model.hpp"
#include "llm/ops.hpp"
#include <vector>
#include <cmath>
#include <random>

// add to tests/test_kvcache.cpp
#include "test_framework.hpp"
#include "llm/model.hpp"
using namespace llm;

TEST(kvcache_allocates_per_layer) {
    ModelConfig c; c.n_layers = 3; c.n_kv_heads = 2; c.head_dim = 4;
    KVCache cache = KVCache::allocate(c, /*max_seq=*/16);
    CHECK((int)cache.k.size() == c.n_layers);
    CHECK((int)cache.v.size() == c.n_layers);
    CHECK(cache.k[0].shape()[0] == 16);                      // max_seq rows
    CHECK(cache.k[0].shape()[1] == c.n_kv_heads * c.head_dim);
    CHECK(cache.length == 0);
}

TEST(kvcache_append_and_advance) {
    ModelConfig c; c.n_layers = 1; c.n_kv_heads = 1; c.head_dim = 2;
    KVCache cache = KVCache::allocate(c, 8);
    const int kvd = c.n_kv_heads * c.head_dim;               // 2
    Tensor knew = Tensor::empty({2, kvd}, DType::F32);       // 2 new positions
    Tensor vnew = Tensor::empty({2, kvd}, DType::F32);
    float kk[4] = {1, 2, 3, 4}, vv[4] = {5, 6, 7, 8};
    for (int i = 0; i < 4; ++i) { knew.data_ptr<float>()[i] = kk[i]; vnew.data_ptr<float>()[i] = vv[i]; }

    cache.append_layer(0, knew, vnew, 2);
    cache.advance(2);
    CHECK(cache.length == 2);
    CHECK_CLOSE(cache.k[0].at<float>({0, 0}), 1.0f, 1e-6);   // first appended row
    CHECK_CLOSE(cache.k[0].at<float>({1, 1}), 4.0f, 1e-6);
    CHECK_CLOSE(cache.v[0].at<float>({1, 0}), 7.0f, 1e-6);

    cache.reset();
    CHECK(cache.length == 0);                                // memory reused for next sequence
}

// add to tests/test_kvcache.cpp  (reuses randt, attn_weights from the model tests)
#include <cstring>

static Tensor rows(const Tensor& x, int r0, int r1) {
    const int hidden = static_cast<int>(x.shape()[1]);
    Tensor s = Tensor::empty({r1 - r0, hidden}, DType::F32);
    std::memcpy(s.data_ptr<float>(), x.data_ptr<float>() + (int64_t)r0 * hidden,
                sizeof(float) * (size_t)(r1 - r0) * hidden);
    return s;
}

TEST(cached_attention_equals_full_when_prefilled_whole) {
    ModelConfig c; c.hidden_size = 8; c.n_heads = 2; c.n_kv_heads = 1; c.head_dim = 4; c.n_layers = 1;  // [BUG FIX] book test omits n_layers -> 0 layers -> segfault
    const LayerWeights w = attn_weights(c, 7);
    const int S = 5;
    Tensor x = randt({S, c.hidden_size}, 3);

    Tensor o_full = Tensor::empty({S, c.hidden_size}, DType::F32);
    attention(o_full, x, w, c, /*pos0=*/0);                 // Unit 5 reference

    KVCache cache = KVCache::allocate(c, 32);
    Tensor o_cached = Tensor::empty({S, c.hidden_size}, DType::F32);
    attention_cached(o_cached, x, w, c, cache, /*layer=*/0);
    cache.advance(S);
    for (int i = 0; i < S * c.hidden_size; ++i)
        CHECK_CLOSE(o_full.data_ptr<float>()[i], o_cached.data_ptr<float>()[i], 1e-4);
}

TEST(cached_attention_incremental_equals_full) {
    ModelConfig c; c.hidden_size = 8; c.n_heads = 2; c.n_kv_heads = 1; c.head_dim = 4; c.n_layers = 1;  // [BUG FIX] book test omits n_layers -> 0 layers -> segfault
    const LayerWeights w = attn_weights(c, 11);
    const int S = 5, hidden = c.hidden_size;
    Tensor x = randt({S, hidden}, 4);

    Tensor o_full = Tensor::empty({S, hidden}, DType::F32);
    attention(o_full, x, w, c, 0);

    KVCache cache = KVCache::allocate(c, 32);
    Tensor pre = rows(x, 0, 3);                              // prefill first 3 positions
    Tensor o_pre = Tensor::empty({3, hidden}, DType::F32);
    attention_cached(o_pre, pre, w, c, cache, 0); cache.advance(3);
    for (int i = 0; i < 3; ++i)
        for (int d = 0; d < hidden; ++d)
            CHECK_CLOSE(o_pre.at<float>({i, d}), o_full.at<float>({i, d}), 1e-4);

    for (int i = 3; i < S; ++i) {                            // decode the rest one at a time
        Tensor xi = rows(x, i, i + 1);
        Tensor oi = Tensor::empty({1, hidden}, DType::F32);
        attention_cached(oi, xi, w, c, cache, 0); cache.advance(1);
        for (int d = 0; d < hidden; ++d)
            CHECK_CLOSE(oi.at<float>({0, d}), o_full.at<float>({i, d}), 1e-4);
    }
}

// add to tests/test_kvcache.cpp
// (reuses randt, full_weights, tiny_model, tiny_cfg from the model tests)
#include <cmath>

TEST(boss_kv_cache_matches_full_recompute) {
    const ModelConfig c = tiny_cfg();
    const ModelWeights m = tiny_model(c, 4);
    const std::vector<int> seq = {3, 7, 1, 9, 2, 5, 8};
    const int S = static_cast<int>(seq.size()), V = c.vocab_size;

    Tensor full = forward(m, c, seq);                       // (A) full recompute

    KVCache cache = KVCache::allocate(c, /*max_seq=*/64);    // (B) prefill + decode
    const int P = 3;
    const std::vector<int> prompt(seq.begin(), seq.begin() + P);
    Tensor pre = forward_cached(m, c, prompt, cache);        // prefill -> [P, V]
    for (int i = 0; i < P; ++i)
        for (int t = 0; t < V; ++t)
            CHECK_CLOSE(pre.at<float>({i, t}), full.at<float>({i, t}), 1e-3);

    for (int i = P; i < S; ++i) {                            // decode one token at a time
        Tensor step = forward_cached(m, c, {seq[i]}, cache); // -> [1, V]
        for (int t = 0; t < V; ++t)
            CHECK_CLOSE(step.at<float>({0, t}), full.at<float>({i, t}), 1e-3);
    }
    CHECK(cache.length == S);                                // whole sequence now cached
}
