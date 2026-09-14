// tests/test_model.cpp
#include "test_framework.hpp"
#include <cstring>   // [fix] std::memcpy used in tests
#include "llm/model.hpp"
using namespace llm;

TEST(embed_gathers_rows) {
    const int vocab = 4, hidden = 3;
    Tensor emb = Tensor::empty({vocab, hidden}, DType::F32);
    for (int i = 0; i < vocab * hidden; ++i) emb.data_ptr<float>()[i] = float(i);
    // row r is [3r, 3r+1, 3r+2]
    std::vector<int> ids = {2, 0, 3};
    Tensor out = Tensor::empty({(int64_t)ids.size(), hidden}, DType::F32);
    embed_tokens(out, ids, emb);
    CHECK_CLOSE(out.at<float>({0, 0}), 6.0f, 1e-6);   // token 2 -> row 2 -> [6,7,8]
    CHECK_CLOSE(out.at<float>({0, 2}), 8.0f, 1e-6);
    CHECK_CLOSE(out.at<float>({1, 0}), 0.0f, 1e-6);   // token 0 -> [0,1,2]
    CHECK_CLOSE(out.at<float>({2, 1}), 10.0f, 1e-6);  // token 3 -> [9,10,11]
}

TEST(linear_matches_dot_products) {
    const int S = 2, K = 3, N = 2;
    Tensor x = Tensor::empty({S, K}, DType::F32);
    Tensor w = Tensor::empty({N, K}, DType::F32);     // [out, in]
    float xs[6] = {1, 2, 3,  4, 5, 6};
    float ws[6] = {1, 0, -1, 2, 2, 2};                // w row0=(1,0,-1), row1=(2,2,2)
    for (int i = 0; i < 6; ++i) { x.data_ptr<float>()[i] = xs[i]; w.data_ptr<float>()[i] = ws[i]; }
    Tensor y = Tensor::empty({S, N}, DType::F32);
    linear(y, x, w);
    CHECK_CLOSE(y.at<float>({0, 0}), 1*1 + 2*0 + 3*-1, 1e-5);   // -2
    CHECK_CLOSE(y.at<float>({0, 1}), 1*2 + 2*2 + 3*2,  1e-5);   // 12
    CHECK_CLOSE(y.at<float>({1, 0}), 4*1 + 5*0 + 6*-1, 1e-5);   // -2
}

// add to tests/test_model.cpp
#include <cmath>

TEST(rope_at_position_zero_is_identity) {
    const int hd = 4;
    Tensor x = Tensor::empty({1, 1, hd}, DType::F32);
    const float orig[hd] = {0.3f, -0.7f, 1.1f, 0.5f};
    for (int i = 0; i < hd; ++i) x.data_ptr<float>()[i] = orig[i];
    apply_rope(x, /*seq=*/1, /*heads=*/1, hd, /*theta=*/10000.0f, /*pos0=*/0);
    for (int i = 0; i < hd; ++i) CHECK_CLOSE(x.data_ptr<float>()[i], orig[i], 1e-6);
}

TEST(rope_preserves_norm) {
    const int hd = 8;
    Tensor x = Tensor::empty({1, 1, hd}, DType::F32);
    float before = 0.0f;
    for (int i = 0; i < hd; ++i) { float v = 0.1f * i - 0.3f; x.data_ptr<float>()[i] = v; before += v * v; }
    apply_rope(x, 1, 1, hd, 10000.0f, /*pos0=*/5);
    float after = 0.0f;
    for (int i = 0; i < hd; ++i) { float v = x.data_ptr<float>()[i]; after += v * v; }
    CHECK_CLOSE(std::sqrt(after), std::sqrt(before), 1e-5);   // rotation is norm-preserving
}

TEST(rope_dot_depends_on_relative_position) {
    // One plane (hd=2): inv_freq[0] = theta^0 = 1, so angle = pos. q=k=(1,0).
    // After RoPE, q_m . k_n = cos(m)cos(n)+sin(m)sin(n) = cos(m-n).
    const int hd = 2;
    auto roped = [&](int pos) {
        Tensor x = Tensor::empty({1, 1, hd}, DType::F32);
        x.data_ptr<float>()[0] = 1.0f; x.data_ptr<float>()[1] = 0.0f;
        apply_rope(x, 1, 1, hd, 10000.0f, pos);
        return std::pair<float,float>(x.data_ptr<float>()[0], x.data_ptr<float>()[1]);
    };
    const auto qm = roped(5), kn = roped(2);                 // m - n = 3
    const float dot = qm.first * kn.first + qm.second * kn.second;
    CHECK_CLOSE(dot, std::cos(3.0f), 1e-5);
}

// add to tests/test_model.cpp
#include <cstdint>
#include <cmath>

static Tensor randt(std::initializer_list<int64_t> shape, uint32_t seed) {
    Tensor t = Tensor::empty(shape, DType::F32);
    uint32_t s = seed; float* p = t.data_ptr<float>();
    for (int64_t i = 0; i < t.numel(); ++i) {
        s = s * 1664525u + 1013904223u;
        p[i] = (s >> 9) * (1.0f / 8388608.0f) - 0.5f;     // ~ [-0.5, 0.5)
    }
    return t;
}
static LayerWeights attn_weights(const ModelConfig& c, uint32_t seed) {
    LayerWeights w;
    const int qd = c.n_heads * c.head_dim, kvd = c.n_kv_heads * c.head_dim;
    w.wq = randt({qd,  c.hidden_size}, seed + 1);
    w.wk = randt({kvd, c.hidden_size}, seed + 2);
    w.wv = randt({kvd, c.hidden_size}, seed + 3);
    w.wo = randt({c.hidden_size, qd},  seed + 4);
    return w;
}

TEST(attention_output_shape) {
    ModelConfig c; c.hidden_size = 8; c.n_heads = 2; c.n_kv_heads = 1; c.head_dim = 4;
    const int S = 3;
    Tensor x = randt({S, c.hidden_size}, 10);
    Tensor o = Tensor::empty({S, c.hidden_size}, DType::F32);
    attention(o, x, attn_weights(c, 1), c, /*pos0=*/0);
    CHECK(o.shape()[0] == S && o.shape()[1] == c.hidden_size);
}

TEST(attention_is_causal) {
    ModelConfig c; c.hidden_size = 8; c.n_heads = 2; c.n_kv_heads = 1; c.head_dim = 4;
    const LayerWeights w = attn_weights(c, 7);
    const int S = 5, hidden = c.hidden_size;
    Tensor x = randt({S, hidden}, 3);

    Tensor o1 = Tensor::empty({S, hidden}, DType::F32);
    attention(o1, x, w, c, 0);

    Tensor x2 = randt({S, hidden}, 3);                    // identical copy...
    for (int d = 0; d < hidden; ++d) x2.data_ptr<float>()[3 * hidden + d] += 1.0f;  // ...perturb pos 3
    Tensor o2 = Tensor::empty({S, hidden}, DType::F32);
    attention(o2, x2, w, c, 0);

    for (int i = 0; i < 3; ++i)                            // positions < 3 unchanged
        for (int d = 0; d < hidden; ++d)
            CHECK_CLOSE(o1.at<float>({i, d}), o2.at<float>({i, d}), 1e-5);

    bool changed = false;                                 // position 4 (attends to 3) changes
    for (int d = 0; d < hidden; ++d)
        if (std::fabs(o1.at<float>({4, d}) - o2.at<float>({4, d})) > 1e-4f) changed = true;
    CHECK(changed);
}

// add to tests/test_model.cpp  (reuses randt, attn_weights)

static LayerWeights full_weights(const ModelConfig& c, uint32_t seed) {
    LayerWeights w = attn_weights(c, seed);
    w.attn_norm = randt({c.hidden_size}, seed + 10);
    w.ffn_norm  = randt({c.hidden_size}, seed + 11);
    w.w_gate = randt({c.intermediate_size, c.hidden_size}, seed + 12);
    w.w_up   = randt({c.intermediate_size, c.hidden_size}, seed + 13);
    w.w_down = randt({c.hidden_size, c.intermediate_size}, seed + 14);
    return w;
}

TEST(block_preserves_shape) {
    ModelConfig c; c.hidden_size = 8; c.n_heads = 2; c.n_kv_heads = 1;
    c.head_dim = 4; c.intermediate_size = 16;
    const int S = 4;
    Tensor x = randt({S, c.hidden_size}, 21);
    transformer_block(x, full_weights(c, 1), c, /*pos0=*/0);
    CHECK(x.shape()[0] == S && x.shape()[1] == c.hidden_size);
}

TEST(block_is_identity_when_outputs_zeroed) {
    ModelConfig c; c.hidden_size = 8; c.n_heads = 2; c.n_kv_heads = 2;
    c.head_dim = 4; c.intermediate_size = 16;
    LayerWeights w = full_weights(c, 5);
    // Zero the two output projections -> each sub-layer contributes exactly 0.
    for (int i = 0; i < w.wo.numel();     ++i) w.wo.data_ptr<float>()[i]     = 0.0f;
    for (int i = 0; i < w.w_down.numel(); ++i) w.w_down.data_ptr<float>()[i] = 0.0f;

    const int S = 4, hidden = c.hidden_size;
    Tensor x  = randt({S, hidden}, 9);
    Tensor x0 = Tensor::empty({S, hidden}, DType::F32);
    std::memcpy(x0.data_ptr<float>(), x.data_ptr<float>(), sizeof(float) * S * hidden);

    transformer_block(x, w, c, 0);
    for (int i = 0; i < S * hidden; ++i)
        CHECK_CLOSE(x.data_ptr<float>()[i], x0.data_ptr<float>()[i], 1e-5);  // residual identity
}

// add to tests/test_model.cpp  (reuses randt, full_weights)
#include <cmath>

static ModelWeights tiny_model(const ModelConfig& c, uint32_t seed) {
    ModelWeights m;
    m.token_embedding = randt({c.vocab_size, c.hidden_size}, seed);
    m.final_norm      = randt({c.hidden_size}, seed + 100);
    m.lm_head         = randt({c.vocab_size, c.hidden_size}, seed + 200);
    for (int l = 0; l < c.n_layers; ++l)
        m.layers.push_back(full_weights(c, seed + 1000u * (l + 1)));
    return m;
}
static ModelConfig tiny_cfg() {
    ModelConfig c; c.vocab_size = 32; c.hidden_size = 8; c.n_layers = 2;
    c.n_heads = 2; c.n_kv_heads = 1; c.head_dim = 4; c.intermediate_size = 16;
    return c;
}

TEST(boss_forward_shape_and_finite) {
    const ModelConfig c = tiny_cfg();
    const ModelWeights m = tiny_model(c, 1);
    const std::vector<int> ids = {3, 7, 1, 9, 2};
    Tensor logits = forward(m, c, ids);
    REQUIRE(logits.shape()[0] == static_cast<int>(ids.size()));
    REQUIRE(logits.shape()[1] == c.vocab_size);
    for (int i = 0; i < logits.numel(); ++i)
        CHECK(std::isfinite(logits.data_ptr<float>()[i]));
}

TEST(boss_forward_is_causal_end_to_end) {
    const ModelConfig c = tiny_cfg();
    const ModelWeights m = tiny_model(c, 4);
    const std::vector<int> a = {5, 2, 8, 1, 6};
    const std::vector<int> b = {5, 2, 8, 4, 6};       // differs ONLY at position 3
    Tensor L1 = forward(m, c, a);
    Tensor L2 = forward(m, c, b);
    const int V = c.vocab_size;

    for (int i = 0; i < 3; ++i)                        // positions < 3: bit-identical
        for (int t = 0; t < V; ++t)
            CHECK_CLOSE(L1.at<float>({i, t}), L2.at<float>({i, t}), 1e-4);

    bool changed = false;                             // position 3: must differ
    for (int t = 0; t < V; ++t)
        if (std::fabs(L1.at<float>({3, t}) - L2.at<float>({3, t})) > 1e-4f) changed = true;
    CHECK(changed);
}

TEST(boss_forward_is_deterministic) {
    const ModelConfig c = tiny_cfg();
    const ModelWeights m = tiny_model(c, 2);
    const std::vector<int> ids = {1, 2, 3};
    Tensor x = forward(m, c, ids);
    Tensor y = forward(m, c, ids);
    for (int i = 0; i < x.numel(); ++i)
        CHECK_CLOSE(x.data_ptr<float>()[i], y.data_ptr<float>()[i], 0.0f);  // exact
}
