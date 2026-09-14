#include "test_framework.hpp"
#include "test_helpers.hpp"
#include "llm/flash.hpp"
#include "llm/model.hpp"
#include <vector>
#include <cmath>

// tests/test_flash.cpp
#include "test_framework.hpp"
#include "llm/flash.hpp"
#include <vector>
#include <cmath>
using namespace llm;

// Reference: full stable softmax over scores, then weighted sum of values.
static std::vector<float> ref_attend(const std::vector<float>& s,
                                     const std::vector<float>& v, int N, int D) {
    float m = s[0]; for (int j = 1; j < N; ++j) m = std::max(m, s[j]);
    std::vector<float> w(N); float sum = 0.0f;
    for (int j = 0; j < N; ++j) { w[j] = std::exp(s[j] - m); sum += w[j]; }
    std::vector<float> out(D, 0.0f);
    for (int j = 0; j < N; ++j) for (int d = 0; d < D; ++d) out[d] += (w[j] / sum) * v[j * D + d];
    return out;
}

TEST(online_softmax_matches_full_one_block) {
    const int N = 6, D = 4;
    std::vector<float> s = {0.5f, -1.0f, 2.0f, 0.0f, 1.5f, -0.5f}, v(N * D);
    for (int i = 0; i < N * D; ++i) v[i] = 0.1f * i - 0.3f;
    OnlineSoftmax st(D);
    online_update(st, s.data(), N, v.data(), D);
    std::vector<float> out(D); online_finalize(st, out.data(), D);
    auto ref = ref_attend(s, v, N, D);
    for (int d = 0; d < D; ++d) CHECK_CLOSE(out[d], ref[d], 1e-5);
}

TEST(online_softmax_block_invariant) {
    const int N = 7, D = 3;
    std::vector<float> s = {1, -2, 3, 0, 2, -1, 4}, v(N * D);
    for (int i = 0; i < N * D; ++i) v[i] = 0.2f * i;
    OnlineSoftmax st(D);
    online_update(st, s.data(),     4, v.data(),         D);   // block 1
    online_update(st, s.data() + 4, 3, v.data() + 4 * D, D);   // block 2
    std::vector<float> out(D); online_finalize(st, out.data(), D);
    auto ref = ref_attend(s, v, N, D);
    for (int d = 0; d < D; ++d) CHECK_CLOSE(out[d], ref[d], 1e-5);   // same as one block
}

TEST(online_softmax_stable_on_large_scores) {
    const int N = 4, D = 2;
    std::vector<float> s = {1000.0f, 999.0f, -1000.0f, 1000.0f}, v(N * D);
    for (int i = 0; i < N * D; ++i) v[i] = float(i + 1);
    OnlineSoftmax st(D);
    online_update(st, s.data(), 2, v.data(),         D);
    online_update(st, s.data() + 2, 2, v.data() + 2 * D, D);
    std::vector<float> out(D); online_finalize(st, out.data(), D);
    for (int d = 0; d < D; ++d) CHECK(std::isfinite(out[d]));        // no overflow
}

// add to tests/test_flash.cpp  (reuses randt, attn_weights from the model tests)
#include "llm/model.hpp"

TEST(flash_attention_matches_naive) {
    ModelConfig c; c.hidden_size = 16; c.n_heads = 4; c.n_kv_heads = 2; c.head_dim = 4;
    const LayerWeights w = attn_weights(c, 7);
    const int S = 9, hidden = c.hidden_size;
    Tensor x = randt({S, hidden}, 3);

    Tensor o_naive = Tensor::empty({S, hidden}, DType::F32);
    attention(o_naive, x, w, c, /*pos0=*/0);                  // Unit 5 reference

    Tensor o_flash = Tensor::empty({S, hidden}, DType::F32);
    attention_flash(o_flash, x, w, c, /*pos0=*/0, /*block=*/4);
    for (int i = 0; i < S * hidden; ++i)
        CHECK_CLOSE(o_naive.data_ptr<float>()[i], o_flash.data_ptr<float>()[i], 1e-4);
}

TEST(flash_attention_block_size_invariant) {
    ModelConfig c; c.hidden_size = 16; c.n_heads = 4; c.n_kv_heads = 4; c.head_dim = 4;
    const LayerWeights w = attn_weights(c, 12);
    const int S = 10, hidden = c.hidden_size;
    Tensor x = randt({S, hidden}, 6);

    Tensor a = Tensor::empty({S, hidden}, DType::F32);
    Tensor b = Tensor::empty({S, hidden}, DType::F32);
    attention_flash(a, x, w, c, 0, /*block=*/2);
    attention_flash(b, x, w, c, 0, /*block=*/8);
    for (int i = 0; i < S * hidden; ++i)
        CHECK_CLOSE(a.data_ptr<float>()[i], b.data_ptr<float>()[i], 1e-5);  // tiling is exact
}

// add to tests/test_flash.cpp  (reuses randt, attn_weights)
#include <cmath>

TEST(boss_flash_exact_at_scale) {
    ModelConfig c; c.hidden_size = 16; c.n_heads = 4; c.n_kv_heads = 2; c.head_dim = 4;
    const LayerWeights w = attn_weights(c, 21);
    const int hidden = c.hidden_size;
    for (int S : {1, 16, 64, 256}) {
        Tensor x = randt({S, hidden}, 100u + (uint32_t)S);
        Tensor o_naive = Tensor::empty({S, hidden}, DType::F32);
        Tensor o_flash = Tensor::empty({S, hidden}, DType::F32);
        attention(o_naive, x, w, c, /*pos0=*/0);
        attention_flash(o_flash, x, w, c, /*pos0=*/0, /*block=*/32);
        double max_abs = 0.0;
        for (int i = 0; i < S * hidden; ++i)
            max_abs = std::max(max_abs,
                      (double)std::fabs(o_naive.data_ptr<float>()[i] - o_flash.data_ptr<float>()[i]));
        CHECK(max_abs < 1e-3);                       // exact within fp noise at every scale
    }
}

TEST(boss_flash_block_size_invariant) {
    ModelConfig c; c.hidden_size = 16; c.n_heads = 4; c.n_kv_heads = 4; c.head_dim = 4;
    const LayerWeights w = attn_weights(c, 31);
    const int S = 128, hidden = c.hidden_size;
    Tensor x = randt({S, hidden}, 7);

    Tensor ref = Tensor::empty({S, hidden}, DType::F32);
    attention_flash(ref, x, w, c, 0, /*block=*/1);   // finest tiling as reference
    for (int blk : {4, 16, 64, 128}) {
        Tensor o = Tensor::empty({S, hidden}, DType::F32);
        attention_flash(o, x, w, c, 0, blk);
        for (int i = 0; i < S * hidden; ++i)
            CHECK_CLOSE(o.data_ptr<float>()[i], ref.data_ptr<float>()[i], 1e-4);
    }
}
