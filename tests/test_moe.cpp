#include "test_framework.hpp"
#include "llm/model.hpp"
#include "llm/ops.hpp"
#include <vector>
#include <cmath>
#include <random>

// add to tests/test_moe.cpp
#include "test_framework.hpp"
#include "llm/model.hpp"
#include <cmath>
using namespace llm;

TEST(router_selects_top_k) {
    const int E = 4, hidden = 1, K = 2;
    // router_w is [E, hidden]; with hidden=1 the logit for expert e is x[0]*router_w[e].
    Tensor rw = Tensor::empty({E, hidden}, DType::F32);
    float w[E] = {0.1f, 0.9f, 0.4f, 0.2f};            // expert 1 highest, then 2
    for (int e = 0; e < E; ++e) rw.data_ptr<float>()[e] = w[e];
    Tensor x = Tensor::empty({1, hidden}, DType::F32);
    x.data_ptr<float>()[0] = 1.0f;                    // logits == w
    Routing r = route(x, rw, E, K);
    REQUIRE((int)r.experts.size() == K);
    CHECK(r.experts[0] == 1);                         // top is expert 1
    CHECK(r.experts[1] == 2);                         // then expert 2
}

TEST(router_weights_sum_to_one) {
    const int E = 8, hidden = 4, K = 2;
    Tensor rw = Tensor::empty({E, hidden}, DType::F32);
    uint32_t s = 5; for (int i = 0; i < E * hidden; ++i) { s = s*1664525u+1013904223u; rw.data_ptr<float>()[i] = (s>>9)*(1.0f/8388608.0f)-0.5f; }
    Tensor x = Tensor::empty({1, hidden}, DType::F32);
    for (int i = 0; i < hidden; ++i) x.data_ptr<float>()[i] = 0.3f * i - 0.2f;
    Routing r = route(x, rw, E, K);
    float sum = 0.0f; for (int j = 0; j < K; ++j) sum += r.weights[j];
    CHECK_CLOSE(sum, 1.0f, 1e-6);                     // softmax over the chosen k
}

// add to tests/test_moe.cpp
#include <cstdint>

static Tensor randt(std::initializer_list<int64_t> shape, uint32_t seed) {
    Tensor t = Tensor::empty(shape, DType::F32);
    uint32_t s = seed; float* p = t.data_ptr<float>();
    for (int64_t i = 0; i < t.numel(); ++i) { s = s*1664525u+1013904223u; p[i] = (s>>9)*(1.0f/8388608.0f)-0.5f; }
    return t;
}

TEST(moe_output_shape) {
    ModelConfig c; c.hidden_size = 8; c.intermediate_size = 16; c.num_experts = 4; c.top_k = 2;
    MoELayerWeights w;
    w.router = randt({c.num_experts, c.hidden_size}, 1);
    for (int e = 0; e < c.num_experts; ++e)
        w.experts.push_back({ randt({c.intermediate_size, c.hidden_size}, 10+e),
                              randt({c.intermediate_size, c.hidden_size}, 20+e),
                              randt({c.hidden_size, c.intermediate_size}, 30+e) });
    const int S = 5;
    Tensor x = randt({S, c.hidden_size}, 7);
    Tensor o = Tensor::empty({S, c.hidden_size}, DType::F32);
    moe_ffn(o, x, w, c);
    CHECK(o.shape()[0] == S && o.shape()[1] == c.hidden_size);
}

TEST(moe_one_expert_equals_dense_ffn) {
    ModelConfig c; c.hidden_size = 8; c.intermediate_size = 16; c.num_experts = 1; c.top_k = 1;
    ExpertWeights e{ randt({c.intermediate_size, c.hidden_size}, 2),
                     randt({c.intermediate_size, c.hidden_size}, 3),
                     randt({c.hidden_size, c.intermediate_size}, 4) };
    MoELayerWeights mw; mw.router = randt({1, c.hidden_size}, 1); mw.experts.push_back(e);
    LayerWeights dw; dw.w_gate = e.w_gate; dw.w_up = e.w_up; dw.w_down = e.w_down;

    const int S = 3;
    Tensor x = randt({S, c.hidden_size}, 9);
    Tensor o_moe   = Tensor::empty({S, c.hidden_size}, DType::F32);
    Tensor o_dense = Tensor::empty({S, c.hidden_size}, DType::F32);
    moe_ffn(o_moe, x, mw, c);          // 1 expert, weight softmax([l]) = 1
    ffn(o_dense, x, dw, c);            // dense SwiGLU FFN from Unit 5
    for (int i = 0; i < S * c.hidden_size; ++i)
        CHECK_CLOSE(o_moe.data_ptr<float>()[i], o_dense.data_ptr<float>()[i], 1e-4);
}

// add to tests/test_moe.cpp  (reuses randt)
#include <set>
#include <cstring>

static Tensor zeros_like(const Tensor& t) {
    Tensor z = Tensor::empty(t.shape(), t.dtype());
    std::memset(z.data_ptr<float>(), 0, sizeof(float) * t.numel());
    return z;
}
static MoELayerWeights moe_weights(const ModelConfig& c, uint32_t seed) {
    MoELayerWeights w;
    w.router = randt({c.num_experts, c.hidden_size}, seed);
    for (int e = 0; e < c.num_experts; ++e)
        w.experts.push_back({ randt({c.intermediate_size, c.hidden_size}, seed + 10 + e),
                              randt({c.intermediate_size, c.hidden_size}, seed + 50 + e),
                              randt({c.hidden_size, c.intermediate_size}, seed + 90 + e) });
    return w;
}

TEST(boss_moe_grouped_matches_naive) {
    ModelConfig c; c.hidden_size = 8; c.intermediate_size = 16; c.num_experts = 6; c.top_k = 2;
    const MoELayerWeights w = moe_weights(c, 3);
    const int S = 7;
    Tensor x  = randt({S, c.hidden_size}, 5);
    Tensor o1 = Tensor::empty({S, c.hidden_size}, DType::F32);
    Tensor o2 = Tensor::empty({S, c.hidden_size}, DType::F32);
    moe_ffn(o1, x, w, c);
    moe_ffn_grouped(o2, x, w, c);
    for (int i = 0; i < S * c.hidden_size; ++i)
        CHECK_CLOSE(o1.data_ptr<float>()[i], o2.data_ptr<float>()[i], 1e-4);
}

TEST(boss_moe_routing_is_sparse) {
    ModelConfig c; c.hidden_size = 8; c.intermediate_size = 16; c.num_experts = 6; c.top_k = 2;
    const MoELayerWeights w = moe_weights(c, 8);
    const int S = 4, hidden = c.hidden_size;
    Tensor x = randt({S, hidden}, 2);

    Tensor base = Tensor::empty({S, hidden}, DType::F32);
    moe_ffn(base, x, w, c);

    const Routing r = route(x, w.router, c.num_experts, c.top_k);
    std::set<int> routed0;
    for (int j = 0; j < c.top_k; ++j) routed0.insert(r.experts[j]);   // token 0's experts
    int non = -1; for (int e = 0; e < c.num_experts; ++e) if (!routed0.count(e)) { non = e; break; }
    const int rt = *routed0.begin();
    REQUIRE(non >= 0);

    auto kill = [&](int e) {                          // copy weights, zero expert e
        MoELayerWeights w2 = w;
        w2.experts[e].w_gate = zeros_like(w.experts[e].w_gate);
        w2.experts[e].w_up   = zeros_like(w.experts[e].w_up);
        w2.experts[e].w_down = zeros_like(w.experts[e].w_down);
        Tensor o = Tensor::empty({S, hidden}, DType::F32);
        moe_ffn(o, x, w2, c);
        return o;
    };

    Tensor o_non = kill(non);                          // non-routed: token 0 unchanged
    for (int h = 0; h < hidden; ++h)
        CHECK_CLOSE(base.at<float>({0, h}), o_non.at<float>({0, h}), 1e-5);

    Tensor o_rt = kill(rt);                            // routed: token 0 changes
    bool changed = false;
    for (int h = 0; h < hidden; ++h)
        if (std::fabs(base.at<float>({0, h}) - o_rt.at<float>({0, h})) > 1e-5f) changed = true;
    CHECK(changed);
}
