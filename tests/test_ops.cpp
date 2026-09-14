// tests/test_ops.cpp
#include "test_framework.hpp"
#include "llm/ops.hpp"
#include <cmath>
using namespace llm;

static Tensor vec(std::initializer_list<float> xs) {
    Tensor t = Tensor::empty({(int64_t)xs.size()}, DType::F32);
    float* p = t.data_ptr<float>();
    int i = 0; for (float x : xs) p[i++] = x;
    return t;
}

TEST(elementwise_add_mul_scale) {
    Tensor a = vec({0, 1, 2, 3});
    Tensor b = vec({0, 2, 4, 6});
    Tensor o = Tensor::empty({4}, DType::F32);
    add(o, a, b);   CHECK_CLOSE(o.at<float>({3}), 9.0f,  1e-6);
    mul(o, a, b);   CHECK_CLOSE(o.at<float>({3}), 18.0f, 1e-6);
    scale(o, a, 10.0f); CHECK_CLOSE(o.at<float>({2}), 20.0f, 1e-6);
}

TEST(add_inplace_is_residual) {
    Tensor x = vec({1, 1, 1});
    Tensor d = vec({2, 3, 4});
    add_inplace(x, d);                      // x <- x + d
    CHECK_CLOSE(x.at<float>({0}), 3.0f, 1e-6);
    CHECK_CLOSE(x.at<float>({2}), 5.0f, 1e-6);
}

TEST(apply_unary_maps_a_function) {
    Tensor in  = vec({-2, 0, 3});
    Tensor out = Tensor::empty({3}, DType::F32);
    apply_unary(out, in, [](float v) { return v * v; });
    CHECK_CLOSE(out.at<float>({0}), 4.0f, 1e-6);
    CHECK_CLOSE(out.at<float>({2}), 9.0f, 1e-6);
}

// add to tests/test_ops.cpp

TEST(sigmoid_is_stable_at_extremes) {
    CHECK(std::isfinite(sigmoidf(-1000.0f)));
    CHECK(std::isfinite(sigmoidf( 1000.0f)));
    CHECK_CLOSE(sigmoidf(-1000.0f), 0.0f, 1e-6);
    CHECK_CLOSE(sigmoidf( 1000.0f), 1.0f, 1e-6);
    CHECK_CLOSE(sigmoidf(0.0f),     0.5f, 1e-6);
}

TEST(silu_matches_definition) {
    CHECK_CLOSE(siluf(0.0f), 0.0f, 1e-6);            // 0 * sigma(0) = 0
    const float x = 1.5f;
    CHECK_CLOSE(siluf(x), x * (1.0f / (1.0f + std::exp(-x))), 1e-6);
}

TEST(gelu_tanh_close_to_exact) {
    for (float x : {-2.0f, -0.5f, 0.0f, 0.7f, 3.0f})
        CHECK_CLOSE(geluf(x), gelu_exactf(x), 2e-3);  // approx within a few 1e-3
}

TEST(swiglu_is_silu_gate_times_up) {
    Tensor g = Tensor::empty({2}, DType::F32);
    Tensor u = Tensor::empty({2}, DType::F32);
    Tensor o = Tensor::empty({2}, DType::F32);
    g.data_ptr<float>()[0] = 1.0f;  g.data_ptr<float>()[1] = -1.0f;
    u.data_ptr<float>()[0] = 3.0f;  u.data_ptr<float>()[1] =  5.0f;
    swiglu(o, g, u);
    CHECK_CLOSE(o.at<float>({0}), siluf( 1.0f) * 3.0f, 1e-6);
    CHECK_CLOSE(o.at<float>({1}), siluf(-1.0f) * 5.0f, 1e-6);
}

// add to tests/test_ops.cpp

TEST(rmsnorm_unit_weight_gives_unit_rms) {
    const int D = 4;
    Tensor x = Tensor::empty({1, D}, DType::F32);
    Tensor w = Tensor::empty({D},    DType::F32);
    Tensor o = Tensor::empty({1, D}, DType::F32);
    for (int j = 0; j < D; ++j) { x.data_ptr<float>()[j] = float(j + 1); // 1,2,3,4
                                  w.data_ptr<float>()[j] = 1.0f; }
    rmsnorm(o, x, w, 0.0f);
    float ss = 0.0f;
    for (int j = 0; j < D; ++j) { float v = o.data_ptr<float>()[j]; ss += v * v; }
    CHECK_CLOSE(std::sqrt(ss / D), 1.0f, 1e-5);       // unit RMS by construction
}

TEST(rmsnorm_weight_scales_each_dim) {
    const int D = 3;
    Tensor x = Tensor::empty({1, D}, DType::F32);
    Tensor w = Tensor::empty({D},    DType::F32);
    Tensor o1 = Tensor::empty({1, D}, DType::F32);
    Tensor o2 = Tensor::empty({1, D}, DType::F32);
    for (int j = 0; j < D; ++j) x.data_ptr<float>()[j] = 1.0f;   // rms(x) = 1
    for (int j = 0; j < D; ++j) w.data_ptr<float>()[j] = 1.0f;
    rmsnorm(o1, x, w, 0.0f);
    w.data_ptr<float>()[1] = 2.0f;                               // double dim 1
    rmsnorm(o2, x, w, 0.0f);
    CHECK_CLOSE(o2.at<float>({0, 1}), 2.0f * o1.at<float>({0, 1}), 1e-6);
    CHECK_CLOSE(o2.at<float>({0, 0}),        o1.at<float>({0, 0}), 1e-6);
}

TEST(rmsnorm_rows_are_independent) {
    const int D = 2;
    Tensor x = Tensor::empty({2, D}, DType::F32);
    Tensor w = Tensor::empty({D},    DType::F32);
    Tensor o = Tensor::empty({2, D}, DType::F32);
    float v[4] = {3, 4, 30, 40};                                 // row1 = 10 * row0
    for (int i = 0; i < 4; ++i) x.data_ptr<float>()[i] = v[i];
    for (int j = 0; j < D; ++j) w.data_ptr<float>()[j] = 1.0f;
    rmsnorm(o, x, w, 0.0f);
    // scaling a row by a constant does not change its RMS-normalized result
    CHECK_CLOSE(o.at<float>({0, 0}), o.at<float>({1, 0}), 1e-5);
    CHECK_CLOSE(o.at<float>({0, 1}), o.at<float>({1, 1}), 1e-5);
}

// add to tests/test_ops.cpp
#include <algorithm>
#include <vector>

// Oracle: the same stable algorithm, in double precision.
static std::vector<double> softmax_ref(const float* x, int n) {
    double m = x[0];
    for (int i = 1; i < n; ++i) m = std::max(m, static_cast<double>(x[i]));
    std::vector<double> e(n);
    double s = 0.0;
    for (int i = 0; i < n; ++i) { e[i] = std::exp(static_cast<double>(x[i]) - m); s += e[i]; }
    for (int i = 0; i < n; ++i) e[i] /= s;
    return e;
}

TEST(boss_softmax_stable_finite_and_correct) {
    const int D = 6;
    float vals[D] = {1000.0f, 999.0f, -1000.0f, 0.0f, 500.0f, 1000.0f};
    Tensor x = Tensor::empty({1, D}, DType::F32);
    Tensor o = Tensor::empty({1, D}, DType::F32);
    for (int j = 0; j < D; ++j) x.data_ptr<float>()[j] = vals[j];

    softmax_lastdim(o, x);

    double sum = 0.0;
    for (int j = 0; j < D; ++j) {
        const float v = o.data_ptr<float>()[j];
        CHECK(std::isfinite(v));                 // 1) no NaN / Inf
        CHECK(v >= 0.0f);
        sum += v;
    }
    CHECK_CLOSE(sum, 1.0, 1e-6);                  // 2) sums to one

    auto ref = softmax_ref(vals, D);             // 3) matches float64 reference
    for (int j = 0; j < D; ++j)
        CHECK_CLOSE(o.data_ptr<float>()[j], ref[j], 1e-5);
}

TEST(boss_softmax_uniform_and_independent_rows) {
    const int D = 3;
    float v[6] = {0, 0, 0,  10000, 0, 0};        // row0 uniform; row1 extreme
    Tensor x = Tensor::empty({2, D}, DType::F32);
    Tensor o = Tensor::empty({2, D}, DType::F32);
    for (int i = 0; i < 6; ++i) x.data_ptr<float>()[i] = v[i];

    softmax_lastdim(o, x);

    double s0 = 0, s1 = 0;
    for (int j = 0; j < D; ++j) { s0 += o.data_ptr<float>()[j]; s1 += o.data_ptr<float>()[D + j]; }
    CHECK_CLOSE(s0, 1.0, 1e-6);
    CHECK_CLOSE(s1, 1.0, 1e-6);
    CHECK_CLOSE(o.data_ptr<float>()[0],     1.0 / 3.0, 1e-6);   // uniform row
    CHECK_CLOSE(o.data_ptr<float>()[D + 0], 1.0,       1e-6);   // extreme row -> ~one-hot
}
