// src/model.cpp
#include "llm/model.hpp"
#include "llm/ops.hpp"
#include <cstring>
#include <cmath>
#include <vector>
#include <limits>
#include "llm/ops.hpp"      // siluf
#include "llm/flash.hpp"
#include <algorithm>

namespace llm {

void embed_tokens(Tensor& out, const std::vector<int>& ids, const Tensor& embedding) {
    const int hidden = static_cast<int>(embedding.shape()[1]);
    const float* table = embedding.data_ptr<float>();
    float* o = out.data_ptr<float>();
    for (size_t i = 0; i < ids.size(); ++i)
        std::memcpy(o + i * hidden,
                    table + static_cast<int64_t>(ids[i]) * hidden,
                    sizeof(float) * hidden);
}

void linear(Tensor& out, const Tensor& x, const Tensor& w) {
    const int S = static_cast<int>(x.shape()[0]);
    const int K = static_cast<int>(x.shape()[1]);
    const int N = static_cast<int>(w.shape()[0]);   // [out, in]
    const float* X = x.data_ptr<float>();
    const float* W = w.data_ptr<float>();
    float*       Y = out.data_ptr<float>();
    for (int i = 0; i < S; ++i)
        for (int o = 0; o < N; ++o) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k)
                acc += X[(int64_t)i * K + k] * W[(int64_t)o * K + k];
            Y[(int64_t)i * N + o] = acc;           // production: dispatch to Unit-3 GEMM
        }
}


// model.cpp

void apply_rope(Tensor& x, int seq, int n_heads, int head_dim, float theta, int pos0) {
    const int half = head_dim / 2;
    std::vector<float> inv_freq(half);
    for (int i = 0; i < half; ++i)
        inv_freq[i] = std::pow(theta, -2.0f * static_cast<float>(i) / head_dim);

    float* X = x.data_ptr<float>();                       // [seq, n_heads, head_dim]
    for (int t = 0; t < seq; ++t) {
        const float pos = static_cast<float>(pos0 + t);
        for (int h = 0; h < n_heads; ++h) {
            float* v = X + (((int64_t)t * n_heads + h) * head_dim);
            for (int i = 0; i < half; ++i) {
                const float ang = pos * inv_freq[i];
                const float c = std::cos(ang), s = std::sin(ang);
                const float x0 = v[i], x1 = v[i + half];  // read both first
                v[i]        = x0 * c - x1 * s;
                v[i + half] = x1 * c + x0 * s;
            }
        }
    }
}


// model.cpp

void attention(Tensor& out, const Tensor& x, const LayerWeights& w,
               const ModelConfig& cfg, int pos0) {
    const int S = static_cast<int>(x.shape()[0]);
    const int H = cfg.n_heads, KVH = cfg.n_kv_heads, D = cfg.head_dim;
    const int qd = H * D, kvd = KVH * D, group = H / KVH;
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));

    Tensor q = Tensor::empty({S, qd},  DType::F32);
    Tensor k = Tensor::empty({S, kvd}, DType::F32);
    Tensor v = Tensor::empty({S, kvd}, DType::F32);
    linear(q, x, w.wq);  linear(k, x, w.wk);  linear(v, x, w.wv);

    apply_rope(q, S, H,   D, cfg.rope_theta, pos0);       // queries
    apply_rope(k, S, KVH, D, cfg.rope_theta, pos0);       // keys (values are NOT rotated)

    const float* Q = q.data_ptr<float>();
    const float* K = k.data_ptr<float>();
    const float* V = v.data_ptr<float>();

    Tensor concat = Tensor::empty({S, qd}, DType::F32);    // [seq, n_heads*head_dim]
    float* O = concat.data_ptr<float>();
    std::vector<float> scores(S);

    for (int h = 0; h < H; ++h) {
        const int kvh = h / group;
        for (int i = 0; i < S; ++i) {
            const float* qi = Q + ((int64_t)i * H + h) * D;
            float m = -INFINITY;                            // scores over keys 0..i
            for (int j = 0; j <= i; ++j) {
                const float* kj = K + ((int64_t)j * KVH + kvh) * D;
                float dot = 0.0f;
                for (int d = 0; d < D; ++d) dot += qi[d] * kj[d];
                dot *= scale;
                scores[j] = dot;
                if (dot > m) m = dot;
            }
            float sum = 0.0f;                               // stable softmax over 0..i
            for (int j = 0; j <= i; ++j) { float e = std::exp(scores[j] - m); scores[j] = e; sum += e; }
            const float inv = 1.0f / sum;

            float* oi = O + ((int64_t)i * H + h) * D;        // weighted sum of values
            for (int d = 0; d < D; ++d) oi[d] = 0.0f;
            for (int j = 0; j <= i; ++j) {
                const float wij = scores[j] * inv;
                const float* vj = V + ((int64_t)j * KVH + kvh) * D;
                for (int d = 0; d < D; ++d) oi[d] += wij * vj[d];
            }
        }
    }
    linear(out, concat, w.wo);                              // project heads back to hidden
}


// model.cpp
void ffn(Tensor& out, const Tensor& x, const LayerWeights& w, const ModelConfig& cfg) {
    const int S = static_cast<int>(x.shape()[0]);
    const int I = cfg.intermediate_size;
    Tensor gate = Tensor::empty({S, I}, DType::F32);
    Tensor up   = Tensor::empty({S, I}, DType::F32);
    linear(gate, x, w.w_gate);
    linear(up,   x, w.w_up);
    Tensor act = Tensor::empty({S, I}, DType::F32);
    swiglu(act, gate, up);                 // SiLU(gate) * up   (Unit 2)
    linear(out, act, w.w_down);            // back to [seq, hidden]
}

void transformer_block(Tensor& x, const LayerWeights& w,
                       const ModelConfig& cfg, int pos0) {
    const int S = static_cast<int>(x.shape()[0]);
    const int hidden = cfg.hidden_size;
    Tensor normed = Tensor::empty({S, hidden}, DType::F32);
    Tensor delta  = Tensor::empty({S, hidden}, DType::F32);

    // attention sub-layer: x += Attention(RMSNorm(x))
    rmsnorm(normed, x, w.attn_norm, cfg.rms_norm_eps);
    attention(delta, normed, w, cfg, pos0);
    add_inplace(x, delta);

    // feed-forward sub-layer: x += FFN(RMSNorm(x))
    rmsnorm(normed, x, w.ffn_norm, cfg.rms_norm_eps);
    ffn(delta, normed, w, cfg);
    add_inplace(x, delta);
}


// model.cpp
Tensor forward(const ModelWeights& model, const ModelConfig& cfg,
               const std::vector<int>& ids) {
    const int S = static_cast<int>(ids.size());
    const int hidden = cfg.hidden_size;

    Tensor x = Tensor::empty({S, hidden}, DType::F32);
    embed_tokens(x, ids, model.token_embedding);          // residual stream, position 0..S-1

    for (int l = 0; l < cfg.n_layers; ++l)
        transformer_block(x, model.layers[l], cfg, /*pos0=*/0);

    Tensor normed = Tensor::empty({S, hidden}, DType::F32);
    rmsnorm(normed, x, model.final_norm, cfg.rms_norm_eps);

    Tensor logits = Tensor::empty({S, cfg.vocab_size}, DType::F32);
    linear(logits, normed, model.lm_head);                // [seq, vocab]
    return logits;
}


// model.cpp

Routing route(const Tensor& x, const Tensor& router_w, int num_experts, int top_k) {
    const int S = static_cast<int>(x.shape()[0]);
    Tensor logits = Tensor::empty({S, num_experts}, DType::F32);
    linear(logits, x, router_w);                       // [seq, num_experts]

    Routing r;
    r.experts.resize((size_t)S * top_k);
    r.weights.resize((size_t)S * top_k);

    std::vector<float> row(num_experts);
    for (int i = 0; i < S; ++i) {
        const float* L = logits.data_ptr<float>() + (int64_t)i * num_experts;
        for (int e = 0; e < num_experts; ++e) row[e] = L[e];

        // top-k by repeated max-and-mask
        float chosen_logit[64];                        // top_k is small
        for (int j = 0; j < top_k; ++j) {
            int best = 0; float bestv = -std::numeric_limits<float>::infinity();
            for (int e = 0; e < num_experts; ++e)
                if (row[e] > bestv) { bestv = row[e]; best = e; }
            r.experts[(size_t)i * top_k + j] = best;
            chosen_logit[j] = bestv;
            row[best] = -std::numeric_limits<float>::infinity();   // mask it out
        }
        // softmax over the k chosen logits
        float m = chosen_logit[0];
        for (int j = 1; j < top_k; ++j) m = std::max(m, chosen_logit[j]);
        float sum = 0.0f;
        for (int j = 0; j < top_k; ++j) { float e = std::exp(chosen_logit[j] - m); chosen_logit[j] = e; sum += e; }
        for (int j = 0; j < top_k; ++j)
            r.weights[(size_t)i * top_k + j] = chosen_logit[j] / sum;
    }
    return r;
}


// model.cpp

// One expert (SwiGLU) on a single token vector x[hidden] -> out[hidden].
static void expert_ffn_token(float* out, const float* x, const ExpertWeights& e,
                             int hidden, int inter) {
    std::vector<float> act(inter);
    const float* Wg = e.w_gate.data_ptr<float>();
    const float* Wu = e.w_up.data_ptr<float>();
    for (int o = 0; o < inter; ++o) {
        float a = 0.0f, b = 0.0f;
        const float* wg = Wg + (int64_t)o * hidden;
        const float* wu = Wu + (int64_t)o * hidden;
        for (int k = 0; k < hidden; ++k) { a += x[k] * wg[k]; b += x[k] * wu[k]; }
        act[o] = siluf(a) * b;                          // SiLU(gate) * up
    }
    const float* Wd = e.w_down.data_ptr<float>();
    for (int h = 0; h < hidden; ++h) {
        float a = 0.0f;
        const float* wd = Wd + (int64_t)h * inter;
        for (int o = 0; o < inter; ++o) a += act[o] * wd[o];
        out[h] = a;
    }
}

void moe_ffn(Tensor& out, const Tensor& x, const MoELayerWeights& w, const ModelConfig& cfg) {
    const int S = static_cast<int>(x.shape()[0]);
    const int hidden = cfg.hidden_size, inter = cfg.intermediate_size, K = cfg.top_k;
    const Routing r = route(x, w.router, cfg.num_experts, K);

    const float* X = x.data_ptr<float>();
    float*       O = out.data_ptr<float>();
    std::vector<float> tmp(hidden);

    for (int i = 0; i < S; ++i) {
        float* oi = O + (int64_t)i * hidden;
        for (int h = 0; h < hidden; ++h) oi[h] = 0.0f;
        for (int j = 0; j < K; ++j) {
            const int   e = r.experts[(size_t)i * K + j];
            const float g = r.weights[(size_t)i * K + j];
            expert_ffn_token(tmp.data(), X + (int64_t)i * hidden, w.experts[e], hidden, inter);
            for (int h = 0; h < hidden; ++h) oi[h] += g * tmp[h];     // weighted combine
        }
    }
}


// model.cpp

void moe_ffn_grouped(Tensor& out, const Tensor& x,
                     const MoELayerWeights& w, const ModelConfig& cfg) {
    const int S = static_cast<int>(x.shape()[0]);
    const int hidden = cfg.hidden_size, inter = cfg.intermediate_size;
    const int E = cfg.num_experts, K = cfg.top_k;
    const Routing r = route(x, w.router, E, K);

    std::vector<std::vector<int>>   toks(E);           // tokens routed to each expert
    std::vector<std::vector<float>> wgts(E);
    for (int i = 0; i < S; ++i)
        for (int j = 0; j < K; ++j) {
            const int e = r.experts[(size_t)i * K + j];
            toks[e].push_back(i);
            wgts[e].push_back(r.weights[(size_t)i * K + j]);
        }

    const float* X = x.data_ptr<float>();
    float*       O = out.data_ptr<float>();
    std::memset(O, 0, sizeof(float) * (size_t)S * hidden);

    for (int e = 0; e < E; ++e) {
        const int n = static_cast<int>(toks[e].size());
        if (n == 0) continue;                          // unused expert: never evaluated

        Tensor xe = Tensor::empty({n, hidden}, DType::F32);   // gather
        for (int t = 0; t < n; ++t)
            std::memcpy(xe.data_ptr<float>() + (int64_t)t * hidden,
                        X + (int64_t)toks[e][t] * hidden, sizeof(float) * hidden);

        Tensor g  = Tensor::empty({n, inter},  DType::F32);   // batched expert FFN
        Tensor u  = Tensor::empty({n, inter},  DType::F32);
        Tensor a  = Tensor::empty({n, inter},  DType::F32);
        Tensor ye = Tensor::empty({n, hidden}, DType::F32);
        linear(g, xe, w.experts[e].w_gate);
        linear(u, xe, w.experts[e].w_up);
        swiglu(a, g, u);
        linear(ye, a, w.experts[e].w_down);

        for (int t = 0; t < n; ++t) {                  // scatter with gate weight
            const float gw  = wgts[e][t];
            float*       dst = O + (int64_t)toks[e][t] * hidden;
            const float* src = ye.data_ptr<float>() + (int64_t)t * hidden;
            for (int h = 0; h < hidden; ++h) dst[h] += gw * src[h];
        }
    }
}


// model.cpp

KVCache KVCache::allocate(const ModelConfig& cfg, int max_seq) {
    KVCache c;
    c.n_layers = cfg.n_layers; c.n_kv_heads = cfg.n_kv_heads;
    c.head_dim = cfg.head_dim; c.max_seq = max_seq; c.length = 0;
    const int kvd = cfg.n_kv_heads * cfg.head_dim;
    c.k.reserve(cfg.n_layers); c.v.reserve(cfg.n_layers);
    for (int l = 0; l < cfg.n_layers; ++l) {
        c.k.push_back(Tensor::empty({max_seq, kvd}, DType::F32));
        c.v.push_back(Tensor::empty({max_seq, kvd}, DType::F32));
    }
    return c;
}

void KVCache::append_layer(int layer, const Tensor& k_new, const Tensor& v_new, int n) {
    const int kvd = n_kv_heads * head_dim;
    LLM_CHECK(length + n <= max_seq, "KVCache: capacity exceeded");
    std::memcpy(k[layer].data_ptr<float>() + (int64_t)length * kvd,
                k_new.data_ptr<float>(), sizeof(float) * (size_t)n * kvd);
    std::memcpy(v[layer].data_ptr<float>() + (int64_t)length * kvd,
                v_new.data_ptr<float>(), sizeof(float) * (size_t)n * kvd);
}


// model.cpp
void attention_cached(Tensor& out, const Tensor& x, const LayerWeights& w,
                      const ModelConfig& cfg, KVCache& cache, int layer) {
    const int n = static_cast<int>(x.shape()[0]);            // new tokens this call
    const int H = cfg.n_heads, KVH = cfg.n_kv_heads, D = cfg.head_dim;
    const int qd = H * D, kvd = KVH * D, group = H / KVH;
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));
    const int pos0 = cache.length;                           // new tokens at pos0..pos0+n-1

    Tensor q     = Tensor::empty({n, qd},  DType::F32);
    Tensor k_new = Tensor::empty({n, kvd}, DType::F32);
    Tensor v_new = Tensor::empty({n, kvd}, DType::F32);
    linear(q, x, w.wq);  linear(k_new, x, w.wk);  linear(v_new, x, w.wv);

    apply_rope(q,     n, H,   D, cfg.rope_theta, pos0);
    apply_rope(k_new, n, KVH, D, cfg.rope_theta, pos0);
    cache.append_layer(layer, k_new, v_new, n);              // history now 0..pos0+n-1

    const float* Q  = q.data_ptr<float>();
    const float* Kc = cache.k[layer].data_ptr<float>();      // [max_seq, kvd]
    const float* Vc = cache.v[layer].data_ptr<float>();

    Tensor concat = Tensor::empty({n, qd}, DType::F32);
    float* O = concat.data_ptr<float>();
    std::vector<float> scores(pos0 + n);

    for (int h = 0; h < H; ++h) {
        const int kvh = h / group;
        for (int i = 0; i < n; ++i) {
            const int qpos = pos0 + i;                        // absolute position
            const float* qi = Q + ((int64_t)i * H + h) * D;
            float m = -INFINITY;
            for (int j = 0; j <= qpos; ++j) {                 // attend cached keys 0..qpos
                const float* kj = Kc + ((int64_t)j * KVH + kvh) * D;
                float dot = 0.0f;
                for (int d = 0; d < D; ++d) dot += qi[d] * kj[d];
                dot *= scale; scores[j] = dot; if (dot > m) m = dot;
            }
            float sum = 0.0f;
            for (int j = 0; j <= qpos; ++j) { float e = std::exp(scores[j] - m); scores[j] = e; sum += e; }
            const float inv = 1.0f / sum;
            float* oi = O + ((int64_t)i * H + h) * D;
            for (int d = 0; d < D; ++d) oi[d] = 0.0f;
            for (int j = 0; j <= qpos; ++j) {
                const float wij = scores[j] * inv;
                const float* vj = Vc + ((int64_t)j * KVH + kvh) * D;
                for (int d = 0; d < D; ++d) oi[d] += wij * vj[d];
            }
        }
    }
    linear(out, concat, w.wo);
}


// model.cpp
void transformer_block_cached(Tensor& x, const LayerWeights& w,
                              const ModelConfig& cfg, KVCache& cache, int layer) {
    const int S = static_cast<int>(x.shape()[0]);
    const int hidden = cfg.hidden_size;
    Tensor normed = Tensor::empty({S, hidden}, DType::F32);
    Tensor delta  = Tensor::empty({S, hidden}, DType::F32);

    rmsnorm(normed, x, w.attn_norm, cfg.rms_norm_eps);
    attention_cached(delta, normed, w, cfg, cache, layer);  // <- the only change
    add_inplace(x, delta);

    rmsnorm(normed, x, w.ffn_norm, cfg.rms_norm_eps);
    ffn(delta, normed, w, cfg);                             // (or moe_ffn, Unit 6)
    add_inplace(x, delta);
}

Tensor forward_cached(const ModelWeights& model, const ModelConfig& cfg,
                      const std::vector<int>& new_ids, KVCache& cache) {
    const int n = static_cast<int>(new_ids.size());
    const int hidden = cfg.hidden_size;

    Tensor x = Tensor::empty({n, hidden}, DType::F32);
    embed_tokens(x, new_ids, model.token_embedding);

    for (int l = 0; l < cfg.n_layers; ++l)
        transformer_block_cached(x, model.layers[l], cfg, cache, l);
    cache.advance(n);                                        // advance once, after all layers

    Tensor normed = Tensor::empty({n, hidden}, DType::F32);
    rmsnorm(normed, x, model.final_norm, cfg.rms_norm_eps);
    Tensor logits = Tensor::empty({n, cfg.vocab_size}, DType::F32);
    linear(logits, normed, model.lm_head);
    return logits;
}


// model.cpp

void flash_attention(float* O, const float* Q, const float* K, const float* V,
                     int seq, int n_heads, int n_kv_heads, int head_dim,
                     int pos0, float scale, int block) {
    const int group = n_heads / n_kv_heads;
    std::vector<float> bscore(block);
    std::vector<float> bval((size_t)block * head_dim);

    for (int h = 0; h < n_heads; ++h) {
        const int kvh = h / group;
        for (int i = 0; i < seq; ++i) {
            const int qpos = pos0 + i;                        // attend keys 0..qpos
            const float* qi = Q + ((int64_t)i * n_heads + h) * head_dim;
            OnlineSoftmax st(head_dim);

            for (int kb = 0; kb <= qpos; kb += block) {
                const int end = std::min(kb + block, qpos + 1);   // causal cap
                const int bn  = end - kb;
                for (int j = 0; j < bn; ++j) {
                    const float* kj = K + ((int64_t)(kb + j) * n_kv_heads + kvh) * head_dim;
                    float dot = 0.0f;
                    for (int d = 0; d < head_dim; ++d) dot += qi[d] * kj[d];
                    bscore[j] = dot * scale;
                    const float* vj = V + ((int64_t)(kb + j) * n_kv_heads + kvh) * head_dim;
                    std::memcpy(bval.data() + (int64_t)j * head_dim, vj, sizeof(float) * head_dim);
                }
                online_update(st, bscore.data(), bn, bval.data(), head_dim);
            }
            online_finalize(st, O + ((int64_t)i * n_heads + h) * head_dim, head_dim);
        }
    }
}

void attention_flash(Tensor& out, const Tensor& x, const LayerWeights& w,
                     const ModelConfig& cfg, int pos0, int block) {
    const int S = static_cast<int>(x.shape()[0]);
    const int H = cfg.n_heads, KVH = cfg.n_kv_heads, D = cfg.head_dim;
    const int qd = H * D, kvd = KVH * D;
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));

    Tensor q = Tensor::empty({S, qd},  DType::F32);
    Tensor k = Tensor::empty({S, kvd}, DType::F32);
    Tensor v = Tensor::empty({S, kvd}, DType::F32);
    linear(q, x, w.wq);  linear(k, x, w.wk);  linear(v, x, w.wv);
    apply_rope(q, S, H,   D, cfg.rope_theta, pos0);
    apply_rope(k, S, KVH, D, cfg.rope_theta, pos0);

    Tensor concat = Tensor::empty({S, qd}, DType::F32);
    flash_attention(concat.data_ptr<float>(), q.data_ptr<float>(),
                    k.data_ptr<float>(), v.data_ptr<float>(),
                    S, H, KVH, D, pos0, scale, block);
    linear(out, concat, w.wo);
}

}  // namespace llm
