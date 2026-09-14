// include/llm/model.hpp
#pragma once
#include <vector>
#include "llm/tensor.hpp"

namespace llm {

struct ModelConfig {
    int   vocab_size       = 0;
    int   hidden_size      = 0;
    int   n_layers         = 0;
    int   n_heads          = 0;
    int   n_kv_heads       = 0;     // <= n_heads (grouped-query attention)
    int   head_dim         = 0;     // usually hidden_size / n_heads
    int   intermediate_size= 0;     // FFN inner width
    float rope_theta       = 10000.0f;
    float rms_norm_eps     = 1e-5f;
    int   num_experts      = 0;     // [Unit 6, per book prose] MoE
    int   top_k            = 0;     // experts routed per token
};

struct LayerWeights {
    Tensor attn_norm;               // [hidden]
    Tensor wq, wk, wv, wo;          // wq:[n_heads*hd, hidden]  wk,wv:[n_kv*hd, hidden]
                                    // wo:[hidden, n_heads*hd]
    Tensor ffn_norm;                // [hidden]
    Tensor w_gate, w_up;            // [intermediate, hidden]
    Tensor w_down;                  // [hidden, intermediate]
};

struct ModelWeights {
    Tensor token_embedding;         // [vocab, hidden]
    std::vector<LayerWeights> layers;
    Tensor final_norm;              // [hidden]
    Tensor lm_head;                 // [vocab, hidden] (may alias token_embedding)
};

// Gather embedding rows named by ids into out [seq, hidden].
void embed_tokens(Tensor& out, const std::vector<int>& ids, const Tensor& embedding);

// Linear layer with [out, in] weight: out[i,o] = sum_k x[i,k]*w[o,k].
void linear(Tensor& out, const Tensor& x, const Tensor& w);

// Full forward pass: token ids -> logits [seq, vocab]. (Completed in 5.5.)
Tensor forward(const ModelWeights& model, const ModelConfig& cfg,
               const std::vector<int>& ids);


// model.hpp
void apply_rope(Tensor& x, int seq, int n_heads, int head_dim, float theta, int pos0 = 0);


// model.hpp
void attention(Tensor& out, const Tensor& x, const LayerWeights& w,
               const ModelConfig& cfg, int pos0 = 0);


// model.hpp
void ffn(Tensor& out, const Tensor& x, const LayerWeights& w, const ModelConfig& cfg);
void transformer_block(Tensor& x, const LayerWeights& w, const ModelConfig& cfg, int pos0 = 0);


// model.hpp  (MoE additions)
struct ExpertWeights {
    Tensor w_gate, w_up;   // [intermediate, hidden]
    Tensor w_down;         // [hidden, intermediate]
};
struct MoELayerWeights {
    Tensor router;                         // [num_experts, hidden]
    std::vector<ExpertWeights> experts;
};

struct Routing {                           // per token, flattened [seq*top_k]
    std::vector<int>   experts;
    std::vector<float> weights;
};

Routing route(const Tensor& x, const Tensor& router_w, int num_experts, int top_k);


// model.hpp
void moe_ffn(Tensor& out, const Tensor& x, const MoELayerWeights& w, const ModelConfig& cfg);


// model.hpp
void moe_ffn_grouped(Tensor& out, const Tensor& x,
                     const MoELayerWeights& w, const ModelConfig& cfg);


// model.hpp
struct KVCache {
    int n_layers = 0, n_kv_heads = 0, head_dim = 0, max_seq = 0;
    int length = 0;                          // live positions
    std::vector<Tensor> k;                   // per layer: [max_seq, n_kv_heads*head_dim]
    std::vector<Tensor> v;

    static KVCache allocate(const ModelConfig& cfg, int max_seq);
    void append_layer(int layer, const Tensor& k_new, const Tensor& v_new, int n);
    void advance(int n) { length += n; }
    void reset()        { length = 0; }
};


// model.hpp
void attention_cached(Tensor& out, const Tensor& x, const LayerWeights& w,
                      const ModelConfig& cfg, KVCache& cache, int layer);


// model.hpp
void   transformer_block_cached(Tensor& x, const LayerWeights& w,
                                const ModelConfig& cfg, KVCache& cache, int layer);
Tensor forward_cached(const ModelWeights& model, const ModelConfig& cfg,
                      const std::vector<int>& new_ids, KVCache& cache);

}  // namespace llm
