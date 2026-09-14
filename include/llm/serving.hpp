// include/llm/serving.hpp
#pragma once
#include <vector>
#include <cmath>
#include <cstring>
#include "llm/model.hpp"
#include "llm/ops.hpp"
#include "llm/sampler.hpp"

namespace llm {

inline void rope_inplace(float* v, int n_heads, int head_dim, float theta, int pos) {
    const int half = head_dim / 2;
    for (int h = 0; h < n_heads; ++h) {
        float* hv = v + h * head_dim;
        for (int i = 0; i < half; ++i) {
            const float freq = std::pow(theta, -2.0f * i / head_dim);
            const float ang = pos * freq, c = std::cos(ang), s = std::sin(ang);
            const float x0 = hv[i], x1 = hv[i + half];
            hv[i] = x0 * c - x1 * s; hv[i + half] = x1 * c + x0 * s;
        }
    }
}

inline void batched_decode_block(Tensor& x, const LayerWeights& w, const ModelConfig& cfg,
                                 std::vector<KVCache*>& caches, int layer) {
    const int B = (int)x.shape()[0], hidden = cfg.hidden_size;
    const int H = cfg.n_heads, KVH = cfg.n_kv_heads, D = cfg.head_dim;
    const int qd = H * D, kvd = KVH * D, group = H / KVH;
    const float scale = 1.0f / std::sqrt((float)D);

    Tensor normed = Tensor::empty({B, hidden}, DType::F32);
    rmsnorm(normed, x, w.attn_norm, cfg.rms_norm_eps);
    Tensor q = Tensor::empty({B, qd},  DType::F32);
    Tensor k = Tensor::empty({B, kvd}, DType::F32);
    Tensor v = Tensor::empty({B, kvd}, DType::F32);
    linear(q, normed, w.wq); linear(k, normed, w.wk); linear(v, normed, w.wv);   // BATCHED

    Tensor attn = Tensor::empty({B, qd}, DType::F32);
    for (int b = 0; b < B; ++b) {                                   // per-sequence attention
        KVCache& cache = *caches[b];
        const int pos0 = cache.length;
        float* qb = q.data_ptr<float>() + (int64_t)b * qd;
        float* kb = k.data_ptr<float>() + (int64_t)b * kvd;
        float* vb = v.data_ptr<float>() + (int64_t)b * kvd;
        rope_inplace(qb, H,   D, cfg.rope_theta, pos0);
        rope_inplace(kb, KVH, D, cfg.rope_theta, pos0);
        std::memcpy(cache.k[layer].data_ptr<float>() + (int64_t)pos0 * kvd, kb, sizeof(float)*kvd);
        std::memcpy(cache.v[layer].data_ptr<float>() + (int64_t)pos0 * kvd, vb, sizeof(float)*kvd);
        const float* Kc = cache.k[layer].data_ptr<float>();
        const float* Vc = cache.v[layer].data_ptr<float>();
        float* ob = attn.data_ptr<float>() + (int64_t)b * qd;
        std::vector<float> sc(pos0 + 1);
        for (int h = 0; h < H; ++h) {
            const int kvh = h / group; const float* qh = qb + h * D;
            float mx = -INFINITY;
            for (int j = 0; j <= pos0; ++j) {
                const float* kj = Kc + ((int64_t)j * KVH + kvh) * D;
                float dot = 0; for (int d = 0; d < D; ++d) dot += qh[d] * kj[d];
                dot *= scale; sc[j] = dot; if (dot > mx) mx = dot;
            }
            float sum = 0; for (int j = 0; j <= pos0; ++j) { float e = std::exp(sc[j]-mx); sc[j]=e; sum+=e; }
            float inv = 1.0f / sum; float* oh = ob + h * D;
            for (int d = 0; d < D; ++d) oh[d] = 0;
            for (int j = 0; j <= pos0; ++j) {
                const float wj = sc[j]*inv; const float* vj = Vc + ((int64_t)j*KVH + kvh)*D;
                for (int d = 0; d < D; ++d) oh[d] += wj * vj[d];
            }
        }
    }
    Tensor ao = Tensor::empty({B, hidden}, DType::F32);
    linear(ao, attn, w.wo); add_inplace(x, ao);                     // BATCHED
    rmsnorm(normed, x, w.ffn_norm, cfg.rms_norm_eps);
    Tensor ff = Tensor::empty({B, hidden}, DType::F32);
    ffn(ff, normed, w, cfg); add_inplace(x, ff);                    // BATCHED
}

inline std::vector<int> batched_decode(const ModelWeights& m, const ModelConfig& cfg,
                                       std::vector<KVCache*>& caches, const std::vector<int>& tokens) {
    const int B = (int)tokens.size(), V = cfg.vocab_size, hidden = cfg.hidden_size;
    Tensor x = Tensor::empty({B, hidden}, DType::F32);
    for (int b = 0; b < B; ++b)
        std::memcpy(x.data_ptr<float>() + (int64_t)b * hidden,
                    m.token_embedding.data_ptr<float>() + (int64_t)tokens[b] * hidden,
                    sizeof(float) * hidden);
    for (int l = 0; l < cfg.n_layers; ++l) batched_decode_block(x, m.layers[l], cfg, caches, l);
    for (int b = 0; b < B; ++b) caches[b]->advance(1);

    Tensor normed = Tensor::empty({B, hidden}, DType::F32);
    rmsnorm(normed, x, m.final_norm, cfg.rms_norm_eps);
    Tensor logits = Tensor::empty({B, V}, DType::F32);
    linear(logits, normed, m.lm_head);

    std::vector<int> next(B);
    for (int b = 0; b < B; ++b) next[b] = argmax(logits.data_ptr<float>() + (int64_t)b * V, V);
    return next;
}


// serving.hpp
struct PagedKVCache {
    int page_size = 0, n_layers = 0, kvd = 0, num_pages = 0;
    std::vector<float> kpool, vpool;     // [num_pages * n_layers * page_size * kvd]
    std::vector<int>   free_list;        // available physical page ids (a stack)

    static PagedKVCache create(const ModelConfig& cfg, int page_size, int num_pages) {
        PagedKVCache p;
        p.page_size = page_size; p.n_layers = cfg.n_layers;
        p.kvd = cfg.n_kv_heads * cfg.head_dim; p.num_pages = num_pages;
        const int64_t cells = (int64_t)num_pages * cfg.n_layers * page_size * p.kvd;
        p.kpool.assign(cells, 0.0f); p.vpool.assign(cells, 0.0f);
        p.free_list.resize(num_pages);
        for (int i = 0; i < num_pages; ++i) p.free_list[i] = num_pages - 1 - i;  // page 0 on top
        return p;
    }
    int  alloc_page() { if (free_list.empty()) return -1; int p = free_list.back(); free_list.pop_back(); return p; }
    void free_page(int p) { free_list.push_back(p); }
    int64_t row_index(int page, int layer, int slot) const {
        return (((int64_t)page * n_layers + layer) * page_size + slot) * kvd;
    }
};

struct SequenceCache {
    std::vector<int> block_table;        // logical page -> physical page id
    int length = 0;
};

inline bool ensure_page(PagedKVCache& pool, SequenceCache& seq) {
    const int lp = seq.length / pool.page_size;
    if (lp < (int)seq.block_table.size()) return true;
    const int pg = pool.alloc_page();
    if (pg < 0) return false;            // pool exhausted
    seq.block_table.push_back(pg);
    return true;
}

inline void write_kv(PagedKVCache& pool, const SequenceCache& seq, int layer,
                     const float* k_row, const float* v_row) {
    const int lp = seq.length / pool.page_size, slot = seq.length % pool.page_size;
    const int page = seq.block_table[lp];
    std::memcpy(pool.kpool.data() + pool.row_index(page, layer, slot), k_row, sizeof(float)*pool.kvd);
    std::memcpy(pool.vpool.data() + pool.row_index(page, layer, slot), v_row, sizeof(float)*pool.kvd);
}

inline const float* read_k(const PagedKVCache& pool, const SequenceCache& seq, int layer, int p) {
    return pool.kpool.data() + pool.row_index(seq.block_table[p/pool.page_size], layer, p%pool.page_size);
}
inline const float* read_v(const PagedKVCache& pool, const SequenceCache& seq, int layer, int p) {
    return pool.vpool.data() + pool.row_index(seq.block_table[p/pool.page_size], layer, p%pool.page_size);
}

inline void free_sequence(PagedKVCache& pool, SequenceCache& seq) {
    for (int pg : seq.block_table) pool.free_page(pg);
    seq.block_table.clear(); seq.length = 0;
}


// serving.hpp
struct Request {
    std::vector<int> prompt;
    int max_new = 0;
    int eos = -1;
    std::vector<int> output;     // filled by serve()
    bool done = false;
};

inline void serve(const ModelWeights& m, const ModelConfig& cfg,
                  std::vector<Request>& reqs, int max_batch, int* iters) {
    struct Active { int req; KVCache cache; int next; int generated; };
    std::vector<Active> active;
    const int V = cfg.vocab_size;
    size_t next_req = 0;
    int iterations = 0;

    while (true) {
        // 1. retire finished sequences (free their slots)
        std::vector<Active> keep;
        for (auto& a : active) {
            Request& r = reqs[a.req];
            if (a.next == r.eos || a.generated >= r.max_new) { r.done = true; continue; }
            keep.push_back(std::move(a));
        }
        active = std::move(keep);

        // 2. admit waiting requests to fill the batch (prefill each)
        while ((int)active.size() < max_batch && next_req < reqs.size()) {
            Request& r = reqs[next_req];
            KVCache cache = KVCache::allocate(cfg, (int)r.prompt.size() + r.max_new + 8);
            Tensor lg = forward_cached(m, cfg, r.prompt, cache);
            int first = argmax(lg.data_ptr<float>() + (int64_t)(r.prompt.size()-1)*V, V);
            active.push_back({(int)next_req, std::move(cache), first, 0});
            ++next_req;
        }

        if (active.empty()) break;                          // nothing left to serve

        // 3. one batched decode step over the active set
        std::vector<KVCache*> ptrs; std::vector<int> feed;
        ptrs.reserve(active.size()); feed.reserve(active.size());
        for (auto& a : active) { ptrs.push_back(&a.cache); feed.push_back(a.next); }
        std::vector<int> nxt = batched_decode(m, cfg, ptrs, feed);
        ++iterations;

        for (size_t i = 0; i < active.size(); ++i) {        // commit fed token, advance
            reqs[active[i].req].output.push_back(active[i].next);
            active[i].generated++;
            active[i].next = nxt[i];
        }
    }
    if (iters) *iters = iterations;
}

}  // namespace llm
