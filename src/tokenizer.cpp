// src/tokenizer.cpp
#include "llm/tokenizer.hpp"
#include <climits>

namespace llm {

int Tokenizer::add_token(const std::string& token) {
    auto it = token_to_id_.find(token);
    if (it != token_to_id_.end()) return it->second;
    const int id = static_cast<int>(id_to_token_.size());
    id_to_token_.push_back(token);
    token_to_id_.emplace(token, id);
    return id;
}

int Tokenizer::id_of(const std::string& token) const {
    auto it = token_to_id_.find(token);
    return it == token_to_id_.end() ? -1 : it->second;
}

const std::string& Tokenizer::token_of(int id) const {
    LLM_ASSERT(id >= 0 && id < static_cast<int>(id_to_token_.size()),
               "token_of: id out of range");
    return id_to_token_[id];
}

Tokenizer Tokenizer::with_byte_vocab() {
    Tokenizer t;
    for (int b = 0; b < 256; ++b)
        t.add_token(std::string(1, static_cast<char>(static_cast<unsigned char>(b))));
    return t;
}


void Tokenizer::add_merge(const std::string& a, const std::string& b, int rank) {
    const int ia = id_of(a), ib = id_of(b);
    LLM_ASSERT(ia >= 0 && ib >= 0, "add_merge: operands must already be in the vocab");
    add_token(a + b);                              // ensure merged token exists
    merge_rank_[pair_key(ia, ib)] = rank;
}

std::vector<int> Tokenizer::bpe_encode_bytes(const std::string& piece) const {
    std::vector<int> ids;
    ids.reserve(piece.size());
    for (unsigned char c : piece) ids.push_back(static_cast<int>(c));   // bytes 0..255

    while (ids.size() >= 2) {
        int best_rank = INT_MAX, best_pos = -1;
        for (size_t i = 0; i + 1 < ids.size(); ++i) {
            auto it = merge_rank_.find(pair_key(ids[i], ids[i + 1]));
            if (it != merge_rank_.end() && it->second < best_rank) {
                best_rank = it->second;
                best_pos  = static_cast<int>(i);
            }
        }
        if (best_pos < 0) break;                   // no applicable rule -> done
        const std::string merged =
            token_of(ids[best_pos]) + token_of(ids[best_pos + 1]);
        ids[best_pos] = id_of(merged);
        ids.erase(ids.begin() + best_pos + 1);
    }
    return ids;
}


static bool is_space_byte(unsigned char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
}

std::vector<std::string> Tokenizer::pretokenize(const std::string& text) {
    std::vector<std::string> pieces;
    size_t i = 0, n = text.size();
    while (i < n) {
        const size_t start = i;
        const bool space = is_space_byte(static_cast<unsigned char>(text[i]));
        while (i < n &&
               is_space_byte(static_cast<unsigned char>(text[i])) == space) ++i;
        pieces.push_back(text.substr(start, i - start));    // a maximal same-class run
    }
    return pieces;
}

std::vector<int> Tokenizer::encode(const std::string& text, bool add_bos) const {
    std::vector<int> out;
    if (add_bos) {
        LLM_ASSERT(bos_id_ >= 0, "encode: add_bos set but no BOS token configured");
        out.push_back(bos_id_);
    }
    for (const std::string& piece : pretokenize(text)) {
        const std::vector<int> ids = bpe_encode_bytes(piece);
        out.insert(out.end(), ids.begin(), ids.end());
    }
    return out;
}


std::string Tokenizer::decode(const std::vector<int>& ids, bool skip_special) const {
    std::string out;
    for (int id : ids) {
        if (skip_special && (id == bos_id_ || id == eos_id_)) continue;
        out += token_of(id);          // append raw bytes; UTF-8 emerges from the whole
    }
    return out;
}

}  // namespace llm
