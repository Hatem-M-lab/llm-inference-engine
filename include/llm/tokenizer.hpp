// include/llm/tokenizer.hpp
#pragma once
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>
#include "llm/common.hpp"

namespace llm {

class Tokenizer {
public:
    // Seed with the 256 single-byte tokens (real vocab + merges arrive in Unit 10).
    static Tokenizer with_byte_vocab();

    int vocab_size() const { return static_cast<int>(id_to_token_.size()); }

    int add_token(const std::string& token);            // existing-or-new id
    int id_of(const std::string& token) const;          // -1 if absent
    const std::string& token_of(int id) const;          // raw bytes for an id

protected:
    std::vector<std::string>             id_to_token_;   // id   -> bytes
    std::unordered_map<std::string,int>  token_to_id_;   // bytes -> id

public:
    // Record "a" + "b" -> "ab" with the given rank (lower rank merges earlier).
    void add_merge(const std::string& a, const std::string& b, int rank);

    // Greedy BPE over a run of raw bytes -> token ids.
    std::vector<int> bpe_encode_bytes(const std::string& piece) const;

protected:
    std::unordered_map<int64_t,int> merge_rank_;   // pair_key(a_id,b_id) -> rank
    static int64_t pair_key(int a, int b) {
        return (static_cast<int64_t>(a) << 32) | static_cast<uint32_t>(b);
    }

public:
    void set_special_tokens(int bos_id, int eos_id) { bos_id_ = bos_id; eos_id_ = eos_id; }
    int  bos_id() const { return bos_id_; }
    int  eos_id() const { return eos_id_; }

    std::vector<int> encode(const std::string& text, bool add_bos = false) const;

protected:
    int bos_id_ = -1;
    int eos_id_ = -1;
    static std::vector<std::string> pretokenize(const std::string& text);

public:
    std::string decode(const std::vector<int>& ids, bool skip_special = true) const;
};

}  // namespace llm
