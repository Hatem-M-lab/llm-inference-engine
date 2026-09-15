import os, struct, numpy as np
from tinyllama import Config, init_model, all_params
from train import cfg, build_model, load_ckpt
from adam import Adam

GGUF_U32, GGUF_F32, GGUF_STRING, GGUF_ARRAY = 4, 6, 8, 9
ALIGN = 32

def w_str(b: bytes) -> bytes:
    return struct.pack('<Q', len(b)) + b

def w_u32_kv(key: str, val: int) -> bytes:
    return w_str(key.encode('ascii')) + struct.pack('<I', GGUF_U32) + struct.pack('<I', val)

def w_f32_kv(key: str, val: float) -> bytes:
    return w_str(key.encode('ascii')) + struct.pack('<I', GGUF_F32) + struct.pack('<f', val)

def w_str_kv(key: str, val: bytes) -> bytes:
    return w_str(key.encode('ascii')) + struct.pack('<I', GGUF_STRING) + w_str(val)

def w_str_array_kv(key: str, items: list) -> bytes:
    out = w_str(key.encode('ascii')) + struct.pack('<I', GGUF_ARRAY)
    out += struct.pack('<I', GGUF_STRING) + struct.pack('<Q', len(items))
    for it in items:
        out += w_str(it)   # each `it` MUST be raw `bytes`, not str (bytes 128-255 must stay single bytes)
    return out

def export(npz_path: str, out_path: str):
    model = build_model()
    opt = Adam(all_params(model))
    step = load_ckpt(model, opt)
    print(f"exporting checkpoint at step {step}")

    kv_blobs = []
    kv_blobs.append(w_str_kv("general.architecture", b"llama"))
    kv_blobs.append(w_u32_kv("general.alignment", ALIGN))
    kv_blobs.append(w_u32_kv("llama.embedding_length", cfg.hidden_size))
    kv_blobs.append(w_u32_kv("llama.block_count", cfg.n_layers))
    kv_blobs.append(w_u32_kv("llama.attention.head_count", cfg.n_heads))
    kv_blobs.append(w_u32_kv("llama.attention.head_count_kv", cfg.n_kv_heads))
    kv_blobs.append(w_u32_kv("llama.feed_forward_length", cfg.intermediate_size))
    kv_blobs.append(w_f32_kv("llama.attention.layer_norm_rms_epsilon", cfg.rms_norm_eps))
    kv_blobs.append(w_f32_kv("llama.rope.freq_base", cfg.rope_theta))
    # byte-level vocab: 256 single RAW bytes (NOT utf-8 encoded -- values 128-255
    # must remain exactly one byte each, matching Tokenizer::with_byte_vocab()).
    tokens = [bytes([i]) for i in range(256)]
    kv_blobs.append(w_str_array_kv("tokenizer.ggml.tokens", tokens))

    n_kv = len(kv_blobs)
    kv_bytes = b"".join(kv_blobs)

    # ---- gather tensors: name -> (numpy array in book row-major [out,in] or [D] convention) ----
    tensors = {}
    tensors["token_embd.weight"] = model["token_embedding"].data
    tensors["output_norm.weight"] = model["final_norm"].data
    tensors["output.weight"] = model["lm_head"].data
    for i, w in enumerate(model["layers"]):
        p = f"blk.{i}."
        tensors[p + "attn_norm.weight"] = w["attn_norm"].data
        tensors[p + "attn_q.weight"] = w["wq"].data
        tensors[p + "attn_k.weight"] = w["wk"].data
        tensors[p + "attn_v.weight"] = w["wv"].data
        tensors[p + "attn_output.weight"] = w["wo"].data
        tensors[p + "ffn_norm.weight"] = w["ffn_norm"].data
        tensors[p + "ffn_gate.weight"] = w["w_gate"].data
        tensors[p + "ffn_up.weight"] = w["w_up"].data
        tensors[p + "ffn_down.weight"] = w["w_down"].data

    # ---- tensor descriptors + compute aligned relative offsets ----
    order = list(tensors.keys())
    desc_blobs = []
    offset = 0
    offsets = {}
    for name in order:
        arr = np.ascontiguousarray(tensors[name].astype(np.float32))
        # book convention: shape (out,in) row-major  -> GGUF dims = reversed = (in,out); ndims=1 for vectors
        dims = list(reversed(arr.shape)) if arr.ndim > 1 else [arr.shape[0]]
        offsets[name] = offset
        nbytes = arr.nbytes
        padded = (nbytes + ALIGN - 1) // ALIGN * ALIGN
        offset += padded
        desc = w_str(name.encode('ascii'))
        desc += struct.pack('<I', len(dims))
        for d in dims:
            desc += struct.pack('<Q', d)
        desc += struct.pack('<I', 0)          # GGML_F32 = 0
        desc += struct.pack('<Q', offsets[name])
        desc_blobs.append(desc)
    desc_bytes = b"".join(desc_blobs)

    header = struct.pack('<I', 0x46554747)      # magic 'GGUF'
    header += struct.pack('<I', 3)              # version
    header += struct.pack('<Q', len(order))     # n_tensors
    header += struct.pack('<Q', n_kv)            # n_kv

    pre_data = header + kv_bytes + desc_bytes
    pad = (-len(pre_data)) % ALIGN
    pre_data += b"\x00" * pad

    with open(out_path, "wb") as f:
        f.write(pre_data)
        for name in order:
            arr = np.ascontiguousarray(tensors[name].astype(np.float32))
            raw = arr.tobytes()
            f.write(raw)
            pad2 = ((len(raw) + ALIGN - 1) // ALIGN * ALIGN) - len(raw)
            f.write(b"\x00" * pad2)

    import os
    print(f"wrote {out_path}  ({os.path.getsize(out_path)} bytes, {len(order)} tensors, step {step})")
    return step

if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    export(os.path.join(here, "ckpt.npz"), os.path.join(here, "tinyshake.gguf"))
