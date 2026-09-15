# training/ — the model used in the book's appendix

This directory trains the tiny Llama-style model that the book's
**Appendix: Training and Running a Real Model** loads and generates from.

## Just want to generate text?
You don't need any of this. The trained model is already here:

```bash
./llm_cli --model training/tinyshake.gguf --prompt "ROMEO:" --max-new 150
```

## Files
| file | what it does |
|---|---|
| `autodiff.py` | minimal reverse-mode autodiff over NumPy |
| `tinyllama.py` | the Llama-style model (RoPE, GQA attention, SwiGLU, RMSNorm) — formulas match `src/model.cpp` exactly |
| `losses.py`, `adam.py` | cross-entropy loss, Adam optimizer |
| `train.py` | the training loop (byte-level, tiny-shakespeare) |
| `gguf_export.py` | writes a real GGUF file the book's `GGUFReader` can load |
| `preview.py` | generate from a checkpoint, in Python |
| `grad_check*.py` | **finite-difference gradient checks — run these first** |
| `tinyshake.gguf` | the trained model used in the appendix (2.6 MB) |

## Verify the math before trusting it
Same discipline as the engine: nothing is taken on faith.

```bash
pip install numpy
python3 grad_check.py            # every primitive vs finite differences
python3 grad_check_attention.py  # RoPE + GQA causal attention
python3 grad_check_fullmodel.py  # the whole model, end to end
```
All three should report `ALL ... OK` (relative errors ~1e-8 or better).

## Train your own
```bash
# fetch the corpus (1.1 MB)
curl -o /tmp/shakespeare.txt \
  https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

python3 train.py 1500            # start; writes ckpt.npz
python3 train.py 1500 resume     # continue from the checkpoint
python3 gguf_export.py           # ckpt.npz -> tinyshake.gguf
```
Training is CPU-only NumPy: roughly 0.07 s/step at the default size
(4 layers, hidden 128, 656k parameters). The appendix's model was trained for
8,700 steps (~10 minutes), reaching a loss of about 1.89.

## Architecture
Matches what `load_config` reads from the GGUF metadata:
vocab 256 (byte-level, BOS = byte 0), hidden 128, 4 layers,
4 heads / 2 KV heads (GQA), FFN 256, rope_theta 10000.
