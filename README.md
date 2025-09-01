# Mini-GPT-Inference-in-C
This project is a **mini GPT-style language model inference engine** written in C (~1000 LOC). It demonstrates the fundamental mechanics of transformer-based text generation without external ML libraries. It is designed for educational purposes, simplicity, and hackability.
# Mini GPT Inference in C

This project is a **mini GPT-style language model inference engine** written in C (~1000 LOC). It demonstrates the fundamental mechanics of transformer-based text generation without external ML libraries. It is designed for educational purposes, simplicity, and hackability.

---

## ✨ Features
- **Byte-level tokenizer** (no complicated BPE).
- **Decoder-only Transformer** architecture with:
  - LayerNorm
  - Causal Self-Attention
  - GELU-based MLP
  - Residual connections
- **Sinusoidal positional encoding** (lightweight & fast).
- **Sampling strategies**:
  - Temperature scaling
  - Top-k sampling
- **Configurable model size** via CLI flags (layers, heads, hidden size, context length).
- **Pure C implementation** with no dependencies beyond `<math.h>`.

---

## ⚙️ Build
```bash
cc -O2 -o mini_gpt mini_gpt_c_1000loc.c -lm
```

---

## 🚀 Run
```bash
./mini_gpt "Hello, world!" --steps 256 --temp 0.9 --topk 50
```

Arguments:
- `--steps N` → number of tokens to generate
- `--temp T` → temperature for sampling (default: 1.0)
- `--topk K` → restricts sampling to top-K logits
- `--layers L` → number of transformer layers
- `--heads H` → number of attention heads
- `--hidden D` → hidden dimension size
- `--ctx C` → context length

---

## 📚 Example
```bash
./mini_gpt "Once upon a time" --steps 50 --temp 0.8 --topk 40
```
Output might look like:
```
Once upon a time there was a small model written in C that tried to tell stories like GPT. It wasn’t perfect, but it worked surprisingly well.
```

---

## 🔮 Future Improvements
- Add **KV-cache** for faster long-sequence generation.
- Switch to **Rotary Positional Embeddings (RoPE)** for smoother scaling.
- Support **binary weight loading** (e.g., LLaMA-style pre-trained weights).
- Add **GPU acceleration** (CUDA/OpenCL).

---

## 🎯 Purpose
This project is **not meant to rival PyTorch/TensorFlow**, but to show that GPT inference can be expressed in **a single C file**. Perfect for learning, tinkering, and embedding into lightweight systems.
