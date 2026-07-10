# GPT-2: From-Scratch PyTorch Implementation 

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/) 
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/) 
[![CUDA](https://img.shields.io/badge/CUDA-Enabled-76b900.svg)](https://developer.nvidia.com/cuda-zone)

## Overview
This repository contains a complete, from-scratch implementation of the GPT-2 (Generative Pre-trained Transformer 2) architecture using Python, PyTorch, and CUDA. Rather than relying on high-level wrapper libraries (such as HuggingFace `transformers`), this project manually implements the core components of the transformer decoder architecture. 

The primary objective of this project was to develop an intimate, tensor-level understanding of large language models. By building the network's internal mechanics—from causal masking to auto-regressive decoding—this codebase serves as a foundational sandbox for mechanistic interpretability and safety evaluations.

## 🧠 Architecture & Implementation Details

The model closely follows the original GPT-2 architecture, emphasizing modularity and computational efficiency:

*   **Multi-Head Causal Self-Attention:** Implemented the core scaled dot-product attention mechanism. Tensor reshaping and transpositions are heavily optimized using `einops` for readable and efficient operations.
    *   Attention formula applied: 
    
$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

*   **Causal Masking:** Engineered lower-triangular masking to ensure the auto-regressive property, strictly preventing future token leakage during the forward pass.
*   **Causal Masking:** Engineered lower-triangular masking to ensure the auto-regressive property, strictly preventing future token leakage during the forward pass.
*   **Embeddings & Normalization:** 
    *   Implemented learned token embeddings and absolute positional embeddings.
    *   Manual integration of Layer Normalization (`LayerNorm`) applied before the attention and feed-forward blocks (pre-norm formulation characteristic of GPT-2).
*   **Residual Connections:** Robust gradient flow maintained across deep layers using standard residual pathways.
*   **Feed-Forward Network (FFN):** Implemented the two-layer linear expansion with GELU activation functions.

## ⚙️ Auto-Regressive Generation & Decoding

To make the model a fully functional generative engine, the inference pipeline supports multiple controllable decoding strategies:

*   **Greedy Decoding:** Selects the highest probability `argmax` token at each step for deterministic outputs.
*   **Multinomial Sampling:** Samples from the raw probability distribution to introduce generation variance.
*   **Top-K Filtering:** Restricts the sampling pool to the $k$ most likely next tokens, truncating the long tail of the distribution to prevent highly improbable (hallucinated) generations.
*   **Temperature Scaling:** Dynamically adjusts the softmax logits via a scalar divisor ($T$) prior to sampling, allowing fine-grained control over the determinism/creativity trade-off.

## 🚀 Training & Fine-Tuning Infrastructure

The model was designed to scale. The training pipeline includes:
*   **Hardware Acceleration:** Full CUDA support for GPU tensor operations.
*   **Distributed Fine-Tuning:** The pre-trained architecture was successfully fine-tuned on a custom text dataset utilizing a high-performance GPU cluster.
*   **Optimization:** Configured with AdamW optimizer, incorporating weight decay and learning rate scheduling for stable convergence.

## 🔬 Relevance to AI Safety & Interpretability

This codebase is specifically structured to facilitate **Mechanistic Interpretability** research. Because the architecture is completely decoupled from opaque third-party libraries, it allows for:
1.  **Activation Patching:** Easy hooks into intermediate residual streams to intervene on hidden states.
2.  **Attention Head Deconstruction:** Direct access to individual $Q$, $K$, and $V$ matrices to trace circuit-level behaviors (e.g., induction heads).
3.  **Adversarial Probing:** Complete visibility into the logit distributions during top-k sampling for automated red-teaming and failure state analysis.

## 📂 Project Structure

```text
├── model/
│   ├── attention.py       # Multi-Head Causal Self-Attention & einops logic
│   ├── transformer.py     # Core GPT-2 block (LayerNorm, FFN, Residuals)
│   ├── embeddings.py      # Token & Positional embeddings
│   └── generation.py      # Decoding strategies (Top-K, Temperature)
├── train/
│   ├── dataloader.py      # Custom dataset parsing and batching
│   └── trainer.py         # Training loop, GPU cluster allocation, loss tracking
├── weights/               # Saved state_dicts from cluster fine-tuning
├── main.py                # Entry point for training or generation
└── README.md
```

## 💻 Usage

**Text Generation Example:**
```python
from model.transformer import GPT2
from model.generation import Generator

# Initialize model and load fine-tuned weights
model = GPT2(vocab_size=50257, d_model=768, n_heads=12, n_layers=12).to('cuda')
model.load_state_dict(torch.load('weights/custom_finetune.pt'))

# Initialize generator
generator = Generator(model)

# Generate text with Top-K and Temperature scaling
prompt = "The fundamental challenge of aligning frontier models is"
output = generator.generate(
    prompt=prompt, 
    max_tokens=50, 
    temperature=0.8, 
    top_k=40
)

print(output)
```
