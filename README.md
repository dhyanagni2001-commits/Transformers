# Transformer Implementation

This project implements a **decoder-only Transformer** model from scratch using **PyTorch** and trains it on the **Tiny Shakespeare** dataset. The goal is to understand the core components of the Transformer architecture, including self-attention, multi-head attention, and autoregressive text generation.

---

## Project Overview
- Implements a tiny decoder-only Transformer
- Uses self-attention with causal masking
- Trains a character-level language model
- Generates Shakespeare-like text after training
- Focuses on understanding Transformer internals using PyTorch

---

## Implementation Details
All implementation is done in:
- `transformer_model.py`

Only this file is modified and graded.

---

## Features Implemented
- Scaled dot-product self-attention
- Multi-head attention
- Transformer blocks with residual connections
- Token and positional embeddings
- Layer normalization
- Cross-entropy loss computation
- Autoregressive text generation

---

## How to Run
Train and evaluate the model:
```bash
python3 train.py
