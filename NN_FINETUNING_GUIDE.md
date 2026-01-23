# Neural Networks & Fine-Tuning: A Comprehensive Guide

An educational deep-dive into the concepts you need to understand for training and fine-tuning language models.

---

## Table of Contents

1. [Neural Network Fundamentals](#1-neural-network-fundamentals)
2. [How Language Models Work](#2-how-language-models-work)
3. [Training vs Fine-Tuning](#3-training-vs-fine-tuning)
4. [LoRA and Parameter-Efficient Fine-Tuning](#4-lora-and-parameter-efficient-fine-tuning)
5. [The Training Process](#5-the-training-process)
6. [Hyperparameters Deep Dive](#6-hyperparameters-deep-dive)
7. [Optimization](#7-optimization)
8. [Regularization](#8-regularization)
9. [Quantization](#9-quantization)
10. [Loss Functions and Metrics](#10-loss-functions-and-metrics)
11. [Overfitting and Underfitting](#11-overfitting-and-underfitting)
12. [Practical Debugging](#12-practical-debugging)
13. [Hardware Considerations](#13-hardware-considerations)

---

## 1. Neural Network Fundamentals

### What Is a Neural Network?

A neural network is a computational system loosely inspired by biological brains. It consists of layers of interconnected "neurons" that process information.

```
Input Layer      Hidden Layers       Output Layer
    ○                ○                   ○
    ○ ───────────→   ○ ───────────→      ○
    ○                ○                   ○
    ○                ○                   
                     ○                   
```

Each connection has a **weight**—a number that determines how much influence one neuron has on another. Training a neural network means adjusting these weights to produce desired outputs.

### Neurons and Activations

A single neuron:
1. Receives inputs (x₁, x₂, x₃, ...)
2. Multiplies each by its weight (w₁, w₂, w₃, ...)
3. Sums them up: z = w₁x₁ + w₂x₂ + w₃x₃ + ... + bias
4. Applies an activation function: output = f(z)

```
     x₁ ──w₁──╲
               ╲
     x₂ ──w₂───→ Σ + bias ──→ f(z) ──→ output
               ╱
     x₃ ──w₃──╱
```

### Activation Functions

Activation functions introduce non-linearity, allowing networks to learn complex patterns.

| Function | Formula | Use Case |
|----------|---------|----------|
| ReLU | max(0, x) | Most hidden layers |
| Sigmoid | 1/(1+e⁻ˣ) | Binary classification |
| Tanh | (eˣ-e⁻ˣ)/(eˣ+e⁻ˣ) | When you need negative values |
| GELU | x·Φ(x) | Modern transformers |
| SiLU/Swish | x·σ(x) | Modern architectures |

```
ReLU:                    Sigmoid:                 GELU:
     │    ╱                  │   ___                  │    ___
     │   ╱                   │  ╱                     │   ╱
─────┼──╱────            ────┼─╱─────             ────┼──╱────
     │                       │                        │ ╱
     │                       │                        │╱
```

### Layers

**Dense (Fully Connected)**: Every neuron connects to every neuron in the next layer. Used in simpler networks and output heads.

**Attention**: The key innovation in transformers. Allows the model to weigh the importance of different parts of the input.

**Embedding**: Converts discrete tokens (words) into continuous vectors.

**Normalization**: Stabilizes training by normalizing activations (LayerNorm, RMSNorm).

---

## 2. How Language Models Work

### The Transformer Architecture

Modern language models (GPT, Llama, Mistral) use the Transformer architecture.

```
┌─────────────────────────────────────────────────────────────┐
│                    TRANSFORMER BLOCK                        │
│                                                             │
│    Input                                                    │
│      ↓                                                      │
│   ┌─────────────────┐                                       │
│   │  Self-Attention │  ← "Which words should I attend to?"  │
│   └────────┬────────┘                                       │
│            ↓                                                │
│   ┌─────────────────┐                                       │
│   │   Layer Norm    │                                       │
│   └────────┬────────┘                                       │
│            ↓                                                │
│   ┌─────────────────┐                                       │
│   │  Feed-Forward   │  ← "Process what I attended to"       │
│   └────────┬────────┘                                       │
│            ↓                                                │
│   ┌─────────────────┐                                       │
│   │   Layer Norm    │                                       │
│   └────────┬────────┘                                       │
│            ↓                                                │
│         Output                                              │
└─────────────────────────────────────────────────────────────┘
              ↓
        (Repeat 32-80 times for large models)
```

### Self-Attention

The key insight: when processing a word, the model should consider other relevant words.

For "The cat sat on the mat because it was tired":
- When processing "it", attention should focus heavily on "cat"

Self-attention computes three things for each token:
- **Query (Q)**: "What am I looking for?"
- **Key (K)**: "What do I contain?"
- **Value (V)**: "What information do I provide?"

```
Attention(Q, K, V) = softmax(QK^T / √d) · V
```

The softmax creates attention weights that sum to 1, determining how much to attend to each position.

### Tokenization

Models don't see words—they see tokens. Tokenization breaks text into subword units.

```
"Understanding" → ["Under", "stand", "ing"]
"consciousness" → ["con", "scious", "ness"]
"Longchenpa"    → ["Long", "chen", "pa"] or ["Longch", "enpa"]
```

Each token maps to an embedding vector (typically 4096 dimensions for large models).

### Autoregressive Generation

Language models generate text one token at a time, each prediction conditioned on previous tokens:

```
Input:  "The shadow represents"
Step 1: "The shadow represents" → "those"
Step 2: "The shadow represents those" → "aspects"
Step 3: "The shadow represents those aspects" → "of"
...
```

### Parameters

The model's knowledge is stored in its parameters (weights). Count:

| Model | Parameters | Approximate Size |
|-------|------------|------------------|
| Llama 3.1 8B | 8 billion | ~16 GB (16-bit) |
| Mistral 7B | 7 billion | ~14 GB (16-bit) |
| Llama 3.1 70B | 70 billion | ~140 GB (16-bit) |
| DeepSeek V3 | 671 billion | ~1.3 TB (16-bit) |

---

## 3. Training vs Fine-Tuning

### Pre-training

Training a model from scratch on massive data (trillions of tokens).

```
Random Weights → Train on Internet Text → General Language Model
                 (months, millions of $)
```

**Objective**: Predict the next token
**Data**: Books, websites, code, conversations
**Cost**: $1M - $100M+ for large models
**Time**: Weeks to months on thousands of GPUs

### Fine-Tuning

Adapting a pre-trained model to a specific task or style.

```
Pre-trained Model → Train on Your Data → Specialized Model
                    (hours, affordable)
```

**Objective**: Same (next token prediction) but on your data
**Data**: Hundreds to thousands of examples
**Cost**: $0 - $1000
**Time**: Minutes to hours

### Why Fine-Tuning Works

Pre-training learns general language patterns. Fine-tuning:
1. Preserves general knowledge
2. Adjusts output distribution toward your domain
3. Learns new patterns from your examples

It's like teaching a fluent English speaker medical terminology—you don't start from scratch.

### Types of Fine-Tuning

| Type | What Changes | Use Case |
|------|--------------|----------|
| Full fine-tuning | All parameters | Maximum quality, high resource cost |
| LoRA | Small adapter matrices | Good balance, practical for most |
| QLoRA | LoRA + quantized base | Memory efficient |
| Prefix tuning | Only prompt embeddings | Lightweight, limited capacity |
| Prompt tuning | Soft prompt vectors | Very lightweight |

---

## 4. LoRA and Parameter-Efficient Fine-Tuning

### The Problem

Full fine-tuning updates all parameters:
- 7B model = 7 billion parameters to update
- Needs full copy of model in memory (for gradients)
- ~50GB+ VRAM for a 7B model

### LoRA Solution

**Low-Rank Adaptation** adds small trainable matrices alongside frozen weights.

Instead of updating weight matrix W directly:
```
Original: W (4096 × 4096 = 16.7M parameters)

LoRA decomposes the update:
  A: 4096 × r (where r = 16 or 32)
  B: r × 4096
  
  New output = W·x + (A·B)·x
```

With rank 16:
- A: 4096 × 16 = 65,536 parameters
- B: 16 × 4096 = 65,536 parameters
- Total: 131,072 vs 16,777,216 (128× smaller!)

### Visual Representation

```
┌─────────────────────────────────────────────────────────────┐
│                         LORA                                │
│                                                             │
│   Input ──→ ┌──────────────┐                                │
│             │   W (frozen) │──────────────┐                 │
│             └──────────────┘              │                 │
│                    │                      ↓                 │
│                    │              ┌───────────────┐         │
│                    │              │   A (r×d_in)  │         │
│                    │              └───────┬───────┘         │
│                    │                      ↓                 │
│                    │              ┌───────────────┐         │
│                    │              │   B (d_out×r) │         │
│                    │              └───────┬───────┘         │
│                    │                      │                 │
│                    └───────→ (+) ←────────┘                 │
│                               ↓                             │
│                            Output                           │
│                                                             │
│   Only A and B are trained. W stays frozen.                 │
└─────────────────────────────────────────────────────────────┘
```

### LoRA Hyperparameters

**Rank (r)**

The "width" of the adapter. Higher = more capacity but more parameters.

| Rank | Parameters (per layer) | Use Case |
|------|------------------------|----------|
| 8 | ~130K | Quick experiments |
| 16 | ~260K | General fine-tuning |
| 32 | ~520K | Style/voice (recommended for your project) |
| 64 | ~1M | Complex tasks |
| 128 | ~2M | Approaching full fine-tuning |

**Alpha (α)**

Scaling factor for LoRA contribution. Usually set to 2× rank.

```
output = W·x + (α/r) · (A·B)·x
```

If r=32 and α=64, the LoRA contribution is scaled by 64/32 = 2.

**Target Modules**

Which layers get LoRA adapters. Common choices:
- `q_proj, v_proj` (query and value attention)
- `q_proj, k_proj, v_proj, o_proj` (all attention)
- All linear layers (maximum adaptation)

**Dropout**

Probability of zeroing LoRA outputs during training. Prevents overfitting.
- 0.0: No dropout
- 0.05: Light regularization (typical)
- 0.1: Stronger regularization

### QLoRA

Combines LoRA with quantization:
- Base model in 4-bit (frozen)
- LoRA adapters in 16-bit (trainable)

```
Memory comparison (7B model):
- Full fine-tuning: ~50 GB
- LoRA (16-bit base): ~30 GB
- QLoRA (4-bit base): ~8 GB
```

---

## 5. The Training Process

### Forward Pass

Data flows through the network to produce a prediction.

```
Input: "The shadow represents"
   ↓
Tokenize: [464, 12261, 11105]
   ↓
Embed: [[0.23, -0.11, ...], [0.87, 0.34, ...], ...]
   ↓
Transformer Layers (×32)
   ↓
Output Logits: [0.1, 0.05, 2.3, -1.2, ...]  (one score per vocab token)
   ↓
Softmax: [0.02, 0.01, 0.58, 0.01, ...]  (probabilities)
   ↓
Prediction: "those" (highest probability)
```

### Loss Calculation

Compare prediction to actual next token.

```
Predicted probability for "those": 0.58
Actual token: "those"

Cross-entropy loss = -log(0.58) = 0.54
```

Lower loss = model assigned higher probability to correct token.

### Backward Pass (Backpropagation)

Calculate how much each weight contributed to the error.

```
Loss
  ↓
Compute gradients (∂Loss/∂w for each weight)
  ↓
Gradients flow backward through layers
  ↓
Each weight gets a gradient: "move this direction to reduce loss"
```

### Weight Update

Adjust weights in the direction that reduces loss.

```
w_new = w_old - learning_rate × gradient
```

### Training Loop

```python
for epoch in range(num_epochs):
    for batch in training_data:
        # Forward pass
        predictions = model(batch.input)
        
        # Calculate loss
        loss = cross_entropy(predictions, batch.target)
        
        # Backward pass
        gradients = compute_gradients(loss)
        
        # Update weights
        for param in model.parameters():
            param -= learning_rate * param.gradient
        
        # Clear gradients for next iteration
        clear_gradients()
```

### Batching

Process multiple examples at once for efficiency and stability.

```
Batch size 1:
  Example 1 → gradient → update
  Example 2 → gradient → update  (noisy, each example pulls in different direction)
  
Batch size 8:
  Examples 1-8 → average gradient → update  (smoother, more stable)
```

---

## 6. Hyperparameters Deep Dive

### Learning Rate

The most important hyperparameter. Controls step size during optimization.

```
w_new = w_old - learning_rate × gradient
```

**Too High (1e-3)**
```
Loss: 2.5 → 3.1 → 8.7 → NaN  ← Divergence!

         Loss
          ↑
          │    ╱╲
          │   ╱  ╲    ╱╲
          │  ╱    ╲  ╱  ╲
          │ ╱      ╲╱    ╲ ...
          └─────────────────→ Steps
```

**Too Low (1e-6)**
```
Loss: 2.5 → 2.49 → 2.48 → 2.47 → ...  ← Too slow!

         Loss
          ↑
          │────────────────
          │                 ← Barely moving
          │
          └─────────────────→ Steps
```

**Just Right (2e-4 typical)**
```
Loss: 2.5 → 2.1 → 1.7 → 1.3 → 1.1 → 1.05  ← Good!

         Loss
          ↑
          │╲
          │ ╲
          │  ╲___
          │      ╲___
          └─────────────────→ Steps
```

**Recommended Values**

| Model Size | Starting LR |
|------------|-------------|
| < 1B | 5e-4 to 1e-3 |
| 1-10B | 1e-4 to 3e-4 |
| 10B+ | 5e-5 to 2e-4 |

For LoRA fine-tuning, `2e-4` is a safe default.

### Learning Rate Schedules

Changing LR during training often improves results.

**Constant**
```
LR: ═══════════════════════
```

**Linear Decay**
```
LR: ╲
     ╲
      ╲
       ╲___
```

**Cosine Decay**
```
LR: ╲
     ╲
      ╲__
         ╲__
            ╲
```

**Warmup + Decay (Recommended)**
```
LR:    ╱╲
      ╱  ╲
     ╱    ╲__
    ╱        ╲___
   ╱
```

Start low (warmup), then decay. Prevents early instability.

### Batch Size

Number of examples processed before weight update.

| Batch Size | Pros | Cons |
|------------|------|------|
| Small (1-4) | Fits in memory, can escape local minima | Noisy gradients, slower convergence |
| Medium (8-32) | Good balance | - |
| Large (64-256) | Stable gradients, faster | Needs more memory, may generalize worse |

**Gradient Accumulation**

Simulate larger batches without more memory:

```python
accumulation_steps = 4
actual_batch_size = 2
effective_batch_size = 2 × 4 = 8

for i, batch in enumerate(data):
    loss = model(batch)
    loss.backward()  # Accumulate gradients
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()  # Update weights
        optimizer.zero_grad()  # Clear gradients
```

### Epochs

One epoch = one complete pass through training data.

```
Dataset: 2000 examples
Batch size: 4
Steps per epoch: 2000 / 4 = 500

3 epochs = 1500 steps
```

**How Many Epochs?**

| Dataset Size | Recommended Epochs |
|--------------|-------------------|
| < 500 | 5-10 |
| 500-2000 | 3-5 |
| 2000-10000 | 1-3 |
| 10000+ | 1 |

More data = fewer epochs needed. Too many epochs = overfitting.

### Sequence Length

Maximum tokens per example. Affects memory and context.

```
"What is the shadow?" (6 tokens)
→ Padded to sequence length or truncated if longer

Sequence length 512:  [tokens...][PAD][PAD][PAD]...
Sequence length 2048: More context, 4× memory
```

| Length | Memory | Use Case |
|--------|--------|----------|
| 512 | Low | Short Q&A |
| 2048 | Medium | Most fine-tuning |
| 4096 | High | Long-form content |
| 8192+ | Very high | Full documents |

Memory scales roughly quadratically with sequence length (due to attention).

### Warmup

Gradually increase LR at the start of training.

```
Without warmup:
Step 1: LR = 2e-4, weights are random-ish
→ Large, random gradient × high LR = chaos

With warmup:
Step 1: LR = 2e-6 (tiny)
Step 100: LR = 2e-4 (full)
→ Model stabilizes before large updates
```

**Settings**
- `warmup_steps = 100` (fixed number)
- `warmup_ratio = 0.1` (10% of training)

### Weight Decay

L2 regularization—penalizes large weights.

```
Loss_total = Loss_prediction + λ × Σ(weights²)
```

Prevents any single weight from dominating. Typical: `0.01` to `0.1`.

---

## 7. Optimization

### Gradient Descent

The fundamental algorithm: move in the direction that reduces loss.

```
repeat:
    gradient = compute_gradient(loss)
    weights = weights - learning_rate × gradient
```

### Stochastic Gradient Descent (SGD)

Compute gradient on a random subset (mini-batch) instead of full dataset.

Pros: Faster, can escape local minima
Cons: Noisy, may oscillate

### Momentum

Remember previous gradient direction, smooth out oscillations.

```
velocity = β × velocity + gradient  (β ≈ 0.9)
weights = weights - learning_rate × velocity
```

Like a ball rolling downhill—builds up speed, doesn't stop at every bump.

### Adam (Adaptive Moment Estimation)

The default optimizer for transformers. Combines:
- Momentum (first moment)
- Adaptive per-parameter learning rates (second moment)

```
m = β₁ × m + (1-β₁) × gradient        # Momentum
v = β₂ × v + (1-β₂) × gradient²       # Squared gradient (for adaptation)

weights = weights - lr × m / (√v + ε)
```

Each parameter gets its own effective learning rate based on gradient history.

**Adam Parameters**
- `β₁ = 0.9` (momentum)
- `β₂ = 0.999` (squared gradient decay)
- `ε = 1e-8` (numerical stability)

### AdamW

Adam with decoupled weight decay. Standard for modern LLMs.

```
# Regular Adam applies weight decay to gradient
# AdamW applies it directly to weights

weights = weights - lr × (adam_update + weight_decay × weights)
```

### Gradient Clipping

Prevent exploding gradients by capping their magnitude.

```python
max_grad_norm = 1.0

total_norm = sqrt(sum(grad² for all grads))
if total_norm > max_grad_norm:
    scale = max_grad_norm / total_norm
    for grad in all_grads:
        grad *= scale
```

Essential for stable training, especially with long sequences.

---

## 8. Regularization

Techniques to prevent overfitting (memorizing training data).

### Dropout

Randomly zero out neurons during training.

```
Training:
  [0.5, 0.3, 0.8, 0.2] → [0.5, 0, 0.8, 0]  (30% dropout)
  
Inference:
  [0.5, 0.3, 0.8, 0.2] → [0.5, 0.3, 0.8, 0.2]  (no dropout)
```

Forces network to not rely on any single neuron. Typical: 0.1 (10%).

### Weight Decay

Penalize large weights (see above). Typical: 0.01.

### Data Augmentation

Create variations of training examples:
- Rephrase questions
- Add/remove context
- Synonym substitution

### Early Stopping

Stop training when validation loss stops improving.

```
         Loss
          ↑
          │╲
   train  │ ╲____
          │      ╲___
          │          ╲___
          │   ╱ validation
          │  ╱
          │ ╱
          └─────────────────→ Epochs
                   ↑
              Stop here (val loss increasing)
```

### Label Smoothing

Instead of hard targets (1 for correct, 0 for wrong), use soft targets:
- Correct: 0.9
- Incorrect: 0.1 / (vocab_size - 1)

Prevents overconfidence.

---

## 9. Quantization

Reducing numerical precision to save memory.

### Number Formats

```
Float32 (FP32):  1 sign + 8 exponent + 23 mantissa = 32 bits
Float16 (FP16):  1 sign + 5 exponent + 10 mantissa = 16 bits
BFloat16 (BF16): 1 sign + 8 exponent + 7 mantissa  = 16 bits
Int8:            8-bit integer
Int4:            4-bit integer
```

### Memory Savings

| Format | Bits/Param | 7B Model Size |
|--------|------------|---------------|
| FP32 | 32 | 28 GB |
| FP16/BF16 | 16 | 14 GB |
| INT8 | 8 | 7 GB |
| INT4 | 4 | 3.5 GB |

### Quantization Methods

**Post-Training Quantization**

Quantize a trained model:
1. Collect calibration data
2. Determine scale factors for each layer
3. Convert weights to lower precision

**Quantization-Aware Training**

Train with quantization in mind:
1. Simulate quantization during training
2. Model learns to be robust to precision loss

### For Fine-Tuning: QLoRA

```
┌─────────────────────────────────────────────────────────────┐
│                         QLoRA                               │
│                                                             │
│   Base Model Weights: 4-bit (frozen, compressed)            │
│   LoRA Adapters: 16-bit (trainable, full precision)         │
│                                                             │
│   Benefit: Train large models on consumer GPUs              │
│   Cost: Slightly lower quality than full precision          │
└─────────────────────────────────────────────────────────────┘
```

### MLX and Quantization

MLX supports 4-bit and 8-bit quantized models:

```bash
# Download pre-quantized model
huggingface-cli download mlx-community/Llama-3.1-8B-Instruct-4bit

# Or quantize yourself
mlx_lm.convert --hf-path meta-llama/Llama-3.1-8B-Instruct -q --q-bits 4
```

---

## 10. Loss Functions and Metrics

### Cross-Entropy Loss

The standard loss for language models.

```
For each token position:
  - Model outputs probability distribution over vocabulary
  - Loss = -log(probability of correct token)
  
If model predicts correct token with 90% confidence:
  Loss = -log(0.9) = 0.105

If model predicts correct token with 10% confidence:
  Loss = -log(0.1) = 2.303
```

Lower loss = higher confidence in correct tokens.

### Perplexity

Exponentiated average loss. Intuition: "How many tokens is the model choosing between?"

```
Perplexity = e^(average_loss)

Loss = 2.0 → Perplexity = e² ≈ 7.4
Loss = 1.0 → Perplexity = e¹ ≈ 2.7
```

Lower is better. A perplexity of 10 means the model is "as confused as if choosing between 10 equally likely tokens."

### Typical Loss Values

| Stage | Loss | Interpretation |
|-------|------|----------------|
| Random init | 10+ | Complete chaos |
| Early training | 3-5 | Learning basics |
| Mid training | 1.5-2.5 | Getting better |
| Well-trained | 0.8-1.5 | Good model |
| Overfit | < 0.5 | Possibly memorizing |

### Training vs Validation Loss

```
         Loss
          ↑
          │╲
   train  │ ╲____
          │      ╲___
          │          ╲___  ← Good: both decrease
          │
          │   _____
   valid  │  ╱     ╲
          │ ╱       ╲___  ← Bad: valid increases (overfitting)
          │╱
          └─────────────────→ Steps
```

Monitor both. If validation loss increases while training loss decreases = overfitting.

---

## 11. Overfitting and Underfitting

### Overfitting

Model memorizes training data, fails on new data.

**Symptoms**
- Training loss very low (< 0.5)
- Validation loss increasing or much higher
- Model repeats exact phrases from training data
- Poor performance on new prompts

**Causes**
- Too many epochs
- Model too large for dataset
- Dataset too small
- Too little regularization

**Solutions**
- Fewer epochs
- Add dropout (0.05-0.1)
- Increase weight decay
- More training data
- Data augmentation
- Early stopping

### Underfitting

Model hasn't learned enough.

**Symptoms**
- Both train and validation loss high
- Outputs are generic, don't capture desired style
- Model ignores fine-tuning, behaves like base model

**Causes**
- Too few epochs
- Learning rate too low
- Model capacity too low (rank too small)
- Training data poor quality

**Solutions**
- More epochs
- Higher learning rate
- Higher LoRA rank
- Better training data
- Longer training

### The Sweet Spot

```
         Error
          ↑
          │╲                      ╱
          │ ╲    Validation     ╱
          │  ╲                ╱
          │   ╲    ___     ╱
          │    ╲__╱   ╲___╱
          │     ↑
          │  Sweet spot
          │         ╲
   Train  │          ╲___
          │              ╲___
          │                  ╲___
          └─────────────────────────→ Model Complexity / Training Time
          
     Underfitting          │         Overfitting
         ←─────────────────┼────────────────→
```

---

## 12. Practical Debugging

### Loss Not Decreasing

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| Loss stays high | LR too low | Increase LR by 2-5× |
| Loss stays high | Data format wrong | Check tokenization, templates |
| Loss oscillates wildly | LR too high | Decrease LR by 2-5× |
| Loss goes to NaN | LR way too high or gradient explosion | Lower LR, add gradient clipping |
| Loss decreases then spikes | Learning rate schedule issue | Add warmup |

### Model Outputs Garbage

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| Completely random text | Training failed | Check loss curve |
| Repeats same phrase | Overfitting | Fewer epochs, more data |
| Ignores fine-tuning | Underfitting | More epochs, higher rank |
| Wrong chat format | Template mismatch | Check tokenizer chat template |

### Memory Errors

| Error | Fix |
|-------|-----|
| CUDA/MPS out of memory | Reduce batch size, use gradient accumulation |
| Still OOM | Reduce sequence length |
| Still OOM | Use 4-bit quantization |
| Still OOM | Use smaller model |

### Monitoring Training

```python
# Log these metrics every N steps:
- step
- train_loss
- validation_loss (periodically)
- learning_rate (if using schedule)
- gradient_norm
- time_per_step
```

Plot loss curves. They tell you almost everything:

```
Good training:           Need more epochs:        Overfitting:
   ↑                        ↑                        ↑
   │╲                       │╲                       │╲ train
   │ ╲                      │ ╲                      │ ╲____
   │  ╲___                  │  ╲___                  │
   │      ╲___              │      (still going)    │    ╱ val
   └───────────→            └───────────→           └───────────→
```

### Checkpointing

Save model periodically so you can:
1. Resume if training crashes
2. Compare different checkpoints
3. Recover if you overtrain

```bash
--save-every 500  # Save every 500 steps
```

---

## 13. Hardware Considerations

### GPU Memory (VRAM)

The main constraint for training.

**What Uses Memory**
1. Model weights
2. Optimizer states (2× weights for Adam)
3. Gradients (1× weights)
4. Activations (scales with batch size × sequence length)

**Rough Estimates (LoRA fine-tuning)**

| Model | 4-bit | 8-bit | 16-bit |
|-------|-------|-------|--------|
| 7B | ~6 GB | ~10 GB | ~18 GB |
| 14B | ~10 GB | ~18 GB | ~35 GB |
| 70B | ~40 GB | ~80 GB | ~160 GB |

### Apple Silicon (M4 Max)

**Advantages**
- Unified memory: CPU and GPU share RAM
- 64GB is usable for training
- MLX optimized for Apple Silicon

**Limitations**
- Slower than NVIDIA for training (but totally usable)
- No CUDA (need MLX or MPS backend)

**Realistic Expectations**

| Model | Fits? | Training Speed |
|-------|-------|----------------|
| 7B 4-bit | ✅ Easy | ~50-100 tokens/sec |
| 14B 4-bit | ✅ Yes | ~25-50 tokens/sec |
| 32B 4-bit | ⚠️ Tight | ~10-20 tokens/sec |

### Cloud Options

| Service | GPU | VRAM | Cost |
|---------|-----|------|------|
| Tinker | Varies | Enough for 70B+ | Pay per use |
| Lambda Labs | A100 | 80 GB | ~$1.50/hr |
| Runpod | A100/H100 | 40-80 GB | ~$1-3/hr |
| Google Colab Pro | A100 | 40 GB | ~$10/month |

### Training Time Estimates

For your project (2000 examples, 3 epochs, batch size 4):

| Hardware | Model | Time |
|----------|-------|------|
| M4 Max 64GB | Llama 8B 4-bit | 1-2 hours |
| A100 80GB | Llama 8B 16-bit | 15-30 min |
| Tinker | DeepSeek V3 | Minutes |

---

## Quick Reference Card

### Hyperparameter Starting Points

```
Learning Rate:     2e-4
Batch Size:        4 (local), 32 (cloud)
LoRA Rank:         32 (for style/voice)
LoRA Alpha:        64 (2× rank)
LoRA Dropout:      0.05
Epochs:            3-5 (for ~2000 examples)
Warmup:            10% of steps
Weight Decay:      0.01
Gradient Clipping: 1.0
Sequence Length:   2048
```

### Debugging Checklist

```
□ Loss decreasing steadily?
□ Validation loss tracking training loss?
□ No NaN or Inf in loss?
□ Outputs make sense on test prompts?
□ Not repeating training data verbatim?
□ Memory usage stable?
```

### Command Templates

```bash
# MLX Training
mlx_lm.lora \
    --model mlx-community/Llama-3.1-8B-Instruct-4bit \
    --train --data ./data/mlx \
    --batch-size 4 --lora-rank 32 --iters 1500 \
    --learning-rate 2e-4 --warmup 0.1 \
    --adapter-path ./adapters/my_model

# MLX Inference
mlx_lm.generate \
    --model mlx-community/Llama-3.1-8B-Instruct-4bit \
    --adapter-path ./adapters/my_model \
    --prompt "Your prompt here"
```

---

## Further Reading

- [Attention Is All You Need (Transformer paper)](https://arxiv.org/abs/1706.03762)
- [LoRA: Low-Rank Adaptation (paper)](https://arxiv.org/abs/2106.09685)
- [QLoRA (paper)](https://arxiv.org/abs/2305.14314)
- [MLX Documentation](https://ml-explore.github.io/mlx/)
- [Andrej Karpathy's Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [Hugging Face Transformers Course](https://huggingface.co/learn/nlp-course/)