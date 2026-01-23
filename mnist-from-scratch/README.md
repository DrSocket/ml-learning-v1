# MNIST From Scratch

Building digit classification twice: first with NumPy only, then with PyTorch.

## Why This Project?

This isn't just about classifying digits—it's about **deeply understanding** how neural networks learn. By implementing backpropagation manually, you'll:

- Understand what PyTorch does "under the hood"
- Build intuition for gradients and the chain rule
- Debug confidently because you know what's happening
- Appreciate abstractions when you use them later

## Project Structure

```
mnist-from-scratch/
├── numpy_net.py      # Part A: Pure NumPy implementation
├── pytorch_net.py    # Part B: PyTorch implementation
├── train.py          # Training script with logging
├── utils.py          # Data loading, visualization
└── README.md         # This file
```

## Part A: NumPy Implementation

**Goal**: Implement a 2-layer neural network using only NumPy

### Architecture
```
Input (784) → Hidden (128, ReLU) → Output (10, Softmax)
```

### What You'll Implement
- [ ] Forward pass (matrix multiplication + activations)
- [ ] Loss function (cross-entropy)
- [ ] Backward pass (gradients via chain rule)
- [ ] SGD optimizer (weight updates)
- [ ] Training loop

### Target: 95%+ accuracy

## Part B: PyTorch Implementation

**Goal**: Rebuild the same network using PyTorch's abstractions

### What You'll Use
- `nn.Module` for model definition
- `nn.Linear` for layers
- `torch.optim` for optimization
- `DataLoader` for batching

### Target: 98%+ accuracy

## Learning Checkpoints

After completing this project, you should be able to:

- [ ] Explain backpropagation without looking anything up
- [ ] Derive gradients for a 2-layer network on paper
- [ ] Write a training loop from memory
- [ ] Explain what `.backward()` does internally
- [ ] Debug shape mismatch errors by tracing dimensions

## Running the Code

```bash
# Train NumPy version
uv run python train.py --model numpy

# Train PyTorch version
uv run python train.py --model pytorch

# With experiment tracking
uv run python train.py --model pytorch --wandb
```

## Resources

- [3Blue1Brown: Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) - Visual intuition
- [Andrej Karpathy: Micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0) - Building autograd from scratch
- [CS231n: Backpropagation](https://cs231n.github.io/optimization-2/) - Math details

## My Learnings

### utils.py

#### one_hot_encode
MNIST labels are just integers 0–9 (e.g., 3).
One-hot = make a length-10 vector with a single 1 at index = label (3 → [0,0,0,1,0,0,0,0,0,0]).
Why bother? The target shape needs to match the model’s 10-class output so the loss makes sense.
Example model output: [0.1, 0.05, 0.02, 0.70, 0.03, 0.02, 0.02, 0.03, 0.02, 0.01] → picks class 3 because 0.70 is highest.

#### create_batches
Takes list of samples and labels and shuffles them changing the order and slicing them into mini-batches, why? because shuffling each epoch helps avoid learning artifacts from a fixed ordering, it doesn't increase variety, for variety you'd use augmentation, now we're just randomizing order and batching.
It preserves pairing: shuffling is done on indices, so each label stays with its sample.
Batches are just slices; the last batch can be smaller than batch_size.

### numpy_net.py


### Things that confused me:

### Aha moments:

### Mistakes I made:

