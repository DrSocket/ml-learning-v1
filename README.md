# Deep Learning Journey: AI-Assisted Self-Directed Learning

<div align="center">

![PyTorch](https://img.shields.io/badge/PyTorch-2.9+-EE4C2C?style=for-the-badge&logo=pytorch)
![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter)
![Open Source](https://img.shields.io/badge/Open%20Source-❤️-brightgreen?style=for-the-badge)

*A structured, AI-guided approach to learning deep learning through hands-on projects*

[📖 Learning Plan](learning_plan.md) • [🤖 AI Learning Guidelines](AGENT.md) • [🚀 Quick Start](#getting-started)

</div>

---

## 🎯 What Makes This Different

This isn't just another deep learning tutorial—it's a **living framework for AI-assisted self-directed learning**. Inspired by:

- **Andrei Karpathy**'s ["Neural Networks: Zero to Hero"](https://karpathy.ai/zero-to-hero.html) series—building intuition through implementation
- **Gabriel Petersson**'s talks on AI as a learning partner rather than just a tool

Instead of passive consumption, this project structures your learning journey with an AI tutor that:
- 🧠 **Deepens understanding** through Socratic questioning
- 🔍 **Builds debugging skills** rather than providing solutions
- 🎯 **Connects theory to intuition** before diving into code
- 📈 **Guides you through progressive projects** from basics to production

## 🗺️ Your Learning Roadmap

This 6-month curriculum takes you from neural network fundamentals to production deployment:

| Phase | Duration | Focus | Key Project |
|-------|----------|-------|-------------|
| **1. Foundations** | Weeks 1-3 | PyTorch basics, backpropagation | MNIST from scratch (NumPy → PyTorch) |
| **2. Computer Vision** | Weeks 4-7 | CNNs, transfer learning | Real-time object detection system |
| **3. Natural Language** | Weeks 8-11 | Transformers, fine-tuning | Retrieval-augmented generation (RAG) |
| **4. Generative AI** | Weeks 12-16 | GANs, diffusion, LLMs | Fine-tuned language model |
| **5. Reinforcement Learning** | Weeks 17-21 | Agents, policy learning | Game-playing AI |
| **6. Production** | Weeks 22-24 | MLOps, deployment | Production-ready API |

Each phase builds on the last, with hands-on projects that teach you to **think like a deep learning engineer**.

## 🛠️ Getting Started

### Prerequisites
- Python 3.12+ (we use modern Python features)
- Basic programming experience (you're good if you know loops and functions)
- Linear algebra from school (we'll refresh as needed)

### Quick Setup

```bash
# Clone this repository
git clone https://github.com/yourusername/ml-learning.git
cd ml-learning

# Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Set up the environment
uv python pin 3.12
uv sync

# Launch Jupyter for exploration
uv run jupyter lab
```

### Your First Steps

1. **Read the [Learning Plan](learning_plan.md)** - Understand your 6-month journey
2. **Review the [AI Guidelines](AGENT.md)** - How your AI tutor will help you learn
3. **Start with Phase 1** - Implement MNIST classification from scratch

```bash
# Run your first project
cd mnist-from-scratch
uv run python train.py --model numpy
```

## 🤖 The AI Learning Partner

This project includes detailed instructions for AI assistants on how to help you learn effectively. Your AI tutor will:

- **Ask before telling** - Guide you to discover solutions yourself
- **Build intuition first** - Explain *why* before *how*
- **Verify understanding** - Check your grasp of concepts through questions
- **Encourage struggle** - Productive confusion leads to deep learning

Read the [AI Learning Guidelines](AGENT.md) to understand this teaching philosophy.

## 📁 Project Structure

```
ml-learning/
├── mnist-from-scratch/          # Phase 1: Neural networks from scratch
│   ├── numpy_net.py            # Pure NumPy implementation
│   ├── pytorch_net.py          # PyTorch implementation
│   ├── train.py                # Training script with W&B logging
│   └── utils.py                # Data loading and visualization
├── learning_plan.md            # Your complete 6-month curriculum
├── AGENT.md                    # AI tutoring guidelines
├── NN_FINETUNING_GUIDE.md      # Transfer learning best practices
└── pyproject.toml              # Modern Python project management
```

## 🎓 Learning Philosophy

### Build, Don't Just Watch
Every concept is implemented from scratch before using libraries. You build intuition through:
- Manual backpropagation implementation
- Custom neural network layers
- Debugging your own training loops

### Progressive Complexity
Start simple, add complexity systematically:
- NumPy → PyTorch
- Single layer → Deep networks
- Classification → Generation → Reinforcement Learning

### Real Projects, Real Skills
Each project teaches production-relevant skills:
- Experiment tracking with Weights & Biases
- Proper train/validation/test splits
- Hyperparameter tuning
- Model debugging and monitoring

## 🤝 Contributing

This is an **open-source learning tool**—your contributions make it better for everyone! Ways to contribute:

### For Learners
- **Share your progress** - Document your learning journey
- **Report bugs** - Help improve the learning experience
- **Suggest improvements** - What made a concept clearer for you?

### For Contributors
- **Improve tutorials** - Make explanations clearer
- **Add new projects** - Extend the curriculum
- **Enhance AI guidelines** - Refine the learning methodology
- **Create visualizations** - Help others build intuition

### Getting Started with Contributions

```bash
# Fork and clone
git clone https://github.com/yourusername/ml-learning.git

# Create a feature branch
git checkout -b improve-phase1-explanations

# Make your changes and test
uv run python mnist-from-scratch/train.py --model numpy

# Submit a pull request
```

## 📚 Resources & Inspiration

### Core Inspiration
- [**Neural Networks: Zero to Hero**](https://karpathy.ai/zero-to-hero.html) - Andrei Karpathy
- [**makemore**](https://github.com/karpathy/makemore) - Building language models from scratch
- **Gabriel Petersson** - AI-assisted learning methodologies

### Recommended Learning Path
1. Start with Karpathy's video series alongside this curriculum
2. Implement everything yourself before looking at solutions
3. Use the AI tutor to deepen your understanding
4. Share your progress and help others

### Additional Resources
- [**Dive into Deep Learning**](https://d2l.ai/) - Free book with PyTorch code
- [**Fast.ai Practical Deep Learning**](https://course.fast.ai/) - Top-down learning approach
- [**Papers with Code**](https://paperswithcode.com/) - State-of-the-art implementations

## 📈 Progress Tracking

Track your journey with the checklists in [learning_plan.md](learning_plan.md):

- ✅ Can explain backpropagation without notes
- ✅ Built a neural network from scratch
- ✅ Debugged a training loop issue independently
- ✅ Fine-tuned a pretrained model effectively
- ✅ Deployed a model to production

## 🙏 Acknowledgments

- **Andrei Karpathy** for showing how to build intuition through implementation
- **Gabriel Petersson** for pioneering AI-assisted learning frameworks
- **The PyTorch team** for making deep learning accessible
- **Fast.ai** for the practical, project-based learning approach

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

<div align="center">

**Ready to start your deep learning journey?** 🚀

[Begin with Phase 1](mnist-from-scratch/) • [Read the Learning Plan](learning_plan.md) • [Contribute](#contributing)

*Happy learning! The journey of building neural networks is as important as the destination.*

</div>