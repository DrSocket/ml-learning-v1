# PyTorch ML/Deep Learning: Project-Based Learning Plan

A hands-on curriculum for software engineers transitioning into AI/ML, designed to build real skills through progressively challenging projects.

---

## Prerequisites & Setup

### What You Already Have
- Python proficiency (you're set here)
- Programming fundamentals and debugging skills
- Basic linear algebra concepts from school (refresh as needed)

### Environment Setup
```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create a new project
uv init ml-learning
cd ml-learning

# Pin Python version (see note below)
uv python pin 3.12

# Core stack
uv add torch torchvision torchaudio
uv add numpy pandas matplotlib seaborn
uv add jupyter jupyterlab
uv add scikit-learn

# Experiment tracking (use from day 1)
uv add wandb

# Later additions (add these when you reach those phases)
uv add transformers datasets accelerate
uv add opencv-python
```

**Why uv?**
- Extremely fast dependency resolution
- Built-in virtual environment management
- Lockfile for reproducible environments (`uv.lock`)
- Drop-in replacement for pip/pip-tools workflows

**Why Python 3.12 instead of 3.14?**

Python 3.14 is faster and stable, but PyTorch 2.9 only has **experimental/preview** support for it. The PyTorch team is still working on full 3.14 wheels across all platforms. For a learning project where you want things to "just work," 3.12 or 3.13 are safer choices right now:

| Python Version | PyTorch Status |
|----------------|----------------|
| 3.10 - 3.13 | ✅ Fully supported |
| 3.14 | ⚠️ Experimental (preview wheels only) |
| 3.14t (free-threaded) | ⚠️ Experimental |

Once PyTorch officially supports 3.14, you can upgrade with `uv python pin 3.14`. Check the [PyTorch compatibility matrix](https://github.com/pytorch/pytorch/blob/main/RELEASE.md) for the latest status.

**Quick Reference**
```bash
uv add <package>        # Add dependency
uv remove <package>     # Remove dependency
uv sync                 # Install from lockfile
uv run python script.py # Run in project environment
uv run jupyter lab      # Launch Jupyter in environment
```

### Compute Resources
| Option | Cost | Best For |
|--------|------|----------|
| Google Colab | Free (with limits) | Learning, small experiments |
| Colab Pro | $10/month | Longer training, better GPUs |
| Kaggle Notebooks | Free | Competitions, datasets |
| Lambda Labs | ~$1-2/hr | Serious training runs |
| Vast.ai | Variable | Budget GPU rental |
| RunPod | Variable | Flexible serverless GPUs |

---

## Phase 1: Foundations (Weeks 1-3)

### Learning Goals
- Understand tensors, automatic differentiation, and computational graphs
- Internalize backpropagation by implementing it manually
- Master PyTorch's core APIs: `torch.Tensor`, `nn.Module`, `optim`, `DataLoader`

### Core Concepts to Master
- Tensor operations and broadcasting
- Autograd and gradient computation
- Loss functions and optimization
- Training loops and batching
- GPU acceleration basics

### 📁 Project 1.1: MNIST from Scratch

**Objective**: Build digit classification twice—first with NumPy only, then with PyTorch.

**Part A - NumPy Implementation**
- Implement forward pass for a 2-layer network
- Implement backward pass manually (chain rule)
- Train with SGD you write yourself
- Target: 95%+ accuracy

**Part B - PyTorch Implementation**
- Rebuild using `nn.Module` and `nn.Linear`
- Use `torch.optim.SGD` or `Adam`
- Add proper train/validation split
- Target: 98%+ accuracy

**Key Files to Create**
```
mnist-from-scratch/
├── numpy_net.py          # Pure NumPy implementation
├── pytorch_net.py        # PyTorch implementation
├── train.py              # Training script with logging
├── utils.py              # Data loading, visualization
└── README.md             # Document your learnings
```

**Stretch Goals**
- Visualize learned weights
- Plot training curves with W&B
- Add dropout and batch normalization
- Experiment with different architectures

### 📁 Project 1.2: Custom Dataset Pipeline

**Objective**: Build a complete data pipeline for a dataset you create or curate.

**Tasks**
- Collect/curate 500+ images of something (screenshots, photos, scraped images)
- Implement a custom `torch.utils.data.Dataset` class
- Add data augmentation with `torchvision.transforms`
- Create efficient `DataLoader` with proper batching
- Handle train/val/test splits correctly

**Why This Matters**: Real ML work is 80% data wrangling. This skill is essential.

### Resources for Phase 1
- **Video**: Andrej Karpathy - "Neural Networks: Zero to Hero" (YouTube)
- **Course**: Fast.ai "Practical Deep Learning" Part 1, Lessons 1-3
- **Book**: "Dive into Deep Learning" - d2l.ai (free, with PyTorch code)
- **Reference**: PyTorch official tutorials (pytorch.org/tutorials)

### Milestone Checklist
- [ ] Can explain backpropagation without looking anything up
- [ ] Comfortable writing training loops from memory
- [ ] Understand when/why to use `.detach()`, `.no_grad()`, `.eval()`
- [ ] Can debug shape mismatch errors efficiently
- [ ] First two projects on GitHub with READMEs

---

## Phase 2: Computer Vision (Weeks 4-7)

### Learning Goals
- Understand convolutional neural networks deeply
- Master transfer learning and fine-tuning
- Handle real-world image data challenges
- Deploy models for inference

### Core Concepts to Master
- Convolution, pooling, stride, padding
- Feature hierarchies and receptive fields
- Batch normalization and residual connections
- Transfer learning strategies (freeze vs fine-tune)
- Data augmentation for images

### 📁 Project 2.1: Niche Image Classifier

**Objective**: Build a production-quality classifier for a specific, useful domain.

**Domain Ideas** (pick something you care about)
- Plant disease detection from leaf photos
- Architectural style classification
- Dog breed identifier (harder than it sounds)
- Food recognition for nutrition tracking
- Bird species from photos
- Skin condition screening (educational only)
- Rock/mineral classification
- Garbage sorting for recycling

**Technical Requirements**
- Use a pretrained backbone (ResNet50, EfficientNet, ConvNeXt)
- Implement proper fine-tuning strategy
- Handle class imbalance if present
- Create train/val/test splits correctly
- Achieve meaningful accuracy on your domain
- Add confidence calibration

**Project Structure**
```
niche-classifier/
├── data/
│   ├── raw/                 # Original images
│   └── processed/           # Cleaned, organized
├── src/
│   ├── dataset.py           # Custom dataset class
│   ├── model.py             # Model architecture
│   ├── train.py             # Training script
│   ├── evaluate.py          # Metrics, confusion matrix
│   └── inference.py         # Single-image prediction
├── notebooks/
│   └── exploration.ipynb    # Data analysis
├── configs/
│   └── train_config.yaml    # Hyperparameters
└── README.md
```

**Stretch Goals**
- Build a Gradio or Streamlit demo
- Add Grad-CAM visualizations (explain predictions)
- Quantize model for faster inference
- Package as a pip-installable library

### 📁 Project 2.2: Real-Time Object Detection

**Objective**: Build a working detection system that runs on live video.

**Implementation Path**
1. Start with pretrained YOLOv8 (ultralytics library)
2. Run inference on images, then video
3. Fine-tune on custom objects
4. Optimize for real-time performance

**Project Ideas**
- Workshop safety monitor (detect PPE)
- Wildlife camera trap analyzer
- Parking spot availability checker
- Package delivery detector
- Pet behavior monitor
- Traffic/pedestrian counter

**Technical Challenges to Solve**
- Handle variable input resolutions
- Optimize for consistent FPS
- Deal with false positives
- Add tracking across frames (optional: SORT/DeepSORT)
- Log detections with timestamps

**Deliverables**
- Working real-time demo
- Performance benchmarks (FPS, mAP)
- Documentation of fine-tuning process

### 📁 Project 2.3: Image Segmentation

**Objective**: Pixel-level classification for a practical use case.

**Ideas**
- Portrait segmentation for background removal
- Lane detection for driving videos
- Medical image segmentation (public datasets)
- Satellite imagery analysis
- Document layout segmentation

**Technical Focus**
- U-Net or DeepLabV3 architectures
- Dice loss and IoU metrics
- Handling imbalanced pixels
- Post-processing predictions

### Resources for Phase 2
- **Course**: CS231n lectures (Stanford, YouTube)
- **Library Docs**: torchvision, timm (PyTorch Image Models)
- **Papers**: ResNet, EfficientNet, YOLO (read for intuition)
- **Datasets**: Kaggle, Roboflow, HuggingFace datasets

### Milestone Checklist
- [ ] Can explain what each layer of a CNN learns
- [ ] Know when to freeze vs fine-tune layers
- [ ] Comfortable with image augmentation strategies
- [ ] Can debug model not learning on images
- [ ] Have a working real-time demo to show

---

## Phase 3: Natural Language Processing (Weeks 8-11)

### Learning Goals
- Understand text representation (tokenization, embeddings)
- Master the Transformer architecture deeply
- Use HuggingFace ecosystem effectively
- Fine-tune language models

### Core Concepts to Master
- Tokenization strategies (BPE, WordPiece, SentencePiece)
- Word embeddings and contextual embeddings
- Attention mechanisms (self-attention, cross-attention)
- Transformer encoder vs decoder
- Transfer learning for NLP

### 📁 Project 3.1: Text Classification System

**Objective**: Build a classifier for text in a useful domain.

**Domain Ideas**
- Support ticket urgency/category classification
- Code review comment tone detector
- Research paper topic classification
- Email intent detection
- Content moderation prototype
- Legal document classification

**Implementation Progression**
1. Baseline with TF-IDF + logistic regression
2. LSTM/GRU approach
3. Fine-tuned BERT/RoBERTa
4. Compare all approaches

**Technical Requirements**
- Proper text preprocessing pipeline
- Handle class imbalance
- Implement proper evaluation (precision, recall, F1)
- Error analysis on misclassifications

### 📁 Project 3.2: Transformer from Scratch

**Objective**: Implement attention and transformer architecture yourself.

**This is crucial for deep understanding. Build:**
- Scaled dot-product attention
- Multi-head attention
- Position encodings (sinusoidal and learned)
- Transformer encoder block
- Transformer decoder block
- Full encoder-decoder model

**Training Task Options**
- Character-level language model (Shakespeare, code)
- Simple machine translation (small parallel corpus)
- Text summarization on CNN/DailyMail

**Target Outcome**
You should be able to whiteboard the transformer architecture and explain every component. This knowledge is essential for interviews and advanced work.

**Key Reference**: "Attention Is All You Need" paper + "The Annotated Transformer" blog post

### 📁 Project 3.3: Retrieval-Augmented Generation (RAG) System

**Objective**: Build a question-answering system over custom documents.

**Components to Build**
1. Document chunking and preprocessing
2. Embedding generation (sentence-transformers)
3. Vector store (FAISS, ChromaDB, or Pinecone)
4. Retrieval pipeline
5. LLM integration for answer generation
6. Simple UI

**Use Cases**
- QA over your company's documentation
- Research paper assistant
- Codebase documentation chatbot
- Personal knowledge base

**Technical Challenges**
- Chunk size optimization
- Retrieval relevance tuning
- Handling multi-hop questions
- Citation/source tracking

### Resources for Phase 3
- **Course**: Stanford CS224N (NLP with Deep Learning)
- **Tutorial**: HuggingFace NLP Course (free)
- **Blog**: Jay Alammar's visual guides (The Illustrated Transformer)
- **Paper**: "Attention Is All You Need"

### Milestone Checklist
- [ ] Can implement attention from scratch
- [ ] Understand tokenization deeply
- [ ] Comfortable with HuggingFace transformers library
- [ ] Can fine-tune BERT-family models
- [ ] Have working RAG system to demo

---

## Phase 4: Generative Models (Weeks 12-16)

### Learning Goals
- Understand generative model fundamentals
- Train GANs and diffusion models
- Fine-tune large language models efficiently
- Handle the unique challenges of generative AI

### Core Concepts to Master
- Generative vs discriminative models
- GAN training dynamics
- Diffusion process (forward and reverse)
- Parameter-efficient fine-tuning (LoRA, QLoRA)
- Sampling strategies for generation

### 📁 Project 4.1: GAN for Custom Domain

**Objective**: Train a GAN to generate images in a specific style.

**Domain Ideas**
- Pixel art sprites
- Album cover art
- Architectural floor plans
- Fashion designs
- Font/letter generation
- Texture synthesis

**Implementation Path**
1. Start with DCGAN on your dataset
2. Progress to StyleGAN2/3 if needed
3. Implement training monitoring
4. Handle mode collapse

**Curate Your Dataset**
- Collect 1000+ images in your chosen domain
- Clean and preprocess consistently
- Consider progressive growing

**Deliverables**
- Working generator
- Interpolation videos (latent space walks)
- FID score or other quality metrics
- Training progression visualization

### 📁 Project 4.2: Diffusion Model Training

**Objective**: Train a diffusion model and understand the process deeply.

**Options**
1. Train small diffusion model from scratch on limited domain
2. Fine-tune Stable Diffusion on custom style/concept
3. Implement DDPM paper from scratch (educational)

**Technical Focus**
- U-Net architecture for diffusion
- Noise schedules
- Classifier-free guidance
- Sampling speed optimizations

**Project Ideas**
- Style-specific image generator
- Texture/pattern generator for games
- Architectural concept generator
- Custom avatar/character generator

### 📁 Project 4.3: LLM Fine-Tuning

**Objective**: Fine-tune an open-source LLM for a specific task.

**Base Models to Consider**
- Llama 3 8B (Meta)
- Mistral 7B
- Phi-3
- Gemma 2

**Fine-Tuning Methods**
- Full fine-tuning (if you have compute)
- LoRA (Low-Rank Adaptation)
- QLoRA (Quantized LoRA)
- Prompt tuning

**Project Ideas**
- Code assistant for specific framework
- Domain-specific chatbot (legal, medical education, etc.)
- Creative writing in specific style
- Structured data extraction
- Function calling specialist

**Technical Requirements**
- Proper dataset preparation (instruction format)
- Training with PEFT library
- Evaluation methodology
- Inference optimization

**Tools**
- HuggingFace PEFT library
- bitsandbytes for quantization
- Axolotl for training pipelines
- vLLM or llama.cpp for inference

### Resources for Phase 4
- **GAN**: "GAN Lab" interactive visualization
- **Diffusion**: "What are Diffusion Models?" (Lil'Log)
- **LLM**: HuggingFace PEFT documentation
- **Papers**: DDPM, Stable Diffusion, LoRA

### Milestone Checklist
- [ ] Can explain GAN training dynamics
- [ ] Understand diffusion forward/reverse process
- [ ] Successfully fine-tuned an LLM
- [ ] Have generative model demos to show
- [ ] Understand evaluation metrics for generative models

---

## Phase 5: Reinforcement Learning (Weeks 17-21)

### Learning Goals
- Understand RL fundamentals (MDPs, policies, value functions)
- Implement core algorithms from scratch
- Train agents in simulated environments
- Appreciate RL's unique challenges

### Core Concepts to Master
- Markov Decision Processes
- Value functions and Q-learning
- Policy gradients
- Actor-Critic methods
- Exploration vs exploitation

### 📁 Project 5.1: Classic Control

**Objective**: Solve classic RL benchmarks from scratch.

**Environments (Gymnasium)**
1. CartPole (start here)
2. LunarLander
3. Acrobot
4. MountainCar

**Algorithms to Implement**
1. Q-Learning (tabular)
2. DQN (Deep Q-Network)
3. REINFORCE (policy gradient)
4. PPO (Proximal Policy Optimization)

**For Each Algorithm**
- Implement from scratch (not just use library)
- Train on multiple environments
- Plot learning curves
- Compare performance

### 📁 Project 5.2: Game-Playing Agent

**Objective**: Train an agent to play a real game competently.

**Options**
- Atari games (via Gymnasium ALE)
- Board games (Connect4, Othello)
- Custom simple game you build
- Retro games (via stable-retro)

**Technical Challenges**
- Frame stacking for temporal info
- Reward shaping
- Training stability
- Hyperparameter sensitivity

**Deliverables**
- Video of trained agent playing
- Learning curves
- Comparison of different approaches
- Write-up of what worked and didn't

### 📁 Project 5.3: Advanced RL Application

**Objective**: Apply RL to a more complex problem.

**Ideas**
- Multi-agent environment (competitive or cooperative)
- Continuous control (MuJoCo, PyBullet)
- Custom environment for real problem
- Combine with LLM (RL from human feedback concepts)

**Stretch: RLHF-Lite**
- Build small preference dataset
- Train reward model
- Fine-tune small LLM with PPO
- Understand how ChatGPT-style training works

### Resources for Phase 5
- **Course**: David Silver's RL course (DeepMind, YouTube)
- **Book**: Sutton & Barto "Reinforcement Learning" (free online)
- **Library**: Stable-Baselines3, CleanRL
- **Environment**: Gymnasium, PettingZoo (multi-agent)

### Milestone Checklist
- [ ] Can explain MDP framework
- [ ] Implemented DQN from scratch
- [ ] Implemented policy gradient from scratch
- [ ] Have video of trained game-playing agent
- [ ] Understand why RL is hard (sample efficiency, stability)

---

## Phase 6: Production & MLOps (Weeks 22-24)

### Learning Goals
- Deploy models as APIs
- Monitor model performance
- Handle ML-specific production challenges
- Understand the full ML lifecycle

### 📁 Project 6.1: Model Deployment

**Deploy one of your models as a production-quality API:**

**Components**
- FastAPI or Flask backend
- Model serialization (TorchScript, ONNX)
- Docker containerization
- Basic load handling
- Input validation
- Logging and monitoring

**Stretch Goals**
- GPU inference server (Triton, TorchServe)
- Kubernetes deployment
- A/B testing infrastructure
- Model versioning

### 📁 Project 6.2: End-to-End ML Pipeline

**Build automated pipeline including:**
- Data versioning (DVC)
- Experiment tracking (W&B, MLflow)
- Automated training
- Model registry
- Automated evaluation
- Deployment pipeline

### Skills to Develop
- Docker for ML
- Cloud platforms (AWS SageMaker, GCP Vertex, Azure ML)
- Model optimization (quantization, pruning, distillation)
- Monitoring and observability

---

## Portfolio Presentation

### GitHub Profile
Each project should have:
- Clear README with problem statement, approach, results
- Requirements.txt or environment.yml
- Example usage / demo instructions
- Sample outputs or screenshots
- Learning reflections

### Best Projects to Highlight
1. **Technical depth**: Transformer from scratch
2. **Practical application**: Niche classifier or RAG system
3. **Visual appeal**: GAN/Diffusion outputs or RL agent video
4. **Production readiness**: Deployed API

### Blog Posts to Write
- "What I learned implementing backprop from scratch"
- "Fine-tuning BERT for [your domain]"
- "Training a GAN: What the tutorials don't tell you"
- "Building a RAG system that actually works"
- "RL is hard: My journey to a game-playing agent"

### Demo Reel
Create a 2-3 minute video showing:
- Your best visual projects (GAN outputs, RL agents)
- Live demos of deployed applications
- Quick walkthrough of code quality

---

## Learning Resources Summary

### Video Courses
| Course | Platform | Best For |
|--------|----------|----------|
| Neural Networks: Zero to Hero | YouTube | Foundations, building intuition |
| Fast.ai Practical Deep Learning | fast.ai | Practical skills, top-down learning |
| CS231n | YouTube | Computer vision depth |
| CS224n | YouTube | NLP depth |
| David Silver RL Course | YouTube | Reinforcement learning |

### Books
- "Dive into Deep Learning" (d2l.ai) - Free, with code
- "Deep Learning" (Goodfellow) - Theory reference
- "Hands-On Machine Learning" (Géron) - Practical reference
- "Reinforcement Learning" (Sutton & Barto) - RL bible

### Communities
- r/MachineLearning, r/LocalLLaMA
- HuggingFace Discord
- PyTorch Forums
- Twitter/X ML community
- Local ML meetups

### Stay Current
- Papers With Code (paperswithcode.com)
- arXiv cs.LG and cs.CV
- The Batch (Andrew Ng's newsletter)
- Import AI newsletter

---

## Timeline Summary

| Phase | Weeks | Focus | Key Deliverable |
|-------|-------|-------|-----------------|
| 1. Foundations | 1-3 | PyTorch basics, backprop | MNIST from scratch |
| 2. Vision | 4-7 | CNNs, transfer learning | Real-time detection system |
| 3. NLP | 8-11 | Transformers, fine-tuning | RAG system |
| 4. Genertic | 12-16 | GANs, diffusion, LLMs | Fine-tuned LLM |
| 5. RL | 17-21 | Agents, policy learning | Game-playing agent |
| 6. Production | 22-24 | Deployment, MLOps | Deployed API |

**Total: ~6 months at 15-20 hours/week**

Adjust based on your pace and interests. It's better to go deep on fewer projects than to rush through everything superficially.

---

## Final Advice

1. **Build, don't just watch**: Tutorial hell is real. Code along, then rebuild without looking.

2. **Embrace confusion**: Not understanding papers immediately is normal. Read them multiple times.

3. **Debug systematically**: When training fails, isolate variables. Overfit on one batch first.

4. **Track everything**: Use W&B from day one. You'll thank yourself later.

5. **Share your work**: Blog posts and GitHub activity matter more than certificates.

6. **Join communities**: Learning is faster with others. Find your people.

7. **Stay curious**: The field moves fast. Enjoy the journey of continuous learning.

Good luck! 🚀