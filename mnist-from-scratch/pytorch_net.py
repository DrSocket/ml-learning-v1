"""
MNIST Neural Network - PyTorch Implementation

After implementing the NumPy version, you'll appreciate how much
PyTorch handles for you. This file teaches you the PyTorch way.

Key concepts:
- nn.Module: base class for all neural network modules
- nn.Linear: fully connected layer (does W @ x + b)
- torch.optim: optimizers that update parameters
- DataLoader: efficient batching and shuffling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class PyTorchNet(nn.Module):
    """
    Same architecture as NumpyNet, but using PyTorch.
    
    Architecture:
        Input (784) → Linear (128) → ReLU → Linear (10) → Output
        
    Note: We don't apply softmax in forward() because:
    - nn.CrossEntropyLoss expects raw logits (it applies softmax internally)
    - This is more numerically stable
    """
    
    def __init__(self, input_size=784, hidden_size=128, output_size=10):
        """
        Initialize the network layers.
        
        TODO: Define the layers
        - self.fc1: Linear layer from input_size to hidden_size
        - self.fc2: Linear layer from hidden_size to output_size
        
        Hint: Use nn.Linear(in_features, out_features)
        """
        super().__init__()  # Always call parent __init__
        
        # YOUR CODE HERE
        raise NotImplementedError("Define the layers")
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: (N, 784) input tensor
            
        Returns:
            (N, 10) logits (raw scores, NOT probabilities)
            
        TODO: Implement forward pass
        - Pass through fc1
        - Apply ReLU activation
        - Pass through fc2
        - Return logits (no softmax!)
        
        Hint: Use F.relu() or torch.relu() for activation
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement forward pass")
    
    def predict(self, x):
        """
        Make predictions (class labels).
        
        Args:
            x: (N, 784) input tensor
            
        Returns:
            (N,) predicted class labels
        """
        with torch.no_grad():  # No gradient needed for inference
            logits = self.forward(x)
            return torch.argmax(logits, dim=1)


def create_data_loaders(batch_size=64):
    """
    Create PyTorch DataLoaders for MNIST.
    
    This shows the "PyTorch way" of handling data:
    - TensorDataset wraps tensors into a dataset
    - DataLoader handles batching, shuffling, parallel loading
    
    Returns:
        train_loader, test_loader
    """
    from torchvision import datasets, transforms
    
    # Transform: convert to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts to [0, 1] and (C, H, W) format
        transforms.Lambda(lambda x: x.view(-1))  # Flatten to (784,)
    ])
    
    # Download and load datasets
    train_dataset = datasets.MNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    test_dataset = datasets.MNIST(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0  # Set > 0 for parallel data loading
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    return train_loader, test_loader


def train_one_epoch(model, train_loader, optimizer, criterion, device):
    """
    Train for one epoch.
    
    Args:
        model: the neural network
        train_loader: DataLoader for training data
        optimizer: torch.optim optimizer
        criterion: loss function
        device: 'cpu' or 'cuda'
        
    Returns:
        average loss for the epoch
        
    TODO: Implement the training loop
    
    The pattern is:
    1. Set model to training mode: model.train()
    2. For each batch:
        a. Move data to device
        b. Zero gradients: optimizer.zero_grad()
        c. Forward pass: outputs = model(inputs)
        d. Compute loss: loss = criterion(outputs, targets)
        e. Backward pass: loss.backward()
        f. Update weights: optimizer.step()
    3. Track total loss
    
    Hint: Labels should NOT be one-hot encoded for nn.CrossEntropyLoss
    """
    model.train()  # Set to training mode (affects dropout, batchnorm, etc.)
    total_loss = 0
    
    # YOUR CODE HERE
    raise NotImplementedError("Implement train_one_epoch")
    
    return total_loss / len(train_loader)


def evaluate(model, test_loader, criterion, device):
    """
    Evaluate model on test set.
    
    Args:
        model: the neural network
        test_loader: DataLoader for test data
        criterion: loss function
        device: 'cpu' or 'cuda'
        
    Returns:
        (average_loss, accuracy)
        
    TODO: Implement evaluation
    
    Key differences from training:
    - Use model.eval() to set evaluation mode
    - Wrap in torch.no_grad() to disable gradient computation
    - Don't call optimizer.step()
    """
    model.eval()  # Set to evaluation mode
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():  # No gradients needed for evaluation
        # YOUR CODE HERE
        raise NotImplementedError("Implement evaluate")
    
    return total_loss / len(test_loader), correct / total


# ==============================================================================
# COMPARISON: NumPy vs PyTorch
# ==============================================================================
"""
After implementing both versions, reflect on these questions:

1. INITIALIZATION
   NumPy: You manually implemented Xavier initialization
   PyTorch: nn.Linear does it automatically
   Question: What initialization does PyTorch use by default?

2. FORWARD PASS
   NumPy: You computed z = x @ W + b explicitly
   PyTorch: nn.Linear does this internally
   Question: Where are the weights stored in nn.Linear?

3. BACKWARD PASS
   NumPy: You derived and implemented every gradient
   PyTorch: .backward() does it automatically
   Question: How does PyTorch know the gradients? (Hint: computational graph)

4. OPTIMIZATION
   NumPy: You wrote param -= lr * grad
   PyTorch: optimizer.step() handles it
   Question: What's the advantage of using torch.optim.Adam over SGD?

5. BATCHING
   NumPy: You wrote create_batches() manually
   PyTorch: DataLoader handles batching, shuffling, parallelization
   Question: What does num_workers do in DataLoader?
"""


# ==============================================================================
# TESTS
# ==============================================================================

def test_model_shapes():
    """Test that model produces correct output shapes."""
    model = PyTorchNet()
    
    x = torch.randn(32, 784)
    output = model(x)
    
    assert output.shape == (32, 10), f"Expected (32, 10), got {output.shape}"
    print("✓ Model shape test passed")


def test_training_step():
    """Test that one training step runs without errors."""
    model = PyTorchNet()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # Fake batch
    x = torch.randn(32, 784)
    y = torch.randint(0, 10, (32,))
    
    # Training step
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    
    assert loss.item() > 0, "Loss should be positive"
    print("✓ Training step test passed")


def test_gradient_flow():
    """Test that gradients flow to all parameters."""
    model = PyTorchNet()
    criterion = nn.CrossEntropyLoss()
    
    x = torch.randn(4, 784)
    y = torch.randint(0, 10, (4,))
    
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"
    
    print("✓ Gradient flow test passed")


if __name__ == "__main__":
    print("Running PyTorch model tests...\n")
    
    try:
        test_model_shapes()
    except NotImplementedError as e:
        print("⏳ Model not yet implemented")
    
    try:
        test_training_step()
    except NotImplementedError as e:
        print("⏳ Training step not yet implemented")
    
    try:
        test_gradient_flow()
    except NotImplementedError as e:
        print("⏳ Gradient flow test skipped")

