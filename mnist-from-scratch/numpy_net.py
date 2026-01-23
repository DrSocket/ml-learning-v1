"""
MNIST Neural Network - Pure NumPy Implementation

This is the core learning exercise. You'll implement:
1. Forward pass (prediction)
2. Loss computation (how wrong are we?)
3. Backward pass (gradients via chain rule)
4. Parameter updates (SGD)

Architecture:
    Input (784) → Linear → ReLU → Linear → Softmax → Output (10)
    
The goal is to understand what PyTorch does under the hood.
"""

import numpy as np


class NumpyNet:
    """
    A 2-layer neural network implemented from scratch with NumPy.
    
    Architecture:
        x (784) → W1 (784, 128) → ReLU → W2 (128, 10) → Softmax → output (10)
    """
    
    def __init__(self, input_size=784, hidden_size=128, output_size=10):
        """
        Initialize network parameters.
        
        Weight initialization matters! We use Xavier/Glorot initialization
        to keep activations from exploding or vanishing.
        
        TODO: Initialize weights and biases
        - W1: shape (input_size, hidden_size)
        - b1: shape (hidden_size,)
        - W2: shape (hidden_size, output_size)
        - b2: shape (output_size,)
        
        Hint: Xavier init uses scale = sqrt(2 / (fan_in + fan_out))
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # YOUR CODE HERE: Initialize W1, b1, W2, b2
        # Example for W1:
        scale = np.sqrt(2.0 / (input_size + hidden_size))
        self.W1 = np.random.randn(input_size, hidden_size) * scale
        self.b1 = np.zeros(hidden_size)
        scale = np.sqrt(2.0 / (hidden_size + output_size))
        self.W2 = np.random.randn(hidden_size, output_size) * scale
        self.b2 = np.zeros(output_size)

        # Cache for backward pass (will store intermediate values)
        self.cache = {}

    def relu(self, z):
        """
        ReLU activation: max(0, z)
        
        Args:
            z: input array of any shape
            
        Returns:
            ReLU applied element-wise
            
        Why ReLU?
        - Simple gradient (1 if z > 0, else 0)
        - Helps with vanishing gradient problem
        - Computationally efficient
        """
        return np.maximum(0, z)
    
    def relu_derivative(self, z):
        """
        Derivative of ReLU.
        
        Args:
            z: input array (the values BEFORE ReLU was applied)
            
        Returns:
            Gradient: 1 where z > 0, else 0
            
        TODO: Implement this
        Hint: This is a simple element-wise operation
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement relu_derivative")
    
    def softmax(self, z):
        """
        Softmax activation: converts logits to probabilities.
        
        softmax(z)_i = exp(z_i) / sum(exp(z_j))
        
        Args:
            z: (N, C) logits for N samples, C classes
            
        Returns:
            (N, C) probabilities (each row sums to 1)
            
        TODO: Implement softmax
        Hint: For numerical stability, subtract max(z) before exp
              This prevents overflow when z has large values
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement softmax")
    
    def forward(self, x):
        """
        Forward pass: compute predictions.
        
        x → (x @ W1 + b1) → ReLU → (hidden @ W2 + b2) → Softmax → output
        
        Args:
            x: (N, 784) input images (flattened)
            
        Returns:
            (N, 10) class probabilities
            
        TODO: Implement the forward pass
        - Store intermediate values in self.cache for backward pass
        - You'll need: z1, a1 (hidden activations), z2, a2 (output probs)
        
        Hint: Matrix multiply is @ or np.dot()
        """
        # YOUR CODE HERE
        # 1. First linear layer: z1 = x @ W1 + b1
        # 2. ReLU activation: a1 = relu(z1)
        # 3. Second linear layer: z2 = a1 @ W2 + b2
        # 4. Softmax: a2 = softmax(z2)
        # 5. Cache everything needed for backward pass
        
        raise NotImplementedError("Implement forward pass")
    
    def cross_entropy_loss(self, predictions, targets):
        """
        Cross-entropy loss for multi-class classification.
        
        L = -1/N * sum(targets * log(predictions))
        
        Args:
            predictions: (N, 10) predicted probabilities from softmax
            targets: (N, 10) one-hot encoded true labels
            
        Returns:
            scalar loss value
            
        TODO: Implement cross-entropy loss
        Hint: Add small epsilon (1e-15) to log to avoid log(0)
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement cross_entropy_loss")
    
    def backward(self, x, targets):
        """
        Backward pass: compute gradients using the chain rule.
        
        This is the heart of backpropagation. We compute:
        - dL/dW2, dL/db2 (gradients for output layer)
        - dL/dW1, dL/db1 (gradients for hidden layer)
        
        Args:
            x: (N, 784) input images
            targets: (N, 10) one-hot encoded true labels
            
        Returns:
            dict of gradients: {'dW1': ..., 'db1': ..., 'dW2': ..., 'db2': ...}
        
        The Chain Rule Applied:
        
        For output layer (softmax + cross-entropy has nice gradient):
            dL/dz2 = predictions - targets  (this is a well-known result)
            dL/dW2 = a1.T @ dz2
            dL/db2 = sum(dz2, axis=0)
        
        For hidden layer:
            dL/da1 = dz2 @ W2.T
            dL/dz1 = dL/da1 * relu_derivative(z1)
            dL/dW1 = x.T @ dz1
            dL/db1 = sum(dz1, axis=0)
        
        TODO: Implement backward pass
        Hint: Retrieve cached values from self.cache
        """
        N = x.shape[0]
        
        # Retrieve cached values from forward pass
        # z1, a1, z2, a2 = self.cache['z1'], self.cache['a1'], ...
        
        # YOUR CODE HERE
        # 1. Output layer gradient (softmax + cross-entropy shortcut)
        # 2. Propagate gradient through W2
        # 3. Propagate through ReLU
        # 4. Hidden layer gradients
        
        raise NotImplementedError("Implement backward pass")
    
    def update_parameters(self, gradients, learning_rate):
        """
        Update parameters using SGD.
        
        For each parameter: param = param - learning_rate * gradient
        
        Args:
            gradients: dict from backward()
            learning_rate: step size
            
        TODO: Implement parameter updates
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement update_parameters")
    
    def predict(self, x):
        """
        Make predictions (class labels, not probabilities).
        
        Args:
            x: (N, 784) input images
            
        Returns:
            (N,) predicted class labels (0-9)
        """
        probs = self.forward(x)
        return np.argmax(probs, axis=1)


# ==============================================================================
# TESTS - Run these to check your implementation
# ==============================================================================

def test_relu():
    """Test ReLU and its derivative."""
    net = NumpyNet.__new__(NumpyNet)  # Create without __init__
    
    z = np.array([-2, -1, 0, 1, 2])
    
    # Test ReLU
    expected_relu = np.array([0, 0, 0, 1, 2])
    assert np.allclose(net.relu(z), expected_relu), "ReLU failed"
    
    # Test derivative
    expected_deriv = np.array([0, 0, 0, 1, 1])
    assert np.allclose(net.relu_derivative(z), expected_deriv), "ReLU derivative failed"
    
    print("✓ ReLU tests passed")


def test_softmax():
    """Test softmax produces valid probabilities."""
    net = NumpyNet.__new__(NumpyNet)
    
    z = np.array([[1, 2, 3], [1, 1, 1]])
    probs = net.softmax(z)
    
    # Check probabilities sum to 1
    assert np.allclose(probs.sum(axis=1), [1, 1]), "Softmax rows don't sum to 1"
    
    # Check all positive
    assert (probs > 0).all(), "Softmax produced non-positive values"
    
    # Check numerical stability (large values shouldn't cause overflow)
    z_large = np.array([[1000, 1001, 1002]])
    probs_large = net.softmax(z_large)
    assert not np.any(np.isnan(probs_large)), "Softmax has numerical stability issues"
    
    print("✓ Softmax tests passed")


def test_forward_shapes():
    """Test that forward pass produces correct shapes."""
    net = NumpyNet()
    
    x = np.random.randn(32, 784)  # Batch of 32 images
    output = net.forward(x)
    
    assert output.shape == (32, 10), f"Expected (32, 10), got {output.shape}"
    assert np.allclose(output.sum(axis=1), 1), "Output probabilities don't sum to 1"
    
    print("✓ Forward pass shape test passed")


def test_gradient_check():
    """
    Numerical gradient checking.
    
    Compares analytical gradients (from backward()) with numerical gradients.
    This is how you verify your backprop implementation is correct.
    """
    net = NumpyNet()
    
    # Small batch for testing
    x = np.random.randn(5, 784)
    y = np.eye(10)[np.random.randint(0, 10, 5)]  # Random one-hot labels
    
    # Get analytical gradients
    net.forward(x)
    grads = net.backward(x, y)
    
    # Numerical gradient check for W1
    epsilon = 1e-5
    numerical_grad = np.zeros_like(net.W1)
    
    # Only check a few elements (full check is slow)
    for i in range(3):
        for j in range(3):
            # f(W + eps)
            net.W1[i, j] += epsilon
            loss_plus = net.cross_entropy_loss(net.forward(x), y)
            
            # f(W - eps)
            net.W1[i, j] -= 2 * epsilon
            loss_minus = net.cross_entropy_loss(net.forward(x), y)
            
            # Restore
            net.W1[i, j] += epsilon
            
            # Numerical gradient
            numerical_grad[i, j] = (loss_plus - loss_minus) / (2 * epsilon)
    
    # Compare
    analytical = grads['dW1'][:3, :3]
    numerical = numerical_grad[:3, :3]
    
    relative_error = np.abs(analytical - numerical) / (np.abs(analytical) + np.abs(numerical) + 1e-8)
    
    if relative_error.max() < 1e-5:
        print("✓ Gradient check passed")
    else:
        print(f"✗ Gradient check FAILED. Max relative error: {relative_error.max()}")
        print(f"Analytical:\n{analytical}")
        print(f"Numerical:\n{numerical}")


if __name__ == "__main__":
    print("Running tests...\n")
    
    try:
        test_relu()
    except NotImplementedError as e:
        print(f"⏳ ReLU not yet implemented")
    
    try:
        test_softmax()
    except NotImplementedError as e:
        print(f"⏳ Softmax not yet implemented")
    
    try:
        test_forward_shapes()
    except NotImplementedError as e:
        print(f"⏳ Forward pass not yet implemented")
    
    try:
        test_gradient_check()
    except NotImplementedError as e:
        print(f"⏳ Backward pass not yet implemented")

