"""
Utility functions for MNIST project.
Data loading, preprocessing, and visualization.
"""

import numpy as np
import matplotlib.pyplot as plt


def load_mnist():
    """
    Load MNIST dataset using torchvision (for convenience),
    then convert to NumPy arrays for our NumPy implementation.
    
    Returns:
        x_train: (60000, 784) - flattened training images, normalized to [0, 1]
        y_train: (60000,) - training labels (0-9)
        x_test: (10000, 784) - flattened test images, normalized to [0, 1]
        y_test: (10000,) - test labels (0-9)
    """
    from torchvision import datasets
    
    # Download MNIST
    train_dataset = datasets.MNIST(root='./data', train=True, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True)
    
    # Convert to numpy and normalize to [0, 1]
    x_train = train_dataset.data.numpy().reshape(-1, 784) / 255.0
    y_train = train_dataset.targets.numpy()
    
    x_test = test_dataset.data.numpy().reshape(-1, 784) / 255.0
    y_test = test_dataset.targets.numpy()
    
    return x_train, y_train, x_test, y_test


def one_hot_encode(labels, num_classes=10):
    """
    Convert integer labels to one-hot encoded vectors.
    
    Args:
        labels: (N,) array of integer labels
        num_classes: number of classes (10 for MNIST)
    
    Returns:
        (N, num_classes) one-hot encoded array
    
    Example:
        >>> one_hot_encode(np.array([0, 3, 9]), num_classes=10)
        array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
    
    TODO: Implement this function
    Hint: Look up np.eye() - it creates an identity matrix
    """
    # YOUR CODE HERE
    # raise NotImplementedError("Implement one_hot_encode")
    return np.eye(num_classes)[labels]


def visualize_samples(images, labels, predictions=None, num_samples=10):
    """
    Visualize MNIST samples with their labels.
    
    Args:
        images: (N, 784) flattened images
        labels: (N,) true labels
        predictions: (N,) predicted labels (optional)
        num_samples: how many to display
    """
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    axes = axes.flatten()
    
    for i in range(min(num_samples, len(images))):
        img = images[i].reshape(28, 28)
        axes[i].imshow(img, cmap='gray')
        
        title = f"Label: {labels[i]}"
        if predictions is not None:
            title += f"\nPred: {predictions[i]}"
            if predictions[i] != labels[i]:
                axes[i].set_title(title, color='red')
            else:
                axes[i].set_title(title, color='green')
        else:
            axes[i].set_title(title)
        
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


def plot_training_curves(train_losses, val_losses=None, train_accs=None, val_accs=None):
    """
    Plot training curves (loss and accuracy over epochs).
    
    Args:
        train_losses: list of training losses per epoch
        val_losses: list of validation losses per epoch (optional)
        train_accs: list of training accuracies per epoch (optional)
        val_accs: list of validation accuracies per epoch (optional)
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    axes[0].plot(train_losses, label='Train Loss')
    if val_losses:
        axes[0].plot(val_losses, label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    if train_accs:
        axes[1].plot(train_accs, label='Train Acc')
    if val_accs:
        axes[1].plot(val_accs, label='Val Acc')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def create_batches(x, y, batch_size, shuffle=True):
    """
    Create mini-batches from data.
    
    Args:
        x: (N, D) input data
        y: (N,) or (N, C) labels
        batch_size: size of each batch
        shuffle: whether to shuffle data
    
    Yields:
        (x_batch, y_batch) tuples
    
    TODO: Implement this generator function
    Hint: 
    - If shuffle=True, create random permutation of indices
    - Yield batches using array slicing
    - Don't forget the last batch (might be smaller than batch_size)
    """
    # YOUR CODE HERE
    # raise NotImplementedError("Implement create_batches")
    if shuffle:
        indices = np.random.permutation(len(x))
        print (f"Shuffling indices: {indices}")
    else:
        indices = np.arange(len(x))

    for i in range(0, len(x), batch_size):
        x_batch = x[indices[i:i+batch_size]]
        y_batch = y[indices[i:i+batch_size]]
        yield x_batch, y_batch

def compute_accuracy(predictions, labels):
    """
    Compute classification accuracy.
    
    Args:
        predictions: (N,) predicted class labels OR (N, C) class probabilities
        labels: (N,) true labels
    
    Returns:
        accuracy as a float between 0 and 1
    """
    # If predictions are probabilities, convert to class labels
    if predictions.ndim == 2:
        predictions = np.argmax(predictions, axis=1)
    
    return np.mean(predictions == labels)


# Quick test when running this file directly
# if __name__ == "__main__":
#     print("Loading MNIST...")
#     x_train, y_train, x_test, y_test = load_mnist()
    
#     print(f"Training set: {x_train.shape}, {y_train.shape}")
#     print(f"Test set: {x_test.shape}, {y_test.shape}")
#     print(f"Pixel range: [{x_train.min():.2f}, {x_train.max():.2f}]")
#     print(f"Labels: {np.unique(y_train)}")
    
#     print("\nVisualizing samples...")
#     visualize_samples(x_train, y_train)

