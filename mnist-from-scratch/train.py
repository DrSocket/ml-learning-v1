"""
Training script for MNIST models.

Supports both NumPy and PyTorch implementations.
Includes experiment tracking with Weights & Biases (optional).

Usage:
    uv run python train.py --model numpy
    uv run python train.py --model pytorch
    uv run python train.py --model pytorch --wandb
"""

import argparse
import time
import numpy as np

from utils import (
    load_mnist, 
    one_hot_encode, 
    create_batches, 
    compute_accuracy,
    plot_training_curves,
    visualize_samples
)


def train_numpy_model(args):
    """
    Train the NumPy implementation.
    
    TODO: Implement the training loop
    
    Steps:
    1. Load data with load_mnist()
    2. Create train/validation split (use last 10k of training as validation)
    3. One-hot encode labels
    4. Initialize NumpyNet
    5. For each epoch:
        - Shuffle and create batches
        - For each batch: forward → loss → backward → update
        - Compute and log training metrics
        - Evaluate on validation set
    6. Final evaluation on test set
    7. Plot training curves
    """
    from numpy_net import NumpyNet
    
    print("=" * 60)
    print("Training NumPy Implementation")
    print("=" * 60)
    
    # Hyperparameters
    learning_rate = args.lr
    batch_size = args.batch_size
    epochs = args.epochs
    
    # Load data
    print("\nLoading MNIST...")
    x_train, y_train, x_test, y_test = load_mnist()
    
    # TODO: Split training data into train and validation sets
    # Hint: Use last 10000 samples for validation
    
    # TODO: One-hot encode labels for training
    # y_train_onehot = one_hot_encode(y_train_split)
    # y_val_onehot = one_hot_encode(y_val)
    
    # Initialize model
    print("\nInitializing model...")
    model = NumpyNet(
        input_size=784,
        hidden_size=args.hidden_size,
        output_size=10
    )
    
    # Training tracking
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    # Optional: W&B logging
    if args.wandb:
        import wandb
        wandb.init(
            project="mnist-from-scratch",
            config={
                "model": "numpy",
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "hidden_size": args.hidden_size,
                "epochs": epochs
            }
        )
    
    print(f"\nTraining for {epochs} epochs...")
    print(f"Learning rate: {learning_rate}, Batch size: {batch_size}")
    print("-" * 60)
    
    # YOUR CODE HERE: Implement training loop
    # 
    # for epoch in range(epochs):
    #     epoch_loss = 0
    #     
    #     # Training
    #     for x_batch, y_batch in create_batches(x_train_split, y_train_onehot, batch_size):
    #         # Forward pass
    #         predictions = model.forward(x_batch)
    #         
    #         # Compute loss
    #         loss = model.cross_entropy_loss(predictions, y_batch)
    #         epoch_loss += loss
    #         
    #         # Backward pass
    #         gradients = model.backward(x_batch, y_batch)
    #         
    #         # Update parameters
    #         model.update_parameters(gradients, learning_rate)
    #     
    #     # Compute metrics
    #     train_acc = compute_accuracy(model.predict(x_train_split), y_train_split_labels)
    #     val_acc = compute_accuracy(model.predict(x_val), y_val)
    #     
    #     # Log progress
    #     ...
    
    raise NotImplementedError("Implement NumPy training loop")
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)
    test_acc = compute_accuracy(model.predict(x_test), y_test)
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Visualize results
    if args.visualize:
        plot_training_curves(train_losses, val_losses, train_accs, val_accs)
        
        predictions = model.predict(x_test[:10])
        visualize_samples(x_test[:10], y_test[:10], predictions)
    
    return model


def train_pytorch_model(args):
    """
    Train the PyTorch implementation.
    
    TODO: Implement the training loop
    
    Steps:
    1. Set up device (CPU/GPU)
    2. Create DataLoaders
    3. Initialize model, optimizer, loss function
    4. For each epoch:
        - Train one epoch
        - Evaluate on test set
        - Log metrics
    5. Plot training curves
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from pytorch_net import PyTorchNet, create_data_loaders, train_one_epoch, evaluate
    
    print("=" * 60)
    print("Training PyTorch Implementation")
    print("=" * 60)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Hyperparameters
    learning_rate = args.lr
    batch_size = args.batch_size
    epochs = args.epochs
    
    # Data
    print("\nLoading MNIST...")
    train_loader, test_loader = create_data_loaders(batch_size)
    print(f"Training batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Model
    print("\nInitializing model...")
    model = PyTorchNet(
        input_size=784,
        hidden_size=args.hidden_size,
        output_size=10
    ).to(device)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # TODO: Initialize optimizer and loss function
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    # or try Adam:
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # criterion = nn.CrossEntropyLoss()
    
    # Training tracking
    train_losses = []
    test_losses = []
    test_accs = []
    
    # Optional: W&B logging
    if args.wandb:
        import wandb
        wandb.init(
            project="mnist-from-scratch",
            config={
                "model": "pytorch",
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "hidden_size": args.hidden_size,
                "epochs": epochs,
                "optimizer": "SGD"  # or "Adam"
            }
        )
    
    print(f"\nTraining for {epochs} epochs...")
    print(f"Learning rate: {learning_rate}, Batch size: {batch_size}")
    print("-" * 60)
    
    # YOUR CODE HERE: Implement training loop
    #
    # for epoch in range(epochs):
    #     start_time = time.time()
    #     
    #     # Train
    #     train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    #     
    #     # Evaluate
    #     test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    #     
    #     # Track metrics
    #     train_losses.append(train_loss)
    #     test_losses.append(test_loss)
    #     test_accs.append(test_acc)
    #     
    #     # Log progress
    #     elapsed = time.time() - start_time
    #     print(f"Epoch {epoch+1}/{epochs} | "
    #           f"Train Loss: {train_loss:.4f} | "
    #           f"Test Loss: {test_loss:.4f} | "
    #           f"Test Acc: {test_acc:.4f} | "
    #           f"Time: {elapsed:.1f}s")
    #     
    #     if args.wandb:
    #         wandb.log({...})
    
    raise NotImplementedError("Implement PyTorch training loop")
    
    # Final results
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Best Test Accuracy: {max(test_accs):.4f}")
    
    # Visualize results
    if args.visualize:
        from utils import plot_training_curves, visualize_samples, load_mnist
        
        # Plot curves
        plot_training_curves(train_losses, test_losses, test_accs=test_accs)
        
        # Show some predictions
        x_test, y_test, _, _ = load_mnist()
        x_test_tensor = torch.tensor(x_test[:10], dtype=torch.float32).to(device)
        predictions = model.predict(x_test_tensor).cpu().numpy()
        visualize_samples(x_test[:10], y_test[:10], predictions)
    
    return model


def main():
    parser = argparse.ArgumentParser(description='Train MNIST classifier')
    
    parser.add_argument('--model', type=str, default='numpy',
                        choices=['numpy', 'pytorch'],
                        help='Which implementation to train')
    
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for training')
    
    parser.add_argument('--lr', type=float, default=0.1,
                        help='Learning rate')
    
    parser.add_argument('--hidden-size', type=int, default=128,
                        help='Hidden layer size')
    
    parser.add_argument('--wandb', action='store_true',
                        help='Enable Weights & Biases logging')
    
    parser.add_argument('--visualize', action='store_true',
                        help='Show training plots and sample predictions')
    
    args = parser.parse_args()
    
    if args.model == 'numpy':
        train_numpy_model(args)
    else:
        train_pytorch_model(args)


if __name__ == "__main__":
    main()
