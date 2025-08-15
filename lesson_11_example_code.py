# -*- coding: utf-8 -*-
"""
Created on Fri Aug 15 12:37:33 2025

@author: taske
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Data loading and preprocessing
def load_cifar10():
    """Load and preprocess CIFAR-10 dataset"""
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                          download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                            shuffle=True, num_workers=2)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                         download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                           shuffle=False, num_workers=2)
    
    return trainloader, testloader

# Model Definitions
class BaselineNet(nn.Module):
    """Simple CNN without regularization - prone to overfitting"""
    def __init__(self):
        super(BaselineNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class RegularizedNet(nn.Module):
    """CNN with dropout regularization"""
    def __init__(self, dropout_rate=0.5):
        super(RegularizedNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.dropout(torch.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

# Training function with validation tracking
def train_model(model, trainloader, testloader, num_epochs=10, lr=0.001, 
                weight_decay=0.0, use_scheduler=False, early_stopping=False):
    """
    Train model with optional regularization techniques
    
    Args:
        model: PyTorch model to train
        trainloader: Training data loader
        testloader: Test data loader for validation
        num_epochs: Maximum number of epochs
        lr: Learning rate
        weight_decay: L2 regularization strength
        use_scheduler: Whether to use learning rate scheduler
        early_stopping: Whether to use early stopping
    """
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Optional learning rate scheduler
    scheduler = None
    if use_scheduler:
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, 
                                    patience=3, verbose=True)
    
    # Early stopping parameters
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 5
    
    # Track metrics
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total_train += labels.size(0)
            correct_train += predicted.eq(labels).sum().item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total_val += labels.size(0)
                correct_val += predicted.eq(labels).sum().item()
        
        # Calculate averages
        avg_train_loss = running_loss / len(trainloader)
        avg_val_loss = val_loss / len(testloader)
        train_acc = 100. * correct_train / total_train
        val_acc = 100. * correct_val / total_val
        
        # Store metrics
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs}: '
              f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}% | '
              f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Learning rate scheduling
        if scheduler:
            scheduler.step(avg_val_loss)
        
        # Early stopping check
        if early_stopping:
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs
    }

# Visualization function
def plot_comparison(results_dict):
    """Plot training curves for model comparison"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    colors = ['blue', 'red', 'green']
    
    for i, (model_name, results) in enumerate(results_dict.items()):
        color = colors[i % len(colors)]
        epochs = range(1, len(results['train_losses']) + 1)
        
        # Training loss
        ax1.plot(epochs, results['train_losses'], f'{color}-', 
                label=f'{model_name} (Train)', alpha=0.7)
        ax1.plot(epochs, results['val_losses'], f'{color}--', 
                label=f'{model_name} (Val)', alpha=0.7)
    
    ax1.set_title('Loss Curves Comparison')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    for i, (model_name, results) in enumerate(results_dict.items()):
        color = colors[i % len(colors)]
        epochs = range(1, len(results['train_accs']) + 1)
        
        # Accuracy curves
        ax2.plot(epochs, results['train_accs'], f'{color}-', 
                label=f'{model_name} (Train)', alpha=0.7)
        ax2.plot(epochs, results['val_accs'], f'{color}--', 
                label=f'{model_name} (Val)', alpha=0.7)
    
    ax2.set_title('Accuracy Curves Comparison')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    # Overfitting analysis
    for i, (model_name, results) in enumerate(results_dict.items()):
        color = colors[i % len(colors)]
        epochs = range(1, len(results['train_losses']) + 1)
        overfitting = np.array(results['val_losses']) - np.array(results['train_losses'])
        ax3.plot(epochs, overfitting, f'{color}-', label=model_name, alpha=0.7)
    
    ax3.set_title('Overfitting Analysis (Val Loss - Train Loss)')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss Difference')
    ax3.legend()
    ax3.grid(True)
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    
    # Final performance comparison
    model_names = list(results_dict.keys())
    final_val_accs = [results['val_accs'][-1] for results in results_dict.values()]
    
    ax4.bar(model_names, final_val_accs, color=colors[:len(model_names)], alpha=0.7)
    ax4.set_title('Final Validation Accuracy')
    ax4.set_ylabel('Accuracy (%)')
    ax4.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig('regularization_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main demonstration function"""
    print("Loading CIFAR-10 dataset...")
    trainloader, testloader = load_cifar10()
    
    # Initialize models
    models = {
        'Baseline': BaselineNet().to(device),
        'Dropout + L2': RegularizedNet(dropout_rate=0.3).to(device),
        'Full Regularization': RegularizedNet(dropout_rate=0.3).to(device)
    }
    
    print("\n" + "="*60)
    print("REGULARIZATION DEMONSTRATION")
    print("="*60)
    
    results = {}
    
    # Train each model variant
    for name, model in models.items():
        print(f"\nTraining {name} model...")
        print("-" * 40)
        
        if name == 'Baseline':
            # No regularization
            result = train_model(model, trainloader, testloader, 
                               num_epochs=12, lr=0.001)
        elif name == 'Dropout + L2':
            # Dropout + L2 regularization
            result = train_model(model, trainloader, testloader, 
                               num_epochs=12, lr=0.001, weight_decay=0.001)
        else:  # Full Regularization
            # All techniques combined
            result = train_model(model, trainloader, testloader, 
                               num_epochs=12, lr=0.001, weight_decay=0.001,
                               use_scheduler=True, early_stopping=True)
        
        results[name] = result
    
    # Create comparison plots
    print("\nGenerating comparison plots...")
    plot_comparison(results)
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY OF RESULTS")
    print("="*60)
    
    for name, result in results.items():
        final_train_acc = result['train_accs'][-1]
        final_val_acc = result['val_accs'][-1]
        overfitting_gap = final_train_acc - final_val_acc
        
        print(f"{name}:")
        print(f"  Final Training Accuracy: {final_train_acc:.2f}%")
        print(f"  Final Validation Accuracy: {final_val_acc:.2f}%")
        print(f"  Overfitting Gap: {overfitting_gap:.2f}%")
        print()
    
    print("Key Observations:")
    print("- Higher overfitting gap indicates more memorization of training data")
    print("- Regularization should reduce this gap and improve generalization")
    print("- Best model balances training performance with validation performance")

if __name__ == "__main__":
    main()