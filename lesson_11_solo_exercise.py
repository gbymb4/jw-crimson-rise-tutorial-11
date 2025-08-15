# -*- coding: utf-8 -*-
"""
Created on Fri Aug 15 12:38:02 2025

@author: taske
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Set device and random seed for reproducibility
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)
print(f"Using device: {device}")

# Data loading (provided for you)
def load_fashion_mnist():
    """Load Fashion-MNIST dataset"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    trainset = torchvision.datasets.FashionMNIST(root='./data', train=True,
                                               download=True, transform=transform)
    # Split training set into train and validation
    train_size = int(0.8 * len(trainset))
    val_size = len(trainset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(trainset, [train_size, val_size])
    
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    valloader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False)
    
    testset = torchvision.datasets.FashionMNIST(root='./data', train=False,
                                              download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)
    
    return trainloader, valloader, testloader

# Class names for reference
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

class BaselineNetwork(nn.Module):
    """Simple 3-layer fully connected network without regularization"""
    def __init__(self, input_size=784, hidden1=512, hidden2=256, num_classes=10):
        super(BaselineNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, num_classes)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the image
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class RegularizedNetwork(nn.Module):
    """3-layer fully connected network with regularization"""
    def __init__(self, input_size=784, hidden1=512, hidden2=256, num_classes=10, dropout_rate=0.5):
        super(RegularizedNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, num_classes)
        
        # TODO 1: Add dropout layers with the given dropout_rate
        # Hint: Use nn.Dropout(dropout_rate)
        # You'll need dropout after fc1 and fc2 layers
        
        # YOUR CODE HERE:
        # self.dropout = ...
        
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the image
        x = torch.relu(self.fc1(x))
        
        # TODO 1 (continued): Apply dropout after first activation
        # YOUR CODE HERE:
        # x = ...
        
        x = torch.relu(self.fc2(x))
        
        # TODO 1 (continued): Apply dropout after second activation
        # YOUR CODE HERE:
        # x = ...
        
        x = self.fc3(x)
        return x

def train_model(model, trainloader, valloader, num_epochs=15, lr=0.001, weight_decay=0.0, 
                use_scheduler=False, early_stopping=False):
    """
    Training function with validation tracking
    
    Returns dictionary with training history
    """
    criterion = nn.CrossEntropyLoss()
    
    # TODO 2: Create optimizer with L2 regularization (weight_decay parameter)
    # Hint: Use optim.Adam with weight_decay parameter
    # YOUR CODE HERE:
    # optimizer = optim.Adam(...)
    
    # TODO 3: Create learning rate scheduler (ReduceLROnPlateau)
    # Only create if use_scheduler is True
    # Hint: Import from torch.optim.lr_scheduler
    scheduler = None
    if use_scheduler:
        # YOUR CODE HERE:
        # scheduler = ...
        pass
    
    # Early stopping variables
    # TODO 4: Initialize early stopping variables
    # YOUR CODE HERE:
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 5  # Stop if no improvement for 5 epochs
    
    # Lists to store training history
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
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
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for inputs, labels in valloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        
        # Calculate metrics
        avg_train_loss = running_loss / len(trainloader)
        avg_val_loss = val_loss / len(valloader)
        train_acc = 100 * correct_train / total_train
        val_acc = 100 * correct_val / total_val
        
        # Store metrics
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        print(f'Epoch [{epoch+1}/{num_epochs}] - '
              f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}% | '
              f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # TODO 3 (continued): Update learning rate scheduler if enabled
        if scheduler is not None:
            # YOUR CODE HERE:
            # scheduler.step(...)
            pass
        
        # TODO 4 (continued): Implement early stopping logic
        if early_stopping:
            # YOUR CODE HERE:
            # Check if validation loss improved
            # Update best_val_loss and patience_counter accordingly
            # Break if patience exceeded
            pass
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies
    }

def evaluate_model(model, testloader):
    """Evaluate model on test set and return accuracy"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

def plot_training_comparison(baseline_history, regularized_history):
    """TODO 5: Create comparison plots of training curves"""
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # TODO 5: Plot training and validation losses for both models
    # Subplot 1 (top-left): Loss curves
    # YOUR CODE HERE:
    # axes[0,0].plot(baseline_history['train_losses'], label='Baseline Train', ...)
    # axes[0,0].plot(baseline_history['val_losses'], label='Baseline Val', ...)
    # axes[0,0].plot(regularized_history['train_losses'], label='Regularized Train', ...)
    # axes[0,0].plot(regularized_history['val_losses'], label='Regularized Val', ...)
    # axes[0,0].set_title('Loss Curves')
    # axes[0,0].set_xlabel('Epoch')
    # axes[0,0].set_ylabel('Loss')
    # axes[0,0].legend()
    # axes[0,0].grid(True)
    
    # TODO 5 (continued): Plot accuracy curves
    # Subplot 2 (top-right): Accuracy curves
    # YOUR CODE HERE - Similar to above but for accuracies
    
    # TODO 5 (continued): Plot overfitting analysis
    # Subplot 3 (bottom-left): Overfitting gap (val_loss - train_loss)
    baseline_gap = np.array(baseline_history['val_losses']) - np.array(baseline_history['train_losses'])
    regularized_gap = np.array(regularized_history['val_losses']) - np.array(regularized_history['train_losses'])
    
    # YOUR CODE HERE:
    # Plot the gaps and add horizontal line at y=0
    
    # TODO 5 (continued): Summary statistics
    # Subplot 4 (bottom-right): Final performance comparison
    models = ['Baseline', 'Regularized']
    final_val_accs = [baseline_history['val_accuracies'][-1], 
                      regularized_history['val_accuracies'][-1]]
    
    # YOUR CODE HERE:
    # Create bar plot of final validation accuracies
    
    plt.tight_layout()
    plt.savefig('fashion_mnist_regularization_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main exercise function"""
    print("Loading Fashion-MNIST dataset...")
    trainloader, valloader, testloader = load_fashion_mnist()
    
    print("Fashion-MNIST classes:", class_names)
    print(f"Training batches: {len(trainloader)}")
    print(f"Validation batches: {len(valloader)}")
    print(f"Test batches: {len(testloader)}")
    
    # Initialize models
    print("\nInitializing models...")
    baseline_model = BaselineNetwork().to(device)
    regularized_model = RegularizedNetwork(dropout_rate=0.3).to(device)
    
    print(f"Baseline model parameters: {sum(p.numel() for p in baseline_model.parameters())}")
    print(f"Regularized model parameters: {sum(p.numel() for p in regularized_model.parameters())}")
    
    # Train baseline model
    print("\n" + "="*50)
    print("TRAINING BASELINE MODEL")
    print("="*50)
    
    baseline_history = train_model(
        baseline_model, trainloader, valloader, 
        num_epochs=15, lr=0.001
    )
    
    # Train regularized model
    print("\n" + "="*50)
    print("TRAINING REGULARIZED MODEL")
    print("="*50)
    
    regularized_history = train_model(
        regularized_model, trainloader, valloader,
        num_epochs=15, lr=0.001, weight_decay=0.001,
        use_scheduler=True, early_stopping=True
    )
    
    # Evaluate on test set
    print("\n" + "="*50)
    print("TEST SET EVALUATION")
    print("="*50)
    
    # TODO 6: Calculate and compare final test accuracies
    baseline_test_acc = evaluate_model(baseline_model, testloader)
    regularized_test_acc = evaluate_model(regularized_model, testloader)
    
    print(f"Baseline Test Accuracy: {baseline_test_acc:.2f}%")
    print(f"Regularized Test Accuracy: {regularized_test_acc:.2f}%")
    print(f"Improvement: {regularized_test_acc - baseline_test_acc:.2f}%")
    
    # Create comparison plots
    print("\nGenerating comparison plots...")
    plot_training_comparison(baseline_history, regularized_history)
    
    # Analysis summary
    print("\n" + "="*50)
    print("ANALYSIS QUESTIONS FOR REFLECTION:")
    print("="*50)
    print("1. Which model shows more overfitting? How can you tell?")
    print("2. Does regularization improve test performance?")
    print("3. What happens to the training curves with regularization?")
    print("4. When might you choose different dropout rates?")
    print("5. How does L2 regularization affect the loss values?")
    
    print(f"\nResults saved to: fashion_mnist_regularization_results.png")
    print("Exercise completed! Review your plots and answer the reflection questions.")

if __name__ == "__main__":
    main()