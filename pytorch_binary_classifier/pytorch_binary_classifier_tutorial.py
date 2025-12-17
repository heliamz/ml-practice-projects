"""
PyTorch Binary Classifier Tutorial
===================================

## Project Information
- **Source**: PyTorch Tutorial / Educational Project
- **Objective**: Learn to build a binary classifier using PyTorch
- **Framework**: PyTorch
- **Dataset**: Synthetic data (for demonstration purposes)

## Overview
This tutorial teaches you how to build a binary classifier using PyTorch.
A binary classifier predicts one of two classes (e.g., spam/not spam, 
sick/healthy, pass/fail).

Key Components:
1. Model Architecture (Neural Network)
2. Loss Function (Binary Cross Entropy)
3. Optimizer (Adam)
4. Training Loop
5. Evaluation

Let's build one step by step!
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ============================================================================
# STEP 1: CREATE SYNTHETIC DATA (for demonstration)
# ============================================================================
# In real projects, you'd load your own data here

print("=" * 60)
print("STEP 1: Creating Synthetic Dataset")
print("=" * 60)

# Generate synthetic binary classification data
n_samples = 1000
n_features = 10

# Create features (X) - random data
X = np.random.randn(n_samples, n_features).astype(np.float32)

# Create labels (y) - binary (0 or 1)
# We'll make it so that features 0-4 influence class 1, and 5-9 influence class 0
# This creates a learnable pattern
y = ((X[:, 0:5].sum(axis=1) - X[:, 5:10].sum(axis=1)) > 0).astype(np.int64)

print(f"Dataset shape: X={X.shape}, y={y.shape}")
print(f"Class distribution: Class 0={np.sum(y==0)}, Class 1={np.sum(y==1)}")

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Normalize features (important for neural networks!)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(f"Train set: {X_train.shape}, Test set: {X_test.shape}\n")

# ============================================================================
# STEP 2: CONVERT TO PYTORCH TENSORS AND CREATE DATALOADERS
# ============================================================================

print("=" * 60)
print("STEP 2: Converting to PyTorch Tensors")
print("=" * 60)

# Convert numpy arrays to PyTorch tensors
X_train_tensor = torch.from_numpy(X_train)
X_test_tensor = torch.from_numpy(X_test)
y_train_tensor = torch.from_numpy(y_train)
y_test_tensor = torch.from_numpy(y_test)

# Create datasets and dataloaders
# DataLoader handles batching and shuffling automatically
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Batch size: {batch_size}")
print(f"Number of batches per epoch: {len(train_loader)}\n")

# ============================================================================
# STEP 3: DEFINE THE NEURAL NETWORK MODEL
# ============================================================================

print("=" * 60)
print("STEP 3: Defining the Neural Network")
print("=" * 60)

class BinaryClassifier(nn.Module):
    """
    A simple feedforward neural network for binary classification.
    
    Architecture:
    - Input layer: n_features neurons
    - Hidden layers: Multiple fully connected (linear) layers with ReLU activation
    - Output layer: 1 neuron with sigmoid activation (outputs probability 0-1)
    
    Why sigmoid?
    - Sigmoid squashes output to [0, 1] range
    - Perfect for binary classification (probability of class 1)
    - Works with Binary Cross Entropy loss
    """
    
    def __init__(self, input_dim, hidden_dims=[64, 32], dropout=0.2):
        """
        Initialize the model.
        
        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer sizes (e.g., [64, 32] = two hidden layers)
            dropout: Dropout probability for regularization (prevents overfitting)
        """
        # WHY super().__init__()?
        # =======================
        # Our BinaryClassifier inherits from nn.Module (the parent class).
        # nn.Module has its own __init__() that sets up important internal machinery:
        #   - Registers layers so PyTorch can track them
        #   - Enables .to(device), .train(), .eval() methods
        #   - Allows .parameters() to find all learnable weights
        #   - Sets up hooks for gradient computation
        #
        # If we don't call super().__init__(), nn.Module won't be properly initialized,
        # and our model won't work! It's like building a house without a foundation.
        #
        # Note: In Python 3+, you can also write: super().__init__()
        #       (Python automatically figures out the class and self)
        super(BinaryClassifier, self).__init__()
        
        # Build layers dynamically
        layers = []
        prev_dim = input_dim
        
        # Create hidden layers
        for hidden_dim in hidden_dims:
            # Linear (fully connected) layer: y = xW^T + b
            layers.append(nn.Linear(prev_dim, hidden_dim))
            # ReLU activation: max(0, x) - introduces non-linearity
            layers.append(nn.ReLU())
            # Batch normalization: normalizes activations (helps training)
            layers.append(nn.BatchNorm1d(hidden_dim))
            # Dropout: randomly sets some neurons to 0 during training (regularization)
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer: single neuron for binary classification
        layers.append(nn.Linear(prev_dim, 1))
        # Sigmoid: outputs probability between 0 and 1
        layers.append(nn.Sigmoid())
        
        # Combine all layers into a sequential model
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass: defines how data flows through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        
        Returns:
            Output tensor of shape (batch_size,) with probabilities
        """
        output = self.model(x)
        # Squeeze removes dimension of size 1: (batch_size, 1) -> (batch_size,)
        return output.squeeze()

# Create the model
input_dim = X_train.shape[1]
model = BinaryClassifier(input_dim=input_dim, hidden_dims=[64, 32], dropout=0.2)

print("Model Architecture:")
print(model)
print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")

# ============================================================================
# STEP 4: SET UP LOSS FUNCTION AND OPTIMIZER
# ============================================================================

print("=" * 60)
print("STEP 4: Setting Up Loss Function and Optimizer")
print("=" * 60)

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Move model to device (GPU or CPU)
model = model.to(device)

# Loss Function: Binary Cross Entropy (BCE)
# Why BCE?
# - Designed for binary classification
# - Measures difference between predicted probabilities and true labels
# - Works perfectly with sigmoid output
criterion = nn.BCELoss()
print(f"Loss function: Binary Cross Entropy (BCE)")

# Optimizer: Adam (Adaptive Moment Estimation)
# Why Adam?
# - Adaptive learning rate (adjusts automatically)
# - Works well in practice
# - Combines benefits of momentum and RMSprop
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
print(f"Optimizer: Adam with learning rate=0.001")
print(f"Weight decay (L2 regularization): 1e-5\n")

# ============================================================================
# STEP 5: TRAINING LOOP
# ============================================================================

print("=" * 60)
print("STEP 5: Training the Model")
print("=" * 60)

num_epochs = 50
train_losses = []
train_accuracies = []

# Set model to training mode (enables dropout, batch norm updates)
model.train()

for epoch in range(num_epochs):
    epoch_loss = 0.0
    correct = 0
    total = 0
    
    # Iterate over batches
    for batch_X, batch_y in train_loader:
        # Move data to device (GPU or CPU)
        batch_X = batch_X.to(device)
        batch_y = batch_y.float().to(device)  # BCE expects float labels
        
        # ===== FORWARD PASS =====
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        
        # Forward pass: compute predictions
        outputs = model(batch_X)  # Shape: (batch_size,)
        
        # Compute loss
        loss = criterion(outputs, batch_y)
        
        # ===== BACKWARD PASS =====
        # Backward pass: compute gradients
        loss.backward()
        
        # Update weights: optimizer takes a step based on gradients
        optimizer.step()
        
        # ===== TRACKING METRICS =====
        # Convert probabilities to predictions (threshold = 0.5)
        predicted = (outputs > 0.5).float()
        
        # Calculate accuracy
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()
        epoch_loss += loss.item()
    
    # Average loss and accuracy for this epoch
    avg_loss = epoch_loss / len(train_loader)
    accuracy = 100 * correct / total
    
    train_losses.append(avg_loss)
    train_accuracies.append(accuracy)
    
    # Print progress every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Loss: {avg_loss:.4f}, '
              f'Accuracy: {accuracy:.2f}%')

print("\nTraining completed!\n")

# ============================================================================
# STEP 6: VISUALIZE TRAINING HISTORY
# ============================================================================

print("=" * 60)
print("STEP 6: Visualizing Training History")
print("=" * 60)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(train_losses)
plt.title('Training Loss Over Time')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(train_accuracies)
plt.title('Training Accuracy Over Time')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.grid(True)

plt.tight_layout()
plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
print("Training history saved to 'training_history.png'\n")

# ============================================================================
# STEP 7: EVALUATE ON TEST SET
# ============================================================================

print("=" * 60)
print("STEP 7: Evaluating on Test Set")
print("=" * 60)

# Set model to evaluation mode (disables dropout, batch norm uses running stats)
model.eval()

all_preds = []
all_probs = []
all_labels = []

# Disable gradient computation for evaluation (saves memory and computation)
with torch.no_grad():
    for batch_X, batch_y in test_loader:
        batch_X = batch_X.to(device)
        
        # Forward pass
        outputs = model(batch_X)
        
        # Convert to numpy
        probs = outputs.cpu().numpy()
        preds = (probs > 0.5).astype(int)
        
        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(batch_y.numpy())

# Convert to numpy arrays
y_pred = np.array(all_preds)
y_proba = np.array(all_probs)
y_true = np.array(all_labels)

# Calculate metrics
accuracy = accuracy_score(y_true, y_pred)

print(f"Test Accuracy: {accuracy:.4f}")
print(f"\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=['Class 0', 'Class 1']))
print(f"\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))

# ============================================================================
# STEP 8: KEY CONCEPTS EXPLAINED
# ============================================================================

print("\n" + "=" * 60)
print("KEY CONCEPTS EXPLAINED")
print("=" * 60)
print("""
0. INHERITANCE AND super().__init__():
   - BinaryClassifier inherits from nn.Module (parent class)
   - nn.Module provides essential functionality:
     * Tracks all layers and parameters
     * Enables .to(device), .train(), .eval() methods
     * Handles gradient computation
   - super().__init__() calls the parent class's __init__()
   - Without it, nn.Module isn't initialized → model breaks!
   - Think of it as: "Hey parent class, set yourself up first!"
   - Modern Python 3: super().__init__() works too (auto-detects class)

1. NEURAL NETWORK LAYERS:
   - Linear (Fully Connected): y = xW^T + b
     * W = weights (learnable parameters)
     * b = bias (learnable parameters)
   - ReLU: max(0, x) - adds non-linearity
   - BatchNorm: normalizes activations (speeds up training)
   - Dropout: randomly zeros neurons (prevents overfitting)
   - Sigmoid: 1/(1+e^(-x)) - outputs probability [0, 1]

2. LOSS FUNCTION (BCE):
   - Measures how wrong our predictions are
   - Lower loss = better predictions
   - Formula: -[y*log(p) + (1-y)*log(1-p)]
   - y = true label (0 or 1)
   - p = predicted probability

3. OPTIMIZER (Adam):
   - Updates model weights to minimize loss
   - Uses gradients (derivatives) to know which direction to update
   - Learning rate controls step size
   - Weight decay adds L2 regularization

4. TRAINING PROCESS:
   a) Forward pass: Input → Model → Predictions
   b) Compute loss: Compare predictions to true labels
   c) Backward pass: Calculate gradients (how to change weights)
   d) Update weights: Optimizer adjusts weights based on gradients
   e) Repeat for many epochs

5. EVALUATION:
   - Set model.eval() to disable dropout/batch norm training behavior
   - Use torch.no_grad() to save memory (no gradient computation needed)
   - Threshold predictions at 0.5 (probability > 0.5 = class 1)

6. HYPERPARAMETERS TO TUNE:
   - Learning rate (lr): How big steps to take (try 0.0001 to 0.01)
   - Batch size: How many samples per update (try 16, 32, 64, 128)
   - Hidden layer sizes: Model capacity (try [32], [64, 32], [128, 64, 32])
   - Dropout: Regularization strength (try 0.1 to 0.5)
   - Number of epochs: How long to train (stop when validation loss stops improving)

7. COMMON ISSUES:
   - Overfitting: Model memorizes training data
     * Solution: More dropout, less model capacity, more data
   - Underfitting: Model too simple
     * Solution: Larger model, more training epochs
   - Vanishing gradients: Deep networks struggle to learn
     * Solution: BatchNorm, better initialization, residual connections
""")

print("=" * 60)
print("Tutorial Complete! You now know how to build a PyTorch binary classifier!")
print("=" * 60)

