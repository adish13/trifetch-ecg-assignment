"""
Training Pipeline for ECG Arrhythmia Classification

Features:
- Focal Loss for handling hard examples
- Learning rate scheduling
- Early stopping
- Model checkpointing
- Cross-validation support

Uses PyTorch for training.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from typing import Dict, Tuple, List, Optional
import os
from datetime import datetime
from copy import deepcopy


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance and hard examples.

    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)

    Args:
        gamma: Focusing parameter (default 2.0). Higher = more focus on hard examples.
        alpha: Balancing factor (default 0.25).
    """
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, weight: torch.Tensor = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.weight = weight

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


def get_class_weights(y: np.ndarray, device: torch.device) -> torch.Tensor:
    """Calculate class weights inversely proportional to class frequency."""
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 100,
    batch_size: int = 16,
    learning_rate: float = 0.001,
    use_focal_loss: bool = True,
    use_class_weights: bool = True,
    model_dir: str = 'ml/models',
    patience: int = 15,
    device: str = None
) -> Dict:
    """
    Train the model with all optimizations.

    Returns:
        Dictionary with training history and final model.
    """
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else
                             'mps' if torch.backends.mps.is_available() else 'cpu')
    else:
        device = torch.device(device)
    print(f"Using device: {device}")

    model = model.to(device)

    # Create data loaders
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Get class weights
    class_weights = get_class_weights(y_train, device) if use_class_weights else None
    if class_weights is not None:
        print(f"Class weights: {class_weights.cpu().numpy()}")

    # Loss function
    if use_focal_loss:
        criterion = FocalLoss(gamma=2.0, alpha=0.25, weight=class_weights)
        print("Using Focal Loss")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print("Using Cross-Entropy Loss")

    # Optimizer and scheduler
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=7)

    # Training history
    history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}

    # Early stopping
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    os.makedirs(model_dir, exist_ok=True)

    print(f"\nTraining for up to {epochs} epochs...")
    print(f"Batch size: {batch_size}, Learning rate: {learning_rate}")

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)
            _, predicted = outputs.max(1)
            train_total += y_batch.size(0)
            train_correct += predicted.eq(y_batch).sum().item()

        train_loss /= train_total
        train_acc = train_correct / train_total

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)

                val_loss += loss.item() * X_batch.size(0)
                _, predicted = outputs.max(1)
                val_total += y_batch.size(0)
                val_correct += predicted.eq(y_batch).sum().item()

        val_loss /= val_total
        val_acc = val_correct / val_total

        # Update history
        history['loss'].append(train_loss)
        history['accuracy'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = deepcopy(model.state_dict())
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), os.path.join(model_dir, 'best_model.pt'))
        else:
            patience_counter += 1

        # Print progress
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Loss: {train_loss:.4f}, Acc: {train_acc:.4f} - "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return {
        'model': model,
        'history': history,
        'best_val_loss': best_val_loss,
        'best_val_accuracy': max(history['val_accuracy']),
        'device': device
    }


def cross_validate(
    build_model_fn,
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    epochs: int = 100,
    batch_size: int = 16,
    learning_rate: float = 0.001,
    use_focal_loss: bool = True,
    augment_fn=None
) -> Dict:
    """
    Perform k-fold cross-validation.

    Args:
        build_model_fn: Function that returns a new model instance
        X: Full training data
        y: Full training labels
        n_folds: Number of folds
        augment_fn: Optional function to augment training data

    Returns:
        Dictionary with fold results and averaged metrics.
    """
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_results = []
    all_val_acc = []
    all_val_loss = []

    print(f"\n{'='*60}")
    print(f"Starting {n_folds}-Fold Cross-Validation")
    print(f"{'='*60}\n")

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
        print(f"\n{'='*40}")
        print(f"Fold {fold + 1}/{n_folds}")
        print(f"{'='*40}")

        # Split data
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        # Augment training data if function provided
        if augment_fn:
            X_train_fold, y_train_fold = augment_fn(X_train_fold, y_train_fold)

        # Build fresh model for this fold
        model = build_model_fn()

        # Train
        result = train_model(
            model=model,
            X_train=X_train_fold,
            y_train=y_train_fold,
            X_val=X_val_fold,
            y_val=y_val_fold,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            use_focal_loss=use_focal_loss,
            model_dir=f'ml/models/fold_{fold+1}'
        )

        fold_results.append(result)
        all_val_acc.append(result['best_val_accuracy'])
        all_val_loss.append(result['best_val_loss'])

        print(f"\nFold {fold + 1} - Best Val Accuracy: {result['best_val_accuracy']:.4f}")

    # Summary
    print(f"\n{'='*60}")
    print("Cross-Validation Summary")
    print(f"{'='*60}")
    print(f"Val Accuracy: {np.mean(all_val_acc):.4f} (+/- {np.std(all_val_acc):.4f})")
    print(f"Val Loss: {np.mean(all_val_loss):.4f} (+/- {np.std(all_val_loss):.4f})")

    return {
        'fold_results': fold_results,
        'mean_accuracy': np.mean(all_val_acc),
        'std_accuracy': np.std(all_val_acc),
        'mean_loss': np.mean(all_val_loss),
        'std_loss': np.std(all_val_loss),
        'all_val_acc': all_val_acc,
        'all_val_loss': all_val_loss
    }

