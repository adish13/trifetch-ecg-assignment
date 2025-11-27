"""
Evaluation Module for ECG Arrhythmia Classification

Provides comprehensive metrics:
- Per-class precision, recall, F1-score
- Confusion matrix
- AUROC curves
- Sensitivity analysis (critical for medical AI)

Works with PyTorch models.
"""

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve,
    f1_score, precision_score, recall_score, accuracy_score
)
from typing import Dict, List, Tuple, Optional
import os


CLASS_NAMES = ['AF', 'VTACH', 'PAUSE']


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: List[str] = CLASS_NAMES,
    save_dir: Optional[str] = None,
    device: torch.device = None
) -> Dict:
    """
    Comprehensive model evaluation.

    Returns:
        Dictionary with all metrics and predictions.
    """
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else
                             'mps' if torch.backends.mps.is_available() else 'cpu')

    model = model.to(device)
    model.eval()

    # Get predictions
    X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)

    with torch.no_grad():
        outputs = model(X_test_t)
        y_pred_proba = F.softmax(outputs, dim=1).cpu().numpy()
        y_pred = outputs.argmax(dim=1).cpu().numpy()

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)

    # Per-class metrics
    precision = precision_score(y_test, y_pred, average=None)
    recall = recall_score(y_test, y_pred, average=None)
    f1 = f1_score(y_test, y_pred, average=None)

    # Macro and weighted averages
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    weighted_f1 = f1_score(y_test, y_pred, average='weighted')

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Classification report
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)

    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"\nOverall Accuracy: {accuracy:.4f}")
    print(f"Macro F1-Score: {macro_f1:.4f}")
    print(f"Weighted F1-Score: {weighted_f1:.4f}")

    print("\n" + "-"*40)
    print("Per-Class Metrics:")
    print("-"*40)
    print(f"{'Class':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-"*40)
    for i, name in enumerate(class_names):
        print(f"{name:<10} {precision[i]:<12.4f} {recall[i]:<12.4f} {f1[i]:<12.4f}")

    print("\n" + "-"*40)
    print("Confusion Matrix:")
    print("-"*40)
    print(f"{'':>10}", end="")
    for name in class_names:
        print(f"{name:>10}", end="")
    print()
    for i, name in enumerate(class_names):
        print(f"{name:>10}", end="")
        for j in range(len(class_names)):
            print(f"{cm[i,j]:>10}", end="")
        print()

    # Medical AI specific metrics
    print("\n" + "-"*40)
    print("Medical AI Critical Metrics (Sensitivity):")
    print("-"*40)
    for i, name in enumerate(class_names):
        sensitivity = recall[i]
        specificity = calculate_specificity(y_test, y_pred, i)
        print(f"{name}: Sensitivity={sensitivity:.4f}, Specificity={specificity:.4f}")

    # Save plots if directory provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plot_confusion_matrix(cm, class_names, save_path=os.path.join(save_dir, 'confusion_matrix.png'))
        plot_roc_curves(y_test, y_pred_proba, class_names, save_path=os.path.join(save_dir, 'roc_curves.png'))
        plot_precision_recall_curves(y_test, y_pred_proba, class_names,
                                     save_path=os.path.join(save_dir, 'pr_curves.png'))

    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'per_class_precision': dict(zip(class_names, precision)),
        'per_class_recall': dict(zip(class_names, recall)),
        'per_class_f1': dict(zip(class_names, f1)),
        'confusion_matrix': cm,
        'classification_report': report,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }


def calculate_specificity(y_true: np.ndarray, y_pred: np.ndarray, class_idx: int) -> float:
    """Calculate specificity (True Negative Rate) for a specific class."""
    y_binary = (y_true == class_idx).astype(int)
    y_pred_binary = (y_pred == class_idx).astype(int)

    tn = np.sum((y_binary == 0) & (y_pred_binary == 0))
    fp = np.sum((y_binary == 0) & (y_pred_binary == 1))

    if tn + fp == 0:
        return 0.0
    return tn / (tn + fp)


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str],
                          save_path: Optional[str] = None):
    """Plot confusion matrix as heatmap."""
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved confusion matrix to {save_path}")
    plt.close()


def plot_roc_curves(y_true: np.ndarray, y_pred_proba: np.ndarray,
                    class_names: List[str], save_path: Optional[str] = None):
    """Plot ROC curves for each class (one-vs-rest)."""
    n_classes = len(class_names)

    plt.figure(figsize=(10, 8))
    colors = ['#2196F3', '#F44336', '#4CAF50']

    for i, (name, color) in enumerate(zip(class_names, colors)):
        # Create binary labels for this class
        y_binary = (y_true == i).astype(int)

        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_binary, y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, color=color, lw=2,
                label=f'{name} (AUC = {roc_auc:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves (One-vs-Rest)')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved ROC curves to {save_path}")
    plt.close()


def plot_precision_recall_curves(y_true: np.ndarray, y_pred_proba: np.ndarray,
                                  class_names: List[str], save_path: Optional[str] = None):
    """Plot Precision-Recall curves for each class."""
    plt.figure(figsize=(10, 8))
    colors = ['#2196F3', '#F44336', '#4CAF50']

    for i, (name, color) in enumerate(zip(class_names, colors)):
        y_binary = (y_true == i).astype(int)

        precision, recall, _ = precision_recall_curve(y_binary, y_pred_proba[:, i])
        pr_auc = auc(recall, precision)

        plt.plot(recall, precision, color=color, lw=2,
                label=f'{name} (AUC = {pr_auc:.3f})')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved PR curves to {save_path}")
    plt.close()


def plot_training_history(history: Dict, save_path: Optional[str] = None):
    """Plot training and validation loss/accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    ax1.plot(history['loss'], label='Train Loss', color='#2196F3')
    ax1.plot(history['val_loss'], label='Val Loss', color='#F44336')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(history['accuracy'], label='Train Accuracy', color='#2196F3')
    ax2.plot(history['val_accuracy'], label='Val Accuracy', color='#F44336')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training history to {save_path}")
    plt.close()


def generate_report(metrics: Dict, save_path: str):
    """Generate a text report of all metrics."""
    with open(save_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("ECG ARRHYTHMIA CLASSIFICATION - EVALUATION REPORT\n")
        f.write("="*60 + "\n\n")

        f.write("OVERALL METRICS\n")
        f.write("-"*40 + "\n")
        f.write(f"Accuracy:        {metrics['accuracy']:.4f}\n")
        f.write(f"Macro F1-Score:  {metrics['macro_f1']:.4f}\n")
        f.write(f"Weighted F1-Score: {metrics['weighted_f1']:.4f}\n\n")

        f.write("PER-CLASS METRICS\n")
        f.write("-"*40 + "\n")
        f.write(f"{'Class':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}\n")
        f.write("-"*40 + "\n")
        for cls in metrics['per_class_precision'].keys():
            f.write(f"{cls:<10} "
                   f"{metrics['per_class_precision'][cls]:<12.4f} "
                   f"{metrics['per_class_recall'][cls]:<12.4f} "
                   f"{metrics['per_class_f1'][cls]:<12.4f}\n")

        f.write("\n\nCONFUSION MATRIX\n")
        f.write("-"*40 + "\n")
        cm = metrics['confusion_matrix']
        class_names = list(metrics['per_class_precision'].keys())
        f.write(f"{'':>10}")
        for name in class_names:
            f.write(f"{name:>10}")
        f.write("\n")
        for i, name in enumerate(class_names):
            f.write(f"{name:>10}")
            for j in range(len(class_names)):
                f.write(f"{cm[i,j]:>10}")
            f.write("\n")

        f.write("\n\nMEDICAL AI NOTES\n")
        f.write("-"*40 + "\n")
        f.write("Sensitivity (Recall) is critical for medical diagnosis.\n")
        f.write("Missing a life-threatening arrhythmia (False Negative) is more\n")
        f.write("dangerous than a false alarm (False Positive).\n\n")

        for cls in class_names:
            sensitivity = metrics['per_class_recall'][cls]
            if sensitivity < 0.90:
                f.write(f"⚠️  {cls}: Sensitivity ({sensitivity:.2%}) below 90% threshold\n")
            else:
                f.write(f"✅ {cls}: Sensitivity ({sensitivity:.2%}) meets medical standards\n")

    print(f"Saved evaluation report to {save_path}")

