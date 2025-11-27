#!/usr/bin/env python3
"""
Main Training Script for ECG Arrhythmia Classification

This script:
1. Loads and preprocesses the ECG data
2. Trains a 1D ResNet + Attention model
3. Evaluates on test set
4. Saves the model and generates reports

Usage:
    python ml/train_ecg_model.py
"""

import os
import sys
import json
import pickle
import numpy as np
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"MPS Available: {torch.backends.mps.is_available()}")

from ml.data_preprocessing import prepare_data, load_all_data, normalize_data, augment_dataset
from ml.model import build_resnet_attention_model, build_simple_cnn
from ml.training import train_model, cross_validate
from ml.evaluation import evaluate_model, plot_training_history, generate_report


def main():
    """Main training pipeline."""

    # Configuration
    config = {
        'data_dir': 'data',
        'model_dir': 'ml/models',
        'results_dir': 'ml/results',
        'epochs': 100,
        'batch_size': 16,
        'learning_rate': 0.001,
        'use_focal_loss': True,
        'use_augmentation': True,
        'augment_factor': 3,
        'test_size': 0.2,
        'val_size': 0.15,
        'random_state': 42
    }

    # Create directories
    os.makedirs(config['model_dir'], exist_ok=True)
    os.makedirs(config['results_dir'], exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(config['results_dir'], f'run_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)

    # Save configuration
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    print("\n" + "="*60)
    print("ECG ARRHYTHMIA CLASSIFICATION - TRAINING PIPELINE")
    print("="*60)
    print(f"\nRun directory: {run_dir}")

    # Prepare data
    print("\n" + "-"*40)
    print("Step 1: Preparing Data")
    print("-"*40)

    data = prepare_data(
        data_dir=config['data_dir'],
        augment=config['use_augmentation'],
        augment_factor=config['augment_factor'],
        test_size=config['test_size'],
        val_size=config['val_size']
    )

    print(f"\nClass distribution in training set:")
    for i, name in enumerate(data['class_names']):
        count = np.sum(data['y_train'] == i)
        print(f"  {name}: {count}")

    # Build model
    print("\n" + "-"*40)
    print("Step 2: Building Model")
    print("-"*40)

    input_shape = (data['X_train'].shape[1], data['X_train'].shape[2])
    print(f"Input shape: {input_shape}")

    model = build_resnet_attention_model(
        input_shape=input_shape,
        num_classes=3,
        filters=[64, 128, 256],
        num_res_blocks=2,
        num_attention_heads=4,
        dropout_rate=0.5
    )

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {model.__class__.__name__}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Train model
    print("\n" + "-"*40)
    print("Step 3: Training Model")
    print("-"*40)

    training_result = train_model(
        model=model,
        X_train=data['X_train'],
        y_train=data['y_train'],
        X_val=data['X_val'],
        y_val=data['y_val'],
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        learning_rate=config['learning_rate'],
        use_focal_loss=config['use_focal_loss'],
        model_dir=config['model_dir']
    )

    # Plot training history
    plot_training_history(
        training_result['history'],
        save_path=os.path.join(run_dir, 'training_history.png')
    )

    print(f"\nBest validation accuracy: {training_result['best_val_accuracy']:.4f}")
    print(f"Best validation loss: {training_result['best_val_loss']:.4f}")

    # Evaluate on test set
    print("\n" + "-"*40)
    print("Step 4: Evaluating on Test Set")
    print("-"*40)

    metrics = evaluate_model(
        model=training_result['model'],
        X_test=data['X_test'],
        y_test=data['y_test'],
        class_names=data['class_names'],
        save_dir=run_dir,
        device=training_result['device']
    )

    # Generate report
    generate_report(metrics, os.path.join(run_dir, 'evaluation_report.txt'))

    # Save final model
    model_path = os.path.join(config['model_dir'], 'final_model.pt')
    torch.save(training_result['model'].state_dict(), model_path)
    print(f"\nSaved final model to {model_path}")

    # Save scaler for inference
    scaler_path = os.path.join(config['model_dir'], 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(data['scaler'], f)
    print(f"Saved scaler to {scaler_path}")

    # Summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"\nResults saved to: {run_dir}")
    print(f"Model saved to: {model_path}")
    print(f"\nFinal Test Metrics:")
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  Macro F1:    {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1: {metrics['weighted_f1']:.4f}")


if __name__ == '__main__':
    main()

