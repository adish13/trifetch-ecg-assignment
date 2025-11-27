"""
ML Module for ECG Arrhythmia Classification

Components:
- data_preprocessing: Data loading, augmentation, and splitting
- model: 1D ResNet + Attention architecture
- training: Training pipeline with focal loss
- evaluation: Comprehensive metrics and visualization
"""

from ml.data_preprocessing import (
    load_all_data,
    normalize_data,
    augment_dataset,
    prepare_data,
    CLASS_NAMES,
    CLASS_MAP
)

from ml.model import (
    build_resnet_attention_model,
    build_simple_cnn
)

from ml.training import (
    FocalLoss,
    train_model,
    cross_validate
)

from ml.evaluation import (
    evaluate_model,
    plot_confusion_matrix,
    plot_roc_curves,
    plot_precision_recall_curves,
    plot_training_history,
    generate_report
)

__all__ = [
    'load_all_data',
    'normalize_data',
    'augment_dataset',
    'prepare_data',
    'CLASS_NAMES',
    'CLASS_MAP',
    'build_resnet_attention_model',
    'build_simple_cnn',
    'FocalLoss',
    'train_model',
    'cross_validate',
    'evaluate_model',
    'plot_confusion_matrix',
    'plot_roc_curves',
    'plot_precision_recall_curves',
    'plot_training_history',
    'generate_report'
]

