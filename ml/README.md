# ECG Arrhythmia Classifier

A deep learning model to classify ECG recordings into AF, VTACH, or PAUSE.

## Results

| Metric | Value |
|--------|-------|
| Accuracy | 96.4% |
| AF Sensitivity | 100% |
| VTACH Sensitivity | 100% |
| PAUSE Sensitivity | 90% |

## Model

**Architecture:** 1D ResNet + Multi-Head Attention

I chose this because:
- **ResNet** - skip connections help with training deeper networks and prevent gradient issues
- **1D Conv** - ECG is a time series, not an image. 1D convolutions capture temporal patterns
- **Attention** - lets the model focus on the important parts of a 90-second recording

The model has ~1.3M parameters.

## Training

```bash
python ml/train_ecg_model.py
```

Key choices:
- **Focal Loss** - focuses training on hard examples instead of easy ones
- **Data Augmentation** - noise, scaling, time shift (3x more training data)
- **Heavy Dropout (0.5)** - prevents overfitting on small dataset
- **Early Stopping** - stopped at epoch 52/100

## Files

```
ml/
├── model.py              # Model architecture
├── training.py           # Training loop + focal loss
├── data_preprocessing.py # Data loading + augmentation
├── evaluation.py         # Metrics + plots
├── train_ecg_model.py    # Main training script
└── models/
    ├── final_model.pt    # Trained weights
    └── scaler.pkl        # Data normalizer
```

## Usage

The backend automatically loads the model:

```python
from ml_classifier import MLECGClassifier
classifier = MLECGClassifier()
result = classifier.classify_and_detect(channel1, channel2, hint_index)
# Returns: (event_type, confidence, event_index, explanation)
```

If the model isn't found, it falls back to the rule-based classifier.

## Why not rule-based?

I started with a rule-based approach using R-R interval variability and heart rate thresholds. It worked (~80% accuracy) but the ML model:
- Learns features I might miss
- Handles edge cases better
- Achieves higher accuracy (96%)

The tradeoff is less interpretability, which is why I keep both.

