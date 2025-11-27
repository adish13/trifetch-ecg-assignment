# ECG Arrhythmia Classifier

A deep learning model to classify ECG recordings into AF, VTACH, or PAUSE.

## Results

| Metric | Value |
|--------|-------|
| Accuracy | 96.4% |
| Macro F1 | 96.5% |
| AF Sensitivity | 100% |
| VTACH Sensitivity | 100% |
| PAUSE Sensitivity | 90% |

## Evaluation Metrics

In medical classification, not all errors are equal. Missing a life-threatening arrhythmia (false negative) is far worse than a false alarm (false positive). This is why I prioritize **sensitivity (recall)** over precision.

**Sensitivity** = detected cases / all actual cases

For VTACH specifically, missing even one case could be fatal, so 100% sensitivity was the goal. The model achieves this for both AF and VTACH, with PAUSE at 90%.

I also track **F1-score** which balances precision and recall. The macro F1 of 96.5% means the model performs consistently across all three classes, not just the majority class.

**Confusion Matrix:**
```
              Predicted
              AF    VTACH   PAUSE
Actual AF      9       0       0
      VTACH    0       9       0
      PAUSE    1       0       9
```

The only errors are 1 PAUSE misclassified as AF. This makes sense clinically - a pause surrounded by irregular beats can look like AF. I'd rather flag it for review than miss it entirely.

**Why not just accuracy?**

Accuracy alone can be misleading with balanced classes. A model predicting everything as one class would still get 33% accuracy. F1-score and per-class sensitivity give a better picture of real-world performance.

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

## Model Selection

I compared a few approaches before settling on 1D ResNet + Attention:

| Approach | Accuracy | Notes |
|----------|----------|-------|
| Rule-based (R-R intervals) | ~80% | Interpretable but misses edge cases |
| Simple CNN | ~88% | Better but struggles with long sequences |
| 1D ResNet + Attention | 96.4% | Best balance of accuracy and efficiency |

The attention mechanism was the key improvement. Without it, the model struggled to find the relevant arrhythmia event in a 90-second recording (18,000 samples). Attention lets it learn which parts of the signal matter most.

I didn't try transformers or LSTMs because:
- Transformers need more data than I have (138 samples)
- LSTMs are slow on long sequences and harder to train

The ResNet backbone is well-understood and the attention layer adds just enough complexity to capture what matters.
