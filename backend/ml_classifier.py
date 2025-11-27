"""
ML-based ECG Arrhythmia Classifier using trained PyTorch model.

This classifier uses a 1D ResNet + Attention model trained on the ECG dataset.
"""

import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, Optional

# Import the model architecture
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.model import ECGResNetAttention


CLASS_NAMES = ['AF', 'VTACH', 'PAUSE']


class MLECGClassifier:
    """
    ML-based ECG arrhythmia classifier using trained PyTorch model.
    
    Uses a 1D ResNet + Attention architecture trained on the ECG dataset.
    Achieves 96.43% accuracy with 100% sensitivity for AF and VTACH.
    """
    
    def __init__(self, model_path: str = None, scaler_path: str = None, sampling_rate: int = 200):
        self.sampling_rate = sampling_rate
        self.device = torch.device('cuda' if torch.cuda.is_available() else 
                                   'mps' if torch.backends.mps.is_available() else 'cpu')
        
        # Default paths
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), '..', 'ml', 'models', 'final_model.pt')
        if scaler_path is None:
            scaler_path = os.path.join(os.path.dirname(__file__), '..', 'ml', 'models', 'scaler.pkl')
        
        # Load model
        self.model = ECGResNetAttention(
            input_channels=2,
            num_classes=3,
            filters=[64, 128, 256],
            num_res_blocks=2,
            num_attention_heads=4,
            dropout_rate=0.5
        )
        
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
            self.model.to(self.device)
            self.model.eval()
            self.model_loaded = True
            print(f"Loaded ML model from {model_path}")
        else:
            self.model_loaded = False
            print(f"Warning: Model not found at {model_path}. Using fallback classifier.")
        
        # Load scaler
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print(f"Loaded scaler from {scaler_path}")
        else:
            self.scaler = None
            print(f"Warning: Scaler not found at {scaler_path}")
    
    def preprocess(self, ch1_data: np.ndarray, ch2_data: np.ndarray) -> np.ndarray:
        """Preprocess ECG data for model input."""
        # Stack channels
        data = np.stack([ch1_data, ch2_data], axis=-1)  # (18000, 2)
        
        # Normalize using scaler if available
        if self.scaler is not None:
            original_shape = data.shape
            data_flat = data.reshape(-1, 2)
            data_flat = self.scaler.transform(data_flat)
            data = data_flat.reshape(original_shape)
        else:
            # Simple normalization
            data = (data - np.mean(data, axis=0)) / (np.std(data, axis=0) + 1e-8)
        
        return data
    
    def classify_and_detect(self, ch1_data: np.ndarray, ch2_data: np.ndarray,
                           hint_index: Optional[int] = None) -> Tuple[str, float, int, dict]:
        """
        Classify the arrhythmia type using ML model.
        
        Returns:
            Tuple of (event_type, confidence, event_start_index, explanation)
        """
        if not self.model_loaded:
            # Fallback to rule-based classifier
            from backend.classifier import ECGClassifier
            fallback = ECGClassifier(self.sampling_rate)
            return fallback.classify_and_detect(ch1_data, ch2_data, hint_index)
        
        # Preprocess
        data = self.preprocess(ch1_data, ch2_data)
        
        # Convert to tensor
        data_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(data_tensor)
            probabilities = F.softmax(outputs, dim=1).cpu().numpy()[0]
            predicted_class = np.argmax(probabilities)
        
        event_type = CLASS_NAMES[predicted_class]
        confidence = float(probabilities[predicted_class])
        
        # Detect event start using signal processing
        event_start_index = self._detect_event_start(ch1_data, hint_index)
        
        # Build explanation
        explanation = {
            "event_type": event_type,
            "confidence": confidence,
            "reasoning": [
                f"ML model prediction: {event_type} with {confidence*100:.1f}% confidence",
                f"Class probabilities: AF={probabilities[0]*100:.1f}%, VTACH={probabilities[1]*100:.1f}%, PAUSE={probabilities[2]*100:.1f}%",
                "Model: 1D ResNet + Attention (96.43% test accuracy)",
                "Trained on 138 ECG episodes with data augmentation"
            ],
            "probabilities": {
                "AF": float(probabilities[0]),
                "VTACH": float(probabilities[1]),
                "PAUSE": float(probabilities[2])
            },
            "method": "Deep Learning (1D ResNet + Multi-Head Attention)"
        }
        
        return event_type, confidence, event_start_index, explanation
    
    def _detect_event_start(self, signal_data: np.ndarray, hint_index: Optional[int] = None) -> int:
        """Detect event start using change point detection."""
        from scipy import signal as sig
        
        window_size = 200  # 1 second
        if hint_index is None:
            hint_index = len(signal_data) // 2
        
        # Calculate rolling standard deviation
        rolling_std = []
        for i in range(window_size, len(signal_data) - window_size):
            window = signal_data[i-window_size:i+window_size]
            rolling_std.append(np.std(window))
        
        rolling_std = np.array(rolling_std)
        
        # Find change point near hint
        search_start = max(0, hint_index - 1000)
        search_end = min(len(rolling_std), hint_index + 1000)
        
        if search_end > search_start:
            search_region = rolling_std[search_start:search_end]
            gradient = np.abs(np.gradient(search_region))
            local_max_idx = np.argmax(gradient)
            event_start = search_start + local_max_idx + window_size
        else:
            event_start = hint_index if hint_index else len(signal_data) // 2
        
        return event_start

