import numpy as np
from scipy import signal
from scipy.stats import entropy
from typing import Tuple, Optional


class ECGClassifier:
    """
    Simple ECG arrhythmia classifier based on signal processing features.

    This classifier uses basic features like:
    - Heart rate variability (HRV)
    - Signal irregularity
    - Amplitude variations
    - Frequency domain features

    Event types:
    - AF (Atrial Fibrillation): Irregular rhythm, variable RR intervals
    - VTACH (Ventricular Tachycardia): Fast, regular rhythm
    - PAUSE: Significant pause in heartbeat
    """

    def __init__(self, sampling_rate: int = 200):
        self.sampling_rate = sampling_rate

    def extract_features(self, signal_data: np.ndarray, window_size: int = 1000) -> dict:
        """
        Extract features from ECG signal.

        Args:
            signal_data: ECG signal array
            window_size: Size of window for feature extraction (samples)

        Returns:
            Dictionary of features
        """
        features = {}

        # Basic statistics
        features['mean'] = np.mean(signal_data)
        features['std'] = np.std(signal_data)
        features['variance'] = np.var(signal_data)

        # Detect peaks (R-peaks approximation)
        peaks, _ = signal.find_peaks(signal_data, distance=self.sampling_rate//4, prominence=100)

        if len(peaks) > 1:
            # RR intervals (time between peaks)
            rr_intervals = np.diff(peaks)

            # Heart rate variability features
            features['mean_rr'] = np.mean(rr_intervals)
            features['std_rr'] = np.std(rr_intervals)
            features['rmssd'] = np.sqrt(np.mean(np.diff(rr_intervals)**2))  # Root mean square of successive differences
            features['cv_rr'] = features['std_rr'] / features['mean_rr'] if features['mean_rr'] > 0 else 0

            # Heart rate
            features['heart_rate'] = 60 * self.sampling_rate / features['mean_rr'] if features['mean_rr'] > 0 else 0

            # Irregularity measure
            features['irregularity'] = entropy(rr_intervals / np.sum(rr_intervals))
        else:
            features['mean_rr'] = 0
            features['std_rr'] = 0
            features['rmssd'] = 0
            features['cv_rr'] = 0
            features['heart_rate'] = 0
            features['irregularity'] = 0

        # Frequency domain features
        freqs, psd = signal.welch(signal_data, fs=self.sampling_rate, nperseg=min(256, len(signal_data)))
        features['dominant_freq'] = freqs[np.argmax(psd)]
        features['spectral_entropy'] = entropy(psd / np.sum(psd))

        return features

    def classify_and_detect(self, ch1_data: np.ndarray, ch2_data: np.ndarray,
                           hint_index: Optional[int] = None) -> Tuple[str, float, int, dict]:
        """
        Classify the arrhythmia type and detect event start time.

        Args:
            ch1_data: Channel 1 ECG data
            ch2_data: Channel 2 ECG data
            hint_index: Approximate index where event occurs (from metadata)

        Returns:
            Tuple of (event_type, confidence, event_start_index, explanation)
        """
        # Use channel 1 for analysis (typically lead I)
        signal_data = ch1_data

        # If we have a hint index, focus analysis around that region
        if hint_index is not None:
            # Analyze a window around the hint
            window_size = 2000  # 10 seconds at 200 Hz
            start_idx = max(0, hint_index - window_size)
            end_idx = min(len(signal_data), hint_index + window_size)
            analysis_window = signal_data[start_idx:end_idx]
            offset = start_idx
        else:
            # Analyze the middle section
            analysis_window = signal_data
            offset = 0

        # Extract features
        features = self.extract_features(analysis_window)

        # Simple rule-based classification with explanations
        event_type = "NORMAL"
        confidence = 0.0
        reasoning = []

        # AF Detection: High irregularity, variable RR intervals
        if features['cv_rr'] > 0.15 and features['irregularity'] > 1.5:
            event_type = "AF"
            confidence = min(0.95, 0.6 + features['cv_rr'] * 2)
            reasoning = [
                f"Heart rate variability (CV) is {features['cv_rr']:.2f} (threshold: 0.15)",
                f"Rhythm irregularity is {features['irregularity']:.2f} (threshold: 1.5)",
                f"Heart rate: {features['heart_rate']:.1f} bpm",
                "Pattern indicates irregular, chaotic rhythm typical of atrial fibrillation"
            ]

        # VTACH Detection: High heart rate, relatively regular
        elif features['heart_rate'] > 120 and features['cv_rr'] < 0.1:
            event_type = "VTACH"
            confidence = min(0.95, 0.7 + (features['heart_rate'] - 120) / 200)
            reasoning = [
                f"Heart rate is {features['heart_rate']:.1f} bpm (threshold: 120)",
                f"Heart rate variability (CV) is {features['cv_rr']:.2f} (threshold: <0.1)",
                f"Rhythm irregularity is {features['irregularity']:.2f}",
                "Pattern indicates fast, regular rhythm typical of ventricular tachycardia"
            ]

        # PAUSE Detection: Very low heart rate or long RR intervals
        elif features['mean_rr'] > 400 or features['heart_rate'] < 40:
            event_type = "PAUSE"
            confidence = min(0.95, 0.7 + (400 - features['heart_rate']) / 400)
            reasoning = [
                f"Mean RR interval is {features['mean_rr']:.1f} samples (threshold: 400)",
                f"Heart rate: {features['heart_rate']:.1f} bpm (threshold: <40)",
                f"Average time between beats: {features['mean_rr']/self.sampling_rate:.2f} seconds",
                "Pattern indicates significant pause or very slow heart rate"
            ]
        else:
            reasoning = [
                f"Heart rate: {features['heart_rate']:.1f} bpm (normal range)",
                f"Heart rate variability (CV): {features['cv_rr']:.2f} (normal)",
                f"Rhythm irregularity: {features['irregularity']:.2f} (normal)",
                "No significant arrhythmia detected in this window"
            ]

        # Detect event start using change point detection
        event_start_index = self.detect_event_start(analysis_window, hint_index - offset if hint_index else None)
        event_start_index += offset

        # Build explanation dictionary
        explanation = {
            "event_type": event_type,
            "confidence": float(confidence),
            "reasoning": reasoning,
            "features": {
                "heart_rate_bpm": float(features['heart_rate']),
                "heart_rate_variability": float(features['cv_rr']),
                "irregularity_score": float(features['irregularity']),
                "mean_rr_interval": float(features['mean_rr']),
                "rmssd": float(features['rmssd'])
            },
            "method": "Rule-based signal processing with HRV analysis"
        }

        return event_type, confidence, event_start_index, explanation

    def detect_event_start(self, signal_data: np.ndarray, hint_index: Optional[int] = None) -> int:
        """
        Detect the start of an arrhythmia event using change point detection.

        Args:
            signal_data: ECG signal
            hint_index: Approximate location of event (relative to signal_data)

        Returns:
            Index where event starts
        """
        # Simple approach: look for significant change in RR interval variability
        window_size = 200  # 1 second

        if hint_index is None:
            hint_index = len(signal_data) // 2

        # Calculate rolling standard deviation
        rolling_std = []
        for i in range(window_size, len(signal_data) - window_size):
            window = signal_data[i-window_size:i+window_size]
            rolling_std.append(np.std(window))

        rolling_std = np.array(rolling_std)

        # Find the point of maximum change near the hint
        search_start = max(0, hint_index - 1000)
        search_end = min(len(rolling_std), hint_index + 1000)

        if search_end > search_start:
            search_region = rolling_std[search_start:search_end]
            # Find maximum gradient (change point)
            gradient = np.abs(np.gradient(search_region))
            local_max_idx = np.argmax(gradient)
            event_start = search_start + local_max_idx + window_size
        else:
            event_start = hint_index if hint_index else len(signal_data) // 2

        return event_start

