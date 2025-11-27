"""
ECG Data Preprocessing Module
Loads, preprocesses, and augments ECG data for training.
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Constants
SAMPLING_RATE = 200  # Hz
SAMPLES_PER_FILE = 6000  # 30 seconds * 200 Hz
TOTAL_SAMPLES = 18000  # 90 seconds * 200 Hz

# Class mapping
CLASS_MAP = {'AF': 0, 'VTACH': 1, 'PAUSE': 2}
CLASS_NAMES = ['AF', 'VTACH', 'PAUSE']


def load_ecg_file(file_path: str) -> np.ndarray:
    """Load a single ECG file and return as numpy array (samples, 2)."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and ',' in line:
                try:
                    ch1, ch2 = line.split(',')
                    data.append([float(ch1), float(ch2)])
                except ValueError:
                    continue
    return np.array(data)


def load_episode(episode_dir: str) -> Tuple[np.ndarray, Dict]:
    """
    Load all ECG files for an episode and concatenate them.
    Returns: (ecg_data, metadata)
    """
    episode_path = Path(episode_dir)

    # Load metadata
    json_files = list(episode_path.glob('*.json'))
    if not json_files:
        raise ValueError(f"No JSON file found in {episode_dir}")

    with open(json_files[0], 'r') as f:
        metadata = json.load(f)

    # Load ECG files (sorted by filename to ensure correct order)
    txt_files = sorted(episode_path.glob('*.txt'))
    if len(txt_files) != 3:
        raise ValueError(f"Expected 3 ECG files, found {len(txt_files)} in {episode_dir}")

    # Concatenate all ECG data
    ecg_segments = [load_ecg_file(str(f)) for f in txt_files]
    ecg_data = np.vstack(ecg_segments)

    return ecg_data, metadata


def load_all_data(data_dir: str = 'data') -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """
    Load all episodes from the data directory.
    Returns: (X, y, metadata_list)
    - X: (n_samples, 18000, 2) - ECG signals
    - y: (n_samples,) - Class labels (0=AF, 1=VTACH, 2=PAUSE)
    """
    data_path = Path(data_dir)
    X_list, y_list, metadata_list = [], [], []

    # Iterate through all category folders
    for category_dir in data_path.iterdir():
        if not category_dir.is_dir():
            continue

        # Extract class from folder name (e.g., "AF_Approved" -> "AF")
        folder_name = category_dir.name
        class_name = folder_name.split('_')[0]

        if class_name not in CLASS_MAP:
            continue

        class_label = CLASS_MAP[class_name]

        # Load each episode in this category
        for episode_dir in category_dir.iterdir():
            if not episode_dir.is_dir():
                continue

            try:
                ecg_data, metadata = load_episode(str(episode_dir))

                # Validate shape
                if ecg_data.shape[0] != TOTAL_SAMPLES:
                    print(f"Warning: {episode_dir} has {ecg_data.shape[0]} samples, expected {TOTAL_SAMPLES}")
                    # Pad or truncate to exact size
                    if ecg_data.shape[0] < TOTAL_SAMPLES:
                        pad_size = TOTAL_SAMPLES - ecg_data.shape[0]
                        ecg_data = np.pad(ecg_data, ((0, pad_size), (0, 0)), mode='edge')
                    else:
                        ecg_data = ecg_data[:TOTAL_SAMPLES]

                X_list.append(ecg_data)
                y_list.append(class_label)
                metadata['class'] = class_name
                metadata['folder'] = folder_name
                metadata_list.append(metadata)

            except Exception as e:
                print(f"Error loading {episode_dir}: {e}")
                continue

    X = np.array(X_list)
    y = np.array(y_list)

    print(f"Loaded {len(X)} episodes")
    print(f"Class distribution: {dict(zip(CLASS_NAMES, [np.sum(y == i) for i in range(3)]))}")

    return X, y, metadata_list


def normalize_data(X: np.ndarray, scaler: Optional[StandardScaler] = None) -> Tuple[np.ndarray, StandardScaler]:
    """
    Normalize ECG data using StandardScaler.
    Each channel is normalized independently.
    """
    n_samples, n_timesteps, n_channels = X.shape
    X_reshaped = X.reshape(-1, n_channels)

    if scaler is None:
        scaler = StandardScaler()
        X_normalized = scaler.fit_transform(X_reshaped)
    else:
        X_normalized = scaler.transform(X_reshaped)

    return X_normalized.reshape(n_samples, n_timesteps, n_channels), scaler


# ============================================================================
# DATA AUGMENTATION
# ============================================================================

def add_noise(X: np.ndarray, noise_factor: float = 0.05) -> np.ndarray:
    """Add Gaussian noise to the signal."""
    noise = np.random.normal(0, noise_factor, X.shape)
    return X + noise


def scale_amplitude(X: np.ndarray, scale_range: Tuple[float, float] = (0.8, 1.2)) -> np.ndarray:
    """Randomly scale the amplitude of the signal."""
    scale = np.random.uniform(scale_range[0], scale_range[1])
    return X * scale


def time_shift(X: np.ndarray, max_shift: int = 400) -> np.ndarray:
    """Shift the signal in time (circular shift)."""
    shift = np.random.randint(-max_shift, max_shift)
    return np.roll(X, shift, axis=0)


def add_baseline_wander(X: np.ndarray, amplitude: float = 0.1, freq: float = 0.5) -> np.ndarray:
    """Add low-frequency baseline wander."""
    t = np.linspace(0, X.shape[0] / SAMPLING_RATE, X.shape[0])
    wander = amplitude * np.sin(2 * np.pi * freq * t)
    return X + wander.reshape(-1, 1)


def augment_sample(X: np.ndarray, augment_prob: float = 0.5) -> np.ndarray:
    """Apply random augmentations to a single sample."""
    X_aug = X.copy()

    if np.random.random() < augment_prob:
        X_aug = add_noise(X_aug, noise_factor=np.random.uniform(0.01, 0.05))

    if np.random.random() < augment_prob:
        X_aug = scale_amplitude(X_aug, scale_range=(0.85, 1.15))

    if np.random.random() < augment_prob:
        X_aug = time_shift(X_aug, max_shift=200)

    if np.random.random() < augment_prob:
        X_aug = add_baseline_wander(X_aug, amplitude=0.05, freq=np.random.uniform(0.1, 0.5))

    return X_aug


def augment_dataset(X: np.ndarray, y: np.ndarray, augment_factor: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Augment the entire dataset by creating augmented copies.
    augment_factor: How many augmented copies to create per original sample.
    """
    X_aug_list = [X]  # Start with original data
    y_aug_list = [y]

    for _ in range(augment_factor):
        X_augmented = np.array([augment_sample(x) for x in X])
        X_aug_list.append(X_augmented)
        y_aug_list.append(y)

    return np.vstack(X_aug_list), np.hstack(y_aug_list)


# ============================================================================
# DATA SPLITTING
# ============================================================================

def create_train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    val_size: float = 0.15,
    random_state: int = 42
) -> Dict[str, np.ndarray]:
    """
    Create stratified train/val/test splits.
    Returns dictionary with X_train, X_val, X_test, y_train, y_val, y_test.
    """
    # First split: train+val vs test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Second split: train vs val
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_ratio, stratify=y_trainval, random_state=random_state
    )

    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test
    }


def get_stratified_kfold(n_splits: int = 5, random_state: int = 42) -> StratifiedKFold:
    """Get a stratified k-fold splitter."""
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


# ============================================================================
# MAIN DATA PIPELINE
# ============================================================================

def prepare_data(
    data_dir: str = 'data',
    augment: bool = True,
    augment_factor: int = 3,
    test_size: float = 0.2,
    val_size: float = 0.15
) -> Dict:
    """
    Complete data preparation pipeline.
    Returns dictionary with all data splits and metadata.
    """
    # Load raw data
    print("Loading data...")
    X, y, metadata = load_all_data(data_dir)

    # Create splits BEFORE augmentation (to prevent data leakage!)
    print("Creating train/val/test splits...")
    splits = create_train_val_test_split(X, y, test_size=test_size, val_size=val_size)

    # Normalize data (fit on train only)
    print("Normalizing data...")
    X_train_norm, scaler = normalize_data(splits['X_train'])
    X_val_norm, _ = normalize_data(splits['X_val'], scaler)
    X_test_norm, _ = normalize_data(splits['X_test'], scaler)

    # Augment training data only
    if augment:
        print(f"Augmenting training data (factor={augment_factor})...")
        X_train_aug, y_train_aug = augment_dataset(X_train_norm, splits['y_train'], augment_factor)
    else:
        X_train_aug, y_train_aug = X_train_norm, splits['y_train']

    print(f"Final shapes:")
    print(f"  Train: {X_train_aug.shape}, {y_train_aug.shape}")
    print(f"  Val:   {X_val_norm.shape}, {splits['y_val'].shape}")
    print(f"  Test:  {X_test_norm.shape}, {splits['y_test'].shape}")

    return {
        'X_train': X_train_aug, 'y_train': y_train_aug,
        'X_val': X_val_norm, 'y_val': splits['y_val'],
        'X_test': X_test_norm, 'y_test': splits['y_test'],
        'scaler': scaler,
        'metadata': metadata,
        'class_names': CLASS_NAMES
    }

