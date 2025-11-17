import json
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List


def load_event_metadata(json_path: Path) -> Dict:
    """Load event metadata from JSON file"""
    with open(json_path, 'r') as f:
        return json.load(f)


def load_ecg_data(ecg_files: List[Path]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load ECG data from multiple files and concatenate them.
    
    Args:
        ecg_files: List of paths to ECG data files (should be 3 files, 30 seconds each)
    
    Returns:
        Tuple of (channel1_data, channel2_data) as numpy arrays
    """
    ch1_all = []
    ch2_all = []
    
    for ecg_file in sorted(ecg_files):
        with open(ecg_file, 'r') as f:
            lines = f.readlines()
            
        # Skip header if present
        start_idx = 1 if 'ch1' in lines[0].lower() else 0
        
        for line in lines[start_idx:]:
            line = line.strip()
            if not line:
                continue
            try:
                ch1, ch2 = line.split(',')
                ch1_all.append(float(ch1))
                ch2_all.append(float(ch2))
            except ValueError:
                continue
    
    return np.array(ch1_all), np.array(ch2_all)


def parse_event_time(event_time_str: str) -> Dict:
    """
    Parse event time string to extract seconds and milliseconds.
    
    Example: "2025-11-08 14:09:19.884" -> {"seconds": 19, "milliseconds": 884}
    """
    time_part = event_time_str.split(' ')[1]  # Get "14:09:19.884"
    time_components = time_part.split(':')
    seconds_with_ms = time_components[2]  # "19.884"
    
    seconds, milliseconds = seconds_with_ms.split('.')
    
    return {
        "seconds": int(seconds),
        "milliseconds": int(milliseconds)
    }

