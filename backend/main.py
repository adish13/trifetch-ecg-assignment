from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import os
import json
import numpy as np
from pathlib import Path

from data_loader import load_ecg_data, load_event_metadata
from classifier import ECGClassifier

app = FastAPI(title="ECG Arrhythmia Detection API")

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize classifier
classifier = ECGClassifier()

# Data directory
DATA_DIR = Path(__file__).parent.parent / "data"


class Episode(BaseModel):
    id: str
    patient_id: str
    event_time: str
    event_name: str
    is_rejected: bool
    predicted_event: Optional[str] = None
    confidence: Optional[float] = None
    event_start_index: Optional[int] = None


class ECGData(BaseModel):
    channel1: List[float]
    channel2: List[float]
    sampling_rate: int
    event_index: int
    predicted_event_index: Optional[int] = None


@app.get("/")
def read_root():
    return {"message": "ECG Arrhythmia Detection API", "status": "running"}


@app.get("/episodes", response_model=List[Episode])
def get_episodes():
    """Get all available episodes"""
    episodes = []

    for category_dir in DATA_DIR.iterdir():
        if not category_dir.is_dir():
            continue

        for event_dir in category_dir.iterdir():
            if not event_dir.is_dir():
                continue

            # Find the event JSON file
            json_files = list(event_dir.glob("event_*.json"))
            if not json_files:
                continue

            metadata = load_event_metadata(json_files[0])
            if metadata:
                episodes.append(Episode(
                    id=event_dir.name,
                    patient_id=metadata["Patient_IR_ID"],
                    event_time=metadata["EventOccuredTime"],
                    event_name=metadata["Event_Name"],
                    is_rejected=metadata["IsRejected"] == "1"
                ))

    return episodes


@app.get("/episodes/{episode_id}/ecg", response_model=ECGData)
def get_ecg_data(episode_id: str):
    """Get ECG data for a specific episode"""

    # Find the episode directory
    episode_dir = None
    for category_dir in DATA_DIR.iterdir():
        if not category_dir.is_dir():
            continue
        potential_dir = category_dir / episode_id
        if potential_dir.exists():
            episode_dir = potential_dir
            break

    if not episode_dir:
        raise HTTPException(status_code=404, detail=f"Episode {episode_id} not found")

    # Load metadata
    json_files = list(episode_dir.glob("event_*.json"))
    if not json_files:
        raise HTTPException(status_code=404, detail="Event metadata not found")

    metadata = load_event_metadata(json_files[0])

    # Load ECG data
    ecg_files = sorted(episode_dir.glob("ECGData_*.txt"))
    if len(ecg_files) != 3:
        raise HTTPException(status_code=404, detail="ECG data files not found")

    ch1_data, ch2_data = load_ecg_data(ecg_files)

    # Get event index from metadata
    event_index = metadata.get("EventIndex", 9000)  # Default to middle if not found

    # Classify and detect event
    predicted_event, confidence, predicted_index, explanation = classifier.classify_and_detect(
        ch1_data, ch2_data, event_index
    )

    return ECGData(
        channel1=ch1_data.tolist(),
        channel2=ch2_data.tolist(),
        sampling_rate=200,
        event_index=event_index,
        predicted_event_index=predicted_index
    )


@app.get("/episodes/{episode_id}/classify")
def classify_episode(episode_id: str):
    """Classify an episode and detect event timing"""

    # Find the episode directory
    episode_dir = None
    for category_dir in DATA_DIR.iterdir():
        if not category_dir.is_dir():
            continue
        potential_dir = category_dir / episode_id
        if potential_dir.exists():
            episode_dir = potential_dir
            break

    if not episode_dir:
        raise HTTPException(status_code=404, detail=f"Episode {episode_id} not found")

    # Load ECG data
    ecg_files = sorted(episode_dir.glob("ECGData_*.txt"))
    ch1_data, ch2_data = load_ecg_data(ecg_files)

    # Load metadata for ground truth
    json_files = list(episode_dir.glob("event_*.json"))
    metadata = load_event_metadata(json_files[0])
    event_index = metadata.get("EventIndex", 9000)

    # Classify
    predicted_event, confidence, predicted_index, explanation = classifier.classify_and_detect(
        ch1_data, ch2_data, event_index
    )

    return {
        "episode_id": episode_id,
        "predicted_event": predicted_event,
        "confidence": float(confidence),
        "predicted_event_index": int(predicted_index),
        "actual_event": metadata["Event_Name"],
        "actual_event_index": int(event_index),
        "explanation": explanation
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

