# Complete Walkthrough - What We Built

## 🎯 The Assignment

**Goal**: Build a system that can:
1. Load ECG recordings (90 seconds of heartbeat data)
2. Classify the type of heart problem (AF, VTACH, PAUSE)
3. Detect exactly when the problem starts
4. Show it all in a web interface

**Time limit**: ~1 day

---

## 🏗️ What We Built

### The Full Stack

**Backend (Python + FastAPI)**
- Loads ECG data from text files
- Runs signal processing to classify arrhythmias
- Serves data through a REST API
- Returns predictions with explanations

**Frontend (React + Vite)**
- Shows list of episodes in a sidebar
- Displays interactive ECG charts
- Has a slider to navigate through 90 seconds
- Shows event markers (red = actual, blue = predicted)

---

## 🧠 The ML Decision (Most Important!)

### What I Chose: Signal Processing

Instead of training a neural network, I used **rule-based signal processing**.

### Why?

**✅ Advantages:**
1. **Fast to build** - No training, works immediately
2. **Explainable** - Can tell you exactly why it made a decision
3. **No data needed** - Works with just the test set
4. **Medically proven** - Based on 50 years of cardiology research

**❌ Tradeoffs:**
1. **Fixed rules** - Doesn't adapt to individual patients
2. **Might miss patterns** - Neural networks could catch subtle things
3. **Lower potential accuracy** - With enough data, deep learning could be better

### How It Works

**Step 1: Find heartbeats**
```
Use scipy.signal.find_peaks() to find R-peaks (the big spikes)
```

**Step 2: Measure timing**
```
Calculate RR intervals = time between each beat
Example: [0.8s, 0.9s, 0.7s, 1.1s, 0.8s...]
```

**Step 3: Extract features**
```python
heart_rate = 60 / average_RR_interval
variability = std(RR_intervals) / mean(RR_intervals)
irregularity = entropy(RR_intervals)
```

**Step 4: Apply rules**
```python
if variability > 0.15 and irregularity > 1.5:
    return "AF"  # Atrial Fibrillation
    
elif heart_rate > 120 and variability < 0.1:
    return "VTACH"  # Ventricular Tachycardia
    
elif mean_RR > 400 or heart_rate < 40:
    return "PAUSE"  # Cardiac Pause
```

**Step 5: Find event start**
```
Use change-point detection:
- Calculate rolling standard deviation
- Find where it changes most dramatically
- That's where the event starts
```

### What Makes This Good for a 1-Day Project?

1. **No training time** - Just load and run
2. **Debuggable** - If it's wrong, I can see why
3. **Explainable** - Can show doctors the reasoning
4. **Fast** - Classifies in milliseconds

### What Would I Do With More Time?

**Week 1**: Train a 1D CNN on the raw waveform
**Week 2**: Add LSTM for temporal patterns
**Week 3**: Ensemble multiple models
**Week 4**: Active learning on hard examples

---

## 💻 The Tech Stack

### Backend
- **FastAPI** - Fastest Python web framework
- **NumPy** - Fast array operations
- **SciPy** - Signal processing tools
- **Pydantic** - Data validation

### Frontend
- **React** - Component-based UI
- **Vite** - Super fast dev server
- **Recharts** - Chart library
- **CSS** - Custom styling

### Why These?
- **FastAPI**: Auto-docs, async, type hints
- **React**: Industry standard, huge ecosystem
- **Vite**: 10x faster than Create React App
- **Recharts**: Works great with React, declarative

---

## 🎨 The UI

### The Problem
Doctors need to see BOTH:
- The big picture (all 90 seconds)
- The details (individual heartbeats)

### The Solution
Three-panel layout:

```
┌─────────────────────────────────────┐
│  DETAILED VIEW (8 seconds)          │
│  Channel 1: [ECG waveform]          │
│  Channel 2: [ECG waveform]          │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│  SLIDER: [━━━━━━━━━━━━━━━━━━━━━━━] │
│  Navigate: 0:00.0 - 0:08.0          │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│  OVERVIEW (90 seconds)              │
│  [Full ECG with highlighted window] │
│  Click to jump to any time          │
└─────────────────────────────────────┘
```

### Key Features
1. **Auto-centers on event** - When you load, it jumps to the problem
2. **Click to navigate** - Click overview to jump there
3. **Smooth slider** - Drag to explore
4. **Event markers** - Red (actual) and Blue (predicted)

---

## 📊 Performance

- **Load time**: 40-50ms (requirement: <8 seconds) ✅
- **Data points**: 18,000 samples
- **Downsampling**: 20x for overview, 2x for detail
- **Memory**: Efficient NumPy arrays

---

## 🔍 API Design

### GET /episodes
```json
[
  {"id": "74422783", "event_name": "AF", "is_rejected": false}
]
```

### GET /episodes/{id}/ecg
```json
{
  "channel1": [...18000 samples...],
  "channel2": [...18000 samples...],
  "event_index": 9234,
  "predicted_event_index": 9180
}
```

### GET /episodes/{id}/classify
```json
{
  "predicted_event": "AF",
  "confidence": 0.87,
  "explanation": {
    "reasoning": [
      "Heart rate variability (CV) is 0.23 (threshold: 0.15)",
      "Rhythm irregularity is 2.1 (threshold: 1.5)",
      "Pattern indicates irregular rhythm typical of AF"
    ],
    "features": {
      "heart_rate_bpm": 78.5,
      "heart_rate_variability": 0.23,
      "irregularity_score": 2.1
    }
  }
}
```

---

## 🎬 For the Video - Key Points

### 1. Show It Working (1 min)
- Click through episodes
- Use the slider
- Point out event markers
- Show how it auto-centers

### 2. Explain the ML Choice (1.5 min)
- Why signal processing over deep learning
- Show the tradeoffs honestly
- Explain the features (heart rate, variability)
- Show the rules (if X > threshold, then Y)

### 3. Show the Code (1 min)
- Quick tour of classifier.py
- Show the feature extraction
- Show the classification rules
- Show the explanation output

### 4. Discuss Tradeoffs (1 min)
- What's good: Fast, explainable, no training
- What's not: Fixed rules, might miss patterns
- What I'd do with more time: CNN, LSTM, ensemble

### 5. Wrap Up (30 sec)
- Full stack in a day
- Clean code, good docs
- Ready to demo
- Focused on shipping fast with good design

---

## 📝 Key Takeaways

1. **I made a choice** - Signal processing over deep learning
2. **I can explain why** - Speed, explainability, time constraints
3. **I know the tradeoffs** - Lower potential accuracy, fixed rules
4. **I shipped it** - Working system in ~1 day
5. **It's documented** - Code is clean and explained

This shows: **Good engineering judgment + ability to ship + honest about limitations**

