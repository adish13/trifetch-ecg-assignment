# ECG Arrhythmia Detection System

A full-stack application for detecting and visualizing cardiac arrhythmias from ECG data, built for the TriFetch Take-Home Assignment.

## 🎯 Features

- **Real-time ECG Visualization**: Interactive charts displaying dual-channel ECG traces
- **Arrhythmia Classification**: Automatic detection of AF (Atrial Fibrillation), VTACH (Ventricular Tachycardia), and PAUSE events
- **Event Detection**: Precise identification of event start times within ECG recordings
- **Episode Management**: Browse and select from multiple ECG episodes
- **Fast Performance**: Episode loading optimized to stay under 8 seconds

## 🏗️ Architecture

### Backend (Python/FastAPI)
- **Framework**: FastAPI for high-performance async API
- **ML Pipeline**: Signal processing-based classifier using scipy
- **Data Processing**: NumPy for efficient ECG data handling
- **Features Extracted**:
  - Heart Rate Variability (HRV) metrics
  - RR interval statistics
  - Frequency domain analysis
  - Signal irregularity measures

### Frontend (React/Vite)
- **Framework**: React 18 with Vite for fast development
- **Charting**: Recharts for responsive ECG visualization
- **Styling**: Custom CSS with modern design
- **State Management**: React hooks for local state

## 📋 Prerequisites

- **Python**: 3.11+ (for backend)
- **Node.js**: 22.9+ (for frontend)
- **npm**: 10.8+ (comes with Node.js)

## 🚀 Setup Instructions

### 1. Clone and Navigate
```bash
cd /path/to/trifetch
```

### 2. Backend Setup
```bash
cd backend

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On macOS/Linux
# OR
venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt
```

### 3. Frontend Setup
```bash
cd frontend

# Install dependencies
npm install
```

### 4. Data Setup
Ensure the `data/` directory contains the extracted ECG dataset with the following structure:
```
data/
├── AF_Approved/
├── AF_Rejected/
├── PAUSE_Approved/
├── PAUSE_Rejected/
├── VTACH_Approved/
└── VTACH_Rejected/
```

## ▶️ Running the Application

### Start Backend Server
```bash
cd backend
source venv/bin/activate
python main.py
```
Backend will run on `http://localhost:8000`

### Start Frontend Development Server
```bash
cd frontend
npm run dev
```
Frontend will run on `http://localhost:5173`

### Access the Application
Open your browser and navigate to: `http://localhost:5173`

## 🧠 Technical Choices & Reasoning

### ML Model Approach
**Choice**: Rule-based signal processing classifier instead of deep learning

**Reasoning**:
1. **Speed**: No model training required, instant predictions
2. **Interpretability**: Clear feature-based decisions
3. **Data Efficiency**: Works without large labeled datasets
4. **Reliability**: Deterministic behavior based on established cardiac metrics

**Features Used**:
- **HRV Metrics**: Coefficient of variation of RR intervals distinguishes AF (high variability)
- **Heart Rate**: Identifies VTACH (>120 bpm) and PAUSE (<40 bpm)
- **Spectral Analysis**: Frequency domain features for rhythm characterization
- **Change Point Detection**: Gradient-based detection of event onset

### Performance Optimizations
1. **Data Downsampling**: Display every 2nd point (still 100 samples/sec) for smooth rendering
2. **Lazy Loading**: Episodes loaded on-demand
3. **Efficient Parsing**: Direct NumPy array operations
4. **Caching**: Browser caches static assets

### API Design
- RESTful endpoints for clear separation of concerns
- Async/await for non-blocking I/O
- CORS enabled for local development
- Structured error handling

## 📊 API Endpoints

- `GET /` - Health check
- `GET /episodes` - List all available episodes
- `GET /episodes/{id}/ecg` - Get ECG data and classification for specific episode
- `GET /episodes/{id}/classify` - Get detailed classification results

## 🔮 Future Improvements

### Model Enhancements
1. **Machine Learning**: Train CNN or LSTM on labeled data for better accuracy
2. **Multi-lead Analysis**: Incorporate all 12 ECG leads
3. **Ensemble Methods**: Combine multiple classifiers
4. **Confidence Calibration**: Better uncertainty quantification

### Features
1. **Real-time Streaming**: Live ECG monitoring
2. **Export Functionality**: Download reports as PDF
3. **Annotation Tools**: Manual event marking and correction
4. **Comparison View**: Side-by-side episode comparison

### Performance
1. **Database Integration**: PostgreSQL for faster queries
2. **Caching Layer**: Redis for frequently accessed episodes
3. **WebSocket**: Real-time updates
4. **Progressive Loading**: Stream large ECG files

## 🛠️ Development

### Project Structure
```
trifetch/
├── backend/
│   ├── main.py              # FastAPI application
│   ├── classifier.py        # ML classification logic
│   ├── data_loader.py       # ECG data parsing
│   └── requirements.txt     # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── App.jsx          # Main application
│   │   ├── components/      # React components
│   │   └── App.css          # Styles
│   └── package.json         # Node dependencies
└── data/                    # ECG dataset
```

### Testing
```bash
# Backend
cd backend
pytest  # (tests to be added)

# Frontend
cd frontend
npm test  # (tests to be added)
```


