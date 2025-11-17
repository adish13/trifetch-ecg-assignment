# How to See ECG Abnormalities in the Viewer

## 🎯 The Event Markers Show You Where to Look!

### What the Lines Mean:
- **🔴 Red Solid Line**: This is the **actual event time** from the medical metadata
  - This marks EXACTLY where the arrhythmia starts
  - This is the ground truth from medical professionals

- **🔵 Blue Dashed Line**: This is the **predicted event time** from our ML algorithm
  - Shows where our classifier thinks the event starts
  - Compare this to the red line to see how accurate our prediction is

## 📍 How to Navigate to See the Abnormality:

### Method 1: Use the Slider
1. Look at the **overview chart** at the bottom
2. Find the **red line** (Event marker)
3. **Drag the slider** until the blue shaded area covers the red line
4. The detailed view will now show the abnormal ECG waveform

### Method 2: Click the Overview
1. **Click directly on the red line** in the overview chart
2. The viewer will automatically jump to that time
3. The detailed view will center around the event

## 🔍 What to Look For in Each Event Type:

### AF (Atrial Fibrillation)
**Visual Signs:**
- **Irregular rhythm**: Heartbeats are NOT evenly spaced
- **No P waves**: The small bump before each QRS complex disappears
- **Chaotic baseline**: The baseline between beats looks "wavy" or irregular
- **Variable RR intervals**: Distance between R peaks varies significantly

**Example:**
```
Normal:  _/\_ _/\_ _/\_ _/\_ (regular spacing)
AF:      _/\_  _/\_ _/\_   _/\_ (irregular spacing, no pattern)
```

### VTACH (Ventricular Tachycardia)
**Visual Signs:**
- **Very fast rate**: Heartbeats come VERY quickly (>120 per minute)
- **Wide QRS**: The main spike is wider than normal
- **Regular but fast**: Unlike AF, the rhythm is regular but extremely fast
- **Looks "different"**: The waveform shape changes dramatically

**Example:**
```
Normal:  _/\_ _/\_ _/\_ (normal speed)
VTACH:   _/\_/\_/\_/\_/\_/\_ (very fast, wide peaks)
```

### PAUSE (Cardiac Pause)
**Visual Signs:**
- **Long flat line**: A section with NO heartbeats
- **Sudden gap**: The regular rhythm suddenly stops
- **Then resumes**: After the pause, the rhythm starts again
- **Very obvious**: This is the easiest to spot!

**Example:**
```
Normal:  _/\_ _/\_ _/\_ _/\_ _/\_
PAUSE:   _/\_ _/\_ __________ _/\_ _/\_
                   ↑ Long gap with no beats
```

## 🎨 Visual Guide in the Viewer:

```
DETAILED VIEW (Top):
┌─────────────────────────────────────────────────────┐
│  Before Event    │ EVENT! │    After Event          │
│  _/\_ _/\_ _/\_  │  ????  │  _/\_ _/\_ _/\_        │
│                  ↑                                   │
│              Red Line marks the start                │
└─────────────────────────────────────────────────────┘

OVERVIEW (Bottom):
┌─────────────────────────────────────────────────────┐
│ ___/\__/\__/\__/\__/\__/\__/\__/\__/\__/\__/\__/\__ │
│                      ↑                               │
│                   Red Line                           │
│              [Blue Shaded Area]                      │
│              ← Current 8-second window →             │
└─────────────────────────────────────────────────────┘
```

## 💡 Pro Tips:

1. **Start at the overview**: Always look at the bottom chart first to see where the event is
2. **Use the slider**: Slowly drag it to see how the waveform changes over time
3. **Compare before/after**: Look at the ECG before and after the red line to see the difference
4. **Check both channels**: Sometimes the abnormality is more obvious in Channel 1 or Channel 2
5. **Zoom in**: The detailed view shows high resolution - you can see individual heartbeats clearly

## 🔬 Technical Details:

- **Sampling Rate**: 200 samples per second
- **Total Duration**: 90 seconds (18,000 data points)
- **Detailed View**: 8 seconds (1,600 data points, downsampled to 800)
- **Event Index**: The exact sample number where the event starts
- **Time Calculation**: `event_time = event_index / 200`

## 📊 Example Episode to Try:

Try episode **74422783** (AF):
1. Click on it in the sidebar
2. Look at the overview - find the red line
3. Click on the red line or drag the slider to it
4. In the detailed view, you should see:
   - Irregular spacing between heartbeats
   - Chaotic baseline
   - No clear pattern

The abnormality IS there - the red line shows you exactly where to look! 🎯

