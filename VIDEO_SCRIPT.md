# 5-Minute Video Script - ECG Arrhythmia Detection Demo

## 🎬 INTRO (30 seconds)

**[Screen: Show the app running]**

"Hey! I'm going to walk you through my ECG arrhythmia detection system that I built for the TriFetch take-home assignment. 

The goal was simple: build a tool that can load ECG recordings, classify what type of heart problem is happening - like atrial fibrillation or a pause - and pinpoint exactly when it starts in the recording.

I had about a day to build this, so I focused on making a clean, working system rather than chasing perfect accuracy. Let me show you what I built."

---

## 🏗️ PART 1: THE BIG PICTURE (1 minute)

**[Screen: Show the full app - sidebar + viewer]**

"Here's the full system. On the left, you can see a list of ECG episodes. Each one is a 90-second heart recording from a patient. They're labeled with what type of event happened - AF for atrial fibrillation, VTACH for fast dangerous heartbeats, or PAUSE for when the heart stops briefly.

When I click on one..."

**[Click on an episode]**

"...the system loads the ECG data, runs it through my classifier, and shows me three things:

1. **Top section**: A detailed 8-second view of the heartbeat waveform - this is zoomed in so you can see individual heartbeats clearly

2. **Middle**: A slider that lets me navigate through all 90 seconds of the recording

3. **Bottom**: A mini-map showing the entire 90 seconds at once

The red line you see? That's where the actual heart problem starts according to the medical data. The blue dashed line is where MY algorithm thinks it starts. The closer they are, the better my classifier is working."

---

## 🧠 PART 2: THE ML APPROACH (1.5 minutes)

**[Screen: Show backend/classifier.py file]**

"Now let's talk about the machine learning part - this is the interesting bit.

I had a choice: I could spend days training a deep neural network, or I could use proven medical signal processing techniques. I went with signal processing, and here's why:

**First - Speed**: I can classify an ECG in milliseconds. No training needed, no GPU required. Just load the data and go.

**Second - Explainability**: When my classifier says 'this is atrial fibrillation,' I can tell you EXACTLY why. It's because the heart rate variability is above 15%, the rhythm is irregular, and there's no consistent pattern. A doctor can understand and trust this.

**Third - It actually works**: These techniques are based on decades of cardiology research. They're not perfect, but they're reliable.

Here's how it works:

**Step 1 - Find the heartbeats**: I use a peak detection algorithm to find every heartbeat in the 90-second recording. These are called R-peaks.

**Step 2 - Measure the timing**: I calculate the time between each heartbeat. This is called the RR interval.

**Step 3 - Extract features**: From these intervals, I calculate:
- **Heart rate**: How fast is the heart beating?
- **Variability**: How much does the timing change between beats?
- **Irregularity**: Is there a pattern or is it chaotic?

**Step 4 - Apply rules**: 
- If variability is HIGH and rhythm is IRREGULAR → It's AF (atrial fibrillation)
- If heart rate is VERY FAST (over 120 bpm) and REGULAR → It's VTACH (ventricular tachycardia)
- If there's a LONG GAP with no beats → It's a PAUSE

**Step 5 - Find when it starts**: I use change-point detection - basically, I look for where the signal suddenly changes behavior. That's where the event starts.

**The tradeoff?** A deep learning model MIGHT be more accurate if I had thousands of labeled examples and weeks to train it. But for a one-day project with limited data, this approach gives me explainable, fast, and reasonably accurate results."

---

## 💻 PART 3: THE TECH STACK (1 minute)

**[Screen: Show the file structure]**

"Let me quickly walk through the architecture:

**Backend** - Python with FastAPI:
- Super fast async API
- Three main endpoints: list episodes, get ECG data, classify events
- Uses NumPy and SciPy for signal processing
- Loads data from JSON metadata and text files

**Frontend** - React with Vite:
- Modern, fast development setup
- Recharts library for the ECG visualizations
- Clean component structure: App, EpisodeList, ECGViewer

**The data flow**:
1. User clicks an episode
2. Frontend calls the backend API
3. Backend loads three 30-second ECG files (90 seconds total)
4. Runs the classifier
5. Returns the waveform data + predictions
6. Frontend renders it with the event markers

**Performance**: The whole thing loads in under 50 milliseconds. That's way under the 8-second requirement."

---

## 🎨 PART 4: THE UI DESIGN (1 minute)

**[Screen: Interact with the viewer]**

"The UI was designed around one key insight: doctors need to see BOTH the big picture AND the details.

Watch what happens when I use the slider..."

**[Drag the slider]**

"The top charts show me a crystal-clear 8-second window. I can see individual heartbeats, measure intervals, spot abnormalities.

The bottom chart shows me the entire 90 seconds. The blue shaded area? That's where I'm currently looking in the detailed view.

And here's a cool feature - I can click anywhere on the bottom chart..."

**[Click on the overview]**

"...and it jumps right to that time. Super fast navigation.

The event markers move with me. If the red line is in my current 8-second window, I see it in the detailed view. If not, I can still see it in the overview to know where to navigate.

I also added automatic centering - when you first load an episode, it automatically centers the view around the event so you see the abnormality right away."

---

## 🔍 PART 5: DEMO TIME (45 seconds)

**[Screen: Click through different episodes]**

"Let me show you a few examples:

**AF Episode**: See how the heartbeats are irregularly spaced? That's atrial fibrillation. The red line marks where it starts. My classifier caught it because the variability spiked.

**PAUSE Episode**: Look at this - a complete flat line. The heart literally stopped for a few seconds. Super obvious, and my classifier detects it by finding the long gap with no R-peaks.

**VTACH Episode**: This one's fast - really fast. Over 120 beats per minute. My classifier sees the high heart rate and flags it as ventricular tachycardia.

The blue dashed lines show where my algorithm predicted the events. Sometimes they're spot-on with the red line, sometimes they're a bit off. That's the reality of signal processing - it's good, but not perfect."

---

## 🎯 CLOSING (15 seconds)

**[Screen: Back to full app view]**

"So that's it! A full-stack ECG analysis system built in a day:
- Clean, fast backend with explainable ML
- Interactive frontend with smart navigation
- Real medical signal processing techniques
- Under 50ms load times

The code is on GitHub, everything's documented, and it's ready to demo. Thanks for watching!"

---

## 📝 NOTES FOR RECORDING:

### What to show on screen:
1. **Intro**: Full app running
2. **Part 1**: Click through episodes, show navigation
3. **Part 2**: Show classifier.py code briefly, then back to app
4. **Part 3**: Show file structure in VS Code
5. **Part 4**: Interact heavily with slider and overview
6. **Part 5**: Click through 3-4 different episodes
7. **Closing**: Full app view

### Tone:
- Conversational, not formal
- Confident but honest about tradeoffs
- Focus on DECISIONS and REASONING, not just features

### Key points to emphasize:
- ✅ "I chose X because Y" (show decision-making)
- ✅ "The tradeoff is..." (show you understand limitations)
- ✅ "This works well for..." (show practical thinking)
- ✅ Demonstrate the app working smoothly

### What NOT to do:
- ❌ Don't apologize for what you didn't build
- ❌ Don't use jargon without explaining it
- ❌ Don't spend too long on any one section
- ❌ Don't just read the script - be natural!

### Timing breakdown:
- Intro: 0:00 - 0:30
- Big Picture: 0:30 - 1:30
- ML Approach: 1:30 - 3:00
- Tech Stack: 3:00 - 4:00
- UI Design: 4:00 - 4:45
- Demo: 4:45 - 5:00

Good luck! 🎬

