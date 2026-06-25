# AutoTuner — Re-Entry File
*Re-entry: AutoTuner*

## What This Is
Real-time vocal pitch correction — standalone Python app with a dark web UI.
Pick key + scale, set correction amount and retune speed, hit ON, sing into mic.

## Re-Entry Phrase
"Re-entry: AutoTuner"

## Current Status
✅ Live locally. Pitch detection + correction working.

## How It Works
1. `sounddevice` captures mic in 4096-sample chunks (~93ms)
2. YIN algorithm (pure numpy) detects fundamental frequency
3. Nearest scale note found; correction amount + retune speed applied (exponential smoothing)
4. `pyrubberband` (Rubber Band Library) pitch-shifts the audio
5. Corrected audio plays out through selected output device
6. Flask UI streams live pitch status via SSE every 80ms

## Scales
- Major, Minor, Pentatonic, Blues, Chromatic

## Ableton Integration
Set Output device to **BlackHole 2ch** → set Ableton audio input to BlackHole.
BlackHole is already installed on this machine.

## Stack
- Python + Flask, port 5576, host 127.0.0.1
- `sounddevice` — audio I/O
- `pyrubberband` + Rubber Band Library (`brew install rubberband`) — pitch shifting
- Pure-numpy YIN — pitch detection (no aubio — fails to build on Python 3.9/macOS)
- Dark theme, CSS variables, SSE for live updates

## File Structure
```
autotuner/
├── app.py          ← Flask server + audio stream management
├── autotune.py     ← YIN pitch detector + AutoTuner engine
├── templates/
│   └── index.html  ← UI
├── requirements.txt
├── Makefile
├── launch.command
└── .env.example
```

## How to Run
```bash
cd ~/autotuner && make run   # → http://127.0.0.1:5576
```

## Key Technical Notes
- `aubio` won't build from pip on Python 3.9/macOS — YIN re-implemented in numpy
- `pyrubberband` requires `rubberband` CLI: `brew install rubberband`
- Flask runs threaded; `sounddevice` callback runs in its own OS thread
- CHUNK=4096 (not 2048) — larger buffer gives YIN more cycles to work with for low pitches
- Confidence threshold: 0.78 — below this, audio passes through unmodified
- Shift smoothing decays to 0 when no signal (prevents clicks on silence)

## GitHub
papjamzzz/autotuner

## Last Session (2026-06-25)
Built from scratch. YIN pitch detection (pure numpy, tested A4=440.5Hz, C4=262.2Hz).
pyrubberband pitch shifting with scipy fallback. Flask SSE for live UI updates.
Dark UI: pitch monitor (cents meter + note display), controls (key/scale/amount/speed),
device selector with BlackHole hint. All working and verified.
