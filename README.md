# GestureCanvas — AR Hand Gesture Drawing System

A real-time augmented reality drawing application controlled entirely by hand gestures — no mouse, keyboard, or stylus needed. Drawings are rendered directly on top of the live camera feed using a transparent overlay pipeline.

![Demo](demo.png)

---

## Problem Statement

Traditional drawing and annotation tools require physical input devices. This system enables touchless, device-free drawing — useful in scenarios where physical contact is undesirable (medical, kiosk, industrial environments), or where immersive AR interaction is preferred (education, creative tools, accessibility).

---

## Features

| Feature | Description |
|---|---|
| Real-time hand tracking | MediaPipe 21-keypoint landmark detection at 30fps |
| Gesture-based mode switching | Draw, Erase, Grab, Transform — all gesture controlled |
| Neon glow rendering | Multi-layer Gaussian blur composited over live camera feed |
| Shape auto-detection | Freehand strokes snap to circle, rectangle, triangle, or line |
| Stroke manipulation | Move, zoom, rotate any stroke via pinch gestures |
| Partial stroke erase | Eraser splits freehand strokes — only touched segment removed |
| Undo / Redo | 60-step stack-based history |
| PNG export | Save drawing + camera frame as PNG |
| Session recording | Record full drawing session as AVI video |
| Live thickness control | Two-hand spread gesture adjusts brush size in real time |

---

## Gestures

| Gesture | Action |
|---|---|
| ☝️ Index finger only | **DRAW** mode — draw freely with fingertip |
| ✌️ Index + Middle fingers | **ERASE** mode — erase with fingertip |
| 🤏 Pinch (one hand) | **GRAB** — select and drag any stroke |
| 🤏🤏 Pinch (both hands) | **TRANSFORM** — zoom and rotate selected stroke |
| ✌✌ Two fingers (both hands) | **THICKNESS** — spread to increase, close to decrease |
| 🖐️ Open palm | Stop drawing / idle |

---

## Shape Auto-Detection

Draw freehand — system automatically detects and snaps to the nearest geometric shape when you lift your finger:

| You Draw | System Detects |
|---|---|
| Rough circle / closed loop | ⭕ Perfect circle |
| Four-cornered closed shape | ▭ Rectangle |
| Three-cornered closed shape | △ Triangle |
| Roughly straight stroke | — Straight line |
| Anything else | ✏️ Freehand (kept as drawn) |

**Algorithm:** Douglas-Peucker polygon approximation to count corners + geometric heuristics (radius variance for circles, aspect ratio, start-end closure distance).

---

## Rendering Pipeline

```
Webcam Frame
     │
     ▼
MediaPipe Hand Landmarker (21 keypoints)
     │
     ▼
Gesture Classifier
(finger-up state + pinch distance + two-hand detection)
     │
     ▼
Mode Dispatch
(DRAW / ERASE / GRAB / TRANSFORM)
     │
     ▼
Stroke Engine
(affine transforms: offset, scale, rotation per stroke)
     │
     ▼
Neon Glow Renderer
  Layer 1: Wide blurred outer glow (Gaussian σ=6)
  Layer 2: Sharp color stroke
  Layer 3: Bright thin center line
     │
     ▼
Alpha Composite over Camera Frame
     │
     ▼
Display (OpenCV window)
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Hand Tracking | MediaPipe Hand Landmarker |
| Computer Vision | OpenCV |
| Rendering | NumPy array ops + Gaussian blur compositing |
| Shape Detection | Douglas-Peucker + geometric heuristics |
| Export | OpenCV PNG write + VideoWriter |

---

## Installation

```bash
git clone https://github.com/ixsntg012-lab/Gesture-Paint-UI-.git
cd Gesture-Paint-UI-
pip install -r requirements.txt
python main.py
```

---

## Keyboard Controls

| Key | Action |
|---|---|
| `S` | Save drawing as PNG → `exports/` |
| `R` | Start / Stop video recording → `exports/` |
| `Z` | Undo |
| `Y` | Redo |
| `Q` | Quit |

**Toolbar (mouse click):**
- 8-colour palette
- Brush size slider (1–30)
- Shape selector: Pen / Line / Rect / Circle
- Undo / Redo buttons

---

## Project Structure

```
GestureCanvas/
├── main.py              ← complete application (single file)
├── requirements.txt
├── exports/             ← PNGs and recordings saved here (auto-created)
├── demo.png             ← screenshot
└── README.md
```

---

## Limitations & Future Work

- Works best with good lighting and a plain background
- Currently supports single-hand drawing (two-hand only for transform/thickness)
- Potential extension: multi-user collaborative drawing over network
- Shape detection could be improved with a learned classifier for more complex shapes

---

## Author

**Swetha Kiran Veernapu**
MS Computer Science

---

## License

MIT License