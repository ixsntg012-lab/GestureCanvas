# GestureCanvas 🎨

<div align="center">

**Real-time AR Hand Gesture Drawing System**

*Draw, erase, transform — entirely with hand gestures. No mouse. No stylus. No touch.*

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Hand_Tracking-0097A7?style=for-the-badge)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer_Vision-5C3EE8?style=for-the-badge&logo=opencv)
![NumPy](https://img.shields.io/badge/NumPy-Rendering-013243?style=for-the-badge&logo=numpy)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

</div>

---

## What is GestureCanvas?

GestureCanvas is a real-time augmented reality drawing application controlled entirely by hand gestures — captured through a standard webcam. Drawings are rendered with neon glow effects directly on top of the live camera feed using a transparent overlay pipeline.

No special hardware. No physical contact required.

> **Use cases:** Touchless annotation in medical/industrial settings, immersive AR drawing, accessibility tools, interactive education displays.

---

## Gestures

| Gesture | Action |
|---------|--------|
| ☝️ Index finger only | **DRAW** — draw freely with fingertip |
| ✌️ Index + Middle fingers | **ERASE** — erase with fingertip |
| 🤏 Pinch (one hand) | **GRAB** — select and drag any stroke |
| 🤏🤏 Pinch (both hands) | **TRANSFORM** — zoom and rotate selected stroke |
| ✌✌ Two fingers (both hands) | **THICKNESS** — spread to increase, close to decrease |
| 🖐️ Open palm | Stop drawing / idle |

---

## Features

| Feature | Description |
|---------|-------------|
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

## Rendering Pipeline

```
Webcam Frame (30fps)
        │
        ▼
MediaPipe Hand Landmarker
(21 hand keypoints — pixel coordinates)
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
  Layer 1 — Wide blurred outer glow (Gaussian σ=6)
  Layer 2 — Sharp color stroke
  Layer 3 — Bright thin center line
        │
        ▼
Alpha Composite over Camera Frame
        │
        ▼
Display (OpenCV window)
```

---

## Shape Auto-Detection

Draw freehand — system automatically snaps to the nearest geometric shape when you lift your finger.

| You Draw | System Detects |
|----------|----------------|
| Rough circle / closed loop | ⭕ Perfect circle |
| Four-cornered closed shape | ▭ Rectangle |
| Three-cornered closed shape | △ Triangle |
| Roughly straight stroke | — Straight line |
| Anything else | ✏️ Freehand (kept as drawn) |

**Algorithm:** Douglas-Peucker polygon approximation to count corners + geometric heuristics (radius variance for circles, aspect ratio, start-end closure distance).

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Hand Tracking | MediaPipe Hand Landmarker |
| Computer Vision | OpenCV |
| Rendering | NumPy array ops + Gaussian blur compositing |
| Shape Detection | Douglas-Peucker + geometric heuristics |
| Export | OpenCV PNG write + VideoWriter (AVI) |

---

## Installation

```bash
git clone https://github.com/ixsntg012-lab/Gesture-Paint-UI-.git
cd Gesture-Paint-UI-
pip install -r requirements.txt
python main.py
```

---

## Controls

**Gestures (primary):** See Gestures table above.

**Keyboard shortcuts:**

| Key | Action |
|-----|--------|
| `S` | Save drawing as PNG → `exports/` |
| `R` | Start / Stop video recording → `exports/` |
| `Z` | Undo |
| `Y` | Redo |
| `Q` | Quit |

**Toolbar (mouse):** 8-colour palette, brush size slider (1–30), shape selector, Undo/Redo buttons.

---

## Project Structure

```
GestureCanvas/
├── main.py              ← complete application
├── hand_tracker.py      ← MediaPipe hand landmark detection
├── gesture_utils.py     ← gesture recognition utilities
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
- Shape detection could be improved with a learned classifier for complex shapes

---

## Author

**Swetha Kiran Veernapu**
MS Computer Science

---

## License

MIT License
