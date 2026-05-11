# AI-Sprayer 🌿

A computer vision system that detects grass/vegetation coverage in video feeds using HSV color segmentation. Designed as a prototype for an AI-controlled sprayer that can estimate how much of an area is covered by grass and adjust spray parameters accordingly.

---

## Features

- Real-time grass detection using HSV color masking
- Adaptive HSV thresholds based on scene brightness
- Morphological noise cleaning (open/close operations)
- Contour filtering to remove small false positives
- ROI (Region of Interest) to ignore sky/upper areas
- Smoothed coverage & spread metrics
- Multiple UI options (Tkinter, PyQt5, pure OpenCV)

---

## Project Structure

```
AI-Sprayer/
├── main_color.py     # Main Tkinter GUI — 4-panel view, adaptive HSV, multi-video loop
├── hsv_gui.py        # PyQt5 GUI — dark-themed UI with live metrics and full slider control
├── hsv.py            # Minimal OpenCV trackbar version for quick HSV tuning
├── adaptive_hsv.py   # Headless script — adaptive HSV + ROI logic, no GUI dependency
└── video/
    ├── grass.mp4
    ├── grass2.mp4
    └── grass3.mp4
```

---

## Requirements

- Python 3.8+
- OpenCV
- NumPy
- Pillow (for `main_color.py`)
- PyQt5 (for `hsv_gui.py`)

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/harishfaqot/AI-Sprayer.git
cd AI-Sprayer
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install opencv-python numpy Pillow PyQt5
```

---

## Usage

### Option A — Tkinter GUI (`main_color.py`) — Recommended starting point

Displays four panels: Original, Mask, Masked RGB, and Result overlay.  
Automatically cycles through all videos in the `video/` folder.

```bash
python main_color.py
```

**Controls (bottom sliders):**

| Slider    | Description                                      |
|-----------|--------------------------------------------------|
| Kernel    | Morphology kernel size (noise removal strength)  |
| MinArea   | Minimum contour area to keep (px²)               |
| ROI %     | Percentage from top to ignore (skip sky)         |
| Smooth %  | Exponential smoothing factor for live metrics    |

---

### Option B — PyQt5 GUI (`hsv_gui.py`) — Full control panel

Dark-themed UI with 4 video panels, full slider control over all HSV and processing parameters, live metrics, pause/resume, and smoothing reset.

```bash
python hsv_gui.py
```

**Controls (right panel):**

| Slider        | Description                              |
|---------------|------------------------------------------|
| Smoothing α   | EMA smoothing factor (0.01–0.99)         |
| Min Contour   | Minimum contour area in pixels           |
| ROI Top %     | Crop top % of frame (ignore sky)         |
| Blur Kernel   | Gaussian blur kernel size                |
| Hue Low/High  | HSV hue range for green detection        |
| Sat Low       | Minimum saturation threshold             |
| Val Low       | Minimum brightness threshold             |
| Morph Kernel  | Morphological operation kernel size      |

**Buttons:**

- **Pause / Resume** — Freeze the video feed
- **Reset Smooth** — Reset smoothed coverage and spread to 0

---

### Option C — OpenCV Trackbar (`hsv.py`) — Quick HSV tuning

Lightweight version using native OpenCV windows and trackbars. Good for dialing in HSV values for a new environment.

```bash
python hsv.py
```

Adjust the H/S/V min and max trackbars in the "Controls" window in real time.  
Press `Q` to quit.

> **Note:** This script expects `video/grass.mp4` to exist.

---

### Option D — Adaptive HSV headless (`adaptive_hsv.py`)

Runs the full detection pipeline without a GUI. Useful for testing the adaptive logic or integrating into a backend system.

```bash
python adaptive_hsv.py
```

> **Note:** This script expects `grass.mp4` in the same directory (not in `video/`).

---

## Adding Your Own Videos

1. Place `.mp4`, `.avi`, or `.mov` files in the `video/` folder.
2. Run `main_color.py` — it will automatically detect and loop through all videos in that folder.

For `hsv.py` and `hsv_gui.py`, edit the `cv2.VideoCapture(...)` path at the top of the respective file.

---

## How It Works

```
Video frame
    │
    ▼
Gaussian Blur          ← reduces noise
    │
    ▼
Convert to HSV         ← hue-based color detection
    │
    ▼
Adaptive Threshold     ← adjusts ranges based on scene brightness (V channel mean)
    │
    ▼
Morphological Ops      ← erode/dilate to clean mask
    │
    ▼
Contour Filtering      ← remove blobs smaller than MinArea
    │
    ▼
ROI Crop               ← ignore top N% of frame (sky, obstacles)
    │
    ▼
Coverage & Spread      ← grass_pixels / total_pixels, horizontal extent
    │
    ▼
EMA Smoothing          ← smooth_val = α × new + (1−α) × prev
    │
    ▼
Pump PWM / Nozzle      ← output signal for sprayer control (hsv_gui.py)
```

---

## License

MIT
