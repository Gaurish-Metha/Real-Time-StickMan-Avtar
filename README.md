# Gaurish Realtime Avatar

**Gaurish** is a highly realistic, real-time interactive avatar system powered by Python and AI. It transforms your webcam feed into a fully animated, neon-styled digital puppet that tracks your body movements, facial expressions, fingers, and voice in real-time.

## ✨ Features

*   **Full-Body Holistic Tracking:** Tracks head, torso, limbs, hands, and fingers with high precision using MediaPipe.
*   **Live Facial Animation:**
    *   **True Blinking:** The avatar blinks exactly when you do.
    *   **Gaze Tracking:** Pupils follow your real eye movements.
    *   **Lip Sync:** Mouth movements sync dynamically with your voice amplitude and face shape.
*   **Detailed Hand & Finger Tracking:** Capture intricate hand gestures and finger movements.
*   **Dynamic Visuals:**
    *   "Neon Android" aesthetic with glowing joints and smooth lines.
    *   Intelligent head scaling based on facial proximity.
    *   **Shoe Rendering:** Tracks feet to ensure the avatar stays grounded.
*   **Interactive UI:**
    *   **Live Webcam Preview:** Toggleable picture-in-picture mode to see yourself alongside the avatar.
    *   **Loading Screen:** Professional gradient loading sequence.
*   **Performance Optimized:** Uses custom smoothing algorithms (EMA) and frame optimization to ensure fluid 60 FPS performance on standard hardware.

## 🛠️ Installation

### Prerequisites
*   Python 3.8 or higher
*   A Webcam
*   A Microphone

### Setup
1.  **Clone or Download** this repository.
2.  Open a terminal in the project folder.
3.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 How to Run

### Windows (Easiest)
Double-click the **`run_stickman.bat`** file in the main directory.

### Command Line
Alternatively, you can run the Python script directly:
```bash
python src/main.py
```

## 🎮 Controls

*   **ESC**: Exit the application.
*   **V** or **Click "Cam: ON/OFF"**: Toggle the live webcam preview in the top-right corner.

## 📂 Project Structure

*   **`src/main.py`**: The core engine loop, window management, and event handling.
*   **`src/tracker.py`**: AI Computer Vision module handling MediaPipe holistic tracking (Body, Face, Hands).
*   **`src/avatar.py`**: The rendering engine that draws the "Gaurish" style avatar, handles physics, shoes, and smoothing.
*   **`src/audio.py`**: Real-time microphone processing for lip-sync.
*   **`src/utils.py`**: Mathematical helper functions (Smoothing/Interpolation).

## ⚙️ Customization

You can tweak the visuals in `src/avatar.py`:
*   Change `self.body_color`, `self.glow_color`, or `self.shoe_color` to customize the look.
*   Adjust `self.smoother = EMASmoother(alpha=0.6)` to change the responsiveness vs. smoothness balance.

---
*Created by Gaurish*
