# Hand Gesture Control using MediaPipe

This Python project utilizes the [MediaPipe](https://mediapipe.dev/) library and [OpenCV](https://opencv.org/) to perform real-time hand gesture recognition. With this code, you can control your computer's cursor, perform clicks, scroll, and even control system volume using hand gestures.

## Features

- Detects and tracks hand landmarks in real time
- Controls mouse cursor movement with smooth tracking
- Simple and complex gesture recognition for various controls
- Visual feedback showing gesture detection status
- Provides a visual control box for easier interaction
- Configurable sensitivity and smoothing parameters
- Low CPU usage and responsive performance
- Multi-hand control with dedicated functions:
  - Right hand for cursor movement and clicking
  - Left hand for scrolling
- Dedicated scripts for specific controls like volume, scrolling, and keyboard input.

## Available Controls

This collection of Python scripts uses MediaPipe hand tracking to control various aspects of your computer with different hand gestures. Each script is specialized for a specific control function.

1.  **Mouse Control** (`main.py`)
    -   Control mouse cursor movement using index and middle fingers
    -   Simple and intuitive pointing interface

2.  **Scroll Control** (`scroll_control.py`)
    -   Dedicated scrolling interface using two fingers
    -   Move hand up/down to scroll pages
    -   Perfect for reading documents and browsing websites

3.  **Volume Control** (`volume_control.py`)
    -   Control system volume by adjusting the distance between thumb and index finger
    -   Visual feedback showing current volume level
    -   Requires `pycaw` library for Windows volume control

4.  **Combo Control** (`combo_control.py`)
    -   All-in-one control interface using both hands
    -   Right hand: Control cursor movement and clicks (index+middle for movement, pinch to click)
    -   Left hand: Control scrolling (index+middle fingers, move up/down to scroll)
    -   Perfect for presentations and browsing

5.  **Keyboard Control** (`keyboardtest.py`)
    -   Control arrow keys using hand swipes
    -   Trigger space key with rock gesture (thumb touching pinky base)
    -   Experimental script for gesture-based keyboard input

## Requirements

-   Python 3.11 (MediaPipe is not compatible with Python 3.13)
-   OpenCV (`pip install opencv-python`)
-   MediaPipe (`pip install mediapipe`)
-   PyAutoGUI (`pip install pyautogui`)
-   PyCaw (for volume control only: `pip install pycaw`)

Install all dependencies with:

```bash
pip install opencv-python mediapipe pyautogui pycaw
```

## How to Use

Each script operates independently with a different control interface. Run them separately based on what functionality you need:

### Mouse Control

```bash
python main.py
```

-   Show ONLY index and middle fingers up to activate mouse control
-   Keep other fingers down for cursor tracking

### Scroll Control

```bash
python scroll_control.py
```

-   Show index and middle fingers up to activate scrolling
-   Move hand up/down within the control box to scroll pages
-   Great for reading articles or browsing web pages

### Volume Control

```bash
python volume_control.py
```

-   Show your hand with thumb and index finger extended
-   Adjust distance between fingers to control volume
-   Move fingers apart to increase volume, closer to decrease volume

### Combo Control

```bash
python combo_control.py
```

-   Right hand: Show index and middle fingers up to control mouse cursor
-   Right hand: Pinch thumb and index finger for left-click
-   Left hand: Show index and middle fingers up to enable scrolling
-   Left hand: Move up/down to scroll pages

### Keyboard Control

```bash
python keyboardtest.py
```

-   Use index finger to swipe in directions for arrow key presses
-   Make a rock gesture (thumb touching pinky base) to press space bar
-   Great for presentations or controlling media playback

## How It Works

1.  The program captures video from your webcam.
2.  During startup, it runs a calibration phase where you move your hand to define the control box.
3.  MediaPipe detects hand landmarks in real-time.
4.  The program checks for specific finger positions to activate different controls:
    -   For cursor control: index and middle fingers raised
    -   For clicking: pinch gesture between thumb and index finger
    -   For scrolling: index and middle fingers raised on left hand, movement up/down
5.  Visual feedback is displayed showing detection status, gesture information, and FPS.

The simplified approach makes it easy to control different computer functions with natural hand positions.

## Customization

Each script has its own set of parameters you can adjust:

-   `SMOOTHING`: Higher values create smoother but slower response
-   `CALIBRATION_TIME`: Duration of the initial hand tracking calibration
-   Script-specific settings (e.g., `SCROLL_SENSITIVITY` in scroll control)

## Troubleshooting

-   **Camera not detected**: Make sure your webcam is properly connected and not being used by another application
-   **Hand not detected**: Ensure proper lighting and try adjusting camera position
-   **Inconsistent tracking**: Recalibrate by pressing 'c' during execution
-   **Volume control not working**: Make sure pycaw is installed correctly

## Acknowledgments

This project uses the [MediaPipe](https://mediapipe.dev/) library for hand landmark detection and tracking.

## Author

-   Nikhil Yadav
-   GitHub: [yadavnikhil17102004](https://github.com/yadavnikhil17102004)

Feel free to contribute or customize these scripts for your own needs!