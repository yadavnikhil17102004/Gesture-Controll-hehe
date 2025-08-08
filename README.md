# Hand Gesture Control using MediaPipe

This Python project utilizes the [MediaPipe](https://mediapipe.dev/) library and [OpenCV](https://opencv.org/) to perform real-time hand gesture recognition. With this code, you can control your computer's cursor using hand gestures within a defined control box.

## Features

- Detects and tracks hand landmarks in real time
- Controls mouse cursor movement with smooth tracking
- Supports mouse clicks using pinch gestures
- Drag and drop functionality with pinch-and-hold gesture
- Tab switching using swipe gestures
- Visual feedback for active gestures
- Provides a visual control box for easier interaction
- Configurable sensitivity and smoothing parameters
- Low CPU usage and responsive performance

## Prerequisites

Before running the code, make sure you have the following dependencies installed:

- Python 3.11 (MediaPipe is not compatible with Python 3.13)
- OpenCV (`pip install opencv-python`)
- MediaPipe (`pip install mediapipe`)
- PyAutoGUI (`pip install pyautogui`)

You can install all dependencies at once using:

```bash
py -3.11 -m pip install opencv-python mediapipe pyautogui
```

## How to Use

1. Clone this repository:

   ```bash
   git clone https://github.com/yadavnikhil17102004/Gesture-Controll-hehe.git
   ```

2. Navigate to the project directory:

   ```bash
   cd Gesture-Controll-hehe
   ```

3. Run the code with Python 3.11:

   ```bash
   py -3.11 main.py
   ```

4. A window titled "Hand Mouse Control" will open showing your webcam feed with a green control box.
   - Move your index finger within the control box to control the mouse cursor
   - Pinch your thumb and index finger together for a left click
   - Hold the pinch gesture for 0.5 seconds to start dragging (mouse button stays down)
   - Release the pinch to drop (release mouse button)
   - Pinch your index and middle fingers together for a right click
   - Swipe left/right for tab switching (ctrl+tab/ctrl+shift+tab)
   - The control box maps to your entire screen for precise control

5. To exit the program, press 'q' in the OpenCV window.
6. To recalibrate the control box, press 'c' in the OpenCV window.

## Customization Options

You can easily customize the program by modifying these parameters in the code:

- `SMOOTHING` (default: 5): Higher values create smoother but slower cursor movement
- `PINCH_ON` (default: 0.30): Threshold for pinch gesture (left click/drag)
- `PINCH_OFF` (default: 0.50): Threshold for releasing pinch
- `PINCH_HOLD_TIME` (default: 0.5): Time to hold pinch for drag operation
- `TWO_PINCH_ON` (default: 0.28): Threshold for two-finger pinch (right click)
- `CLICK_COOLDOWN` (default: 0.35): Minimum time between clicks
- `SWIPE_WINDOW` (default: 0.35): Time window for swipe detection
- `SWIPE_MIN_DIST_FRAC` (default: 0.25): Minimum distance for swipe as fraction of control box
- `SWIPE_VEL_THRESH` (default: 600): Velocity threshold for swipe detection
- `CALIBRATION_TIME` (default: 3.0): Duration of calibration in seconds

## How It Works

1. The program captures video from your webcam
2. During startup, it runs a calibration phase where you move your hand to define the control box
3. MediaPipe detects hand landmarks in real-time
4. The program tracks several key gestures:
   - **Mouse Movement**: Index finger position within the control box is mapped to screen coordinates
   - **Left Click**: Detected when thumb and index finger pinch together briefly
   - **Drag and Drop**: Detected when thumb and index finger pinch is held for more than 0.5 seconds
   - **Right Click**: Detected when index and middle fingers pinch together
   - **Tab Switching**: Swipe gestures trigger browser tab navigation shortcuts
5. Visual feedback is displayed showing detection status, gesture information, and FPS

The application uses normalized distances between landmarks to detect gestures, making it work consistently regardless of hand size or distance from the camera.

## Troubleshooting

- **Camera not detected**: Make sure your webcam is properly connected and not being used by another application
- **Jerky cursor movement**: Increase the `SMOOTHING` value for smoother movement
- **Hand not detected**: Ensure proper lighting and try adjusting `min_detection_confidence` to a lower value
- **Clicks not registering**: Try adjusting the `PINCH_ON` and `PINCH_OFF` thresholds
- **Too many accidental clicks**: Increase the `CLICK_COOLDOWN` value
- **Control box too small/large**: Press 'c' to recalibrate, and move your hand more/less during calibration

## Acknowledgments

This project uses the [MediaPipe](https://mediapipe.dev/) library for hand landmark detection and tracking.

## License

This project is licensed under the MIT License.

## Author

- Nikhil Yadav
- GitHub: [yadavnikhil17102004](https://github.com/yadavnikhil17102004)

Feel free to contribute to this project and make it even more awesome! If you have any questions or suggestions, please open an issue or pull request.

Enjoy controlling your computer with hand gestures! 🖐️🖥️
