# Hand Gesture Control using MediaPipe

This Python project utilizes the [MediaPipe](https://mediapipe.dev/) library and [OpenCV](https://opencv.org/) to perform real-time hand gesture recognition. With this code, you can control your computer's cursor using hand gestures within a defined control box.

## Features

- Detects and tracks hand landmarks in real time
- Controls mouse cursor movement with smooth tracking
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
   - The control box maps to your entire screen for precise control
   - Movement is smoothed for better usability

5. To exit the program, press 'q' in the OpenCV window.

## Customization Options

You can easily customize the program by modifying these parameters in the code:

- `GAP` (default: 100): Controls the margin from the edges of the camera frame
- `SMOOTHING` (default: 5): Higher values create smoother but slower cursor movement
- `min_detection_confidence` (default: 0.7): Adjust sensitivity of hand detection

## How It Works

1. The program captures video from your webcam
2. A control box is defined within the frame with margins set by the `GAP` parameter
3. MediaPipe detects hand landmarks, particularly the index finger tip
4. When your index finger is within the control box:
   - Its position is mapped to your screen coordinates
   - Movement is smoothed using a weighted average algorithm
   - The cursor is moved to the calculated position

## Troubleshooting

- **Camera not detected**: Make sure your webcam is properly connected and not being used by another application
- **Jerky cursor movement**: Increase the `SMOOTHING` value for smoother movement
- **Hand not detected**: Ensure proper lighting and increase the camera resolution if available

## Acknowledgments

This project uses the [MediaPipe](https://mediapipe.dev/) library for hand landmark detection and tracking.

## License

This project is licensed under the MIT License.

## Author

- Nikhil Yadav
- GitHub: [yadavnikhil17102004](https://github.com/yadavnikhil17102004)

Feel free to contribute to this project and make it even more awesome! If you have any questions or suggestions, please open an issue or pull request.

Enjoy controlling your computer with hand gestures! 🖐️🖥️
