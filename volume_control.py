#!/usr/bin/env python3
import time
import math
import cv2
import numpy as np
import mediapipe as mp # type: ignore
import pyautogui
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.75, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Initialize volume control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volume_range = volume.GetVolumeRange()
min_volume = volume_range[0]
max_volume = volume_range[1]

# Configuration parameters
SMOOTHING = 5  # Smoothing factor for volume changes
CALIBRATION_TIME = 3.0  # Duration of calibration in seconds
FPS_SMOOTH = 0.9  # Smoothing factor for FPS calculation
MIN_HAND_DISTANCE = 30  # Minimum distance between thumb and index for volume
MAX_HAND_DISTANCE = 250  # Maximum distance between thumb and index for volume

# Initialize video capture
cap = cv2.VideoCapture(0)

def distance(a, b):
    """Calculate Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def landmark_to_pixel(lm, w, h):
    """Convert a MediaPipe landmark to pixel coordinates."""
    return int(lm.x * w), int(lm.y * h)

def calibrate_control_box(cap, hands, seconds=3.0):
    """Calibrate the control box based on hand movement."""
    start = time.time()
    xs = []
    ys = []
    while time.time() - start < seconds:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)
        if res.multi_hand_landmarks:
            lm = res.multi_hand_landmarks[0]
            ix, iy = landmark_to_pixel(lm.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP], w, h)
            xs.append(ix)
            ys.append(iy)
        overlay = frame.copy()
        cv2.putText(overlay, f'Calibrating... move your hand around ({int(seconds - (time.time()-start))}s)', 
                   (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.imshow('Calibrate', overlay)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyWindow('Calibrate')
    
    if not xs or not ys:
        h, w = 480, 640
        ret, frame = cap.read()
        if ret:
            h, w, _ = frame.shape
        margin = int(min(w, h) * 0.15)
        return margin, margin, w - margin, h - margin
    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)
    pad_x = int((max_x - min_x) * 0.12) if (max_x - min_x) > 10 else int(w * 0.05)
    pad_y = int((max_y - min_y) * 0.12) if (max_y - min_y) > 10 else int(h * 0.05)
    x1 = max(0, min_x - pad_x)
    y1 = max(0, min_y - pad_y)
    x2 = min(w - 1, max_x + pad_x)
    y2 = min(h - 1, max_y + pad_y)
    if x2 - x1 < 50 or y2 - y1 < 50:
        margin = int(min(w, h) * 0.12)
        return margin, margin, w - margin, h - margin
    return x1, y1, x2, y2

# Run calibration to set the control box
x1, y1, x2, y2 = calibrate_control_box(cap, hands, CALIBRATION_TIME)

# Initialize variables
prev_time = time.time()
fps = 0.0
prev_volume = 0
smoothed_volume = 0

print("=== Volume Control with Hand Gesture ===")
print("INSTRUCTIONS:")
print("- Show your hand with thumb and index finger extended")
print("- Adjust the distance between thumb and index to control volume")
print("- Close fingers to lower volume, spread them to increase volume")
print("- Press 'q' to quit")
print("- Press 'c' to recalibrate control box")

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    # Flip frame horizontally for a more intuitive interaction
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    # Convert to RGB for MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame with MediaPipe
    res = hands.process(rgb)
    
    # Create a copy for drawing
    overlay = frame.copy()
    
    # Calculate FPS
    now = time.time()
    instant_fps = 1.0 / max(0.001, now - prev_time)  # Prevent division by zero
    fps = fps * FPS_SMOOTH + instant_fps * (1.0 - FPS_SMOOTH)
    
    # Default status
    status_text = "No Hand Detected"
    
    # Get current volume
    current_volume = volume.GetMasterVolumeLevelScalar()
    volume_percent = int(current_volume * 100)
    
    if res.multi_hand_landmarks:
        # Get the first hand detected
        hand_landmarks = res.multi_hand_landmarks[0]
        
        # Draw the hand landmarks
        mp_draw.draw_landmarks(overlay, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # Get the positions of thumb and index finger
        thumb_tip = landmark_to_pixel(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP], w, h)
        index_tip = landmark_to_pixel(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP], w, h)
        
        # Calculate the distance between thumb and index finger
        finger_distance = distance(thumb_tip, index_tip)
        
        # Map the finger distance to volume
        # Constrain the distance to our min/max range
        finger_distance = max(MIN_HAND_DISTANCE, min(finger_distance, MAX_HAND_DISTANCE))
        
        # Convert distance to volume (0.0 to 1.0)
        target_volume = (finger_distance - MIN_HAND_DISTANCE) / (MAX_HAND_DISTANCE - MIN_HAND_DISTANCE)
            
            # Apply smoothing
            smoothed_volume = prev_volume + (target_volume - prev_volume) / SMOOTHING
            smoothed_volume = max(0.0, min(1.0, smoothed_volume))
            
            # Set system volume
            volume.SetMasterVolumeLevelScalar(smoothed_volume, None)
            prev_volume = smoothed_volume
            
            volume_percent = int(smoothed_volume * 100)
            status_text = f"Volume: {volume_percent}%"
            
            # Draw a line between thumb and index finger
            cv2.line(overlay, thumb_tip, index_tip, (255, 0, 255), 3)
            
            # Draw circles at the tips for better visibility
            cv2.circle(overlay, thumb_tip, 10, (255, 0, 0), cv2.FILLED)
            cv2.circle(overlay, index_tip, 10, (0, 255, 0), cv2.FILLED)
            
            # Draw a volume bar
            bar_x = 50
            bar_y = h - 50
            bar_width = 200
            bar_height = 20
            
            cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 2)
            filled_width = int(smoothed_volume * bar_width)
            cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + filled_width, bar_y + bar_height), (0, 255, 0), cv2.FILLED)
            
            # Draw volume percentage
            cv2.putText(overlay, f"{volume_percent}%", (bar_x + bar_width + 10, bar_y + 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Display status information
    cv2.putText(overlay, f"Status: {status_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(overlay, f"FPS: {fps:.1f}", (w - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Show the final image
    cv2.imshow('Volume Control with Hand Gesture', overlay)
    prev_time = now
    
    # Check for key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('c'):
        x1, y1, x2, y2 = calibrate_control_box(cap, hands, CALIBRATION_TIME)

# Clean up
cap.release()
cv2.destroyAllWindows()
print("Application closed.")
