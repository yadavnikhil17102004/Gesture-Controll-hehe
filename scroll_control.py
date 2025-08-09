#!/usr/bin/env python3
import time
import math
import cv2
import numpy as np
import mediapipe as mp # type: ignore
import pyautogui

pyautogui.FAILSAFE = False

# Configuration parameters
SMOOTHING = 4  # Lower value for quicker response to scroll gestures
SCROLL_SENSITIVITY = 30  # Scroll sensitivity (higher = more scroll per movement)
SCROLL_THRESHOLD = 10  # Minimum pixel movement to trigger scroll
CALIBRATION_TIME = 3.0  # Duration of calibration in seconds
FPS_SMOOTH = 0.9  # Smoothing factor for FPS calculation

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.75, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

def distance(a, b):
    """Calculate Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def landmark_to_pixel(lm, w, h):
    """Convert a MediaPipe landmark to pixel coordinates."""
    return int(lm.x * w), int(lm.y * h)

def is_finger_up(landmarks, finger_tip_id, finger_pip_id, finger_mcp_id, w, h):
    """Check if a finger is pointing up"""
    tip = landmark_to_pixel(landmarks[finger_tip_id], w, h)
    pip = landmark_to_pixel(landmarks[finger_pip_id], w, h)
    mcp = landmark_to_pixel(landmarks[finger_mcp_id], w, h)
    
    # Vertical check (y-coordinate comparison)
    is_up = tip[1] < pip[1] < mcp[1]
    
    return is_up

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
scroll_active = False
prev_middle_y = None
prev_index_y = None
prev_hand_center_y = None

print("=== Two-Finger Scroll Control ===")
print("INSTRUCTIONS:")
print("- Show index and middle fingers up to activate scrolling")
print("- Move fingers up/down to scroll")
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
    
    # Calculate control box dimensions
    control_w = max(1, x2 - x1)
    control_h = max(1, y2 - y1)
    
    # Create a copy for drawing
    overlay = frame.copy()
    
    # Calculate FPS
    now = time.time()
    instant_fps = 1.0 / max(0.001, now - prev_time)  # Prevent division by zero
    fps = fps * FPS_SMOOTH + instant_fps * (1.0 - FPS_SMOOTH)
    
    # Default status
    scroll_active = False
    status_text = "Inactive"
    
    if res.multi_hand_landmarks:
        # Extract hand landmarks
        landmarks = res.multi_hand_landmarks[0].landmark
        
        # Get key finger positions for scroll control
        index_tip = landmark_to_pixel(landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP], w, h)
        middle_tip = landmark_to_pixel(landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP], w, h)
        
        # Calculate center point between fingers for scroll reference
        hand_center_x = (index_tip[0] + middle_tip[0]) // 2
        hand_center_y = (index_tip[1] + middle_tip[1]) // 2
        
        # Check if index and middle fingers are up
        is_index_up = is_finger_up(
            landmarks, 
            mp_hands.HandLandmark.INDEX_FINGER_TIP,
            mp_hands.HandLandmark.INDEX_FINGER_PIP,
            mp_hands.HandLandmark.INDEX_FINGER_MCP,
            w, h
        )
        
        is_middle_up = is_finger_up(
            landmarks, 
            mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
            mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
            mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
            w, h
        )
        
        is_ring_up = is_finger_up(
            landmarks, 
            mp_hands.HandLandmark.RING_FINGER_TIP,
            mp_hands.HandLandmark.RING_FINGER_PIP,
            mp_hands.HandLandmark.RING_FINGER_MCP,
            w, h
        )
        
        is_pinky_up = is_finger_up(
            landmarks, 
            mp_hands.HandLandmark.PINKY_TIP,
            mp_hands.HandLandmark.PINKY_PIP,
            mp_hands.HandLandmark.PINKY_MCP,
            w, h
        )
        
        # Check thumb separately
        thumb_tip = landmark_to_pixel(landmarks[mp_hands.HandLandmark.THUMB_TIP], w, h)
        thumb_cmc = landmark_to_pixel(landmarks[mp_hands.HandLandmark.THUMB_CMC], w, h)
        is_thumb_out = thumb_tip[0] > thumb_cmc[0]  # For right hand
        
        # Activate scroll only when both index and middle are up, others are down
        scroll_active = (is_index_up and is_middle_up and 
                         not is_ring_up and not is_pinky_up and not is_thumb_out)
        
        # Draw the hand landmarks
        mp_draw.draw_landmarks(overlay, res.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
        
        # Handle scroll when active
        if scroll_active:
            # Draw the scroll control points
            cv2.circle(overlay, (hand_center_x, hand_center_y), 10, (0, 255, 255), -1)
            cv2.line(overlay, index_tip, middle_tip, (0, 255, 255), 2)
            
            # Calculate and perform scroll
            if prev_hand_center_y is not None:
                # Calculate vertical movement
                y_diff = hand_center_y - prev_hand_center_y
                
                # Apply scroll only if movement exceeds threshold
                if abs(y_diff) > SCROLL_THRESHOLD:
                    # Determine scroll direction and amount
                    scroll_amount = int(y_diff / SMOOTHING) 
                    
                    # Perform scroll (positive = down, negative = up)
                    pyautogui.scroll(-scroll_amount * SCROLL_SENSITIVITY)
                    
                    # Show scroll direction
                    direction = "DOWN" if y_diff > 0 else "UP"
                    status_text = f"Scrolling {direction}"
                    
                    # Draw scroll direction indicator
                    arrow_start = (w - 50, h - 70)
                    arrow_end = (w - 50, h - 70 - (20 if y_diff < 0 else -20))
                    cv2.arrowedLine(overlay, arrow_start, arrow_end, (0, 0, 255), 2, tipLength=0.5)
            
            # Update previous positions
            prev_hand_center_y = hand_center_y
            prev_index_y = index_tip[1]
            prev_middle_y = middle_tip[1]
        else:
            # Reset previous positions when not scrolling
            prev_hand_center_y = None
            prev_index_y = None
            prev_middle_y = None
    else:
        # Reset all previous positions when no hand is detected
        prev_hand_center_y = None
        prev_index_y = None 
        prev_middle_y = None
    
    # Display status information
    cv2.putText(overlay, f"Status: {status_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                (0, 255, 0) if scroll_active else (0, 0, 255), 2)
    cv2.putText(overlay, f"FPS: {fps:.1f}", (w - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Show the final image
    cv2.imshow('Two-Finger Scroll Control', overlay)
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
