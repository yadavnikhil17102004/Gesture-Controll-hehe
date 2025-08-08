#!/usr/bin/env python3
import time
from collections import deque
import math
import cv2
import numpy as np
import mediapipe as mp # type: ignore
import pyautogui

pyautogui.FAILSAFE = False

# Configuration parameters
SMOOTHING = 5  # Higher values create smoother cursor movement
PINCH_ON = 0.30  # Threshold for pinch gesture (left click/drag)
PINCH_OFF = 0.50  # Threshold for releasing pinch
PINCH_HOLD_TIME = 0.5  # Time to hold pinch for drag operation
TWO_PINCH_ON = 0.28  # Threshold for two-finger pinch (right click)
CLICK_COOLDOWN = 0.35  # Minimum time between clicks
SWIPE_WINDOW = 0.35  # Time window for swipe detection
SWIPE_MIN_DIST_FRAC = 0.25  # Minimum distance for swipe as fraction of control box
SWIPE_VEL_THRESH = 600  # Velocity threshold for swipe detection
CALIBRATION_TIME = 3.0  # Duration of calibration in seconds
FPS_SMOOTH = 0.9  # Smoothing factor for FPS calculation

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.75, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

screen_w, screen_h = pyautogui.size()
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
        cv2.putText(overlay, f'Calibrating... move your hand around ({int(seconds - (time.time()-start))}s)', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
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

x1, y1, x2, y2 = calibrate_control_box(cap, hands, CALIBRATION_TIME)

prev_screen_x, prev_screen_y = pyautogui.position()
prev_finger_x, prev_finger_y = None, None
prev_time = time.time()
last_left_click = 0
last_right_click = 0
left_released = True
right_released = True
last_action = 'idle'
swipe_buffer = deque()
fps = 0.0
pinch_start_time = 0  # Time when pinch gesture started
is_dragging = False   # Flag to track drag operations

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
    
    # Draw control box with a clearer color and label
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(overlay, "Control Area", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # Calculate FPS
    now = time.time()
    instant_fps = 1.0 / max(0.001, now - prev_time)  # Prevent division by zero
    fps = fps * FPS_SMOOTH + instant_fps * (1.0 - FPS_SMOOTH)
    
    action_label = ''
    if res.multi_hand_landmarks:
        # Extract hand landmarks
        lm = res.multi_hand_landmarks[0]
        landmarks = lm.landmark
        
        # Get key finger positions
        ix, iy = landmark_to_pixel(landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP], w, h)
        tx, ty = landmark_to_pixel(landmarks[mp_hands.HandLandmark.THUMB_TIP], w, h)
        mtx, mty = landmark_to_pixel(landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP], w, h)
        wrist_x, wrist_y = landmark_to_pixel(landmarks[mp_hands.HandLandmark.WRIST], w, h)
        mcp_x, mcp_y = landmark_to_pixel(landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP], w, h)
        
        # Calculate hand size for normalization
        hand_size = max(1.0, distance((wrist_x, wrist_y), (mcp_x, mcp_y)))
        
        # Calculate ratios for gesture detection
        pinch_dist = distance((ix, iy), (tx, ty))
        pinch_ratio = pinch_dist / hand_size  # Normalized pinch distance
        
        two_dist = distance((ix, iy), (mtx, mty))
        two_ratio = two_dist / hand_size  # Normalized two-finger distance
        
        # Draw lines for active gestures
        if pinch_ratio < PINCH_ON:
            cv2.line(overlay, (ix, iy), (tx, ty), (0, 255, 0), 2)  # Green line for pinch
        
        if two_ratio < TWO_PINCH_ON:
            cv2.line(overlay, (ix, iy), (mtx, mty), (0, 0, 255), 2)  # Red line for two-finger pinch
        # Process swipe gestures
        if ix >= x1 and ix <= x2 and iy >= y1 and iy <= y2:
            # Add current position and time to swipe buffer
            swipe_buffer.append((ix, now))
            
            # Remove old entries outside the time window
            while swipe_buffer and (now - swipe_buffer[0][1]) > SWIPE_WINDOW:
                swipe_buffer.popleft()
                
            # Check for swipe gesture if we have enough data points
            if len(swipe_buffer) >= 2:
                start_x, start_time = swipe_buffer[0]
                end_x, end_time = swipe_buffer[-1]
                dx = end_x - start_x
                dt = end_time - start_time
                
                # Calculate velocity and check against thresholds
                if dt > 0 and abs(dx) > SWIPE_MIN_DIST_FRAC * control_w and abs(dx / dt) > SWIPE_VEL_THRESH:
                    if dx < 0:  # Swipe left
                        pyautogui.hotkey('ctrl', 'shift', 'tab')
                        action_label = 'Swipe Left'
                    else:  # Swipe right
                        pyautogui.hotkey('ctrl', 'tab')
                        action_label = 'Swipe Right'
                    
                    last_action = f'swipe_{"left" if dx < 0 else "right"}'
                    swipe_buffer.clear()  # Reset buffer after swipe
        else:
            swipe_buffer.clear()  # Clear buffer when hand is outside control box
            
        # Process click gestures
        
        # Left click with pinch gesture (index finger and thumb)
        if pinch_ratio < PINCH_ON:
            if left_released:
                # Start tracking pinch time for potential drag operation
                if pinch_start_time == 0:
                    pinch_start_time = now
                
                # Check if this is a click or the start of a drag
                if not is_dragging and (now - pinch_start_time) > PINCH_HOLD_TIME:
                    # Start drag operation
                    pyautogui.mouseDown(button='left')
                    is_dragging = True
                    last_action = 'drag_start'
                    action_label = 'Drag Start'
                elif not is_dragging and (now - last_left_click) > CLICK_COOLDOWN:
                    # Single click if not held long enough for drag
                    if (now - pinch_start_time) <= PINCH_HOLD_TIME:
                        pyautogui.click()
                        last_left_click = now
                        last_action = 'left_click'
                        action_label = 'Left Click'
            left_released = False
        elif pinch_ratio > PINCH_OFF:
            # End drag operation if active
            if is_dragging:
                pyautogui.mouseUp(button='left')
                is_dragging = False
                last_action = 'drag_end'
                action_label = 'Drag End'
            
            left_released = True
            pinch_start_time = 0  # Reset pinch timer
            
        # Right click with two-finger pinch (index and middle finger)
        if two_ratio < TWO_PINCH_ON and right_released and (now - last_right_click) > CLICK_COOLDOWN:
            pyautogui.rightClick()
            last_right_click = now
            right_released = False
            last_action = 'right_click'
            action_label = 'Right Click'
        elif two_ratio > PINCH_OFF:
            right_released = True
            
        # Mouse movement within control box
        if ix >= x1 and ix <= x2 and iy >= y1 and iy <= y2:
            # Normalize position within control box to screen coordinates
            nx = max(0.0, min(1.0, (ix - x1) / control_w))
            ny = max(0.0, min(1.0, (iy - y1) / control_h))
            
            # Map to screen coordinates
            target_x = nx * screen_w
            target_y = ny * screen_h
            
            # Apply smoothing
            smooth_x = prev_screen_x + (target_x - prev_screen_x) / SMOOTHING
            smooth_y = prev_screen_y + (target_y - prev_screen_y) / SMOOTHING
            
            try:
                pyautogui.moveTo(smooth_x, smooth_y)
                prev_screen_x, prev_screen_y = smooth_x, smooth_y
                last_action = 'move'
                action_label = 'Move'
            except Exception:
                pass
        
        prev_finger_x, prev_finger_y = ix, iy
        mp_draw.draw_landmarks(overlay, lm, mp_hands.HAND_CONNECTIONS)
        cv2.circle(overlay, (ix, iy), 8, (255, 0, 0), -1)
        cv2.putText(overlay, f'PinchR:{pinch_ratio:.2f}', (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(overlay, f'TwoR:{two_ratio:.2f}', (10, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        
        # Display drag status if active
        if is_dragging:
            cv2.putText(overlay, "DRAGGING", (w - 150, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    else:
        swipe_buffer.clear()
        prev_finger_x, prev_finger_y = None, None
        
        # End drag operation if hand is lost during drag
        if is_dragging:
            pyautogui.mouseUp(button='left')
            is_dragging = False
        
    # Display the frame with information
    cv2.putText(overlay, f'Action:{action_label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.putText(overlay, f'FPS:{fps:.1f}', (w - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    
    # Show the final image
    cv2.imshow('Hand Mouse Control', overlay)
    prev_time = now
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        # Ensure we release any held mouse buttons before exiting
        if is_dragging:
            pyautogui.mouseUp(button='left')
        break
    if key == ord('c'):
        x1, y1, x2, y2 = calibrate_control_box(cap, hands, CALIBRATION_TIME)

cap.release()
cv2.destroyAllWindows()
