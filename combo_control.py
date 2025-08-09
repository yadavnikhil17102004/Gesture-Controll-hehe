#!/usr/bin/env python3
import time
import math
import cv2
import numpy as np
import mediapipe as mp # type: ignore
import pyautogui

pyautogui.FAILSAFE = False

# Configuration parameters
SMOOTHING = 8  # Higher values create smoother cursor movement
FINGER_DETECTION_THRESHOLD = 0.6  # Threshold for finger up detection
CALIBRATION_TIME = 3.0  # Duration of calibration in seconds
FPS_SMOOTH = 0.9  # Smoothing factor for FPS calculation
PINCH_THRESHOLD = 0.15  # Threshold for pinch detection (thumb-index touch)
PINCH_RELEASE_THRESHOLD = 0.25  # Threshold for releasing pinch
CLICK_COOLDOWN = 15  # Frames to wait between clicks to prevent multiple clicks
SCROLL_SENSITIVITY = 30  # Scroll sensitivity (higher = more scroll per movement)
SCROLL_THRESHOLD = 10  # Minimum pixel movement to trigger scroll

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.75, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

screen_w, screen_h = pyautogui.size()
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

def is_pinch(landmarks, w, h):
    """Detect pinch gesture between thumb and index finger"""
    thumb_tip = landmark_to_pixel(landmarks[mp_hands.HandLandmark.THUMB_TIP], w, h)
    index_tip = landmark_to_pixel(landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP], w, h)
    wrist = landmark_to_pixel(landmarks[mp_hands.HandLandmark.WRIST], w, h)
    
    # Calculate hand size for normalization (using wrist to middle finger base)
    middle_mcp = landmark_to_pixel(landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP], w, h)
    hand_size = max(1.0, distance(wrist, middle_mcp))
    
    # Calculate distance between thumb tip and index finger tip
    pinch_distance = distance(thumb_tip, index_tip)
    
    # Normalize the pinch distance by hand size
    pinch_ratio = pinch_distance / hand_size
    
    return pinch_ratio, thumb_tip, index_tip

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
prev_screen_x, prev_screen_y = pyautogui.position()
prev_time = time.time()
fps = 0.0
cursor_active = False
prev_scroll_y = None
pinch_active = False
prev_pinch_active = False
click_cooldown_counter = 0

print("=== Multi-Gesture Control with SEPARATE Pinch Click ===")
print("INSTRUCTIONS:")
print("RIGHT HAND - TWO SEPARATE GESTURES:")
print("1. CURSOR CONTROL:")
print("   - Show ONLY index and middle fingers UP")
print("   - Keep ring and pinky fingers DOWN")
print("   - Thumb position doesn't matter for cursor")
print("2. CLICK ACTION (works independently):")
print("   - PINCH thumb and index finger together")
print("   - Middle finger can be up or down")
print("   - Keep pinch brief to avoid multiple clicks")
print("LEFT HAND:")
print("- Show index and middle fingers UP to enable scrolling")
print("- Move up/down to scroll")
print("CONTROLS:")
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
    
    # Draw control box
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(overlay, "Control Area", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # Calculate FPS
    now = time.time()
    instant_fps = 1.0 / max(0.001, now - prev_time)  # Prevent division by zero
    fps = fps * FPS_SMOOTH + instant_fps * (1.0 - FPS_SMOOTH)
    
    # Default statuses
    cursor_active = False
    scroll_active = False
    cursor_status = "Inactive"
    scroll_status = "Inactive"
    right_hand_info = None
    left_hand_info = None
    
    # Decrease click cooldown counter
    if click_cooldown_counter > 0:
        click_cooldown_counter -= 1
    
    if res.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(res.multi_hand_landmarks):
            # Draw the hand landmarks
            mp_draw.draw_landmarks(overlay, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Get handedness information
            handedness = res.multi_handedness[i].classification[0].label
            is_right = (handedness == "Right")
            is_left = (handedness == "Left")
            
            # Extract finger landmarks
            landmarks = hand_landmarks.landmark
            
            # Check finger states
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
            is_thumb_out = thumb_tip[0] > thumb_cmc[0] if is_right else thumb_tip[0] < thumb_cmc[0]
            
            # Get key finger positions
            index_tip = landmark_to_pixel(landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP], w, h)
            middle_tip = landmark_to_pixel(landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP], w, h)
            
            # Calculate center point between index and middle fingers
            center_x = (index_tip[0] + middle_tip[0]) // 2
            center_y = (index_tip[1] + middle_tip[1]) // 2
            
            # Check for pinch gesture
            pinch_ratio, thumb_tip_pos, index_tip_pos = is_pinch(landmarks, w, h)
            
            # Store hand information based on handedness
            hand_info = {
                "index_tip": index_tip,
                "middle_tip": middle_tip,
                "center_x": center_x,
                "center_y": center_y,
                "is_index_up": is_index_up,
                "is_middle_up": is_middle_up,
                "is_ring_up": is_ring_up,
                "is_pinky_up": is_pinky_up,
                "is_thumb_out": is_thumb_out,
                "pinch_ratio": pinch_ratio,
                "thumb_tip": thumb_tip_pos,
                "index_tip_pos": index_tip_pos
            }
            
            if is_right:
                right_hand_info = hand_info
            elif is_left:
                left_hand_info = hand_info
                
            # Display handedness text
            cv2.putText(overlay, f"{handedness} Hand", 
                        (center_x - 40, center_y - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Process right hand for cursor control
        if right_hand_info:
            # Activate cursor when index and middle are up (thumb position doesn't matter for cursor)
            cursor_active = (right_hand_info["is_index_up"] and 
                             right_hand_info["is_middle_up"] and 
                             not right_hand_info["is_ring_up"] and 
                             not right_hand_info["is_pinky_up"])
            
            if cursor_active:
                # Draw cursor indicator
                cv2.circle(overlay, (right_hand_info["center_x"], right_hand_info["center_y"]), 10, (0, 0, 255), -1)
                cv2.circle(overlay, (right_hand_info["center_x"], right_hand_info["center_y"]), 12, (255, 255, 255), 2)
                cv2.line(overlay, right_hand_info["index_tip"], right_hand_info["middle_tip"], (0, 255, 255), 2)
                
                # Check if hand is in control box
                in_control_box = (right_hand_info["center_x"] >= x1 and 
                                 right_hand_info["center_x"] <= x2 and 
                                 right_hand_info["center_y"] >= y1 and 
                                 right_hand_info["center_y"] <= y2)
                
                if in_control_box:
                    # Map cursor position to screen coordinates
                    nx = max(0.0, min(1.0, (right_hand_info["center_x"] - x1) / control_w))
                    ny = max(0.0, min(1.0, (right_hand_info["center_y"] - y1) / control_h))
                    
                    target_x = nx * screen_w
                    target_y = ny * screen_h
                    
                    # Apply smoothing
                    smooth_x = prev_screen_x + (target_x - prev_screen_x) / SMOOTHING
                    smooth_y = prev_screen_y + (target_y - prev_screen_y) / SMOOTHING
                    
                    try:
                        pyautogui.moveTo(smooth_x, smooth_y)
                        prev_screen_x, prev_screen_y = smooth_x, smooth_y
                        cursor_status = "Active"
                    except Exception as e:
                        print(f"Error moving cursor: {e}")
            
            # SEPARATE PINCH DETECTION - works independently of cursor control
            # Check for pinch (THUMB touching INDEX finger for click) - regardless of cursor state
            pinch_active = right_hand_info["pinch_ratio"] < PINCH_THRESHOLD
            
            # Enhanced pinch detection with cooldown and better visual feedback
            if pinch_active and not prev_pinch_active and click_cooldown_counter == 0:
                try:
                    # Perform click
                    pyautogui.click()
                    print(f"Thumb-Index Click performed! Pinch ratio: {right_hand_info['pinch_ratio']:.3f}")
                    click_cooldown_counter = CLICK_COOLDOWN
                except Exception as e:
                    print(f"Error clicking: {e}")
            
            # Visual feedback for pinch gesture (THUMB to INDEX) - always show when pinching
            if right_hand_info["pinch_ratio"] < PINCH_RELEASE_THRESHOLD:
                # Draw pinch visual indicator between THUMB and INDEX
                pinch_color = (0, 255, 0) if pinch_active else (0, 255, 255)
                line_thickness = 4 if pinch_active else 2
                
                # Draw line between thumb tip and index finger tip
                cv2.line(overlay, right_hand_info["thumb_tip"], right_hand_info["index_tip_pos"], pinch_color, line_thickness)
                cv2.circle(overlay, right_hand_info["thumb_tip"], 8, pinch_color, -1)
                cv2.circle(overlay, right_hand_info["index_tip_pos"], 8, pinch_color, -1)
                
                # Show pinch strength indicator
                pinch_strength = max(0, int((1 - right_hand_info["pinch_ratio"] / PINCH_RELEASE_THRESHOLD) * 100))
                cv2.putText(overlay, f"Thumb-Index Pinch: {pinch_strength}%", 
                            (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, pinch_color, 2)
                
                # Draw click indicator when pinch is strong enough
                if pinch_active:
                    cv2.putText(overlay, "THUMB-INDEX CLICK!", 
                                (right_hand_info["center_x"] - 100, right_hand_info["center_y"] + 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Update click status if click was performed
            if pinch_active and click_cooldown_counter > 10:  # Show click status briefly
                cursor_status = "CLICK!"
            
            prev_pinch_active = pinch_active
            
            # Show detailed pinch ratio for debugging
            cv2.putText(overlay, f"Thumb-Index Distance: {right_hand_info['pinch_ratio']:.3f}", 
                        (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Process left hand for scrolling
        if left_hand_info:
            # Activate scroll only when index and middle are up, others are down
            scroll_active = (left_hand_info["is_index_up"] and 
                            left_hand_info["is_middle_up"] and 
                            not left_hand_info["is_ring_up"] and 
                            not left_hand_info["is_pinky_up"])
            
            if scroll_active:
                # Draw scroll indicator
                cv2.circle(overlay, (left_hand_info["center_x"], left_hand_info["center_y"]), 10, (0, 255, 255), -1)
                cv2.line(overlay, left_hand_info["index_tip"], left_hand_info["middle_tip"], (0, 255, 255), 2)
                
                # Calculate and perform scroll
                if prev_scroll_y is not None:
                    # Calculate vertical movement
                    y_diff = left_hand_info["center_y"] - prev_scroll_y
                    
                    # Apply scroll only if movement exceeds threshold
                    if abs(y_diff) > SCROLL_THRESHOLD:
                        # Determine scroll direction and amount
                        scroll_amount = int(y_diff / SMOOTHING)
                        
                        # Perform scroll (positive = down, negative = up)
                        pyautogui.scroll(-scroll_amount * SCROLL_SENSITIVITY)
                        
                        # Show scroll direction
                        direction = "DOWN" if y_diff > 0 else "UP"
                        scroll_status = f"Scrolling {direction}"
                        
                        # Draw scroll direction indicator
                        arrow_start = (w - 50, h - 70)
                        arrow_end = (w - 50, h - 70 - (20 if y_diff < 0 else -20))
                        cv2.arrowedLine(overlay, arrow_start, arrow_end, (0, 0, 255), 2, tipLength=0.5)
                
                prev_scroll_y = left_hand_info["center_y"]
            else:
                prev_scroll_y = None
    
    # Display status information
    cv2.putText(overlay, f"Cursor: {cursor_status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                (0, 255, 0) if cursor_active else (0, 0, 255), 2)
    cv2.putText(overlay, f"Scroll: {scroll_status}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                (0, 255, 0) if scroll_active else (0, 0, 255), 2)
    cv2.putText(overlay, f"FPS: {fps:.1f}", (w - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Show the final image
    cv2.imshow('Multi-Gesture Control', overlay)
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
