#!/usr/bin/env python3
import time
import math
import cv2
import numpy as np
import mediapipe as mp
import pyautogui

pyautogui.FAILSAFE = False

class GestureController:
    def __init__(self):
        # Configuration parameters
        self.SMOOTHING = 4 # Reduced smoothing for quicker response
        self.FINGER_DETECTION_THRESHOLD = 0.6
        self.CALIBRATION_TIME = 3.0
        self.FPS_SMOOTH = 0.9
        self.PINCH_THRESHOLD = 0.15
        self.PINCH_RELEASE_THRESHOLD = 0.25
        self.CLICK_COOLDOWN = 15
        self.SCROLL_SENSITIVITY = 30
        self.SCROLL_THRESHOLD = 10

        # MediaPipe setup (increased max_num_hands to 2 for left/right hand detection)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.75, min_tracking_confidence=0.7)
        self.mp_draw = mp.solutions.drawing_utils

        # Screen and camera setup
        self.screen_w, self.screen_h = pyautogui.size()
        self.cap = cv2.VideoCapture(0)

        # State variables
        self.prev_screen_x, self.prev_screen_y = pyautogui.position()
        self.prev_time = time.time()
        self.fps = 0.0
        self.cursor_active = False
        self.prev_scroll_y = None
        self.pinch_active = False
        self.prev_pinch_active = False
        self.click_cooldown_counter = 0
        self.x1, self.y1, self.x2, self.y2 = 0, 0, 0, 0

    def distance(self, a, b):
        """Calculate Euclidean distance between two points."""
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def landmark_to_pixel(self, lm, w, h):
        """Convert a MediaPipe landmark to pixel coordinates."""
        return int(lm.x * w), int(lm.y * h)

    def is_finger_up(self, landmarks, finger_tip_id, finger_pip_id, finger_mcp_id, w, h):
        """Check if a finger is pointing up"""
        tip = self.landmark_to_pixel(landmarks[finger_tip_id], w, h)
        pip = self.landmark_to_pixel(landmarks[finger_pip_id], w, h)
        mcp = self.landmark_to_pixel(landmarks[finger_mcp_id], w, h)
        return tip[1] < pip[1] < mcp[1]

    def is_pinch(self, landmarks, w, h):
        """Detect pinch gesture between thumb and index finger"""
        thumb_tip = self.landmark_to_pixel(landmarks[self.mp_hands.HandLandmark.THUMB_TIP], w, h)
        index_tip = self.landmark_to_pixel(landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_TIP], w, h)
        wrist = self.landmark_to_pixel(landmarks[self.mp_hands.HandLandmark.WRIST], w, h)
        middle_mcp = self.landmark_to_pixel(landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP], w, h)
        hand_size = max(1.0, self.distance(wrist, middle_mcp))
        pinch_distance = self.distance(thumb_tip, index_tip)
        pinch_ratio = pinch_distance / hand_size
        return pinch_ratio, thumb_tip, index_tip

    def calibrate_control_box(self):
        """Calibrate the control box based on hand movement."""
        start = time.time()
        xs, ys = [], []
        while time.time() - start < self.CALIBRATION_TIME:
            ret, frame = self.cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = self.hands.process(rgb)
            if res.multi_hand_landmarks:
                lm = res.multi_hand_landmarks[0]
                ix, iy = self.landmark_to_pixel(lm.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP], w, h)
                xs.append(ix)
                ys.append(iy)
            overlay = frame.copy()
            cv2.putText(overlay, f'Calibrating... move your hand around ({int(self.CALIBRATION_TIME - (time.time() - start))}s)',
                       (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow('Calibrate', overlay)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyWindow('Calibrate')

        if not xs or not ys:
            h, w = 480, 640
            ret, frame = self.cap.read()
            if ret:
                h, w, _ = frame.shape
            margin = int(min(w, h) * 0.15)
            self.x1, self.y1, self.x2, self.y2 = margin, margin, w - margin, h - margin
            return

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        pad_x = int((max_x - min_x) * 0.12) if (max_x - min_x) > 10 else int(w * 0.05)
        pad_y = int((max_y - min_y) * 0.12) if (max_y - min_y) > 10 else int(h * 0.05)
        self.x1 = max(0, min_x - pad_x)
        self.y1 = max(0, min_y - pad_y)
        self.x2 = min(w - 1, max_x + pad_x)
        self.y2 = min(h - 1, max_y + pad_y)

        if self.x2 - self.x1 < 50 or self.y2 - self.y1 < 50:
            margin = int(min(w, h) * 0.12)
            self.x1, self.y1, self.x2, self.y2 = margin, margin, w - margin, h - margin

    def print_instructions(self):
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

    def process_hands(self, frame, h, w):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.hands.process(rgb)
        right_hand_info, left_hand_info = None, None

        if res.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(res.multi_hand_landmarks):
                handedness = res.multi_handedness[i].classification[0].label
                is_right = (handedness == "Right")
                
                landmarks = hand_landmarks.landmark
                
                is_index_up = self.is_finger_up(landmarks, self.mp_hands.HandLandmark.INDEX_FINGER_TIP, self.mp_hands.HandLandmark.INDEX_FINGER_PIP, self.mp_hands.HandLandmark.INDEX_FINGER_MCP, w, h)
                is_middle_up = self.is_finger_up(landmarks, self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP, self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP, self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP, w, h)
                is_ring_up = self.is_finger_up(landmarks, self.mp_hands.HandLandmark.RING_FINGER_TIP, self.mp_hands.HandLandmark.RING_FINGER_PIP, self.mp_hands.HandLandmark.RING_FINGER_MCP, w, h)
                is_pinky_up = self.is_finger_up(landmarks, self.mp_hands.HandLandmark.PINKY_TIP, self.mp_hands.HandLandmark.PINKY_PIP, self.mp_hands.HandLandmark.PINKY_MCP, w, h)

                index_tip = self.landmark_to_pixel(landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_TIP], w, h)
                middle_tip = self.landmark_to_pixel(landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP], w, h)
                center_x = (index_tip[0] + middle_tip[0]) // 2
                center_y = (index_tip[1] + middle_tip[1]) // 2

                pinch_ratio, thumb_tip_pos, index_tip_pos = self.is_pinch(landmarks, w, h)

                hand_info = {
                    "landmarks": hand_landmarks,
                    "index_tip": index_tip, "middle_tip": middle_tip,
                    "center_x": center_x, "center_y": center_y,
                    "is_index_up": is_index_up, "is_middle_up": is_middle_up,
                    "is_ring_up": is_ring_up, "is_pinky_up": is_pinky_up,
                    "pinch_ratio": pinch_ratio, "thumb_tip": thumb_tip_pos,
                    "index_tip_pos": index_tip_pos
                }

                if is_right:
                    right_hand_info = hand_info
                else:
                    left_hand_info = hand_info
        
        return right_hand_info, left_hand_info

    def handle_right_hand(self, right_hand_info, overlay, h, w):
        cursor_status = "Inactive"
        if not right_hand_info:
            return cursor_status

        self.cursor_active = (right_hand_info["is_index_up"] and
                              right_hand_info["is_middle_up"] and
                              not right_hand_info["is_ring_up"] and
                              not right_hand_info["is_pinky_up"]) # Removed thumb check for cursor activation

        if self.cursor_active:
            cursor_status = self.move_cursor(right_hand_info, overlay, w, h)

        self.handle_pinch_click(right_hand_info, overlay, h)
        
        if self.pinch_active and self.click_cooldown_counter > 10:
            cursor_status = "CLICK!"
            
        return cursor_status

    def move_cursor(self, hand_info, overlay, w, h):
        center_x, center_y = hand_info["center_x"], hand_info["center_y"]
        cv2.circle(overlay, (center_x, center_y), 10, (0, 0, 255), -1)
        cv2.circle(overlay, (center_x, center_y), 12, (255, 255, 255), 2)
        cv2.line(overlay, hand_info["index_tip"], hand_info["middle_tip"], (0, 255, 255), 2)

        control_w = max(1, self.x2 - self.x1)
        control_h = max(1, self.y2 - self.y1)

        in_control_box = (self.x1 <= center_x <= self.x2 and self.y1 <= center_y <= self.y2)
        if not in_control_box:
            return "Inactive"

        nx = max(0.0, min(1.0, (center_x - self.x1) / control_w))
        ny = max(0.0, min(1.0, (center_y - self.y1) / control_h))
        target_x, target_y = nx * self.screen_w, ny * self.screen_h

        smooth_x = self.prev_screen_x + (target_x - self.prev_screen_x) / self.SMOOTHING
        smooth_y = self.prev_screen_y + (target_y - self.prev_screen_y) / self.SMOOTHING

        try:
            pyautogui.moveTo(smooth_x, smooth_y)
            self.prev_screen_x, self.prev_screen_y = smooth_x, smooth_y
            return "Active"
        except Exception as e:
            print(f"Error moving cursor: {e}")
            return "Error"

    def handle_pinch_click(self, hand_info, overlay, h):
        self.pinch_active = hand_info["pinch_ratio"] < self.PINCH_THRESHOLD
        if self.pinch_active and not self.prev_pinch_active and self.click_cooldown_counter == 0:
            try:
                pyautogui.click() # Perform a left click
                print(f"Thumb-Index Click performed! Pinch ratio: {hand_info['pinch_ratio']:.3f}")
                self.click_cooldown_counter = self.CLICK_COOLDOWN
            except Exception as e:
                print(f"Error clicking: {e}")

        if hand_info["pinch_ratio"] < self.PINCH_RELEASE_THRESHOLD:
            pinch_color = (0, 255, 0) if self.pinch_active else (0, 255, 255)
            line_thickness = 4 if self.pinch_active else 2
            cv2.line(overlay, hand_info["thumb_tip"], hand_info["index_tip_pos"], pinch_color, line_thickness)
            cv2.circle(overlay, hand_info["thumb_tip"], 8, pinch_color, -1)
            cv2.circle(overlay, hand_info["index_tip_pos"], 8, pinch_color, -1)
            
            pinch_strength = max(0, int((1 - hand_info["pinch_ratio"] / self.PINCH_RELEASE_THRESHOLD) * 100))
            cv2.putText(overlay, f"Thumb-Index Pinch: {pinch_strength}%", (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, pinch_color, 2)
            if self.pinch_active:
                cv2.putText(overlay, "THUMB-INDEX CLICK!", (hand_info["center_x"] - 100, hand_info["center_y"] + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        self.prev_pinch_active = self.pinch_active
        cv2.putText(overlay, f"Thumb-Index Distance: {hand_info['pinch_ratio']:.3f}", (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def handle_left_hand(self, left_hand_info, overlay, h, w):
        scroll_status = "Inactive"
        if not left_hand_info:
            self.prev_scroll_y = None
            return scroll_status

        scroll_active = (left_hand_info["is_index_up"] and
                         left_hand_info["is_middle_up"] and
                         not left_hand_info["is_ring_up"] and # Ensure ring and pinky are down
                         not left_hand_info["is_pinky_up"]) # for scroll activation

        if scroll_active:
            scroll_status = self.perform_scroll(left_hand_info, overlay, h, w)
        else:
            self.prev_scroll_y = None
            
        return scroll_status

    def perform_scroll(self, hand_info, overlay, h, w):
        center_x, center_y = hand_info["center_x"], hand_info["center_y"]
        cv2.circle(overlay, (center_x, center_y), 10, (0, 255, 255), -1)
        cv2.line(overlay, hand_info["index_tip"], hand_info["middle_tip"], (0, 255, 255), 2)

        if self.prev_scroll_y is not None:
            y_diff = center_y - self.prev_scroll_y
            if abs(y_diff) > self.SCROLL_THRESHOLD:
                scroll_amount = int(y_diff / self.SMOOTHING) # Use smoothing for scroll amount
                pyautogui.scroll(-scroll_amount * self.SCROLL_SENSITIVITY)
                direction = "DOWN" if y_diff > 0 else "UP"
                
                arrow_start = (w - 50, h - 70)
                arrow_end = (w - 50, h - 70 - (20 if y_diff < 0 else -20))
                cv2.arrowedLine(overlay, arrow_start, arrow_end, (0, 0, 255), 2, tipLength=0.5)
                self.prev_scroll_y = center_y
                return f"Scrolling {direction}"

        if self.prev_scroll_y is None:
            self.prev_scroll_y = center_y
        
        return "Active"

    def run(self):
        self.print_instructions()
        self.calibrate_control_box()

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            overlay = frame.copy()

            # FPS calculation
            now = time.time()
            instant_fps = 1.0 / max(0.001, now - self.prev_time)
            self.fps = self.fps * self.FPS_SMOOTH + instant_fps * (1.0 - self.FPS_SMOOTH)
            self.prev_time = now

            if self.click_cooldown_counter > 0:
                self.click_cooldown_counter -= 1

            right_hand_info, left_hand_info = self.process_hands(frame, h, w)
            
            if right_hand_info:
                self.mp_draw.draw_landmarks(overlay, right_hand_info["landmarks"], self.mp_hands.HAND_CONNECTIONS)
            if left_hand_info:
                self.mp_draw.draw_landmarks(overlay, left_hand_info["landmarks"], self.mp_hands.HAND_CONNECTIONS)

            cursor_status = self.handle_right_hand(right_hand_info, overlay, h, w)
            scroll_status = self.handle_left_hand(left_hand_info, overlay, h, w)

            # Display status
            cv2.rectangle(overlay, (self.x1, self.y1), (self.x2, self.y2), (0, 255, 0), 2)
            cv2.putText(overlay, "Control Area", (self.x1, self.y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(overlay, f"Cursor: {cursor_status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if self.cursor_active else (0, 0, 255), 2)
            cv2.putText(overlay, f"Scroll: {scroll_status}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if scroll_status != "Inactive" else (0, 0, 255), 2)
            cv2.putText(overlay, f"FPS: {self.fps:.1f}", (w - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.imshow('Multi-Gesture Control', overlay)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('c'):
                self.calibrate_control_box()

        self.cap.release()
        cv2.destroyAllWindows()
        print("Application closed.")

if __name__ == '__main__':
    controller = GestureController()
    controller.run()