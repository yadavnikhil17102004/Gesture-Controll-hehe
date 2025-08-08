import cv2
import mediapipe as mp
import pyautogui

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

screen_w, screen_h = pyautogui.size()

cap = cv2.VideoCapture(0)

GAP = 100  # margin from edges
SMOOTHING = 5  # higher = smoother but slower

# Previous cursor position for smoothing
prev_x, prev_y = 0, 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    CONTROL_BOX_X = GAP
    CONTROL_BOX_Y = GAP
    CONTROL_BOX_W = w - (GAP * 2)
    CONTROL_BOX_H = h - (GAP * 2)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    cv2.rectangle(frame, (CONTROL_BOX_X, CONTROL_BOX_Y),
                  (CONTROL_BOX_X + CONTROL_BOX_W, CONTROL_BOX_Y + CONTROL_BOX_H),
                  (0, 255, 0), 2)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            finger_x = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * w)
            finger_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * h)

            if (CONTROL_BOX_X <= finger_x <= CONTROL_BOX_X + CONTROL_BOX_W and
                CONTROL_BOX_Y <= finger_y <= CONTROL_BOX_Y + CONTROL_BOX_H):

                mapped_x = (finger_x - CONTROL_BOX_X) / CONTROL_BOX_W * screen_w
                mapped_y = (finger_y - CONTROL_BOX_Y) / CONTROL_BOX_H * screen_h

                # Smoothing formula
                smooth_x = prev_x + (mapped_x - prev_x) / SMOOTHING
                smooth_y = prev_y + (mapped_y - prev_y) / SMOOTHING

                pyautogui.moveTo(smooth_x, smooth_y)

                prev_x, prev_y = smooth_x, smooth_y

    cv2.imshow("Hand Mouse Control", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
