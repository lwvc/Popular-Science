import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import time
import numpy as np

def result_callback(result, output_image, timestamp_ms):
    global latest_gesture, latest_landmarks

    if result.gestures and len(result.gestures[0]) > 0:
        latest_gesture = result.gestures[0][0]
    else:
        latest_gesture = None

    if result.hand_landmarks and len(result.hand_landmarks) > 0:
        latest_landmarks = result.hand_landmarks[0]
    else:
        latest_landmarks = None

def draw_hand_skeleton(image, landmarks):
    height, width, _ = image.shape
    points = [(int(lm.x * width), int(lm.y * height)) for lm in landmarks]

    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9,10), (10,11), (11,12),
        (0,13), (13,14), (14,15), (15,16),
        (0,17), (17,18), (18,19), (19,20),
        (5, 9), (9,13), (13,17), (17,5)
    ]

    for start_idx, end_idx in HAND_CONNECTIONS:
        cv2.line(image, points[start_idx], points[end_idx], (0, 255, 0), 2)
    for point in points:
        cv2.circle(image, point, 4, (0, 0, 255), -1)

def hand_center(image, landmarks):
    height, width, _ = image.shape
    lm_dw = landmarks[0]
    lm_up = landmarks[9]
    hand_x = int(((lm_dw.x + lm_up.x) / 2) * width)
    hand_y = int(((lm_dw.y + lm_up.y) / 2) * height)
    cv2.circle(image, (hand_x, hand_y), 8, (0, 255, 255), -1)
    return hand_x, hand_y

# Mediapipe 設定
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

model_path = 'model/gesture_recognizer.task'
options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=result_callback
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

latest_gesture = None
latest_landmarks = None

with GestureRecognizer.create_from_options(options) as recognizer:
    while True:
        ret, img = cap.read()
        if not ret:
            print("Cannot receive frame")
            break

        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        recognizer.recognize_async(mp_image, int(time.time() * 1000))

        # 畫出手部骨架與掌心中心點
        if latest_landmarks:
            draw_hand_skeleton(img, latest_landmarks)
            hand_center(img, latest_landmarks)

        # 顯示手勢辨識名稱
        if latest_gesture:
            cv2.putText(img, latest_gesture.category_name, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # 顯示畫面（放大顯示）
        img_display = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
        cv2.imshow('oxxostudio', img_display)

        if cv2.waitKey(1) == 27:  # ESC 鍵離開
            break

cap.release()
cv2.destroyAllWindows()
