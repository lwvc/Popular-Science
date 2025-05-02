import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import time
import random
import numpy as np

# 遊戲時間秒數
play_time = 30
high_score = 0  # 全域最高分紀錄

# 儲存即時手勢與關鍵點
latest_gesture = None
latest_landmarks = None

# Mediapipe 初始化
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

model_path = 'model/gesture_recognizer.task'
options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=lambda result, output_image, timestamp_ms: result_callback(result)
)

def result_callback(result):
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

def practice_mode(cap):
    global latest_gesture, latest_landmarks
    ball_x, ball_y = random.randint(50, 590), 180
    ball_radius = 20
    holding_ball = False

    with GestureRecognizer.create_from_options(options) as recognizer:
        while True:
            ret, img = cap.read()
            if not ret:
                break
            img = cv2.flip(img, 1)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            recognizer.recognize_async(mp_image, int(time.time() * 1000))

            frame_h, frame_w = img.shape[:2]
            cv2.circle(img, (ball_x, ball_y), ball_radius, (200, 200, 200), -1)
            cv2.putText(img, "Practice Mode - Press Enter to Start Game", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if latest_gesture:
                cv2.putText(img, f"Gesture: {latest_gesture.category_name}", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            if latest_landmarks:
                draw_hand_skeleton(img, latest_landmarks)
                hand_x, hand_y = hand_center(img, latest_landmarks)
                distance = ((hand_x - ball_x) ** 2 + (hand_y - ball_y) ** 2) ** 0.5

                if latest_gesture and latest_gesture.category_name == 'Closed_Fist':
                    if distance < ball_radius + 30:
                        holding_ball = True
                else:
                    holding_ball = False

                if holding_ball:
                    ball_x, ball_y = hand_x, hand_y

            img_display = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
            cv2.imshow("oxxostudio", img_display)
            key = cv2.waitKey(1)
            if key == 13 or key == 27:
                break

    return key

def game_mode(cap):
    global latest_gesture, latest_landmarks, high_score
    score = 0
    holding_ball = False
    ball_radius = 20
    ball_speed = 5
    ball_x = random.randint(50, 590)
    ball_y = 0

    with GestureRecognizer.create_from_options(options) as recognizer:
        start_time = time.time()
        while True:
            ret, img = cap.read()
            if not ret:
                break
            img = cv2.flip(img, 1)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            recognizer.recognize_async(mp_image, int(time.time() * 1000))

            frame_h, frame_w = img.shape[:2]
            box_w, box_h = 160, 100
            box_x1 = frame_w // 2 - box_w // 2
            box_y1 = frame_h - box_h
            box_x2 = frame_w // 2 + box_w // 2
            box_y2 = frame_h

            elapsed_time = time.time() - start_time
            remaining_time = max(0, int(play_time - elapsed_time))

            cv2.rectangle(img, (box_x1, box_y1), (box_x2, box_y2), (0, 255, 255), 3)
            cv2.putText(img, 'Drop Here!', (box_x1, box_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
            cv2.putText(img, f'Score: {score}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.putText(img, f'High Score: {high_score}', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)
            cv2.putText(img, f'Time: {remaining_time}s', (frame_w - 180, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

            if not holding_ball:
                ball_y += ball_speed
                if ball_y > frame_h:
                    ball_x = random.randint(50, frame_w - 50)
                    ball_y = 0
            cv2.circle(img, (ball_x, ball_y), ball_radius, (255, 0, 0), -1)

            if latest_landmarks:
                draw_hand_skeleton(img, latest_landmarks)
                hand_x, hand_y = hand_center(img, latest_landmarks)
                distance = ((hand_x - ball_x) ** 2 + (hand_y - ball_y) ** 2) ** 0.5
                if distance < ball_radius + 30:
                    if latest_gesture and latest_gesture.category_name == 'Closed_Fist':
                        holding_ball = True
                        ball_x, ball_y = hand_x, hand_y
                else:
                    holding_ball = False

            if holding_ball and box_x1 < ball_x < box_x2 and box_y1 < ball_y < box_y2:
                score += 1
                holding_ball = False
                ball_x = random.randint(50, frame_w - 50)
                ball_y = 0

            if elapsed_time >= play_time:
                if score > high_score:
                    high_score = score
                    print(f"\U0001f389 新高分紀錄！High Score: {high_score}")
                print(f"\u23F0 時間到！得分: {score}")
                cv2.putText(img, "Game Over - Press Enter to Restart", (80, frame_h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
                cv2.imshow('oxxostudio', img)
                cv2.waitKey(0)
                break

            cv2.imshow('oxxostudio', img)
            if cv2.waitKey(1) == 27:
                break

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    # ✅ 設定全螢幕視窗一次即可
    cv2.namedWindow("oxxostudio", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("oxxostudio", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        # 顯示起始畫面
        start_img = 255 * np.ones((360, 640, 3), dtype=np.uint8)
        cv2.putText(start_img, "Press Enter to Start", (130, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        cv2.imshow("oxxostudio", start_img)
        key = cv2.waitKey(1)

        if key == 13:
            key = practice_mode(cap)
            if key == 13:
                game_mode(cap)
        elif key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
