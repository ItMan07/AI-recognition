# import cv2
# import numpy as np
# import mediapipe as mp
# import time
# import os
#
# # Подключаем камеру
# cap = cv2.VideoCapture(0)
# cap.set(3, 640)  # Width
# cap.set(4, 480)  # Lenght
# cap.set(10, 100)  # Brightness
#
# mp_drawing = mp.solutions.drawing_utils
# mp_holistic = mp.solutions.holistic
#
# pTime = 0
# cTime = 0
#
# # Зацикливаем получение кадров от камеры
# while True:
#     with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
#         ret, frame = cap.read()
#         # frame = cv2.imread("../recognized.png")
#         # Recolor Feed
#         image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         # image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         # Make Detections
#         results = holistic.process(image)
#         # print(results.face_landmarks)
#         # face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks
#         # Recolor image back to BGR for rendering
#         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#
#         # Draw face landmarks
#         # mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
#         #                           mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
#         #                           mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
#         #                           )
#         #
#         # # Right hand
#         # mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
#         #
#         # # Left Hand
#         # mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
#
#         # Pose Detections
#         mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
#
#         # cTime = time.time()
#         # fps = 1 / (cTime - pTime)
#         # pTime = cTime
#         # cv2.putText(image, str(int(fps)), (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)  # ФреймРейт
#
#         cv2.imshow('python', image)
#
#     if cv2.waitKey(1) == 27:  # exit on ESC
#         break
#
# cv2.destroyWindow("python")
# cap.release()
# cv2.waitKey(1)


import mediapipe as mp
import cv2

model_path = "pose_landmarker_lite.task"

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a pose landmarker instance with the video mode:
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_poses=10,
)

camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)

# with PoseLandmarker.create_from_options(options) as landmarker:
with mp_holistic.Holistic(
    min_detection_confidence=0.5, min_tracking_confidence=0.5
) as holistic:
    while camera.isOpened():
        ret, frame = camera.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Make Detections
        results = holistic.process(image)
        # results = landmarker.detect(image)
        print(results.pose_landmarks)

        # Recolor image back to BGR for rendering
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
        )

        cv2.imshow("Raw Webcam Feed", image)

        if cv2.waitKey(1) == 27:
            break

camera.release()
cv2.destroyAllWindows()

# while camera.isOpened():
#     success_code, img = camera.read()
#
#     if not success_code:
#         cv2.waitKey()
#         print("Ошибка получения изображения с камеры")
#         break
#
#     cv2.imshow("image", img)
#     if cv2.waitKey(1) == 27:
#         break
#
#     image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# camera.release()
# cv2.destroyAllWindows()
#
# with PoseLandmarker.create_from_options(options) as landmarker:
