import cv2
import numpy as np
from mediapipe.python import solutions

camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2

model_path = "pose_landmarker_full.task"

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


def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    # print(pose_landmarks_list, end="\n\n")

    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        # Draw the pose landmarks.
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend(
            [
                landmark_pb2.NormalizedLandmark(
                    x=landmark.x, y=landmark.y, z=landmark.z
                )
                for landmark in pose_landmarks
            ]
        )
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style(),
        )

    return annotated_image


with PoseLandmarker.create_from_options(options) as landmarker:
    while camera.isOpened():
        success_code, frame = camera.read()

        if not success_code:
            cv2.waitKey()
            print("Ошибка получения изображения с камеры")
            break

        cv2.imshow("window", frame)

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

        timestamps = camera.get(cv2.CAP_PROP_POS_MSEC)
        pose_landmarker_result = landmarker.detect_for_video(mp_image, int(timestamps))
        print(len(pose_landmarker_result.pose_landmarks), end="\n\n")

        annotated_image = draw_landmarks_on_image(
            mp_image.numpy_view(), pose_landmarker_result
        )
        cv2.imshow("window", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

        if cv2.waitKey(1) == 27:
            break

    camera.release()
    cv2.destroyAllWindows()
