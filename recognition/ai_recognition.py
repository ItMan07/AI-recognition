import cv2
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats import landmark_pb2
from mediapipe.python import solutions


class AiRecognition:
    def __init__(
            self,
            camera_frame_width: int = 1280,
            camera_frame_height: int = 720,
            flip_code: int = None,
            camera_id: int = 0,
            window_title: str = "Распознавание",
    ):
        # self.face_cascade = cv2.CascadeClassifier("haarcascade_fontalface_default.xml")
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        self.cam_width = camera_frame_width
        self.cam_height = camera_frame_height
        self.flip_code = flip_code
        self.window_title = window_title

        self.camera = cv2.VideoCapture(camera_id)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, camera_frame_width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_frame_height)

        model_path = "people/pose_landmarker_full.task"

        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        self.PoseLandmarker = mp.tasks.vision.PoseLandmarker

        self.options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_poses=10,
        )

    @staticmethod
    def draw_landmarks_on_image(rgb_image, detection_result):
        pose_landmarks_list = detection_result.pose_landmarks
        annotated_image = np.copy(rgb_image)
        # print(pose_landmarks_list, end="\n\n")

        for idx in range(len(pose_landmarks_list)):
            pose_landmarks = pose_landmarks_list[idx]

            # Draw the pose landmarks
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

    def face_recognition(self, img):
        """
        Распознавание лиц с изображения
        :param img:
        :return: img
        """
        image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(image_gray, 1.3, 5)

        for x, y, width, height in faces:
            cv2.rectangle(
                img, (x, y), (x + width, y + height), color=(255, 0, 0), thickness=3
            )

        img = cv2.putText(
            img,
            str(len(faces)),
            (100, 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            3,
            (0, 0, 255),
            4,
        )

        return img

    def human_recognition(self, img):
        """
        Распознавание силуэтов людей с изображения
        :param img:
        :return: img
        """
        with self.PoseLandmarker.create_from_options(self.options) as landmarker:
            img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_RGB)

            timestamps = self.camera.get(cv2.CAP_PROP_POS_MSEC)
            pose_landmarker_result = landmarker.detect_for_video(
                mp_image, int(timestamps)
            )
            num_of_people = len(pose_landmarker_result.pose_landmarks)

            annotated_image = self.draw_landmarks_on_image(
                mp_image.numpy_view(), pose_landmarker_result
            )

            annotated_image = cv2.putText(
                annotated_image,
                str(num_of_people),
                (100, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                3,
                (255, 0, 0),
                4,
            )
            img = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

            return img

    def camera_processing(self):
        """
        Получение изображения с камеры и последующая обработка
        :return: None
        """
        with self.PoseLandmarker.create_from_options(self.options) as landmarker:
            while self.camera.isOpened():
                success_code, img_bgr = self.camera.read()

                if self.flip_code is not None:
                    img_bgr = cv2.flip(img_bgr, self.flip_code)

                if not success_code:
                    cv2.waitKey()
                    print("Ошибка получения изображения с камеры")
                    break

                # img_bgr = self.face_recognition(img_bgr)
                # img_bgr = self.human_recognition(img_bgr)

                image_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(image_gray, 1.3, 5)
                for x, y, width, height in faces:
                    cv2.rectangle(
                        img_bgr, (x, y), (x + width, y + height), color=(255, 0, 0), thickness=3
                    )

                image_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                mp_image_rgb = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
                timestamps = self.camera.get(cv2.CAP_PROP_POS_MSEC)
                pose_landmarker_result = landmarker.detect_for_video(
                    mp_image_rgb, int(timestamps)
                )
                annotated_image_rgb = self.draw_landmarks_on_image(
                    mp_image_rgb.numpy_view(), pose_landmarker_result
                )

                num_of_faces = len(faces)
                num_of_people = len(pose_landmarker_result.pose_landmarks)
                img_bgr = cv2.cvtColor(annotated_image_rgb, cv2.COLOR_RGB2BGR)
                img_bgr = cv2.putText(
                    img_bgr,
                    f"People: {num_of_people} | Faces: {num_of_faces}",
                    (int(self.cam_height * 0.05), int(self.cam_width * 0.05)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (0, 0, 250),
                    4,
                )
                cv2.imshow("Изображение с камеры", img_bgr)
                if cv2.waitKey(1) == 27:
                    break

        self.camera.release()
        cv2.destroyAllWindows()

    def image_recognize(self, img_path):
        """
        Распознавание лиц с фотографии
        :param img_path:
        :return: None
        """
        img = cv2.imread(img_path)

        image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(image_gray, 1.3, 5)

        for x, y, width, height in faces:
            cv2.rectangle(
                img, (x, y), (x + width, y + height), color=(255, 0, 0), thickness=3
            )

        cv2.imshow("image", img)
        while True:
            if cv2.waitKey(1) == 27:
                break
