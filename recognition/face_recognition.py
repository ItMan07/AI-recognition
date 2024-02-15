import cv2


class AiFace:
    def __init__(
        self,
        camera_frame_width: int = 1280,
        camera_frame_height: int = 720,
        flip_code: int = None,
        camera_id: int = 0,
        window_title: str = "Распознавание лиц",
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

    def processing(self):
        while True:
            success_code, img = self.camera.read()

            if self.flip_code is not None:
                img = cv2.flip(img, self.flip_code)

            if not success_code:
                cv2.waitKey()
                print("Ошибка получения изображения с камеры")
                break

            image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(image_gray, 1.3, 5)

            for x, y, width, height in faces:
                cv2.rectangle(
                    img, (x, y), (x + width, y + height), color=(255, 0, 0), thickness=3
                )

            cv2.imshow("image", img)
            if cv2.waitKey(1) == 27:
                break

        self.camera.release()
        cv2.destroyAllWindows()
