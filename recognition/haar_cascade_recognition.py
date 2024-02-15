import cv2

camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

face_cascade = cv2.CascadeClassifier("haarcascade_fontalface_default.xml")


def start_fr():
    while True:
        success_code, img = camera.read()
        if not success_code:
            cv2.waitKey()
            break

        image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(image_gray, 1.3, 5)

        for x, y, width, height in faces:
            cv2.rectangle(img, (x, y), (x + width, y + height), color=(255, 0, 0), thickness=3)

        cv2.imshow("image", img)

        if cv2.waitKey(1) == 27:
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    start_fr()
