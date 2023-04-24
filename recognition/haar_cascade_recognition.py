import cv2

camera = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier("data/haarcascade_fontalface_default.xml")


def start_fr():
    while True:
        _, img = camera.read()

        image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(image_gray, 1.3, 5)

        for x, y, width, height in faces:
            cv2.rectangle(img, (x, y), (x + width, y + height), color=(255, 0, 0), thickness=3)
            print('Распознано: человеческое лицо')

        cv2.imshow("image", img)

        if cv2.waitKey(1) == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


# start_fr()
