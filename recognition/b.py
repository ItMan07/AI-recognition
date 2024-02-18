import cv2
import mediapipe as mp


def find_people_silhouettes(image):
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic()

    # Преобразование изображения в черно-белый формат
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Обнаружение ключевых точек
    results = holistic.process(gray)

    # Отрисовка силуэтов
    if results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            h, w, _ = image.shape
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(image, (cx, cy), 5, (0, 255, 0), -1)

    cv2.imshow("People Silhouettes", image)


# Запуск веб-камеры
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    find_people_silhouettes(frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
