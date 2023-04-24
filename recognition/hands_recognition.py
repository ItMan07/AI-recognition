import cv2
import mediapipe as mp

# import assistant.voiceAssistant as va

camera = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

p = [0 for i in range(21)]  # создаем массив из 21 ячейки для хранения высоты каждой точки
finger = [0 for i in range(5)]  # создаем массив из 5 ячеек для хранения положения каждого пальца


def distance(point1, point2):
    return abs(point1 - point2)


def start_hr():
    while True:
        good, img = camera.read()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(imgRGB)
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

                for id, point in enumerate(handLms.landmark):
                    width, height, color = img.shape
                    width, height = int(point.x * height), int(point.y * width)

                    p[id] = height  # заполняем массив высотой каждой точки

                    cv2.putText(img, str(id), (width, height),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                    if id == 4:
                        cv2.circle(img, (width, height), 15, (255, 0, 255), cv2.FILLED)
                    if id == 8:
                        cv2.circle(img, (width, height), 15, (0, 255, 255), cv2.FILLED)
                    if id == 20:
                        cv2.circle(img, (width, height), 15, (255, 255, 0), cv2.FILLED)

                distanceGood = distance(p[0], p[5]) + (distance(p[0], p[5]) / 2)

                finger[1] = 1 if distance(p[0], p[8]) > distanceGood else 0
                finger[2] = 1 if distance(p[0], p[12]) > distanceGood else 0
                finger[3] = 1 if distance(p[0], p[16]) > distanceGood else 0
                finger[4] = 1 if distance(p[0], p[20]) > distanceGood else 0
                finger[0] = 1 if distance(p[4], p[17]) > distanceGood else 0

                # 0 - большой палец, 1 - указательный, 2 - средний, 3 - безымянный, 4 - мизинец

                if finger[0] and not finger[1] and not finger[2] and not finger[3] and not finger[4]:
                    print('Распознано: большой палец')
                    # va.speak("движение распознанноо")

        cv2.imshow("Image", img)

        if cv2.waitKey(1) == ord('q'):
            break
    camera.release()
    cv2.destroyAllWindows()

# start_hr()
