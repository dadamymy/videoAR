import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

while True:
    ret, img = cap.read()
    # print(f"frame size0:{img.shape[0]}")
    # print(f"frame size1:{img.shape[1]}")
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(imgRGB)
    #print(result.multi_hand_landmarks)
    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms,mpHands.HAND_CONNECTIONS)
            for num, lm in enumerate(handLms.landmark):
                if num == 8:
                    print(f"coordinate:x{lm.x*img.shape[1]}, y{lm.y*img.shape[0]} ")

    cv2.imshow('img', img)
    cv2.waitKey(1)