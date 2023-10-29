import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

while True:
    ret, frame = cap.read()
    # print(f"frame size0:{img.shape[0]}")
    # print(f"frame size1:{img.shape[1]}")
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(imgRGB)
    #print(result.multi_hand_landmarks)
    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mpDraw.draw_landmarks(frame, handLms,mpHands.HAND_CONNECTIONS)
            for num, lm in enumerate(handLms.landmark):
                if num == 8:
                    lm.x*frame.shape[1]
                    print(f"coordinate:x{lm.x*frame.shape[1]}, y{lm.y*frame.shape[0]} ")

    cv2.imshow('img', frame)
    cv2.waitKey(1)