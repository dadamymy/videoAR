import cv2
import numpy as np
import mediapipe as mp
import distinguish_area

# =========================================
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils
# =========================================
det = 0
def fitting(frame,video_frame, matches, referenceImagePts, sourceImagePts, corners, det):
    detection = True
    # Get the good key points positions
    sourcePoints = np.float32(
        [referenceImagePts[m.queryIdx].pt for m in matches]
    ).reshape(-1, 1, 2)
    destinationPoints = np.float32(
        [sourceImagePts[m.trainIdx].pt for m in matches]
    ).reshape(-1, 1, 2)
    # Obtain the homography matrix
    homography, mask = cv2.findHomography(
         sourcePoints, destinationPoints, cv2.RANSAC, 5.0
     )

    #This can know the object corners matrixes on camera frame
    transformedCorners = cv2.perspectiveTransform(corners, homography)
    # print(transformedCorners)

    # =============================
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(imgRGB)
    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)
            for num, lm in enumerate(handLms.landmark):
                if num == 8:
                    finger_m = []
                    finger_m.append(lm.x * frame.shape[1])
                    finger_m.append(lm.y * frame.shape[0])
                    det, detection = distinguish_area.get_direction(transformedCorners, finger_m, det)
                    #print(f"coordinate:x{lm.x * frame.shape[1]}, y{lm.y * frame.shape[0]} ")
        # =============================
    warp_img = cv2.warpPerspective(video_frame, homography, (frame.shape[1],frame.shape[0]))

    #mask
    new_mask = np.zeros((frame.shape[0],frame.shape[1]), np.uint8)
    cv2.fillPoly(new_mask, [np.int32(transformedCorners)], (255,0,255))
    mask_Inv = cv2.bitwise_not(new_mask)

    frame = cv2.bitwise_and(frame,frame,mask=mask_Inv)
    frame = cv2.bitwise_or(warp_img, frame)
    return frame, video_frame, detection, det
