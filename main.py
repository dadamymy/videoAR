import cv2
import numpy as np
import distinguish_area
import mediapipe as mp

# =========================================
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils
# =========================================
det = 0
detection = False
match_mean = 0
times = 1
# Minimum number of matches
MIN_MATCHES = 260
# ============== Reference Image Asuna ==============
# Load reference image and convert it to gray scale
referenceImage = cv2.imread("img/asuna_yuuki__ref_img.jpg", cv2.IMREAD_GRAYSCALE)
h, w = referenceImage.shape[:2]

# ============== Reference Image Kirito =============
referenceImage_kirito = cv2.imread("img/kirito_ref_img.jpg", cv2.IMREAD_GRAYSCALE)
h_k, w_k = referenceImage_kirito[:2]

# =============== Set Video Asuna ===================
Video_cap = cv2.VideoCapture("img/kirito.mp4")
_, video_frame = Video_cap.read()

video_cap1 = cv2.VideoCapture("img/happy_asuna.mp4") # Top

video_cap2 = cv2.VideoCapture("img/innocence.mp4") # Bottom

video_cap3 = cv2.VideoCapture("img/unhappy_asuna.mp4") # Right

video_cap4 = cv2.VideoCapture("img/ADAMAS.mp4") # Left

# =============== Set Video Kirito ===================
Video_cap_kirito = cv2.VideoCapture("img/kirito.mp4")




# ============== Recognize ================
# Initiate ORB detector
orb = cv2.ORB_create(nfeatures=1000)
# create brute force  matcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# Compute model keypoints and its descriptors
referenceImagePts, referenceImageDsc = orb.detectAndCompute(referenceImage, None)
referenceImagePts_kirito, referenceImageDsc_kirito = orb.detectAndCompute(referenceImage_kirito, None)

# on laptop
cap = cv2.VideoCapture(1)

# on pc
# cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
while True:
    # read the current frame
    _, frame = cap.read()

    # Compute scene keypoints and its descriptors
    sourceImagePts, sourceImageDsc = orb.detectAndCompute(frame, None)

    # Match frame descriptors with model descriptors
    matches_asuna = bf.match(referenceImageDsc, sourceImageDsc)
    # matches_kirito = bf.match(referenceImageDsc_kirito, sourceImageDsc)
    # Sort them in the order of their distance
    matches_asuna = sorted(matches_asuna, key=lambda x: x.distance)
    # matches_kirito = sorted(matches_kirito, Key=lambda x: x.distance)


    # test press button
    # if cv2.waitKey(1) & 0xFF == ord("w"):
    #     detection = False
    #     det = 1
    # elif cv2.waitKey(1) & 0xFF == ord("s"):
    #     detection = False
    #     det = 2
    # elif cv2.waitKey(1) & 0xFF == ord("a"):
    #     detection = False
    #     det = 3
    # elif cv2.waitKey(1) & 0xFF == ord("d"):
    #     detection = False
    #     det = 4
    # elif cv2.waitKey(1) & 0xFF == ord("o"):
    #     detection = False
    #     det = 0

    # Change Video
    if det == "up":
        _, video_frame = video_cap1.read()
        if detection == False:
            video_cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_counter = 0
        else:
            # print(Video_cap.get(cv2.CAP_PROP_FRAME_COUNT)-3)
            if frame_counter == video_cap1.get(cv2.CAP_PROP_FRAME_COUNT) - 3:
                video_cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_counter = 0
                det=0
            # make video rotate 90 degress
            # video_frame = cv2.rotate(video_frame, cv2.ROTATE_90_CLOCKWISE)
            # make video size as big as Target Image
            video_frame = cv2.resize(video_frame, (w, h))
    elif det == "down":
        _, video_frame = video_cap2.read()
        if detection == False:
            video_cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_counter = 0
        else:
            # print(Video_cap.get(cv2.CAP_PROP_FRAME_COUNT)-3)
            if frame_counter == video_cap2.get(cv2.CAP_PROP_FRAME_COUNT) - 3:
                video_cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_counter = 0
                det=0
            # make video rotate 90 degress
            video_frame = cv2.rotate(video_frame, cv2.ROTATE_90_CLOCKWISE)
            # make video size as big as Target Image
            video_frame = cv2.resize(video_frame, (w, h))
    elif det == "left":
        _, video_frame = video_cap3.read()
        if detection == False:
            video_cap3.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_counter = 0
        else:
            # print(Video_cap.get(cv2.CAP_PROP_FRAME_COUNT)-3)
            if frame_counter == video_cap3.get(cv2.CAP_PROP_FRAME_COUNT) - 3:
                video_cap3.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_counter = 0
                det=0
                print("hi")
            # make video rotate 90 degress
            # video_frame = cv2.rotate(video_frame, cv2.ROTATE_90_CLOCKWISE)
            # make video size as big as Target Image
            video_frame = cv2.resize(video_frame, (w, h))
    elif det == "right":
        _, video_frame = video_cap4.read()
        if detection == False:
            video_cap4.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_counter = 0
        else:
            # print(Video_cap.get(cv2.CAP_PROP_FRAME_COUNT)-3)
            if frame_counter == video_cap4.get(cv2.CAP_PROP_FRAME_COUNT) - 3:
                video_cap4.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_counter = 0
                det=0
            # make video rotate 90 degress
            video_frame = cv2.rotate(video_frame, cv2.ROTATE_90_CLOCKWISE)
            # make video size as big as Target Image
            video_frame = cv2.resize(video_frame, (w, h))
    elif det == 0:
        _, video_frame = Video_cap.read()
        if detection == False:
            Video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_counter = 0
        else:
            # print(Video_cap.get(cv2.CAP_PROP_FRAME_COUNT)-3)
            if frame_counter == Video_cap.get(cv2.CAP_PROP_FRAME_COUNT) - 3:
                Video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_counter = 0
            # make video rotate 90 degress
            # video_frame = cv2.rotate(video_frame, cv2.ROTATE_90_CLOCKWISE)
            # make video size as big as Target Image
            video_frame = cv2.resize(video_frame, (w, h))

    # Apply the homography transformation if we have enough good matches
    if len(matches_asuna) > MIN_MATCHES:
        # match_mean+=len(matches)
        # print(f"means:{match_mean / times}\nnow:{len(matches)}")
        # times+=1

        detection = True
        # Get the good key points positions
        sourcePoints = np.float32(
            [referenceImagePts[m.queryIdx].pt for m in matches_asuna]
        ).reshape(-1, 1, 2)
        destinationPoints = np.float32(
            [sourceImagePts[m.trainIdx].pt for m in matches_asuna]
        ).reshape(-1, 1, 2)
        # Obtain the homography matrix
        homography, mask = cv2.findHomography(
            sourcePoints, destinationPoints, cv2.RANSAC, 5.0
        )
        # Apply the perspective transformation to the source image corners
        corners = np.float32(
            [[0, 0], [0, h], [w, h], [w, 0]]
        ).reshape(-1, 1, 2)

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

    # ===================== Display ====================
    # show result
    frame_counter += 1
    cv2.imshow("video", video_frame)
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()