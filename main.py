import cv2
import numpy as np

detection = False
match_mean = 0
times = 1
# Minimum number of matches
MIN_MATCHES = 300
# ============== Reference Image ==============
# Load reference image and convert it to gray scale
referenceImage = cv2.imread("img/sword-art-online-12.jpg", cv2.IMREAD_GRAYSCALE)
h, w = referenceImage.shape[:2]
# print(h,w)
# =============== Set Video ===================
Video_cap = cv2.VideoCapture("img/alicization-rising-steel.mp4")
success, video_frame = Video_cap.read()
# make video rotate 90 degress
video_frame = cv2.rotate(video_frame, cv2.ROTATE_90_CLOCKWISE)
# make video size as big as Target Image
video_frame = cv2.resize(video_frame, (w, h))

# ============== Recognize ================
# Initiate ORB detector
orb = cv2.ORB_create(nfeatures=1000)
# create brute force  matcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# Compute model keypoints and its descriptors
referenceImagePts, referenceImageDsc = orb.detectAndCompute(referenceImage, None)


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
while True:
    # read the current frame
    _, frame = cap.read()
    success, video_frame = Video_cap.read()

    if detection == False:
        Video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_counter = 0
    else:
        #print(Video_cap.get(cv2.CAP_PROP_FRAME_COUNT)-3)
        if frame_counter == Video_cap.get(cv2.CAP_PROP_FRAME_COUNT)-3:
            Video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_counter = 0
        # make video rotate 90 degress
        video_frame = cv2.rotate(video_frame, cv2.ROTATE_90_CLOCKWISE)
        # make video size as big as Target Image
        video_frame = cv2.resize(video_frame, (w, h))



    # Compute scene keypoints and its descriptors
    sourceImagePts, sourceImageDsc = orb.detectAndCompute(frame, None)

    # Match frame descriptors with model descriptors
    matches = bf.match(referenceImageDsc, sourceImageDsc)
    # Sort them in the order of their distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Apply the homography transformation if we have enough good matches
    if len(matches) > MIN_MATCHES:
        # match_mean+=len(matches)
        # print(f"means:{match_mean / times}\nnow:{len(matches)}")
        # times+=1

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
        # Apply the perspective transformation to the source image corners
        corners = np.float32(
            [[0, 0], [0, h], [w, h], [w, 0]]
        ).reshape(-1, 1, 2)

        #This can know the object corners matrixes on camera frame
        transformedCorners = cv2.perspectiveTransform(corners, homography)
        print(transformedCorners)

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
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()