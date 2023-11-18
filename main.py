import cv2
import numpy as np
from video_fit import fitting
from ffpyplayer.player import MediaPlayer


a = 1
asuna_total= 0
b = 1
kirito_total= 0

det = 0
detection = False
match_mean = 0
times = 1
# Minimum number of matches
MIN_MATCHES = 280
# ============== Reference Image Asuna ==============
# Load reference image and convert it to gray scale
referenceImage = cv2.imread("img/asuna_yuuki__ref_img.jpg")
h, w = referenceImage.shape[:2]

# ============== Reference Image Kirito =============
referenceImage_kirito = cv2.imread("img/kirito_ref_img.jpg")
h_k, w_k = referenceImage_kirito.shape[:2]

# =============== Set Video Asuna ===================
Video_cap = cv2.VideoCapture("img/asuna_dance.mp4")
_, video_frame = Video_cap.read()

video_cap1 = cv2.VideoCapture("img/react_asuna.mp4") # Top
video_cap2 = cv2.VideoCapture("img/asuna_glare.mp4") # Bottom
video_cap3 = cv2.VideoCapture("img/asuna_smile.mp4") # Left
video_cap4 = cv2.VideoCapture("img/happy_asuna.mp4") # Right

# =============== Set Video Kirito ===================
Video_cap_kirito = cv2.VideoCapture("img/kirito_drink_tea.mp4")

video_cap_kirito1 = cv2.VideoCapture("img/starburst_stream.mp4") # Top
video_cap_kirito2 = cv2.VideoCapture("img/shy_kirito.mp4") # Left
video_cap_kirito3 = cv2.VideoCapture("img/suprise_kirito.mp4") # Right
video_cap_kirito4 = cv2.VideoCapture("img/angry_kirito.mp4") # Bottom


# ============== Recognize ================
# Initiate ORB detector
orb = cv2.ORB_create(nfeatures=1000)
orb1 = cv2.ORB_create(nfeatures=1000)
# create brute force  matcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
bf1 = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# Compute model keypoints and its descriptors
referenceImagePts, referenceImageDsc = orb.detectAndCompute(referenceImage, None)
referenceImagePts_kirito, referenceImageDsc_kirito = orb1.detectAndCompute(referenceImage_kirito, None)

# on laptop
# cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

# on pc
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
while True:
    # read the current frame
    _, frame = cap.read()

    # Compute scene keypoints and its descriptors
    sourceImagePts, sourceImageDsc = orb.detectAndCompute(frame, None)
    try:
        # Match frame descriptors with model descriptors
        matches_asuna = bf.match(referenceImageDsc, sourceImageDsc)
        matches_kirito = bf1.match(referenceImageDsc_kirito, sourceImageDsc)
    except:
        # Match frame descriptors with model descriptors
        matches_kirito = bf1.match(referenceImageDsc_kirito, sourceImageDsc)
        matches_asuna = bf.match(referenceImageDsc, sourceImageDsc)
    # Sort them in the order of their distance
    matches_asuna = sorted(matches_asuna, key=lambda x: x.distance)
    matches_kirito = sorted(matches_kirito, key=lambda x: x.distance)

    # ===============================================
    # Here is counting good match points average
    # print("=====asuna========")
    # print(len(matches_asuna))
    asuna_total+=len(matches_asuna)
    mean_asuna = asuna_total/a
    # print(mean_asuna)
    a+=1
    if len(matches_asuna)<MIN_MATCHES:
        asuna_total=0
        a=1


    # print("=====kirito=======")
    # print(len(matches_kirito))
    kirito_total += len(matches_kirito)
    mean_kirito = kirito_total / b
    # print(mean_kirito)
    b += 1
    if len(matches_kirito) < MIN_MATCHES:
        kirito_total = 0
        b = 1
    # =========================================

    # Apply the homography transformation if we have enough good matches
    if mean_asuna > MIN_MATCHES:
        if a == 1:
            detection=False
        # Apply the perspective transformation to the source image corners
        corners = np.float32(
            [[0, 0], [0, h], [w, h], [w, 0]]
        ).reshape(-1, 1, 2)
        # Change Video
        if det == "up":
            _, video_frame = video_cap1.read()
            video_frame = cv2.resize(video_frame, (w, h))
            if detection == False:
                video_cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_counter = 0
                detection=True
            else:
                # print(Video_cap.get(cv2.CAP_PROP_FRAME_COUNT)-3)
                if frame_counter == video_cap1.get(cv2.CAP_PROP_FRAME_COUNT) - 3:
                    video_cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    frame_counter = 0
                    det = 0
                    detection = False
                else:
                    video_frame = cv2.resize(video_frame, (w, h))
        elif det == "down":
            _, video_frame = video_cap2.read()
            video_frame = cv2.resize(video_frame, (w, h))
            if detection == False:
                video_cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_counter = 0
                detection=True
            else:
                # print(Video_cap.get(cv2.CAP_PROP_FRAME_COUNT)-3)
                if frame_counter == video_cap2.get(cv2.CAP_PROP_FRAME_COUNT) - 3:
                    video_cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    frame_counter = 0
                    det = 0
                    detection = False
                else:
                    # make video size as big as Target Image
                    video_frame = cv2.resize(video_frame, (w, h))
        elif det == "left":
            _, video_frame = video_cap3.read()
            video_frame = cv2.rotate(video_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            # make video size as big as Target Image
            video_frame = cv2.resize(video_frame, (w, h))
            if detection == False:
                video_cap3.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_counter = 0
                detection=True
            else:
                # print(Video_cap.get(cv2.CAP_PROP_FRAME_COUNT)-3)
                if frame_counter == video_cap3.get(cv2.CAP_PROP_FRAME_COUNT) - 3:
                    video_cap3.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    frame_counter = 0
                    det = 0
                    detection = False
                else:
                    # make video size as big as Target Image
                    video_frame = cv2.resize(video_frame, (w, h))
        elif det == "right":
            _, video_frame = video_cap4.read()
            # make video size as big as Target Image size
            video_frame = cv2.resize(video_frame, (w, h))
            if detection == False:
                video_cap4.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_counter = 0
                detection=True
            else:
                # print(Video_cap.get(cv2.CAP_PROP_FRAME_COUNT)-3)
                if frame_counter == video_cap4.get(cv2.CAP_PROP_FRAME_COUNT) - 3:
                    video_cap4.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    frame_counter = 0
                    det = 0
                    detection = False
                else:
                    # make video size as big as Target Image size
                    video_frame = cv2.resize(video_frame, (w, h))
        elif det == 0:
            _, video_frame = Video_cap.read()
            # make video rotate 90 degress
            video_frame = cv2.rotate(video_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            video_frame = cv2.resize(video_frame, (w, h))
            if detection == False:
                Video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_counter = 0
                detection=True
            else:
                # print(Video_cap.get(cv2.CAP_PROP_FRAME_COUNT)-3)
                if frame_counter == Video_cap.get(cv2.CAP_PROP_FRAME_COUNT) - 4:
                    Video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    frame_counter = 0
                    detection=False
                else:
                    video_frame = cv2.resize(video_frame, (w, h))
        frame, video_frame, detection, det = fitting(frame, video_frame, matches_asuna, referenceImagePts, sourceImagePts, corners, det, detection)
        cv2.imshow("video", video_frame)
        frame_counter += 1

    elif mean_kirito>MIN_MATCHES:
        if b == 1:
            detection=False
        # Apply the perspective transformation to the source image corners
        corners = np.float32(
            [[0, 0], [0, h_k], [w_k, h_k], [w_k, 0]]
        ).reshape(-1, 1, 2)
        # Change Video
        if det == "up":
            _, video_frame = video_cap_kirito1.read()
            video_frame = cv2.resize(video_frame, (w_k, h_k))
            if detection == False:
                video_cap_kirito1.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_counter = 0
                detection = True
            else:
                if frame_counter == video_cap_kirito1.get(cv2.CAP_PROP_FRAME_COUNT) - 4:
                    video_cap_kirito1.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    frame_counter = 0
                    det = 0
                    detection = False
                else:
                    video_frame = cv2.resize(video_frame, (w_k, h_k))

        elif det == "down":
            _, video_frame = video_cap_kirito4.read()
            video_frame = cv2.resize(video_frame, (w_k, h_k))
            if detection == False:
                video_cap_kirito4.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_counter = 0
                detection=True
            else:
                # print(Video_cap.get(cv2.CAP_PROP_FRAME_COUNT)-3)
                if frame_counter == video_cap_kirito4.get(cv2.CAP_PROP_FRAME_COUNT) - 3:
                    video_cap_kirito4.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    frame_counter = 0
                    det = 0
                    detection = False
                else:
                    # make video size as big as Target Image
                    video_frame = cv2.resize(video_frame, (w_k, h_k))
        elif det == "left":
            _, video_frame = video_cap_kirito2.read()
            # make video size as big as Target Image
            video_frame = cv2.resize(video_frame, (w_k, h_k))
            if detection == False:
                video_cap_kirito2.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_counter = 0
                detection=True
            else:
                # print(Video_cap.get(cv2.CAP_PROP_FRAME_COUNT)-3)
                if frame_counter == video_cap_kirito2.get(cv2.CAP_PROP_FRAME_COUNT) - 3:
                    video_cap_kirito2.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    frame_counter = 0
                    det = 0
                    detection = False
                else:
                    # make video size as big as Target Image
                    video_frame = cv2.resize(video_frame, (w_k, h_k))
        elif det == "right":
            _, video_frame = video_cap_kirito3.read()
            # make video size as big as Target Image
            video_frame = cv2.resize(video_frame, (w_k, h_k))
            if detection == False:
                video_cap_kirito3.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_counter = 0
                detection=True
            else:
                if frame_counter == video_cap_kirito3.get(cv2.CAP_PROP_FRAME_COUNT) - 3:
                    video_cap_kirito3.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    frame_counter = 0
                    det = 0
                    detection = False
                else:
                    # make video size as big as Target Image
                    video_frame = cv2.resize(video_frame, (w_k, h_k))
        elif det == 0:
            _, video_frame = Video_cap_kirito.read()
            video_frame = cv2.rotate(video_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            video_frame = cv2.resize(video_frame, (w_k, h_k))
            if detection == False:
                Video_cap_kirito.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_counter = 0
                detection=True
                # drink_sound = MediaPlayer("audio/drink_sound.m4a")
            else:
                if frame_counter == Video_cap_kirito.get(cv2.CAP_PROP_FRAME_COUNT) - 10:
                    Video_cap_kirito.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    frame_counter = 0
                    detection=False
                else:
                    video_frame = cv2.resize(video_frame, (w_k, h_k))
                    # drink_sound.set_pause(True)

        frame, video_frame, detection, det = fitting(frame, video_frame, matches_kirito, referenceImagePts_kirito, sourceImagePts, corners, det, detection)
        # cv2.resize(video_frame, ())
        print(det)
        cv2.imshow("video", video_frame)
        frame_counter += 1
    # ===================== Display ====================
    # show result


    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()