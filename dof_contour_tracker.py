import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

#cap = cv.VideoCapture(cv.samples.findFile("data/highway.mp4"))
#cap = cv.VideoCapture("./data/traj/Goodwine/episodeRGB01_%05d.jpg")
#cap = cv.VideoCapture("./data/episode_00/episode_00_frame_%05d.jpeg")
cap = cv.VideoCapture("./data/CoralScanEnv4/CoralFrame_%04d.png")
ret, frame1 = cap.read()
prvs = cv.cvtColor(frame1,cv.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[...,1] = 255

# Object detection from Stable camera
#object_detector = cv.createBackgroundSubtractorMOG2(history=100, varThreshold=150, detectShadows=False)
#min_area = 100.

object_detector = cv.createBackgroundSubtractorMOG2(history=100, varThreshold=32, detectShadows=False)
min_area = 10.

dofs = []
cnts = []

fps = 30
max_t = 59.
max_num_frames = max_t * fps
count = 0
dim = 64


while(1):
    if count > max_num_frames:
        break
    ret, frame2 = cap.read()
    if frame2 is None:
        break
    height, width, _ = frame2.shape
    next = cv.cvtColor(frame2,cv.COLOR_BGR2GRAY)
    flow = cv.calcOpticalFlowFarneback(prvs, next, None, pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
    mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv.normalize(mag,None,0,255,cv.NORM_MINMAX)
    bgr = cv.cvtColor(hsv,cv.COLOR_HSV2BGR)
    gray = cv.cvtColor(bgr,cv.COLOR_BGR2GRAY)
    cnt_bgr = np.zeros((width,height,3), np.uint8)
    dof = cv.resize(gray, (dim, dim), interpolation = cv.INTER_AREA)

    # Extract Region of interest
    mask = object_detector.apply(bgr)
    _, mask = cv.threshold(mask, 254, 255, cv.THRESH_BINARY)
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        # Calculate area and remove small elements
        area = cv.contourArea(cnt)
        if area > min_area:
            cv.drawContours(cnt_bgr, [cnt], -1, (0, 255, 0), 2)

    cnt_gray = cv.cvtColor(cnt_bgr,cv.COLOR_BGR2GRAY)
    cnt = cv.resize(cnt_gray, (dim, dim), interpolation = cv.INTER_AREA)
    
    dofs.append(dof.reshape((dim,dim,1)))
    cnts.append(cnt.reshape((dim,dim,1)))
    prvs = next
    count += 1
    print(count)
    
dof_array = np.dstack(tuple(dofs))
np.save('dof_array', dof_array)
cnt_array = np.dstack(tuple(cnts))
np.save('cnt_array', cnt_array)

plt.imshow(dof_array, aspect='auto')
plt.show()

plt.imshow(cnt_array, aspect='auto')
plt.show()

