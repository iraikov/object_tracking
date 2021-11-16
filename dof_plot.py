import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

#cap = cv.VideoCapture(cv.samples.findFile("data/highway.mp4"))
cap = cv.VideoCapture("./data/traj/Goodwine/episodeRGB01_%05d.jpg")
ret, frame1 = cap.read()
prvs = cv.cvtColor(frame1,cv.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[...,1] = 255

# Object detection from Stable camera
object_detector = cv.createBackgroundSubtractorMOG2(history=100, varThreshold=150, detectShadows=False)
min_area = 100.

dofs = []

fps = 30
max_t = 30.
max_num_frames = max_t * fps
count = 0

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
    bgr = cv.resize(bgr, (64, 64), interpolation = cv.INTER_AREA)
    gray = cv.cvtColor(bgr,cv.COLOR_BGR2GRAY)
    dof = gray

    dofs.append(dof.ravel())
    prvs = next
    count += 1

dof_array = np.column_stack(dofs)
np.save('dof_array', dof_array)
plt.imshow(dof_array, aspect='auto')
plt.show()

