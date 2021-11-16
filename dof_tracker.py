import numpy as np
import cv2 as cv

#cap = cv.VideoCapture("./data/traj/Goodwine/episodeRGB01_%05d.jpg")
cap = cv.VideoCapture("./data/CoralScanEnv4/CoralFrame_%04d.png")

ret, frame1 = cap.read()
height, width, _ = frame1.shape
print(f"frame height:width is {height}:{width}")
prvs = cv.cvtColor(frame1,cv.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[...,1] = 255

# Object detection from Stable camera
object_detector = cv.createBackgroundSubtractorMOG2(history=100, varThreshold=32, detectShadows=False)
min_area = 10.

dof_out = cv.VideoWriter('dof_tracker.avi', cv.VideoWriter_fourcc(*'DIVX'), 15, (width, height))
rgb_out = cv.VideoWriter('rgb_tracker.avi', cv.VideoWriter_fourcc(*'DIVX'), 15, (width, height))
cnt_out = cv.VideoWriter('cnt_tracker.avi', cv.VideoWriter_fourcc(*'DIVX'), 15, (width, height))

cv.namedWindow("frame", cv.WINDOW_NORMAL)
count = 0
while(1):
    ret, frame2 = cap.read()
    if frame2 is None:
        break
    print(count)
    height, width, _ = frame2.shape
    next = cv.cvtColor(frame2,cv.COLOR_BGR2GRAY)
    flow = cv.calcOpticalFlowFarneback(prvs, next, None, pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
    mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv.normalize(mag,None,0,255,cv.NORM_MINMAX)
    bgr = cv.cvtColor(hsv,cv.COLOR_HSV2BGR)
    dof_frame = bgr
    bgr = cv.resize(bgr, (64, 64), interpolation = cv.INTER_AREA)
    #gray = cv.cvtColor(bgr,cv.COLOR_BGR2GRAY)
    dof = bgr
    cnt_frame = np.zeros((width,height,3), np.uint8)
    cv.rectangle( cnt_frame, ( 0,0 ), ( width, height), ( 0,0,0 ), -1, 8 )
    
    # Extract Region of interest
    mask = object_detector.apply(dof)
    _, mask = cv.threshold(mask, 254, 255, cv.THRESH_BINARY)
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    detections_bb = []
    for cnt in contours:
        # Calculate area and remove small elements
        area = cv.contourArea(cnt)
        print(f"area = {area}")
        if area > min_area:
            cv.drawContours(cnt_frame, [cnt], -1, (0, 255, 0), 2)
            x, y, w, h = cv.boundingRect(cnt)
            detections_bb.append([x, y, w, h])

    for x, y, w, h in detections_bb:
        cv.rectangle(dof, (x, y), (x + w, y + h), (0, 255, 0), 3)
    cv.imshow('frame',cnt_frame)
    cnt_out.write(cnt_frame)
    dof_out.write(dof_frame)
    rgb_out.write(frame2)
    
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    prvs = next
    count += 1
    
dof_out.release()
rgb_out.release()
cnt_out.release()
