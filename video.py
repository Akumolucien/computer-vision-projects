import cv2 as cv
import numpy as np

capture=cv.VideoCapture('VIDEOS/cat.mp4')
while True:
    isTrue, frame = capture.read()
    cv.imshow('video', frame)

    if cv.waitKey(20) & 0xFF==ord('d'):
        break
   
capture.release()
cv.destroyAllWindows()