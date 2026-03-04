import cv2 as cv

cap = cv.VideoCapture(0)  # 0 = default camera

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    cv.imshow("Webcam", frame)
     
    print(frame[100,100])  # prints BGR values

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cv.imshow("Webcam", gray)
    
    print(gray[100,100])   # prints brightness value
    if cv.waitKey(20) & 0xFF == ord('q'):  # press 'q' to quit
        break

cap.release()
cv.destroyAllWindows()
