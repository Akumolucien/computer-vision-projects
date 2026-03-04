import cv2 as cv

cap = cv.VideoCapture(0)

ret, frame1 = cap.read()
prev_frame = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
prev_frame = cv.GaussianBlur(prev_frame,(5,5),0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray,(5,5),0)

    diff = cv.absdiff(prev_frame, blur)
    _, thresh = cv.threshold(diff,25,255,cv.THRESH_BINARY)

    contours, _ = cv.findContours(
        thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
    )

    for contour in contours:
        if cv.contourArea(contour) < 500:
            continue

        x,y,w,h = cv.boundingRect(contour)
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

    cv.imshow("Security Camera", frame)
    cv.imshow("Motion", thresh)

    prev_frame = blur

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()