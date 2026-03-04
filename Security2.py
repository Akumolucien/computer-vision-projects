import cv2 as cv
import time

cap = cv.VideoCapture(0)

last_capture_time = 0

# Read first frame
ret, frame1 = cap.read()
prev_frame = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
prev_frame = cv.GaussianBlur(prev_frame, (5,5), 0)

recording = False
out = None
last_motion_time = 0
object_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5,5), 0)

    # Detect motion
    diff = cv.absdiff(prev_frame, blur)
    _, thresh = cv.threshold(diff, 25, 255, cv.THRESH_BINARY)
    thresh = cv.dilate(thresh, None, iterations=2)

    contours, _ = cv.findContours(
        thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
    )

    motion_detected = False

    for contour in contours:
        if cv.contourArea(contour) < 800:
            continue

        motion_detected = True
        x, y, w, h = cv.boundingRect(contour)
        cv.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

    # -------- ACTIONS WHEN MOTION DETECTED --------
    if motion_detected:
        last_motion_time = time.time()

        # Warning text
        cv.putText(frame, "INTRUDER DETECTED!",
                   (20,50),
                   cv.FONT_HERSHEY_SIMPLEX,
                   1, (0,0,255), 3)

        # Count object
        object_count += 1

        # Take screenshot
        filename = f"intruder_{int(time.time())}.jpg"
        cv.imwrite(filename, frame)

        # Start recording
        if not recording:
            fourcc = cv.VideoWriter_fourcc(*'XVID')
            out = cv.VideoWriter(
                'security_record.avi',
                fourcc,
                20.0,
                (frame.shape[1], frame.shape[0])
            )
            recording = True

    # -------- STOP RECORDING AFTER NO MOTION --------
    if recording:
        out.write(frame)

        # stop after 5 seconds without motion
        if time.time() - last_motion_time > 5:
            recording = False
            out.release()
            print("Recording stopped")

    # Display object count
    cv.putText(frame, f"Objects: {object_count}",
               (20,90),
               cv.FONT_HERSHEY_SIMPLEX,
               0.8, (255,0,0), 2)

    cv.imshow("Security Camera", frame)
    cv.imshow("Motion Mask", thresh)

    prev_frame = blur

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()