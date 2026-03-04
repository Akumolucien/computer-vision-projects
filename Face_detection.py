import cv2
import os

# Path to video file (relative to repo root)
video_path = os.path.join('VIDEOS', 'people.mp4')

if not os.path.exists(video_path):
    print(f"Error: video file not found: {os.path.abspath(video_path)}")
    raise SystemExit(1)

capture = cv2.VideoCapture(0)


import os
import time


# Load Haar cascade for face detection (bundled with OpenCV)
haar_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
if not os.path.exists(haar_path):
    print(f"Error: Haar cascade not found at {haar_path}")
    raise SystemExit(1)

face_cascade = cv2.CascadeClassifier(haar_path)

# Optional: toggle to blur faces instead of drawing boxes
BLUR_FACES = False

# Simple FPS calculation
prev_time = time.time()
frame_count = 0

try:
    while True:
        isTrue, frame = capture.read()

        # If frame read failed (end of file or error), break the loop
        if not isTrue or frame is None:
            break

        # Convert to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces (scaleFactor and minNeighbors can be tuned)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Annotate detected faces
        for (x, y, w, h) in faces:
            if BLUR_FACES:
                # Blur the face region for privacy
                face_region = frame[y:y+h, x:x+w]
                # Apply a strong Gaussian blur
                face_region = cv2.GaussianBlur(face_region, (99, 99), 30)
                frame[y:y+h, x:x+w] = face_region
            else:
                # Draw rectangle and label
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, 'Face', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # FPS overlay
        frame_count += 1
        if frame_count >= 10:
            now = time.time()
            fps = frame_count / (now - prev_time)
            prev_time = now
            frame_count = 0
        else:
            fps = 1

        if fps is not None:
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        cv2.imshow('Real time face detection', frame)

        # Key handling: 'q' to quit, 'b' to toggle blur
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('b'):
            BLUR_FACES = not BLUR_FACES
            print(f"BLUR_FACES = {BLUR_FACES}")
finally:
    # Release and close windows after the loop
    capture.release()
    cv2.destroyAllWindows()



