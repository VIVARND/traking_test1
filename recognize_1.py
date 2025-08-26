from picamera2 import Picamera2
import cv2

# 카메라 초기화
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
picam2.start()

# Haar Cascade 로 얼굴 검출
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)

while True:
    frame = picam2.capture_array()
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    # 얼굴 위치에 사각형 표시 (트래킹)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame_bgr, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    cv2.imshow("Camera", frame_bgr)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
picam2.stop()
