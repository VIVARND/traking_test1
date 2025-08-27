from picamera2 import Picamera2
import cv2

# -----------------------------
# 1. 카메라 설정 (원본 해상도)
# -----------------------------
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (1280, 720)}))
picam2.start()

# -----------------------------
# 2. 얼굴 검출용 Haar Cascade
# -----------------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# -----------------------------
# 3. OpenCV 창 설정 (원본 크기)
# -----------------------------
cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Camera", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# -----------------------------
# 4. 루프 시작
# -----------------------------
while True:
    # 원본 프레임 캡처
    frame = picam2.capture_array()
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # -----------------------------
    # 5. 얼굴 검출용 작은 프레임 생성 (속도 향상)
    # -----------------------------
    small_frame = cv2.resize(frame_bgr, (320, 180))  # 1280x720 -> 320x180
    gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # -----------------------------
    # 6. 얼굴 좌표 원본 화면으로 변환
    # -----------------------------
    scale_x = frame_bgr.shape[1] / small_frame.shape[1]
    scale_y = frame_bgr.shape[0] / small_frame.shape[0]
    faces = [(int(x*scale_x), int(y*scale_y), int(w*scale_x), int(h*scale_y)) for (x, y, w, h) in faces]

    # -----------------------------
    # 7. 얼굴 위치에 사각형 표시
    # -----------------------------
    for (x, y, w, h) in faces:
        cv2.rectangle(frame_bgr, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # -----------------------------
    # 8. 화면 출력
    # -----------------------------
    cv2.imshow("Camera", frame_bgr)

    # 'q' 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -----------------------------
# 9. 종료 처리
# -----------------------------
cv2.destroyAllWindows()
picam2.stop()
