import cv2
from picamera2 import Picamera2
import mediapipe as mp

# -----------------------------
# 1. Mediapipe 설정
# -----------------------------
mp_face = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detector = mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5)

# -----------------------------
# 2. Picamera2 설정 (원본 해상도)
# -----------------------------
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (1280, 720), "format": "RGB888"}))
picam2.start()

# -----------------------------
# 3. OpenCV 창 설정 (풀스크린)
# -----------------------------
cv2.namedWindow("Face Tracking", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Face Tracking", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# -----------------------------
# 4. 루프 시작
# -----------------------------
while True:
    frame = picam2.capture_array()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 얼굴 탐지
    results = face_detector.process(frame_rgb)

    # 얼굴 위치 그리기
    if results.detections:
        for detection in results.detections:
            mp_drawing.draw_detection(frame, detection)

    cv2.imshow("Face Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -----------------------------
# 5. 종료 처리
# -----------------------------
cv2.destroyAllWindows()
picam2.stop()
