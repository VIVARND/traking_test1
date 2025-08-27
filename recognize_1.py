import cv2
from picamera2 import Picamera2
import mediapipe as mp

# 카메라 해상도 설정
CAM_WIDTH, CAM_HEIGHT = 1280, 720

# Mediapipe 얼굴 검출 초기화
mp_face = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils
face_detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# 카메라 설정
picam2 = Picamera2()
preview_config = picam2.create_preview_configuration(main={"size": (CAM_WIDTH, CAM_HEIGHT), "format": "RGB888"})
picam2.configure(preview_config)
picam2.start()

# OpenCV 창 설정
cv2.namedWindow("Face Tracking", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Face Tracking", CAM_WIDTH, CAM_HEIGHT)

while True:
    frame = picam2.capture_array()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Mediapipe 얼굴 탐지
    results = face_detector.process(frame_rgb)

    if results.detections:
        for det in results.detections:
            mp_draw.draw_detection(frame_rgb, det)

    # 화면 출력
    cv2.imshow("Face Tracking", frame_rgb)

    # q 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
picam2.stop()
