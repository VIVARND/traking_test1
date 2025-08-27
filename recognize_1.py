from picamera2 import Picamera2
import cv2

# 카메라 초기화
picam2 = Picamera2()
# 해상도 1280x720 설정 (큰 화면용)
picam2.configure(picam2.create_preview_configuration(main={"size": (1280, 720)}))
picam2.start()

# Haar Cascade 얼굴 검출기 로드
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# 출력창 설정 (크기 조절 가능 + 1280x720)
cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Camera", 1280, 720)

while True:
    # 카메라 프레임 가져오기
    frame = picam2.capture_array()
    # 색상 변환 (RGB → BGR, 파란 화면 방지)
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # 얼굴 검출은 흑백으로 진행
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # 얼굴 위치에 초록 박스 그리기
    for (x, y, w, h) in faces:
        cv2.rectangle(frame_bgr, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # 화면에 출력
    cv2.imshow("Camera", frame_bgr)

    # 'q' 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 리소스 해제
cv2.destroyAllWindows()
picam2.stop()
