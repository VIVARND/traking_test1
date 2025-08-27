from picamera2 import Picamera2
import cv2

# 카메라 초기화
picam2 = Picamera2()
# 해상도 1280x720으로 설정
picam2.configure(picam2.create_preview_configuration(main={"size": (1280, 720)}))
picam2.start()

# 얼굴 인식용 Haar Cascade 불러오기
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# OpenCV 창을 전체화면으로 설정
cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Camera", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    # 카메라 프레임 읽기
    frame = picam2.capture_array()
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # 흑백 변환 후 얼굴 검출
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # 얼굴 위치에 사각형 그리기
    for (x, y, w, h) in faces:
        cv2.rectangle(frame_bgr, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # 풀스크린으로 출력
    cv2.imshow("Camera", frame_bgr)

    # q 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 종료 처리
cv2.destroyAllWindows()
picam2.stop()
