import cv2
from picamera2 import Picamera2
import mediapipe as mp
import RPi.GPIO as GPIO
import time

# --------------------
# 서보 설정
# --------------------
SERVO_X = 17   # GPIO 핀 번호 (X축 서보)
SERVO_Y = 27   # GPIO 핀 번호 (Y축 서보)

GPIO.setmode(GPIO.BCM)
GPIO.setup(SERVO_X, GPIO.OUT)
GPIO.setup(SERVO_Y, GPIO.OUT)

pwm_x = GPIO.PWM(SERVO_X, 50)  # 50Hz
pwm_y = GPIO.PWM(SERVO_Y, 50)

pwm_x.start(7.5)  # 초기 위치 (90도)
pwm_y.start(7.5)

# 서보 각도 → 듀티비율 변환 함수
def set_servo_angle(pwm, angle):
    duty = 2.5 + (angle / 18)  # 0~180° → duty 2.5~12.5
    pwm.ChangeDutyCycle(duty)

# 초기 각도
angle_x, angle_y = 90, 90

# --------------------
# 카메라 설정
# --------------------
CAM_WIDTH, CAM_HEIGHT = 640, 480

mp_face = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils
face_detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)

picam2 = Picamera2()
preview_config = picam2.create_preview_configuration(main={"size": (CAM_WIDTH, CAM_HEIGHT), "format": "RGB888"})
picam2.configure(preview_config)
picam2.start()

cv2.namedWindow("Face Tracking", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Face Tracking", CAM_WIDTH, CAM_HEIGHT)

# --------------------
# 메인 루프
# --------------------
while True:
    frame = picam2.capture_array()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    results = face_detector.process(frame_rgb)

    if results.detections:
        for det in results.detections:
            mp_draw.draw_detection(frame_rgb, det)

            # 얼굴 중심 좌표 계산
            bboxC = det.location_data.relative_bounding_box
            cx = int((bboxC.xmin + bboxC.width / 2) * CAM_WIDTH)
            cy = int((bboxC.ymin + bboxC.height / 2) * CAM_HEIGHT)

            # 화면 중앙과 비교
            error_x = cx - (CAM_WIDTH // 2)
            error_y = cy - (CAM_HEIGHT // 2)

            # 오차를 각도에 반영 (비율 조절 필요)
            if abs(error_x) > 20:  # X축 오차 허용 범위
                angle_x -= error_x * 0.01
                angle_x = max(0, min(180, angle_x))
                set_servo_angle(pwm_x, angle_x)

            if abs(error_y) > 20:  # Y축 오차 허용 범위
                angle_y += error_y * 0.01  # Y축 반대 방향 보정
                angle_y = max(0, min(180, angle_y))
                set_servo_angle(pwm_y, angle_y)

    cv2.imshow("Face Tracking", frame_rgb)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --------------------
# 종료
# --------------------
picam2.stop()
cv2.destroyAllWindows()
pwm_x.stop()
pwm_y.stop()
GPIO.cleanup()
