import cv2
from picamera2 import Picamera2
import mediapipe as mp
import RPi.GPIO as GPIO
import time

# ======================
# 설정 값
# ======================
CAM_WIDTH, CAM_HEIGHT = 1280, 720
SERVO_PIN_X = 17
SERVO_PIN_Y = 18

# Dead Zone 크기 (화면 중앙 ±픽셀)
DEAD_ZONE = 40

# 서보 기본 위치
angle_x = 90
angle_y = 90

# ======================
# 서보 세팅
# ======================
GPIO.setmode(GPIO.BCM)
GPIO.setup(SERVO_PIN_X, GPIO.OUT)
GPIO.setup(SERVO_PIN_Y, GPIO.OUT)

pwm_x = GPIO.PWM(SERVO_PIN_X, 50)
pwm_y = GPIO.PWM(SERVO_PIN_Y, 50)

pwm_x.start(7.5)  # 90도 기준
pwm_y.start(7.5)

def set_servo_angle(pwm, angle):
    duty = 2.5 + (angle / 18)  # 0~180도 → 2.5~12.5 duty 변환
    pwm.ChangeDutyCycle(duty)

# ======================
# Mediapipe 초기화
# ======================
mp_face = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils
face_detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# ======================
# 카메라 초기화
# ======================
picam2 = Picamera2()
preview_config = picam2.create_preview_configuration(main={"size": (CAM_WIDTH, CAM_HEIGHT), "format": "RGB888"})
picam2.configure(preview_config)
picam2.start()

# OpenCV 창 → 전체 화면 모드
cv2.namedWindow("Face Tracking", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Face Tracking", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# ======================
# 메인 루프
# ======================
try:
    while True:
        frame = picam2.capture_array()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        results = face_detector.process(frame_rgb)

        if results.detections:
            for det in results.detections:
                mp_draw.draw_detection(frame_rgb, det)

                # 얼굴 중앙 좌표 추출
                bbox = det.location_data.relative_bounding_box
                cx = int((bbox.xmin + bbox.width / 2) * CAM_WIDTH)
                cy = int((bbox.ymin + bbox.height / 2) * CAM_HEIGHT)

                # 화면 중앙과 오차 계산
                error_x = cx - (CAM_WIDTH // 2)
                error_y = cy - (CAM_HEIGHT // 2)

                # Dead Zone 적용
                target_angle_x = angle_x
                target_angle_y = angle_y

                if abs(error_x) > DEAD_ZONE:
                    target_angle_x -= error_x * 0.02  # 민감도 조절
                if abs(error_y) > DEAD_ZONE:
                    target_angle_y += error_y * 0.02

                # 각도 제한
                target_angle_x = max(0, min(180, target_angle_x))
                target_angle_y = max(0, min(180, target_angle_y))

                # 보간 (부드럽게 이동)
                angle_x += (target_angle_x - angle_x) * 0.1
                angle_y += (target_angle_y - angle_y) * 0.1

                # 서보 제어
                set_servo_angle(pwm_x, angle_x)
                set_servo_angle(pwm_y, angle_y)

        cv2.imshow("Face Tracking", frame_rgb)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(0.05)  # 서보 안정화를 위해 약간 딜레이

except KeyboardInterrupt:
    pass

finally:
    cv2.destroyAllWindows()
    picam2.stop()
    pwm_x.stop()
    pwm_y.stop()
    GPIO.cleanup()
