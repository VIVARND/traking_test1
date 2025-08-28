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

DEAD_ZONE = 80   # Dead Zone 키움
SENSITIVITY = 0.005  # 민감도 낮춤

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

pwm_x.start(7.5)
pwm_y.start(7.5)

def set_servo_angle(pwm, angle):
    duty = 2.5 + (angle / 18)
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

# 전체화면
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

                bbox = det.location_data.relative_bounding_box
                cx = int((bbox.xmin + bbox.width / 2) * CAM_WIDTH)
                cy = int((bbox.ymin + bbox.height / 2) * CAM_HEIGHT)

                error_x = cx - (CAM_WIDTH // 2)
                error_y = cy - (CAM_HEIGHT // 2)

                target_angle_x = angle_x
                target_angle_y = angle_y

                if abs(error_x) > DEAD_ZONE:
                    target_angle_x -= error_x * SENSITIVITY
                if abs(error_y) > DEAD_ZONE:
                    target_angle_y += error_y * SENSITIVITY

                target_angle_x = max(0, min(180, target_angle_x))
                target_angle_y = max(0, min(180, target_angle_y))

                angle_x += (target_angle_x - angle_x) * 0.1
                angle_y += (target_angle_y - angle_y) * 0.1

                set_servo_angle(pwm_x, angle_x)
                set_servo_angle(pwm_y, angle_y)
        else:
            # 얼굴 없으면 현재 위치 유지
            set_servo_angle(pwm_x, angle_x)
            set_servo_angle(pwm_y, angle_y)

        # 카메라 영상 자체도 전체화면 크기로 확대
        frame_rgb = cv2.resize(frame_rgb, (1920, 1080))
        cv2.imshow("Face Tracking", frame_rgb)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(0.02)  # 프레임 속도 조정

except KeyboardInterrupt:
    pass

finally:
    cv2.destroyAllWindows()
    picam2.stop()
    pwm_x.stop()
    pwm_y.stop()
    GPIO.cleanup()
