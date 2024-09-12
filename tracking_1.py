import cv2
import numpy as np
import RPi.GPIO as GPIO
import time

# 서보 모터 핀 설정
SERVO_PIN_X = 17
SERVO_PIN_Y = 18

# GPIO 설정
GPIO.setmode(GPIO.BCM)
GPIO.setup(SERVO_PIN_X, GPIO.OUT)
GPIO.setup(SERVO_PIN_Y, GPIO.OUT)

# 서보 모터 PWM 제어 (50Hz)
servo_x = GPIO.PWM(SERVO_PIN_X, 50)
servo_y = GPIO.PWM(SERVO_PIN_Y, 50)
servo_x.start(7.5)  # 초기 위치
servo_y.start(7.5)

# 카메라 설정 (ArduCam UC-367)
cap = cv2.VideoCapture(0)  # 카메라 인덱스 0

# 색상 범위 설정 (흰색 예시)
lower_color = np.array([0, 0, 200], dtype=np.uint8)
upper_color = np.array([180, 30, 255], dtype=np.uint8)

# 서보 모터 각도 설정 함수
def set_servo_angle(servo, angle):
    duty = (angle / 18.0) + 2.5
    servo.ChangeDutyCycle(duty)

# 카메라 트래킹 루프
try:
    while True:
        # 카메라로부터 프레임 읽기
        ret, frame = cap.read()
        if not ret:
            print("카메라에서 프레임을 읽을 수 없습니다.")
            break

        # 이미지 크기 가져오기
        height, width, _ = frame.shape

        # 이미지를 HSV 색 공간으로 변환
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 흰색 영역 마스크 생성
        mask = cv2.inRange(hsv, lower_color, upper_color)

        # 흰색 물체의 윤곽선 찾기
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # 윤곽선이 존재하면 추적
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # 물체의 중심 좌표 계산
            obj_center_x = x + w // 2
            obj_center_y = y + h // 2

            # 카메라 화면의 중심과 비교하여 서보 모터 각도 조정
            frame_center_x = width // 2
            frame_center_y = height // 2

            # X축 서보 제어 (좌우 이동)
            if obj_center_x < frame_center_x - 20:
                set_servo_angle(servo_x, 100)  # 왼쪽
            elif obj_center_x > frame_center_x + 20:
                set_servo_angle(servo_x, 80)  # 오른쪽
            else:
                set_servo_angle(servo_x, 90)  # 중앙

            # Y축 서보 제어 (상하 이동)
            if obj_center_y < frame_center_y - 20:
                set_servo_angle(servo_y, 80)  # 위쪽
            elif obj_center_y > frame_center_y + 20:
                set_servo_angle(servo_y, 100)  # 아래쪽
            else:
                set_servo_angle(servo_y, 90)  # 중앙

            # 추적하는 물체에 사각형 그리기
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 결과 영상 출력
        cv2.imshow("Tracking", frame)

        # ESC 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == 27:
            break

except KeyboardInterrupt:
    pass

finally:
    # 종료 후 자원 해제
    cap.release()
    cv2.destroyAllWindows()
    servo_x.stop()
    servo_y.stop()
    GPIO.cleanup()
