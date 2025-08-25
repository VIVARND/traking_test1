import cv2

# libcamera 스트리밍 주소
cap = cv2.VideoCapture("tcp://127.0.0.1:8554")  

# Haar Cascade 파일 경로 (홈 디렉토리)
face_cascade = cv2.CascadeClassifier("/home/user/haarcascade_frontalface_default.xml")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
