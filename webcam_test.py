from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # force DirectShow

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't read frame")
        break

    results = model(frame)
    annotated = results[0].plot()
    cv2.imshow("YOLOv8", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
