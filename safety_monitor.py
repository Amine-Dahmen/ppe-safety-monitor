from ultralytics import YOLO
import cv2
import csv
from datetime import datetime

if __name__ == '__main__':
    model = YOLO("runs/detect/ppe_runs/ppe_v1/weights/best.pt")

    # violation classes — these trigger an alert
    VIOLATION_CLASSES = {'NO-Hardhat', 'NO-Safety-Vest', 'NO-Mask'}

    CLASS_NAMES = [
        'Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask',
        'NO-Safety-Vest', 'Person', 'Safety-Cone', 'Safety-Vest', 'vehicle'
    ]

    # colors: green for safe, red for violation
    COLORS = {
        'Hardhat':       (0, 255, 0),
        'Mask':          (0, 255, 0),
        'Safety-Vest':   (0, 255, 0),
        'Safety-Cone':   (0, 255, 0),
        'Person':        (255, 165, 0),
        'vehicle':       (255, 165, 0),
        'NO-Hardhat':    (0, 0, 255),
        'NO-Mask':       (0, 0, 255),
        'NO-Safety-Vest':(0, 0, 255),
    }

    # setup CSV log
    log_file = "violations_log.csv"
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'violation', 'confidence'])

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    print("Safety monitor running — press Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=0.5, verbose=False)
        detections = results[0].boxes

        violations_this_frame = []

        for box in detections:
            class_id = int(box.cls[0])
            class_name = CLASS_NAMES[class_id]
            confidence = float(box.conf[0])

            # draw bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color = COLORS.get(class_name, (255, 255, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{class_name} {confidence:.2f}",
                       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, color, 2)

            # log violations
            if class_name in VIOLATION_CLASSES:
                violations_this_frame.append((class_name, confidence))

        # log to CSV and show alert
        if violations_this_frame:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                for violation, conf in violations_this_frame:
                    writer.writerow([timestamp, violation, f"{conf:.2f}"])

            # red alert banner on screen
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 255), -1)
            cv2.putText(frame, f"VIOLATION DETECTED: {violations_this_frame[0][0]}",
                       (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        else:
            # green safe banner
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 180, 0), -1)
            cv2.putText(frame, "ALL CLEAR — PPE COMPLIANT",
                       (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Industrial Safety Monitor", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Session ended. Violations logged to {log_file}")