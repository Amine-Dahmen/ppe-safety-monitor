from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolov8s.pt")

    model.train(
        data="merged_dataset/data.yaml",
        epochs=50,
        imgsz=640,
        batch=8,
        workers=2,
        device=0,
        patience=15,
        save=True,
        project="ppe_runs",
        name="ppe_v1",
        exist_ok=True
    )

    print("Training complete!")
    print("Best model saved to: ppe_runs/ppe_v1/weights/best.pt")