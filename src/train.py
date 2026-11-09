from ultralytics import YOLO
from common import ROOT, get_dataset_config


def main():
    model = YOLO("yolov8n.pt")

    model.train(
        data=get_dataset_config(),
        epochs=20,
        imgsz=640,
        batch=4,
        device="cpu",
        workers=0,
        project=str(ROOT / "runs"),
        name="shelf_experiment_03",
        exist_ok=True,
        plots=True,
    )

    print("Training completed!")


if __name__ == "__main__":
    main()