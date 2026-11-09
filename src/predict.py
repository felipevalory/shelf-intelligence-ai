
import argparse
from pathlib import Path

from ultralytics import YOLO

from common import ROOT, MODEL_PATH, IMAGE_EXTENSIONS, require_file


def main():
    parser = argparse.ArgumentParser(
        description="YOLO inference on retail shelf images"
    )

    parser.add_argument(
        "--source",
        type=str,
        default=str(ROOT / "data" / "test" / "images"),
        help="Image file or directory containing images",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of images (0 for all)",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold",
    )

    args = parser.parse_args()

    if not 0 <= args.conf <= 1:
        parser.error("--conf must be between 0 and 1")

    if args.limit < 0:
        parser.error("--limit must be greater than or equal to 0")

    source = Path(args.source)

    if source.is_dir():
        images = sorted(
            p for p in source.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )

        if args.limit > 0:
            images = images[:args.limit]

        if not images:
            raise ValueError(f"No images found in {source}")

        inputs = [str(p) for p in images]

    else:
        require_file(source)

        if source.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image format: {source.suffix}")

        inputs = [str(source)]

    model = YOLO(str(require_file(MODEL_PATH)))

    results = model.predict(
        source=inputs,
        imgsz=640,
        conf=args.conf,
        device="cpu",
        save=True,
        project=str(ROOT / "runs"),
        name="test_inference_03",
        exist_ok=True,
        verbose=False,
    )

    for result in results:
        counts = {"shelf": 0, "void": 0}

        for box in result.boxes:
            class_id = int(box.cls.item())
            class_name = model.names[class_id]

            counts[class_name] = counts.get(class_name, 0) + 1

        print(f"\nImage: {Path(result.path).name}")
        print(f"Shelves detected: {counts['shelf']}")
        print(f"Empty spaces detected: {counts['void']}")

    print("\nInference completed!")


if __name__ == "__main__":
    main()
