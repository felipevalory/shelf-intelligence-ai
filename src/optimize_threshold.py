from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
from common import ROOT, MODEL_PATH, require_file

THRESHOLDS = np.round(np.arange(0.10, 0.61, 0.05), 2)
IOU_MATCH = 0.50
CLASS_VOID = 1


def calculate_iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2-x1) * max(0, y2-y1)
    area_a = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    area_b = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def load_gt(image_path, shape):
    label = ROOT / 'data' / 'valid' / 'labels' / f'{Path(image_path).stem}.txt'
    require_file(label)
    h, w = shape
    boxes = []
    for line in label.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        cls, xc, yc, bw, bh = map(float, line.split())
        if int(cls) == CLASS_VOID:
            boxes.append([
                (xc-bw/2)*w, 
                (yc-bh/2)*h,
                (xc+bw/2)*w, 
                (yc+bh/2)*h
            ])
    return boxes


def match_boxes(pred, gt):
    candidates = [(calculate_iou(p, g), i, j)
                  for i, p in enumerate(pred) for j, g in enumerate(gt)]
    candidates = sorted((c for c in candidates if c[0] >= IOU_MATCH), reverse=True)
    used_pred, used_gt = set(), set()
    for _, i, j in candidates:
        if i not in used_pred and j not in used_gt:
            used_pred.add(i)
            used_gt.add(j)
    tp = len(used_pred)
    return tp, len(pred)-tp, len(gt)-tp


def main():
    output = ROOT / 'runs' / 'threshold_evaluation_04'
    output.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(require_file(MODEL_PATH)))
    image_dir = ROOT / 'data' / 'valid' / 'images'
    
    if not image_dir.is_dir():
        raise FileNotFoundError(f'Directory not found: {image_dir}')
    predictions = model.predict(
        source=str(image_dir), 
        imgsz=640, 
        conf=0.001,
        iou=0.70, 
        max_det=1000, 
        device='cpu', 
        verbose=False
        )
    data = []
    for result in predictions:
        gt = load_gt(result.path, result.orig_shape)
        classes = result.boxes.cls.cpu().numpy()
        mask = classes == CLASS_VOID
        data.append((
            gt, 
            result.boxes.xyxy.cpu().numpy()[mask],
            result.boxes.conf.cpu().numpy()[mask]
        ))

    rows = []
    for threshold in THRESHOLDS:
        tp = fp = fn = 0
        for gt, boxes, confs in data:
            a, b, c = match_boxes(boxes[confs >= threshold], gt)
            tp, fp, fn = tp+a, fp+b, fn+c
        precision = tp/(tp+fp) if tp+fp else 0
        recall = tp/(tp+fn) if tp+fn else 0
        f1 = 2*precision*recall/(precision+recall) if precision+recall else 0
        rows.append(dict(
            threshold=threshold, 
            precision=precision, 
            recall=recall,
            f1=f1, 
            TP=tp, 
            FP=fp, 
            FN=fn, 
            detections=tp+fp
        ))
        
    df = pd.DataFrame(rows)
    best = df.loc[df.f1.idxmax()]

    print(
        f"\nSelected confidence threshold: {best['threshold']:.2f}"
        f" | Precision: {best['precision']:.4f}"
        f" | Recall: {best['recall']:.4f}"
        f" | F1: {best['f1']:.4f}"
    )

    print('\n===== RESULTS BY THRESHOLD =====')
    print(df.round(4).to_string(index=False))
    print('\n===== BEST THRESHOLD (F1) =====')
    print(best.to_string())

    df.to_csv(output/'threshold_metrics.csv', index=False, sep=';', decimal=',')
    plt.figure(figsize=(10, 6))

    for metric in ['precision', 'recall', 'f1']:
        plt.plot(
            df.threshold, 
            df[metric], 
            marker='o', 
            label=metric
        )
    plt.axvline(
        best.threshold, 
        linestyle='--', 
        label=f'Best F1: {best.threshold:.2f}'
    )

    plt.xlabel('Confidence threshold')
    plt.ylabel('Metric')
    plt.title('Threshold optimization — void class')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output/'threshold_curves.png', dpi=150)
    plt.close()
    print('\nExperiment 04 completed!')


if __name__ == '__main__':
    main()
