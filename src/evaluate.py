import pandas as pd
from ultralytics import YOLO
from common import ROOT, MODEL_PATH, get_dataset_config, require_file


def main():
    model = YOLO(str(require_file(MODEL_PATH)))

    metrics = model.val(
        data=get_dataset_config(),
        split="test",
        imgsz=640,
        batch=4,
        device="cpu",
        workers=0,
        plots=True,
        project=str(ROOT / "runs"),
        name="test_evaluation_03",
        exist_ok=True,
        verbose=True,
    )

    print('\n===== OVERALL METRICS =====')
    
    for label, value in [
        ('Precision', metrics.box.mp), 
        ('Recall', metrics.box.mr),
        ('mAP@50', metrics.box.map50), 
        ('mAP@50-95', metrics.box.map)
    ]:
        print(f'{label}: {value:.4f}')
    
    rows = []
    # class_result expects the position in metrics.box.ap_class_index.
    for idx, class_id in enumerate(metrics.box.ap_class_index):
        p, r, m50, m5095 = metrics.box.class_result(idx)
        
        rows.append({
            'class': model.names[int(class_id)], 
            'precision': p,
            'recall': r, 
            'map50': m50, 
            'map50_95': m5095
        })

    df = pd.DataFrame(rows)

    print('\n===== PER-CLASS METRICS =====')
    print(df.to_string(index=False))
    
    output = ROOT / 'runs' / 'model_evaluation_03.csv'
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False, sep=';', decimal=',')
    print(f'\nResults saved to: {output}')


if __name__ == '__main__':
    main()
