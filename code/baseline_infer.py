from __future__ import annotations

import argparse
import csv
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description='交通标志检测推理脚本')
    parser.add_argument("--model", default="runs/detect/train_optimized/weights/best.pt", help="Path to best.pt")
    parser.add_argument("--test-dir", default="test/images", help="Directory of test images")
    parser.add_argument("--output", default="submission.csv", help="Output CSV path")
    parser.add_argument("--conf", type=float, default=0.0005, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--augment", action="store_true", help="Use test-time augmentation")
    args = parser.parse_args()

    model = YOLO(args.model)
    image_paths = sorted(
        [p for p in Path(args.test_dir).iterdir() if p.is_file()]
    )

    with Path(args.output).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image_id", "class_id", "x_center", "y_center", "width", "height", "confidence"],
        )
        writer.writeheader()
        
        for i, img_path in enumerate(image_paths):
            if (i + 1) % 20 == 0:
                print(f"处理进度: {i+1}/{len(image_paths)}")
            
            result = model.predict(
                source=str(img_path), 
                conf=args.conf, 
                iou=args.iou,
                imgsz=args.imgsz,
                augment=args.augment,
                save=False, 
                verbose=False
            )[0]
            
            image_id = Path(result.path).name
            if result.boxes is None:
                continue
            for box in result.boxes:
                x_center, y_center, width, height = box.xywhn[0].tolist()
                conf = float(box.conf[0].item())
                if conf > args.conf:
                    writer.writerow(
                        {
                            "image_id": image_id,
                            "class_id": int(box.cls[0].item()),
                            "x_center": x_center,
                            "y_center": y_center,
                            "width": width,
                            "height": height,
                            "confidence": conf,
                        }
                    )
    
    print(f"\n推理完成！结果已保存到: {args.output}")


if __name__ == "__main__":
    main()
