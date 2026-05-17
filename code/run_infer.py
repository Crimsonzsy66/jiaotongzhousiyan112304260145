import csv
from pathlib import Path
import numpy as np
from ultralytics import YOLO

def main():
    model_path = Path("runs/detect/train_optimized/weights/best.pt")
    test_dir = Path("test/images")
    output_path = Path("submission_optimized.csv")
    
    if not model_path.exists():
        print(f"警告: 找不到优化模型 {model_path}")
        print("尝试查找其他训练结果...")
        alt_paths = list(Path("runs/detect").glob("*/weights/best.pt"))
        if alt_paths:
            model_path = alt_paths[0]
            print(f"使用替代模型: {model_path}")
        else:
            print("错误: 找不到任何训练模型！")
            return
    
    print(f"加载模型: {model_path}")
    model = YOLO(model_path)
    
    image_paths = sorted([p for p in test_dir.iterdir() if p.is_file()])
    print(f"找到 {len(image_paths)} 张测试图片")
    
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image_id", "class_id", "x_center", "y_center", "width", "height", "confidence"],
        )
        writer.writeheader()
        
        for i, img_path in enumerate(image_paths):
            image_id = img_path.name
            if (i + 1) % 20 == 0:
                print(f"处理进度: {i+1}/{len(image_paths)}")
            
            results = model.predict(
                source=str(img_path),
                conf=0.0005,
                iou=0.5,
                imgsz=640,
                augment=True,
                max_det=100,
                agnostic_nms=False,
                save=False,
                verbose=False
            )
            
            for result in results:
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    x_center, y_center, width, height = box.xywhn[0].tolist()
                    conf = float(box.conf[0].item())
                    if conf > 0.0005:
                        writer.writerow({
                            "image_id": image_id,
                            "class_id": int(box.cls[0].item()),
                            "x_center": x_center,
                            "y_center": y_center,
                            "width": width,
                            "height": height,
                            "confidence": conf,
                        })
    
    print(f"\n优化推理完成！结果已保存到: {output_path}")
    print(f"请将此文件提交到比赛平台！")

if __name__ == "__main__":
    main()
