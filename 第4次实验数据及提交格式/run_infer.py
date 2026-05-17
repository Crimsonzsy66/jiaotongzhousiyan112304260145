import csv
from pathlib import Path

from ultralytics import YOLO

def main():
    # 设置路径
    model_path = Path("runs/detect/train3/weights/best.pt")
    test_dir = Path("test/images")
    output_path = Path("submission.csv")
    
    # 加载模型
    model = YOLO(model_path)
    
    # 获取所有图片
    image_paths = sorted([p for p in test_dir.iterdir() if p.is_file()])
    
    # 进行预测并生成提交文件
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image_id", "class_id", "x_center", "y_center", "width", "height", "confidence"],
        )
        writer.writeheader()
        
        # 逐个预测，确保文件名正确
        for img_path in image_paths:
            image_id = img_path.name
            results = model.predict(source=str(img_path), conf=0.001, save=False, verbose=False)
            
            for result in results:
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    x_center, y_center, width, height = box.xywhn[0].tolist()
                    writer.writerow({
                        "image_id": image_id,
                        "class_id": int(box.cls[0].item()),
                        "x_center": x_center,
                        "y_center": y_center,
                        "width": width,
                        "height": height,
                        "confidence": float(box.conf[0].item()),
                    })
    
    print(f"预测完成！结果已保存到: {output_path}")

if __name__ == "__main__":
    main()
