import os
from ultralytics import YOLO
import sys

if __name__ == '__main__':
    os.chdir("d:/Trae programes/交通标志检测/第4次实验数据及提交格式")
    
    print("Starting training...")
    
    try:
        # 尝试加载预训练模型
        model = YOLO('yolov8n.pt')
        print("Loaded yolov8n.pt successfully")
    except Exception as e:
        print(f"Error loading yolov8n.pt: {e}")
        print("Trying to create a new model from scratch...")
        model = YOLO('yolov8n.yaml')
        print("Created new model from yolov8n.yaml")
    
    # 训练模型
    print("Starting training with data.yaml...")
    results = model.train(
        data='data.yaml',
        epochs=50,
        imgsz=416,
        device=0,
        patience=50,
        verbose=True,
        save=True,
        plots=True,
        workers=0  # 禁用多进程避免问题
    )
    
    print("Training completed!")
    print(f"Results: {results}")
