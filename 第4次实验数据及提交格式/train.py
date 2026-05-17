import os
import sys
from ultralytics import YOLO

os.chdir("d:/Trae programes/交通标志检测/第4次实验数据及提交格式")

model = YOLO('yolov8n.pt')

results = model.train(
    data='data.yaml',
    epochs=50,
    imgsz=416,
    device=0,
    verbose=True
)

print("Training completed!")