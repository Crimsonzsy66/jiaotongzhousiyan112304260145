# Traffic Sign Detection Challenge

## 项目概述
本项目使用YOLOv8模型进行交通标志目标检测，实现高精度的交通标志识别与定位。

## Task
Train an object detection model with the provided YOLO dataset and predict objects on the hidden-label test set.

## 实验环境
- **GPU**: NVIDIA GeForce RTX 4060 Laptop GPU
- **框架**: Ultralytics YOLOv8
- **Python版本**: Python 3.12
- **训练轮数**: 50 epochs
- **图像尺寸**: 416x416

## Classes
| 类别ID | 类别名称 |
|--------|----------|
| 0 | Green Light |
| 1 | Red Light |
| 2 | Speed Limit 10 |
| 3 | Speed Limit 100 |
| 4 | Speed Limit 110 |
| 5 | Speed Limit 120 |
| 6 | Speed Limit 20 |
| 7 | Speed Limit 30 |
| 8 | Speed Limit 40 |
| 9 | Speed Limit 50 |
| 10 | Speed Limit 60 |
| 11 | Speed Limit 70 |
| 12 | Speed Limit 80 |
| 13 | Speed Limit 90 |
| 14 | Stop |

## Directory
- `train/`: training images and labels
- `val/`: validation images and labels
- `test/images/`: test images only
- `data.yaml`: Ultralytics training config
- `sample_submission.csv`: submission schema
- `baseline_infer.py`: example inference-to-CSV script
- `train_model.py`: 训练脚本
- `run_infer.py`: 推理脚本
- `submission.csv`: 提交结果文件
- `runs/`: 训练结果目录

## 实验结果

### 性能指标
| 指标 | 数值 |
|------|------|
| mAP@0.5 | 94.43% |
| mAP@0.5-0.95 | 81.23% |
| Precision | 94.84% |
| Recall | 87.81% |

### 训练说明
- 模型：YOLOv8n (nano版本)
- 预训练权重：yolov8n.pt
- 训练时间：约2242秒
- 最佳模型：`runs/detect/train3/weights/best.pt`

## Submission
Submit one `submission.csv` file with these columns:
- `image_id`
- `class_id`
- `x_center`
- `y_center`
- `width`
- `height`
- `confidence`

All coordinates must be YOLO-style normalized values in `[0, 1]`.

## Metric
Ranking metric: `mAP@0.5`

## 使用方法

### 训练模型
```bash
# 使用提供的脚本训练
python train_model.py
```

### 生成提交文件
```bash
# 使用run_infer.py推理
python run_infer.py

# 或使用baseline_infer.py
python baseline_infer.py --model runs/detect/train3/weights/best.pt --test-dir test/images --output submission.csv
```

## Example training
```bash
yolo detect train data=data.yaml model=yolov8n.pt epochs=50 imgsz=416
```

## Example submission generation
```bash
python baseline_infer.py --model runs/detect/train/weights/best.pt --test-dir test/images --output submission.csv
```
