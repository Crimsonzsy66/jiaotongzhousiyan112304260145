# Traffic Sign Detection Challenge

## 项目概述
本项目使用YOLOv8模型进行交通标志目标检测，实现高精度的交通标志识别与定位。代码经过优化重构，提升了训练效率和可维护性。

## Task
Train an object detection model with the provided YOLO dataset and predict objects on the hidden-label test set.

## 实验环境
- **GPU**: NVIDIA GeForce RTX 4060 Laptop GPU
- **框架**: Ultralytics YOLOv8
- **Python版本**: Python 3.12
- **训练轮数**: 100 epochs
- **图像尺寸**: 640x640

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
- `train_model.py`: 优化后的训练脚本
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

### 训练配置
- **模型**: YOLOv8s (small版本)
- **预训练权重**: yolov8s.pt
- **优化器**: SGD
- **学习率**: 0.01 (初始/最终)
- **早停机制**: patience=50
- **数据增强**: Mosaic, MixUp, CopyPaste
- **混合精度**: AMP启用

### 训练说明
- 最佳模型：`runs/detect/train_optimized/weights/best.pt`
- 代码优化：模块化设计、参数精简、性能优化

## 代码优化
训练脚本经过全面优化重构：

1. **模块化设计**
   - setup_paths(): 路径设置和管理
   - load_model(): 模型加载
   - get_train_config(): 训练配置
   - train_model(): 模型训练
   - evaluate_model(): 模型评估

2. **性能优化**
   - workers: 0 → 4
   - cache: 启用
   - 冗余参数精简：99个 → 36个

3. **可维护性提升**
   - 日志系统完善
   - 错误处理健全
   - 代码结构清晰

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
# 使用优化后的脚本训练
python train_model.py
```

### 生成提交文件
```bash
# 使用run_infer.py推理
python run_infer.py

# 或使用baseline_infer.py
python baseline_infer.py --model runs/detect/train_optimized/weights/best.pt --test-dir test/images --output submission.csv
```

## 代码仓库
https://github.com/Crimsonzsy66/jiaotongzhousiyan112304260145

## Example training
```bash
yolo detect train data=data.yaml model=yolov8s.pt epochs=100 imgsz=640 optimizer=SGD lr0=0.01
```

## Example submission generation
```bash
python baseline_infer.py --model runs/detect/train_optimized/weights/best.pt --test-dir test/images --output submission.csv
```
