import os
import sys
from pathlib import Path
from ultralytics import YOLO
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_paths():
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    data_dir = project_root / "第4次实验数据及提交格式"
    
    if not data_dir.exists():
        logger.error(f"数据目录不存在: {data_dir}")
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")
    
    os.chdir(data_dir)
    logger.info(f"工作目录: {data_dir}")
    return data_dir


def load_model():
    model_name = 'yolov8s.pt'
    try:
        logger.info(f"尝试加载 {model_name}...")
        model = YOLO(model_name)
        logger.info(f"成功加载 {model_name}")
        return model
    except Exception as e:
        logger.warning(f"加载 {model_name} 失败: {e}")
        fallback_model = 'yolov8n.pt'
        logger.info(f"回退使用 {fallback_model}")
        return YOLO(fallback_model)


def get_train_config():
    return {
        'data': 'data.yaml',
        'epochs': 100,
        'imgsz': 640,
        'device': 0,
        'patience': 50,
        'save': True,
        'plots': True,
        'workers': 4,
        'cache': True,
        'lr0': 0.01,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'augment': True,
        'mosaic': 1.0,
        'mixup': 0.1,
        'copy_paste': 0.1,
        'close_mosaic': 10,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 10.0,
        'translate': 0.1,
        'scale': 0.5,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'amp': True,
        'scheduler': 'cosine',
        'cos_lr': True,
        'iou': 0.7,
        'max_det': 300,
        'half': False,
        'optimizer': 'SGD',
        'seed': 0,
        'deterministic': True,
        'project': 'runs/detect',
        'name': 'train_optimized',
        'exist_ok': False,
        'pretrained': True,
        'resume': False,
        'single_cls': False,
    }


def train_model(model):
    logger.info("=" * 70)
    logger.info("开始优化训练 - YOLOv8s")
    logger.info("=" * 70)
    
    config = get_train_config()
    logger.info(f"训练配置: {len(config)} 个参数")
    
    try:
        logger.info("训练中，请稍候...")
        results = model.train(**config)
        logger.info("训练完成")
        return results
    except Exception as e:
        logger.error(f"训练失败: {e}")
        raise


def evaluate_model(model):
    logger.info("=" * 70)
    logger.info("开始模型评估...")
    logger.info("=" * 70)
    
    try:
        metrics = model.val()
        logger.info(f"mAP@0.5: {metrics.box.map50:.4f}")
        logger.info(f"mAP@0.5-0.95: {metrics.box.map:.4f}")
        logger.info(f"Precision: {metrics.box.mp:.4f}")
        logger.info(f"Recall: {metrics.box.mr:.4f}")
        return metrics
    except Exception as e:
        logger.error(f"评估失败: {e}")
        raise


if __name__ == '__main__':
    try:
        setup_paths()
        model = load_model()
        train_model(model)
        evaluate_model(model)
        
        logger.info("=" * 70)
        logger.info("训练和评估完成！")
        logger.info("=" * 70)
        
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        sys.exit(1)
