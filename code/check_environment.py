import os
import sys
from pathlib import Path
import subprocess

def check_environment():
    print("=" * 70)
    print("Environment Check Report")
    print("=" * 70)
    print()
    
    # 1. Check Python version
    print("1. Python Environment")
    print("-" * 70)
    print(f"   Python Version: {sys.version}")
    print(f"   Python Path: {sys.executable}")
    print()
    
    # 2. Check key libraries
    print("2. Installed Libraries")
    print("-" * 70)
    libraries = ['ultralytics', 'torch', 'torchvision', 'cv2', 'numpy', 'pandas']
    for lib in libraries:
        try:
            if lib == 'cv2':
                import cv2
                print(f"   [OK] {lib}: {cv2.__version__}")
            elif lib == 'pandas':
                import pandas
                print(f"   [OK] {lib}: {pandas.__version__}")
            else:
                module = __import__(lib)
                version = getattr(module, '__version__', 'unknown')
                print(f"   [OK] {lib}: {version}")
        except ImportError:
            print(f"   [MISSING] {lib}: not installed")
        except Exception as e:
            print(f"   [ERROR] {lib}: {str(e)[:50]}")
    print()
    
    # 3. Check data paths
    print("3. Data Path Configuration")
    print("-" * 70)
    base_path = Path("c:/Users/ASUS/Downloads/数字识别器/交通标志检测/第4次实验数据及提交格式")
    paths_to_check = {
        'Base Directory': base_path,
        'Train Images': base_path / "train" / "images",
        'Val Images': base_path / "val" / "images",
        'Test Images': base_path / "test" / "images",
        'Config File': base_path / "data.yaml",
    }
    
    for name, path in paths_to_check.items():
        if path.exists():
            if path.is_dir():
                count = len(list(path.iterdir()))
                print(f"   [OK] {name}: {path} ({count} items)")
            else:
                print(f"   [OK] {name}: {path}")
        else:
            print(f"   [MISSING] {name}: {path}")
    print()
    
    # 4. Check GPU
    print("4. GPU Availability")
    print("-" * 70)
    try:
        import torch
        if torch.cuda.is_available():
            print(f"   [OK] CUDA Available")
            print(f"   [OK] GPU Count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"   [OK] GPU {i}: {torch.cuda.get_device_name(i)}")
                props = torch.cuda.get_device_properties(i)
                print(f"     - VRAM: {props.total_memory / 1024**3:.2f} GB")
        else:
            print(f"   [WARNING] CUDA not available, will use CPU")
    except Exception as e:
        print(f"   [ERROR] GPU check failed: {e}")
    print()
    
    # 5. Check code files
    print("5. Code Files Integrity")
    print("-" * 70)
    code_path = Path("c:/Users/ASUS/Downloads/数字识别器/交通标志检测/code")
    code_files = ['train_model.py', 'run_infer.py', 'baseline_infer.py', 'data.yaml']
    
    for file in code_files:
        file_path = code_path / file
        if file_path.exists():
            size = file_path.stat().st_size
            print(f"   [OK] {file}: {size} bytes")
        else:
            print(f"   [MISSING] {file}")
    print()
    
    print("=" * 70)
    print("Environment Check Complete!")
    print("=" * 70)
    print()
    print("Next Steps:")
    print("1. If all checks show [OK], you can start training")
    print("2. If there are [WARNING] or [ERROR], resolve those first")
    print("3. Training command:")
    print("   cd \"c:/Users/ASUS/Downloads/数字识别器/交通标志检测/code\"")
    print("   python train_model.py")
    print("4. Inference command:")
    print("   python run_infer.py")

if __name__ == "__main__":
    check_environment()
