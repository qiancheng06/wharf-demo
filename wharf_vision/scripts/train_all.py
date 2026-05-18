"""
一键训练所有模型 - 测试版
依次训练三个模型，每个3个epoch
"""

import sys
import time
from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def train_model(name, model_yaml, data_yaml, epochs=3, imgsz=640, batch=4):
    """训练单个模型（从yaml构建，不下载预训练权重）"""
    print(f"\n{'='*60}")
    print(f"训练: {name}")
    print(f"模型配置: {model_yaml}")
    print(f"数据: {data_yaml}")
    print(f"{'='*60}")
    
    start = time.time()
    
    # 从yaml构建模型（无需下载预训练权重）
    model = YOLO(model_yaml)
    
    # 训练
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device='cpu',
        patience=50,
        save=True,
        save_period=epochs,
        cache=False,
        workers=0,
        project=str(PROJECT_ROOT / "runs" / "train"),
        name=name,
        exist_ok=True,
        pretrained=False,
        verbose=True
    )
    
    elapsed = time.time() - start
    print(f"\n{name} 训练完成! 耗时: {elapsed:.1f}s")
    
    # 复制最佳模型到models目录
    best_pt = PROJECT_ROOT / "runs" / "train" / name / "weights" / "best.pt"
    if best_pt.exists():
        import shutil
        dst = MODELS_DIR / f"{name}.pt"
        shutil.copy(str(best_pt), str(dst))
        print(f"最佳模型已保存: {dst}")
    
    return results


def main():
    print("=" * 60)
    print("数字码头 - 一键训练 (测试版)")
    print("=" * 60)
    
    total_start = time.time()
    
    # 1. 一级全局检测模型 (YOLOv8s)
    train_model(
        name="global_detector",
        model_yaml="yolov8s.yaml",
        data_yaml=DATA_DIR / "global_dataset.yaml",
        epochs=3,
        imgsz=640,
        batch=4
    )
    
    # 2. 二级人员属性模型 (YOLOv8n)
    train_model(
        name="person_attr_detector",
        model_yaml="yolov8n.yaml",
        data_yaml=DATA_DIR / "person_attr_dataset.yaml",
        epochs=3,
        imgsz=320,
        batch=4
    )
    
    # 3. 二级烟火属性模型 (YOLOv8n)
    train_model(
        name="fire_attr_detector",
        model_yaml="yolov8n.yaml",
        data_yaml=DATA_DIR / "fire_attr_dataset.yaml",
        epochs=3,
        imgsz=320,
        batch=4
    )
    
    total_elapsed = time.time() - total_start
    
    print(f"\n{'='*60}")
    print(f"全部训练完成! 总耗时: {total_elapsed:.1f}s")
    print(f"{'='*60}")
    print(f"\n模型文件:")
    for f in MODELS_DIR.glob("*.pt"):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name}: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
