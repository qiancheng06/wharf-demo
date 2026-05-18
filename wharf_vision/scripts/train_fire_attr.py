"""
训练二级烟火属性检测模型
Usage: python scripts/train_fire_attr.py --data config/fire_attr.yaml
"""

import argparse
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def train_fire_attr_model(data_yaml, epochs=100, imgsz=320, batch=32, device='auto'):
    """
    训练二级烟火属性检测模型
    
    Args:
        data_yaml: 数据配置文件路径
        epochs: 训练轮数
        imgsz: 输入图像尺寸
        batch: batch大小
        device: 训练设备
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("错误: 未安装ultralytics，请先安装: pip install ultralytics")
        return
    
    print("=" * 60)
    print("数字码头 - 二级烟火属性检测模型训练")
    print("=" * 60)
    print(f"模型: YOLOv8n")
    print(f"数据配置: {data_yaml}")
    print(f"训练轮数: {epochs}")
    print(f"图像尺寸: {imgsz}")
    print(f"Batch大小: {batch}")
    print(f"设备: {device}")
    print("=" * 60)
    print("\n检测类别:")
    print("  0: fire (明火)")
    print("  1: smoke (烟雾)")
    print("  2: light_interference (灯光干扰)")
    print("  3: welding_interference (电焊干扰)")
    print("\n注意: 干扰类别用于降低误报")
    print("=" * 60)
    
    # 加载预训练模型
    model = YOLO('yolov8n.pt')
    
    # 开始训练
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        patience=50,
        save=True,
        save_period=10,
        cache=True,
        workers=8,
        project='runs/train',
        name='fire_attr_detector',
        exist_ok=True,
        pretrained=True,
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        momentum=0.9,
        weight_decay=0.0005,
        # 数据增强（针对烟火场景）
        hsv_h=0.02,   # 增强色调变化
        hsv_s=0.8,    # 增强饱和度
        hsv_v=0.5,    # 增强亮度（模拟昼夜）
        degrees=0.0,  # 烟火方向固定，不旋转
        translate=0.1,
        scale=0.4,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=0.5,   # 降低mosaic
        mixup=0.0,
        copy_paste=0.0,
        verbose=True
    )
    
    print("\n" + "=" * 60)
    print("训练完成!")
    print(f"最佳模型: {results.best}")
    print("=" * 60)
    
    return results


def validate_model(model_path, data_yaml, device='auto'):
    """
    验证模型
    
    Args:
        model_path: 模型路径
        data_yaml: 数据配置文件路径
        device: 验证设备
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("错误: 未安装ultralytics")
        return
    
    print("=" * 60)
    print("模型验证")
    print("=" * 60)
    
    model = YOLO(model_path)
    metrics = model.val(
        data=data_yaml,
        device=device,
        verbose=True
    )
    
    print("\n验证结果:")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    
    # 各类别详细结果
    print("\n各类别AP50:")
    class_names = ['fire', 'smoke', 'light_interference', 'welding_interference']
    for i, name in enumerate(class_names):
        if i < len(metrics.box.ap50):
            print(f"  {name}: {metrics.box.ap50[i]:.4f}")
    
    return metrics


def export_model(model_path, format='onnx'):
    """
    导出模型
    
    Args:
        model_path: 模型路径
        format: 导出格式
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("错误: 未安装ultralytics")
        return
    
    print("=" * 60)
    print(f"导出模型为 {format} 格式")
    print("=" * 60)
    
    model = YOLO(model_path)
    model.export(format=format)
    
    print("导出完成!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='训练二级烟火属性检测模型')
    parser.add_argument('--data', type=str, default='config/fire_attr.yaml',
                       help='数据配置文件路径')
    parser.add_argument('--epochs', type=int, default=100,
                       help='训练轮数')
    parser.add_argument('--imgsz', type=int, default=320,
                       help='输入图像尺寸')
    parser.add_argument('--batch', type=int, default=32,
                       help='batch大小')
    parser.add_argument('--device', type=str, default='auto',
                       help='训练设备')
    parser.add_argument('--val', action='store_true',
                       help='仅进行验证')
    parser.add_argument('--export', type=str, default=None,
                       help='导出格式')
    
    args = parser.parse_args()
    
    if args.val:
        best_model = 'runs/train/fire_attr_detector/weights/best.pt'
        validate_model(best_model, args.data, args.device)
    elif args.export:
        best_model = 'runs/train/fire_attr_detector/weights/best.pt'
        export_model(best_model, args.export)
    else:
        train_fire_attr_model(
            data_yaml=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device
        )
