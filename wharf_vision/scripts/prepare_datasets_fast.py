"""
快速数据集准备脚本
创建合成数据集用于快速开始训练
"""

import os
import sys
import cv2
import numpy as np
import shutil
from pathlib import Path
from tqdm import tqdm

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def create_global_dataset(output_dir, num_samples=200):
    """
    创建全局检测数据集（person, container, fire_region, obstacle）
    
    Args:
        output_dir: 输出目录
        num_samples: 样本数量
    """
    print(f"创建全局检测数据集: {num_samples} 样本")
    
    output_dir = Path(output_dir)
    train_images = output_dir / "images" / "train"
    train_labels = output_dir / "labels" / "train"
    val_images = output_dir / "images" / "val"
    val_labels = output_dir / "labels" / "val"
    
    for d in [train_images, train_labels, val_images, val_labels]:
        d.mkdir(parents=True, exist_ok=True)
    
    # 类别定义
    classes = ['person', 'container', 'fire_region', 'obstacle']
    
    for i in tqdm(range(num_samples), desc="生成全局检测数据"):
        # 创建场景背景
        img = np.ones((640, 640, 3), dtype=np.uint8) * 240
        
        # 添加地面线
        cv2.line(img, (0, 500), (640, 500), (200, 200, 200), 2)
        
        labels = []
        num_objects = np.random.randint(2, 6)
        
        for j in range(num_objects):
            class_id = np.random.randint(0, len(classes))
            class_name = classes[class_id]
            
            if class_name == 'person':
                # 绘制简化的行人
                x = np.random.randint(50, 550)
                y = np.random.randint(300, 450)
                w = np.random.randint(40, 80)
                h = np.random.randint(100, 180)
                
                # 身体（矩形）
                cv2.rectangle(img, (x, y), (x+w, y+h), (100, 100, 200), -1)
                # 头（圆形）
                cv2.circle(img, (x+w//2, y-15), 15, (150, 150, 250), -1)
                
            elif class_name == 'container':
                # 绘制集装箱
                x = np.random.randint(50, 500)
                y = np.random.randint(350, 480)
                w = np.random.randint(100, 200)
                h = np.random.randint(60, 100)
                
                cv2.rectangle(img, (x, y), (x+w, y+h), (200, 150, 100), -1)
                cv2.rectangle(img, (x, y), (x+w, y+h), (150, 100, 50), 2)
                
            elif class_name == 'fire_region':
                # 绘制火焰区域
                x = np.random.randint(50, 550)
                y = np.random.randint(400, 550)
                w = np.random.randint(60, 120)
                h = np.random.randint(60, 120)
                
                # 火焰颜色渐变
                for dy in range(h):
                    intensity = int(255 * (1 - dy/h))
                    color = (0, intensity//2, intensity)  # 红色调
                    cv2.line(img, (x, y+dy), (x+w, y+dy), color, 1)
                    
            else:  # obstacle
                # 绘制障碍物
                x = np.random.randint(50, 550)
                y = np.random.randint(400, 550)
                w = np.random.randint(50, 100)
                h = np.random.randint(50, 100)
                
                cv2.rectangle(img, (x, y), (x+w, y+h), (128, 128, 128), -1)
            
            # YOLO格式标签
            x_center = (x + w/2) / 640
            y_center = (y + h/2) / 640
            w_norm = w / 640
            h_norm = h / 640
            
            labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
        
        # 划分训练集和验证集
        is_val = i < num_samples // 5
        img_dir = val_images if is_val else train_images
        lbl_dir = val_labels if is_val else train_labels
        
        # 保存
        img_path = img_dir / f"global_{i:04d}.jpg"
        cv2.imwrite(str(img_path), img)
        
        lbl_path = lbl_dir / f"global_{i:04d}.txt"
        with open(lbl_path, 'w') as f:
            f.write('\n'.join(labels))
    
    print(f"✓ 全局检测数据集创建完成")
    print(f"  训练集: {num_samples - num_samples//5} 张")
    print(f"  验证集: {num_samples//5} 张")


def create_person_attr_dataset(output_dir, num_samples=200):
    """
    创建人员属性数据集（helmet, vest, jacket, work_clothes, military, cigarette）
    
    Args:
        output_dir: 输出目录
        num_samples: 样本数量
    """
    print(f"创建人员属性数据集: {num_samples} 样本")
    
    output_dir = Path(output_dir)
    train_images = output_dir / "images" / "train"
    train_labels = output_dir / "labels" / "train"
    val_images = output_dir / "images" / "val"
    val_labels = output_dir / "labels" / "val"
    
    for d in [train_images, train_labels, val_images, val_labels]:
        d.mkdir(parents=True, exist_ok=True)
    
    # 属性类别
    attr_classes = ['helmet', 'reflective_vest', 'life_jacket', 'work_clothes', 'military_uniform', 'suspected_cigarette']
    
    for i in tqdm(range(num_samples), desc="生成人员属性数据"):
        # 创建人员裁剪区域背景
        img = np.ones((320, 320, 3), dtype=np.uint8) * 220
        
        labels = []
        
        # 绘制人体轮廓
        body_x, body_y = 100, 80
        body_w, body_h = 120, 200
        
        # 身体
        cv2.rectangle(img, (body_x, body_y+50), (body_x+body_w, body_y+body_h), (100, 100, 150), -1)
        
        # 随机添加属性
        num_attrs = np.random.randint(1, 4)
        selected_attrs = np.random.choice(len(attr_classes), num_attrs, replace=False)
        
        for attr_id in selected_attrs:
            attr_name = attr_classes[attr_id]
            
            if attr_name == 'helmet':
                # 安全帽（头部上方）
                helmet_y = body_y - 20
                cv2.ellipse(img, (body_x+body_w//2, helmet_y), (35, 25), 0, 180, 360, (0, 255, 255), -1)
                # 标签
                labels.append(f"0 {(body_x+body_w//2)/320:.6f} {helmet_y/320:.6f} {70/320:.6f} {50/320:.6f}")
                
            elif attr_name == 'reflective_vest':
                # 反光衣
                vest_y = body_y + 60
                cv2.rectangle(img, (body_x+10, vest_y), (body_x+body_w-10, vest_y+80), (0, 255, 255), -1)
                # 反光条
                cv2.rectangle(img, (body_x+10, vest_y+30), (body_x+body_w-10, vest_y+35), (255, 255, 255), -1)
                labels.append(f"1 {(body_x+body_w//2)/320:.6f} {(vest_y+40)/320:.6f} {(body_w-20)/320:.6f} {80/320:.6f}")
                
            elif attr_name == 'life_jacket':
                # 救生衣
                jacket_y = body_y + 50
                cv2.rectangle(img, (body_x+15, jacket_y), (body_x+body_w-15, jacket_y+90), (0, 100, 255), -1)
                labels.append(f"2 {(body_x+body_w//2)/320:.6f} {(jacket_y+45)/320:.6f} {(body_w-30)/320:.6f} {90/320:.6f}")
                
            elif attr_name == 'work_clothes':
                # 工作服
                cv2.rectangle(img, (body_x+5, body_y+50), (body_x+body_w-5, body_y+150), (100, 50, 50), -1)
                labels.append(f"3 {(body_x+body_w//2)/320:.6f} {(body_y+100)/320:.6f} {(body_w-10)/320:.6f} {100/320:.6f}")
                
            elif attr_name == 'military_uniform':
                # 军装（绿色）
                cv2.rectangle(img, (body_x+5, body_y+50), (body_x+body_w-5, body_y+150), (50, 100, 50), -1)
                labels.append(f"4 {(body_x+body_w//2)/320:.6f} {(body_y+100)/320:.6f} {(body_w-10)/320:.6f} {100/320:.6f}")
                
            elif attr_name == 'suspected_cigarette':
                # 疑似烟支（手部附近）
                hand_x = body_x + body_w + 10
                hand_y = body_y + 100
                cv2.rectangle(img, (hand_x, hand_y), (hand_x+20, hand_y+5), (200, 200, 200), -1)
                # 烟雾效果
                cv2.circle(img, (hand_x+25, hand_y-5), 8, (180, 180, 180), -1)
                labels.append(f"5 {(hand_x+10)/320:.6f} {(hand_y)/320:.6f} {30/320:.6f} {20/320:.6f}")
        
        # 划分训练集和验证集
        is_val = i < num_samples // 5
        img_dir = val_images if is_val else train_images
        lbl_dir = val_labels if is_val else train_labels
        
        # 保存
        img_path = img_dir / f"person_attr_{i:04d}.jpg"
        cv2.imwrite(str(img_path), img)
        
        lbl_path = lbl_dir / f"person_attr_{i:04d}.txt"
        with open(lbl_path, 'w') as f:
            f.write('\n'.join(labels))
    
    print(f"✓ 人员属性数据集创建完成")
    print(f"  训练集: {num_samples - num_samples//5} 张")
    print(f"  验证集: {num_samples//5} 张")


def create_fire_attr_dataset(output_dir, num_samples=200):
    """
    创建烟火属性数据集（fire, smoke, light_interference, welding_interference）
    
    Args:
        output_dir: 输出目录
        num_samples: 样本数量
    """
    print(f"创建烟火属性数据集: {num_samples} 样本")
    
    output_dir = Path(output_dir)
    train_images = output_dir / "images" / "train"
    train_labels = output_dir / "labels" / "train"
    val_images = output_dir / "images" / "val"
    val_labels = output_dir / "labels" / "val"
    
    for d in [train_images, train_labels, val_images, val_labels]:
        d.mkdir(parents=True, exist_ok=True)
    
    # 烟火属性类别
    fire_classes = ['fire', 'smoke', 'light_interference', 'welding_interference']
    
    for i in tqdm(range(num_samples), desc="生成烟火属性数据"):
        # 创建烟火区域背景
        img = np.ones((320, 320, 3), dtype=np.uint8) * 50  # 深色背景
        
        labels = []
        
        # 随机选择一种主要类型
        main_class = np.random.randint(0, len(fire_classes))
        class_name = fire_classes[main_class]
        
        if class_name == 'fire':
            # 明火 - 红色/橙色渐变
            center_x = np.random.randint(100, 220)
            center_y = np.random.randint(200, 280)
            
            for r in range(60, 0, -5):
                intensity = int(255 * (60-r) / 60)
                color = (0, intensity//2, intensity)  # BGR: 红色
                cv2.circle(img, (center_x, center_y), r, color, -1)
            
            labels.append(f"0 {center_x/320:.6f} {center_y/320:.6f} {120/320:.6f} {120/320:.6f}")
            
        elif class_name == 'smoke':
            # 烟雾 - 灰色扩散
            center_x = np.random.randint(80, 240)
            center_y = np.random.randint(100, 200)
            
            for r in range(80, 0, -10):
                gray = int(100 + 100 * (80-r) / 80)
                cv2.circle(img, (center_x, center_y), r, (gray, gray, gray), -1)
            
            labels.append(f"1 {center_x/320:.6f} {center_y/320:.6f} {160/320:.6f} {160/320:.6f}")
            
        elif class_name == 'light_interference':
            # 灯光干扰 - 白色强光
            center_x = np.random.randint(100, 220)
            center_y = np.random.randint(100, 220)
            
            # 光晕效果
            for r in range(100, 0, -5):
                intensity = int(200 + 55 * (100-r) / 100)
                cv2.circle(img, (center_x, center_y), r, (intensity, intensity, intensity), -1)
            
            labels.append(f"2 {center_x/320:.6f} {center_y/320:.6f} {200/320:.6f} {200/320:.6f}")
            
        else:  # welding_interference
            # 电焊干扰 - 蓝白色闪光
            center_x = np.random.randint(100, 220)
            center_y = np.random.randint(150, 250)
            
            # 电焊火花效果
            for _ in range(20):
                spark_x = center_x + np.random.randint(-40, 40)
                spark_y = center_y + np.random.randint(-40, 40)
                cv2.circle(img, (spark_x, spark_y), np.random.randint(2, 8), (255, 255, 200), -1)
            
            # 核心强光
            cv2.circle(img, (center_x, center_y), 30, (255, 255, 255), -1)
            
            labels.append(f"3 {center_x/320:.6f} {center_y/320:.6f} {100/320:.6f} {100/320:.6f}")
        
        # 划分训练集和验证集
        is_val = i < num_samples // 5
        img_dir = val_images if is_val else train_images
        lbl_dir = val_labels if is_val else train_labels
        
        # 保存
        img_path = img_dir / f"fire_attr_{i:04d}.jpg"
        cv2.imwrite(str(img_path), img)
        
        lbl_path = lbl_dir / f"fire_attr_{i:04d}.txt"
        with open(lbl_path, 'w') as f:
            f.write('\n'.join(labels))
    
    print(f"✓ 烟火属性数据集创建完成")
    print(f"  训练集: {num_samples - num_samples//5} 张")
    print(f"  验证集: {num_samples//5} 张")


def create_dataset_yaml_files():
    """创建数据集配置文件"""
    
    # 全局检测数据集配置
    global_yaml = DATA_DIR / "global_dataset.yaml"
    with open(global_yaml, 'w') as f:
        f.write("""# 数字码头 - 一级全局检测数据集
path: ./data/global_dataset
train: images/train
val: images/val

names:
  0: person
  1: container
  2: fire_region
  3: obstacle
""")
    
    # 人员属性数据集配置
    person_yaml = DATA_DIR / "person_attr_dataset.yaml"
    with open(person_yaml, 'w') as f:
        f.write("""# 数字码头 - 二级人员属性数据集
path: ./data/person_attr_dataset
train: images/train
val: images/val

names:
  0: helmet
  1: reflective_vest
  2: life_jacket
  3: work_clothes
  4: military_uniform
  5: suspected_cigarette
""")
    
    # 烟火属性数据集配置
    fire_yaml = DATA_DIR / "fire_attr_dataset.yaml"
    with open(fire_yaml, 'w') as f:
        f.write("""# 数字码头 - 二级烟火属性数据集
path: ./data/fire_attr_dataset
train: images/train
val: images/val

names:
  0: fire
  1: smoke
  2: light_interference
  3: welding_interference
""")
    
    print("✓ 数据集配置文件创建完成")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='快速准备训练数据集')
    parser.add_argument('--samples', type=int, default=200,
                       help='每个数据集的样本数量')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("数字码头 - 快速数据集准备")
    print("=" * 60)
    print(f"每个数据集样本数: {args.samples}")
    print("=" * 60)
    
    # 创建数据目录
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. 全局检测数据集
    print("\n[1/3] 创建全局检测数据集...")
    create_global_dataset(DATA_DIR / "global_dataset", args.samples)
    
    # 2. 人员属性数据集
    print("\n[2/3] 创建人员属性数据集...")
    create_person_attr_dataset(DATA_DIR / "person_attr_dataset", args.samples)
    
    # 3. 烟火属性数据集
    print("\n[3/3] 创建烟火属性数据集...")
    create_fire_attr_dataset(DATA_DIR / "fire_attr_dataset", args.samples)
    
    # 创建配置文件
    print("\n创建数据集配置文件...")
    create_dataset_yaml_files()
    
    print("\n" + "=" * 60)
    print("数据集准备完成!")
    print("=" * 60)
    print(f"数据目录: {DATA_DIR}")
    print("\n数据集结构:")
    print("  data/global_dataset/         - 一级全局检测数据集")
    print("  data/person_attr_dataset/    - 二级人员属性数据集")
    print("  data/fire_attr_dataset/      - 二级烟火属性数据集")
    print("\n配置文件:")
    print("  data/global_dataset.yaml")
    print("  data/person_attr_dataset.yaml")
    print("  data/fire_attr_dataset.yaml")
    print("\n开始训练:")
    print("  python scripts/train_global.py --data data/global_dataset.yaml")
    print("  python scripts/train_person_attr.py --data data/person_attr_dataset.yaml")
    print("  python scripts/train_fire_attr.py --data data/fire_attr_dataset.yaml")


if __name__ == '__main__':
    main()
