"""
下载小型真实数据集用于测试训练
使用公开可访问的图像资源
"""

import os
import sys
import urllib.request
import ssl
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# 禁用SSL验证（用于下载）
ssl._create_default_https_context = ssl._create_unverified_context

# 小型真实图像URL列表（公开可访问）
# 使用Unsplash和Pexels的公开图片
SAMPLE_IMAGES = {
    "person": [
        "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=640&h=640&fit=crop",
        "https://images.unsplash.com/photo-1500648767791-00dcc994a43e?w=640&h=640&fit=crop",
        "https://images.unsplash.com/photo-1438761681033-6461ffad8d80?w=640&h=640&fit=crop",
        "https://images.unsplash.com/photo-1472099645785-5658abf4ff4e?w=640&h=640&fit=crop",
    ],
    "container": [
        "https://images.unsplash.com/photo-1494412574643-ff11b0a5c1c3?w=640&h=640&fit=crop",
        "https://images.unsplash.com/photo-1578575437130-527eed3abbec?w=640&h=640&fit=crop",
        "https://images.unsplash.com/photo-1586528116311-ad8dd3c8310d?w=640&h=640&fit=crop",
    ],
    "fire": [
        "https://images.unsplash.com/photo-1543005472-1b1d37fa4eae?w=640&h=640&fit=crop",
        "https://images.unsplash.com/photo-1560169573-5b09f7c1d1b2?w=640&h=640&fit=crop",
        "https://images.unsplash.com/photo-1517594422361-5eeb8ae275a9?w=640&h=640&fit=crop",
    ],
}


def download_image(url, output_path, timeout=30):
    """下载单张图片"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as response:
            with open(output_path, 'wb') as f:
                f.write(response.read())
        return True
    except Exception as e:
        print(f"下载失败 {url}: {e}")
        return False


def create_dataset_with_downloads(dataset_name, class_names, num_samples_per_class=20):
    """
    创建数据集：下载真实图片 + 生成标注
    
    Args:
        dataset_name: 数据集名称
        class_names: 类别名称列表
        num_samples_per_class: 每个类别的样本数
    """
    print(f"\n创建数据集: {dataset_name}")
    print(f"类别: {class_names}")
    
    dataset_dir = DATA_DIR / dataset_name
    train_img_dir = dataset_dir / "images" / "train"
    train_lbl_dir = dataset_dir / "labels" / "train"
    val_img_dir = dataset_dir / "images" / "val"
    val_lbl_dir = dataset_dir / "labels" / "val"
    
    for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    sample_idx = 0
    
    # 为每个类别创建样本
    for class_id, class_name in enumerate(class_names):
        print(f"  处理类别 {class_name}...")
        
        for i in range(num_samples_per_class):
            # 尝试下载图片
            img = None
            if class_name in SAMPLE_IMAGES:
                urls = SAMPLE_IMAGES[class_name]
                url = urls[i % len(urls)]
                
                temp_path = train_img_dir / f"temp_{sample_idx}.jpg"
                if download_image(url, temp_path):
                    img = cv2.imread(str(temp_path))
                    temp_path.unlink()  # 删除临时文件
            
            # 如果下载失败，生成合成图像
            if img is None:
                img = generate_synthetic_image(class_name, 640, 640)
            
            # 调整图像大小
            img = cv2.resize(img, (640, 640))
            
            # 生成标注（模拟目标在图像中的位置）
            h, w = img.shape[:2]
            
            # 随机生成1-3个目标
            num_objects = np.random.randint(1, 4)
            labels = []
            
            for _ in range(num_objects):
                # 随机生成边界框
                box_w = np.random.randint(100, 300)
                box_h = np.random.randint(100, 300)
                x1 = np.random.randint(50, w - box_w - 50)
                y1 = np.random.randint(50, h - box_h - 50)
                
                # YOLO格式: class_id x_center y_center width height
                x_center = (x1 + box_w / 2) / w
                y_center = (y1 + box_h / 2) / h
                w_norm = box_w / w
                h_norm = box_h / h
                
                labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
                
                # 在图像上绘制矩形（可视化用）
                cv2.rectangle(img, (x1, y1), (x1 + box_w, y1 + box_h), (0, 255, 0), 2)
            
            # 划分训练集和验证集 (80/20)
            is_val = (i % 5 == 0) and (i > 0)
            
            if is_val:
                img_dir = val_img_dir
                lbl_dir = val_lbl_dir
            else:
                img_dir = train_img_dir
                lbl_dir = train_lbl_dir
            
            # 保存图像
            img_path = img_dir / f"{dataset_name}_{sample_idx:04d}.jpg"
            cv2.imwrite(str(img_path), img)
            
            # 保存标签
            lbl_path = lbl_dir / f"{dataset_name}_{sample_idx:04d}.txt"
            with open(lbl_path, 'w') as f:
                f.write('\n'.join(labels))
            
            sample_idx += 1
    
    print(f"  完成: {sample_idx} 张图像")
    return sample_idx


def generate_synthetic_image(class_name, width, height):
    """生成合成图像作为备选"""
    img = np.ones((height, width, 3), dtype=np.uint8) * 240
    
    if class_name == "person":
        # 绘制简化的行人
        cv2.rectangle(img, (200, 200), (400, 500), (100, 100, 200), -1)
        cv2.circle(img, (300, 150), 50, (150, 150, 250), -1)
    elif class_name == "container":
        # 绘制集装箱
        cv2.rectangle(img, (150, 250), (490, 450), (200, 150, 100), -1)
        cv2.rectangle(img, (150, 250), (490, 450), (150, 100, 50), 3)
    elif class_name == "fire_region":
        # 绘制火焰
        for r in range(100, 0, -5):
            intensity = int(255 * (100 - r) / 100)
            cv2.circle(img, (320, 400), r, (0, intensity // 2, intensity), -1)
    elif class_name == "obstacle":
        # 绘制障碍物
        cv2.rectangle(img, (220, 300), (420, 500), (128, 128, 128), -1)
    elif class_name == "helmet":
        # 安全帽
        cv2.ellipse(img, (320, 200), (80, 60), 0, 180, 360, (0, 255, 255), -1)
        cv2.rectangle(img, (240, 250), (400, 500), (100, 100, 150), -1)
    elif class_name == "reflective_vest":
        # 反光衣
        cv2.rectangle(img, (240, 250), (400, 500), (100, 100, 150), -1)
        cv2.rectangle(img, (250, 300), (390, 450), (0, 255, 255), -1)
        cv2.rectangle(img, (250, 350), (390, 360), (255, 255, 255), -1)
    elif class_name == "fire":
        # 明火
        for r in range(80, 0, -5):
            intensity = int(255 * (80 - r) / 80)
            cv2.circle(img, (320, 350), r, (0, intensity // 2, intensity), -1)
    elif class_name == "smoke":
        # 烟雾
        for r in range(100, 0, -10):
            gray = int(100 + 100 * (100 - r) / 100)
            cv2.circle(img, (320, 300), r, (gray, gray, gray), -1)
    else:
        # 默认
        cv2.rectangle(img, (200, 200), (440, 440), (150, 150, 150), -1)
    
    return img


def create_dataset_yaml(dataset_name, class_names):
    """创建数据集YAML配置文件"""
    yaml_path = DATA_DIR / f"{dataset_name}.yaml"
    
    names_str = "\n".join([f"  {i}: {name}" for i, name in enumerate(class_names)])
    
    content = f"""# 数字码头 - {dataset_name}数据集
path: ./data/{dataset_name}
train: images/train
val: images/val

names:
{names_str}
"""
    
    with open(yaml_path, 'w') as f:
        f.write(content)
    
    print(f"  配置文件: {yaml_path}")


def main():
    """主函数"""
    print("=" * 60)
    print("数字码头 - 小型真实数据集准备")
    print("=" * 60)
    
    # 创建数据目录
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. 全局检测数据集 (person, container, fire_region, obstacle)
    print("\n[1/3] 创建全局检测数据集...")
    global_classes = ["person", "container", "fire_region", "obstacle"]
    create_dataset_with_downloads("global_dataset", global_classes, num_samples_per_class=15)
    create_dataset_yaml("global_dataset", global_classes)
    
    # 2. 人员属性数据集 (helmet, reflective_vest)
    print("\n[2/3] 创建人员属性数据集...")
    person_classes = ["helmet", "reflective_vest", "life_jacket", "work_clothes", "military_uniform", "suspected_cigarette"]
    create_dataset_with_downloads("person_attr_dataset", person_classes, num_samples_per_class=10)
    create_dataset_yaml("person_attr_dataset", person_classes)
    
    # 3. 烟火属性数据集 (fire, smoke)
    print("\n[3/3] 创建烟火属性数据集...")
    fire_classes = ["fire", "smoke", "light_interference", "welding_interference"]
    create_dataset_with_downloads("fire_attr_dataset", fire_classes, num_samples_per_class=10)
    create_dataset_yaml("fire_attr_dataset", fire_classes)
    
    print("\n" + "=" * 60)
    print("数据集准备完成!")
    print("=" * 60)
    print(f"数据目录: {DATA_DIR}")
    print("\n数据集统计:")
    
    for dataset_name in ["global_dataset", "person_attr_dataset", "fire_attr_dataset"]:
        dataset_dir = DATA_DIR / dataset_name
        train_imgs = list((dataset_dir / "images" / "train").glob("*.jpg"))
        val_imgs = list((dataset_dir / "images" / "val").glob("*.jpg"))
        print(f"  {dataset_name}: 训练集 {len(train_imgs)} 张, 验证集 {len(val_imgs)} 张")
    
    print("\n开始训练测试:")
    print("  python scripts/train_global.py --data data/global_dataset.yaml --epochs 5")
    print("  python scripts/train_person_attr.py --data data/person_attr_dataset.yaml --epochs 5")
    print("  python scripts/train_fire_attr.py --data data/fire_attr_dataset.yaml --epochs 5")


if __name__ == '__main__':
    main()
