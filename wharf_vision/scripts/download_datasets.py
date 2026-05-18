"""
数据集下载脚本
自动下载并准备训练所需的数据集
"""

import os
import sys
import urllib.request
import zipfile
import tarfile
import shutil
from pathlib import Path
from tqdm import tqdm

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# 数据集下载链接
DATASET_URLS = {
    # 人员检测数据集 - 使用COCO子集
    "person_detection": {
        "name": "Person Detection Dataset",
        "urls": [
            # COCO 2017 验证集（用于快速测试）
            ("http://images.cocodataset.org/zips/val2017.zip", "coco_val2017.zip"),
            ("http://images.cocodataset.org/annotations/annotations_trainval2017.zip", "coco_annotations.zip"),
        ],
        "target_dir": "global_dataset"
    },
    
    # 安全帽检测数据集
    "helmet_detection": {
        "name": "Safety Helmet Detection Dataset",
        "urls": [
            # 使用Roboflow公开数据集
            ("https://public.roboflow.com/ds/ safety-helmet-detection", "safety_helmet.zip"),
        ],
        "target_dir": "person_attr_dataset"
    },
    
    # 烟火检测数据集
    "fire_detection": {
        "name": "Fire Detection Dataset",
        "urls": [
            # 使用公开烟火数据集
            ("https://github.com/cair/Fire-Detection-Image-Dataset/archive/refs/heads/master.zip", "fire_dataset.zip"),
        ],
        "target_dir": "fire_attr_dataset"
    },
}


class DownloadProgressBar(tqdm):
    """下载进度条"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url, output_path):
    """
    下载文件并显示进度
    
    Args:
        url: 下载链接
        output_path: 保存路径
    """
    print(f"下载: {url}")
    print(f"保存到: {output_path}")
    
    try:
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path.name) as t:
            urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)
        print(f"✓ 下载完成: {output_path}")
        return True
    except Exception as e:
        print(f"✗ 下载失败: {e}")
        return False


def extract_archive(archive_path, extract_to):
    """
    解压压缩文件
    
    Args:
        archive_path: 压缩文件路径
        extract_to: 解压目标目录
    """
    print(f"解压: {archive_path}")
    
    try:
        if archive_path.suffix == '.zip':
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        elif archive_path.suffix in ['.tar', '.gz', '.tgz']:
            with tarfile.open(archive_path, 'r:*') as tar_ref:
                tar_ref.extractall(extract_to)
        else:
            print(f"不支持的压缩格式: {archive_path.suffix}")
            return False
        
        print(f"✓ 解压完成: {extract_to}")
        return True
    except Exception as e:
        print(f"✗ 解压失败: {e}")
        return False


def create_synthetic_dataset(output_dir, dataset_type="person"):
    """
    创建合成数据集（用于测试和演示）
    
    Args:
        output_dir: 输出目录
        dataset_type: 数据集类型
    """
    import cv2
    import numpy as np
    
    print(f"创建合成数据集: {dataset_type}")
    
    output_dir = Path(output_dir)
    images_dir = output_dir / "images" / "train"
    labels_dir = output_dir / "labels" / "train"
    
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建合成图像
    num_samples = 100
    
    for i in range(num_samples):
        # 创建空白图像
        img = np.ones((640, 640, 3), dtype=np.uint8) * 200
        
        # 添加随机矩形（模拟目标）
        labels = []
        num_objects = np.random.randint(1, 5)
        
        for j in range(num_objects):
            x1 = np.random.randint(50, 500)
            y1 = np.random.randint(50, 500)
            w = np.random.randint(50, 150)
            h = np.random.randint(50, 150)
            
            # 根据数据集类型设置颜色
            if dataset_type == "person":
                color = (100, 100, 200)  # 蓝色调
                class_id = 0
            elif dataset_type == "fire":
                color = (0, 100, 255)  # 红色调
                class_id = 0
            elif dataset_type == "helmet":
                color = (100, 200, 100)  # 绿色调
                class_id = 0
            else:
                color = (200, 100, 100)
                class_id = 0
            
            cv2.rectangle(img, (x1, y1), (x1+w, y1+h), color, -1)
            
            # YOLO格式标签: class_id x_center y_center width height (归一化)
            x_center = (x1 + w/2) / 640
            y_center = (y1 + h/2) / 640
            w_norm = w / 640
            h_norm = h / 640
            
            labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
        
        # 保存图像
        img_path = images_dir / f"sample_{i:04d}.jpg"
        cv2.imwrite(str(img_path), img)
        
        # 保存标签
        label_path = labels_dir / f"sample_{i:04d}.txt"
        with open(label_path, 'w') as f:
            f.write('\n'.join(labels))
    
    # 创建验证集（复制部分训练集）
    val_images_dir = output_dir / "images" / "val"
    val_labels_dir = output_dir / "labels" / "val"
    val_images_dir.mkdir(parents=True, exist_ok=True)
    val_labels_dir.mkdir(parents=True, exist_ok=True)
    
    # 复制20%作为验证集
    import random
    val_indices = random.sample(range(num_samples), num_samples // 5)
    
    for idx in val_indices:
        src_img = images_dir / f"sample_{idx:04d}.jpg"
        src_label = labels_dir / f"sample_{idx:04d}.txt"
        dst_img = val_images_dir / f"sample_{idx:04d}.jpg"
        dst_label = val_labels_dir / f"sample_{idx:04d}.txt"
        
        shutil.copy(src_img, dst_img)
        shutil.copy(src_label, dst_label)
    
    print(f"✓ 合成数据集创建完成: {output_dir}")
    print(f"  训练集: {num_samples - len(val_indices)} 张")
    print(f"  验证集: {len(val_indices)} 张")


def setup_coco_person_dataset():
    """
    设置COCO人员检测数据集
    下载并转换为YOLO格式
    """
    print("=" * 60)
    print("设置COCO人员检测数据集")
    print("=" * 60)
    
    coco_dir = DATA_DIR / "coco"
    coco_dir.mkdir(parents=True, exist_ok=True)
    
    # 下载COCO验证集（较小，适合快速开始）
    val_zip = coco_dir / "val2017.zip"
    anno_zip = coco_dir / "annotations_trainval2017.zip"
    
    # 下载图像
    if not val_zip.exists():
        download_file(
            "http://images.cocodataset.org/zips/val2017.zip",
            val_zip
        )
        extract_archive(val_zip, coco_dir)
    
    # 下载标注
    if not anno_zip.exists():
        download_file(
            "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
            anno_zip
        )
        extract_archive(anno_zip, coco_dir)
    
    # 转换为YOLO格式（仅提取person类别）
    convert_coco_to_yolo(coco_dir)
    
    print("✓ COCO数据集设置完成")


def convert_coco_to_yolo(coco_dir):
    """
    将COCO格式转换为YOLO格式
    
    Args:
        coco_dir: COCO数据集目录
    """
    import json
    
    print("转换COCO格式为YOLO格式...")
    
    coco_dir = Path(coco_dir)
    
    # 创建输出目录
    output_dir = DATA_DIR / "global_dataset"
    (output_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
    (output_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)
    
    # 加载COCO标注
    anno_file = coco_dir / "annotations" / "instances_val2017.json"
    if not anno_file.exists():
        print(f"标注文件不存在: {anno_file}")
        return
    
    with open(anno_file, 'r') as f:
        coco_data = json.load(f)
    
    # COCO类别ID映射（只保留person）
    # COCO person类别ID是1
    person_category_id = 1
    
    # 处理每张图像
    images_dir = coco_dir / "val2017"
    
    for img_info in tqdm(coco_data['images'], desc="处理图像"):
        img_id = img_info['id']
        img_name = img_info['file_name']
        img_w = img_info['width']
        img_h = img_info['height']
        
        # 查找该图像的所有person标注
        annotations = [a for a in coco_data['annotations'] 
                      if a['image_id'] == img_id and a['category_id'] == person_category_id]
        
        if not annotations:
            continue
        
        # 复制图像
        src_img = images_dir / img_name
        dst_img = output_dir / "images" / "val" / img_name
        
        if src_img.exists():
            shutil.copy(src_img, dst_img)
        
        # 创建YOLO格式标签
        label_name = img_name.replace('.jpg', '.txt')
        label_path = output_dir / "labels" / "val" / label_name
        
        with open(label_path, 'w') as f:
            for anno in annotations:
                bbox = anno['bbox']  # [x, y, width, height]
                x_center = (bbox[0] + bbox[2] / 2) / img_w
                y_center = (bbox[1] + bbox[3] / 2) / img_h
                w = bbox[2] / img_w
                h = bbox[3] / img_h
                
                # YOLO格式: class_id x_center y_center width height
                f.write(f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")
    
    print(f"✓ 转换完成: {output_dir}")


def download_roboflow_dataset(workspace, project, version, api_key=None):
    """
    从Roboflow下载数据集
    
    Args:
        workspace: 工作空间名称
        project: 项目名称
        version: 版本号
        api_key: API密钥（可选）
    """
    try:
        from roboflow import Roboflow
        
        print(f"从Roboflow下载: {workspace}/{project}")
        
        if api_key:
            rf = Roboflow(api_key=api_key)
        else:
            # 尝试无API key下载公开数据集
            rf = Roboflow()
        
        project_obj = rf.workspace(workspace).project(project)
        dataset = project_obj.version(version).download("yolov8")
        
        print(f"✓ 下载完成: {dataset.location}")
        return dataset.location
        
    except ImportError:
        print("未安装roboflow包，使用替代方法...")
        return None
    except Exception as e:
        print(f"下载失败: {e}")
        return None


def setup_all_datasets(use_synthetic=True):
    """
    设置所有数据集
    
    Args:
        use_synthetic: 如果下载失败，是否使用合成数据
    """
    print("=" * 60)
    print("数字码头 - 数据集准备")
    print("=" * 60)
    
    # 创建数据目录
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. 全局检测数据集（人员、集装箱等）
    print("\n[1/3] 准备全局检测数据集...")
    try:
        setup_coco_person_dataset()
    except Exception as e:
        print(f"COCO数据集准备失败: {e}")
        if use_synthetic:
            create_synthetic_dataset(DATA_DIR / "global_dataset", "person")
    
    # 2. 人员属性数据集（PPE、安全帽等）
    print("\n[2/3] 准备人员属性数据集...")
    # 尝试从Roboflow下载
    rf_path = download_roboflow_dataset("roboflow-100", "safety-helmet-detection", 1)
    if rf_path:
        # 移动到新位置
        target = DATA_DIR / "person_attr_dataset"
        if target.exists():
            shutil.rmtree(target)
        shutil.move(rf_path, target)
    elif use_synthetic:
        create_synthetic_dataset(DATA_DIR / "person_attr_dataset", "helmet")
    
    # 3. 烟火检测数据集
    print("\n[3/3] 准备烟火检测数据集...")
    try:
        fire_zip = DATA_DIR / "fire_dataset.zip"
        if not fire_zip.exists():
            download_file(
                "https://github.com/cair/Fire-Detection-Image-Dataset/archive/refs/heads/master.zip",
                fire_zip
            )
        
        fire_dir = DATA_DIR / "fire_attr_dataset"
        if fire_zip.exists() and not fire_dir.exists():
            extract_archive(fire_zip, DATA_DIR)
            # 重命名目录
            extracted_dir = DATA_DIR / "Fire-Detection-Image-Dataset-master"
            if extracted_dir.exists():
                shutil.move(extracted_dir, fire_dir)
        elif use_synthetic:
            create_synthetic_dataset(fire_dir, "fire")
            
    except Exception as e:
        print(f"烟火数据集准备失败: {e}")
        if use_synthetic:
            create_synthetic_dataset(DATA_DIR / "fire_attr_dataset", "fire")
    
    print("\n" + "=" * 60)
    print("数据集准备完成!")
    print("=" * 60)
    print(f"数据目录: {DATA_DIR}")
    
    # 列出数据集
    for dataset_dir in DATA_DIR.iterdir():
        if dataset_dir.is_dir():
            print(f"  - {dataset_dir.name}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='下载并准备训练数据集')
    parser.add_argument('--synthetic', action='store_true', default=True,
                       help='使用合成数据作为备选')
    parser.add_argument('--coco-only', action='store_true',
                       help='仅下载COCO数据集')
    
    args = parser.parse_args()
    
    if args.coco_only:
        setup_coco_person_dataset()
    else:
        setup_all_datasets(use_synthetic=args.synthetic)


if __name__ == '__main__':
    main()
