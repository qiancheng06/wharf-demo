"""
数据处理工具函数
"""

import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Union
from datetime import datetime


def load_image(image_path: Union[str, Path]) -> np.ndarray:
    """
    加载图像
    
    Args:
        image_path: 图像路径
        
    Returns:
        图像数组 (BGR格式)
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"无法加载图像: {image_path}")
    return image


def save_image(image: np.ndarray, output_path: Union[str, Path]):
    """
    保存图像
    
    Args:
        image: 图像数组
        output_path: 输出路径
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def save_results_json(
    results: Dict[str, Any],
    output_path: Union[str, Path],
    indent: int = 2
):
    """
    保存结果为JSON文件
    
    Args:
        results: 结果字典
        output_path: 输出路径
        indent: JSON缩进
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 转换numpy类型为Python类型
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(item) for item in obj]
        return obj
    
    results = convert(results)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=indent, ensure_ascii=False)
    
    print(f"结果已保存: {output_path}")


def load_results_json(input_path: Union[str, Path]) -> Dict[str, Any]:
    """
    从JSON文件加载结果
    
    Args:
        input_path: 输入路径
        
    Returns:
        结果字典
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_coco_annotation(
    image_id: int,
    category_id: int,
    bbox: List[float],
    score: float = 1.0,
    segmentation: List[float] = None
) -> Dict[str, Any]:
    """
    创建COCO格式的标注
    
    Args:
        image_id: 图像ID
        category_id: 类别ID
        bbox: 边界框 [x, y, width, height]
        score: 置信度
        segmentation: 分割掩码（可选）
        
    Returns:
        COCO格式标注字典
    """
    annotation = {
        'image_id': image_id,
        'category_id': category_id,
        'bbox': bbox,
        'score': score,
        'area': bbox[2] * bbox[3]
    }
    
    if segmentation:
        annotation['segmentation'] = segmentation
    
    return annotation


def xyxy_to_xywh(bbox: List[int]) -> List[float]:
    """
    将XYXY格式转换为XYWH格式
    
    Args:
        bbox: [x1, y1, x2, y2]
        
    Returns:
        [x, y, width, height]
    """
    x1, y1, x2, y2 = bbox
    return [x1, y1, x2 - x1, y2 - y1]


def xywh_to_xyxy(bbox: List[float]) -> List[int]:
    """
    将XYWH格式转换为XYXY格式
    
    Args:
        bbox: [x, y, width, height]
        
    Returns:
        [x1, y1, x2, y2]
    """
    x, y, w, h = bbox
    return [int(x), int(y), int(x + w), int(y + h)]


def crop_image(
    image: np.ndarray,
    bbox: List[int],
    padding: int = 0
) -> np.ndarray:
    """
    根据边界框裁剪图像
    
    Args:
        image: 输入图像
        bbox: 边界框 [x1, y1, x2, y2]
        padding: 填充像素
        
    Returns:
        裁剪后的图像
    """
    x1, y1, x2, y2 = bbox
    
    # 添加填充
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(image.shape[1], x2 + padding)
    y2 = min(image.shape[0], y2 + padding)
    
    return image[y1:y2, x1:x2]


def resize_image(
    image: np.ndarray,
    target_size: int,
    keep_aspect: bool = True
) -> tuple:
    """
    调整图像尺寸
    
    Args:
        image: 输入图像
        target_size: 目标尺寸
        keep_aspect: 是否保持宽高比
        
    Returns:
        (调整后的图像, 缩放比例)
    """
    h, w = image.shape[:2]
    
    if keep_aspect:
        scale = target_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
    else:
        new_h, new_w = target_size, target_size
        scale = target_size / max(h, w)
    
    resized = cv2.resize(image, (new_w, new_h))
    return resized, scale


def generate_report(
    results: List[Dict[str, Any]],
    output_path: Union[str, Path]
):
    """
    生成检测报告
    
    Args:
        results: 检测结果列表
        output_path: 输出路径
    """
    report = {
        'generated_at': datetime.now().isoformat(),
        'total_frames': len(results),
        'summary': {},
        'details': results
    }
    
    # 统计各类别
    class_counts = {}
    for frame_result in results:
        for det in frame_result.get('detections', []):
            class_name = det.get('class', 'unknown')
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    report['summary'] = {
        'class_counts': class_counts,
        'total_detections': sum(class_counts.values())
    }
    
    save_results_json(report, output_path)
