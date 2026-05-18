"""
可视化工具函数
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Tuple


# 颜色映射表
COLOR_MAP = {
    'person': (0, 255, 0),           # 绿色
    'container': (255, 165, 0),      # 橙色
    'fire_region': (0, 0, 255),      # 红色
    'obstacle': (128, 128, 128),     # 灰色
    'helmet': (255, 255, 0),         # 青色
    'reflective_vest': (255, 0, 255), # 紫色
    'life_jacket': (0, 255, 255),    # 黄色
    'work_clothes': (128, 0, 128),   # 深紫
    'military_uniform': (0, 128, 0), # 深绿
    'suspected_cigarette': (128, 128, 0), # 橄榄
    'fire': (0, 0, 255),             # 红色
    'smoke': (128, 128, 128),        # 灰色
    'light_interference': (255, 255, 0), # 青色
    'welding_interference': (0, 255, 255), # 黄色
}


def get_color(class_name: str) -> Tuple[int, int, int]:
    """获取类别对应的颜色"""
    return COLOR_MAP.get(class_name, (255, 255, 255))


def draw_detections(
    image: np.ndarray,
    detections: List[Dict[str, Any]],
    thickness: int = 2,
    font_scale: float = 0.5,
    show_attributes: bool = True
) -> np.ndarray:
    """
    在图像上绘制检测结果
    
    Args:
        image: 输入图像
        detections: 检测结果列表
        thickness: 线条粗细
        font_scale: 字体大小
        show_attributes: 是否显示属性
        
    Returns:
        绘制后的图像
    """
    result = image.copy()
    
    for det in detections:
        class_name = det.get('class', 'unknown')
        bbox = det.get('bbox', [0, 0, 0, 0])
        conf = det.get('conf', 0.0)
        attributes = det.get('attributes', {})
        
        x1, y1, x2, y2 = bbox
        color = get_color(class_name)
        
        # 绘制检测框
        cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)
        
        # 构建标签
        label = f"{class_name}: {conf:.2f}"
        
        # 添加属性信息
        if show_attributes and attributes:
            attr_texts = []
            
            if class_name == 'person':
                if attributes.get('is_military'):
                    attr_texts.append("军人")
                if attributes.get('has_helmet'):
                    attr_texts.append("安全帽")
                if attributes.get('has_reflective_vest'):
                    attr_texts.append("反光衣")
                if attributes.get('has_life_jacket'):
                    attr_texts.append("救生衣")
                if attributes.get('suspected_cigarette'):
                    attr_texts.append("疑似烟支")
            
            elif class_name == 'fire_region':
                if attributes.get('is_fire'):
                    attr_texts.append("明火")
                if attributes.get('is_smoke'):
                    attr_texts.append("烟雾")
                if attributes.get('is_light_interference'):
                    attr_texts.append("灯光干扰")
                if attributes.get('is_welding_interference'):
                    attr_texts.append("电焊干扰")
            
            if attr_texts:
                label += f" [{', '.join(attr_texts)}]"
        
        # 计算标签尺寸
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        label_w, label_h = label_size
        
        # 绘制标签背景
        cv2.rectangle(result, 
                     (x1, y1 - label_h - 10), 
                     (x1 + label_w, y1), 
                     color, -1)
        
        # 绘制标签文字
        cv2.putText(result, label, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
    
    return result


def create_summary_image(
    image: np.ndarray,
    detections: List[Dict[str, Any]],
    frame_id: int = 0,
    fps: float = 0.0
) -> np.ndarray:
    """
    创建带统计信息的汇总图像
    
    Args:
        image: 输入图像
        detections: 检测结果
        frame_id: 帧ID
        fps: 帧率
        
    Returns:
        汇总图像
    """
    # 绘制检测结果
    result = draw_detections(image, detections)
    
    # 添加统计信息面板
    h, w = result.shape[:2]
    panel_height = 100
    
    # 创建信息面板
    panel = np.zeros((panel_height, w, 3), dtype=np.uint8)
    panel[:] = (50, 50, 50)  # 深灰色背景
    
    # 统计各类别数量
    class_counts = {}
    for det in detections:
        class_name = det.get('class', 'unknown')
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    # 绘制统计信息
    info_text = f"Frame: {frame_id} | FPS: {fps:.1f} | "
    info_text += " | ".join([f"{k}: {v}" for k, v in class_counts.items()])
    
    cv2.putText(panel, info_text, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # 合并图像和面板
    result = np.vstack([result, panel])
    
    return result


def draw_roi(
    image: np.ndarray,
    roi_points: List[Tuple[int, int]],
    color: Tuple[int, int, int] = (0, 255, 255),
    thickness: int = 2,
    label: str = None
) -> np.ndarray:
    """
    绘制ROI区域
    
    Args:
        image: 输入图像
        roi_points: ROI多边形点坐标
        color: 线条颜色
        thickness: 线条粗细
        label: 标签文字
        
    Returns:
        绘制后的图像
    """
    result = image.copy()
    
    # 绘制多边形
    points = np.array(roi_points, np.int32)
    points = points.reshape((-1, 1, 2))
    cv2.polylines(result, [points], True, color, thickness)
    
    # 绘制标签
    if label:
        # 在ROI中心位置绘制标签
        center_x = int(np.mean([p[0] for p in roi_points]))
        center_y = int(np.mean([p[1] for p in roi_points]))
        
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(result,
                     (center_x - label_size[0]//2, center_y - label_size[1] - 5),
                     (center_x + label_size[0]//2, center_y + 5),
                     color, -1)
        cv2.putText(result, label,
                   (center_x - label_size[0]//2, center_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return result
