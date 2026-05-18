"""
数字码头单帧视觉感知模块
Wharf Single-Frame Vision Perception Module

基于YOLOv8的两级检测架构:
- 一级: Global Detection (YOLOv8s) - 全图基础目标检测
- 二级A: Person Attr (YOLOv8n) - 人员属性识别
- 二级B: Fire Attr (YOLOv8n) - 烟火属性识别
"""

__version__ = "1.0.0"
__author__ = "AI Assistant"

from .inference import WharfVisionInference
from .models.global_detector import GlobalDetector
from .models.person_attr_detector import PersonAttrDetector
from .models.fire_attr_detector import FireAttrDetector

__all__ = [
    'WharfVisionInference',
    'GlobalDetector',
    'PersonAttrDetector',
    'FireAttrDetector',
]
