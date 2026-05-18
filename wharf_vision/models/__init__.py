"""
模型模块
"""

from .global_detector import GlobalDetector
from .person_attr_detector import PersonAttrDetector
from .fire_attr_detector import FireAttrDetector

__all__ = [
    'GlobalDetector',
    'PersonAttrDetector',
    'FireAttrDetector',
]
