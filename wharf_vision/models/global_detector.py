"""
一级全局检测器 - Global Detection
使用YOLOv8s进行全图基础目标检测

检测类别:
- person: 人员
- container: 集装箱
- fire_region: 疑似烟火区域
- obstacle: 障碍物
"""

from .base_detector import BaseDetector


class GlobalDetector(BaseDetector):
    """
    一级全局检测器
    
    职责:
    - 全图基础目标检测
    - 输出全局视觉实体
    
    模型选型: YOLOv8s
    原因:
    - 全图检测需要更稳定召回
    - 夜间与复杂背景鲁棒性更强
    - 比nano模型更适合码头场景
    """
    
    # 类别定义
    CLASS_NAMES = {
        0: 'person',
        1: 'container',
        2: 'fire_region',
        3: 'obstacle'
    }
    
    def __init__(
        self,
        model_path: str = None,
        device: str = 'auto',
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45
    ):
        """
        初始化全局检测器
        
        Args:
            model_path: 模型路径，默认使用yolov8s
            device: 运行设备
            conf_threshold: 置信度阈值
            iou_threshold: NMS IoU阈值
        """
        # 默认使用yolov8s预训练模型
        if model_path is None:
            model_path = 'yolov8s.pt'
        
        super().__init__(
            model_path=model_path,
            device=device,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            imgsz=640  # 一级模型使用640x640
        )
        
        self.class_names = self.CLASS_NAMES
        
        print(f"[GlobalDetector] 一级全局检测器初始化完成")
        print(f"[GlobalDetector] 检测类别: {list(self.CLASS_NAMES.values())}")
    
    def detect(self, image):
        """
        执行全局检测
        
        Args:
            image: 输入图像
            
        Returns:
            检测结果列表
        """
        results = super().detect(image)
        
        # 过滤只保留我们关心的类别
        valid_classes = set(self.CLASS_NAMES.values())
        filtered_results = [
            r for r in results 
            if r['class'] in valid_classes
        ]
        
        return filtered_results
