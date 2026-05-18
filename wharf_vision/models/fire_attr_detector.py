"""
二级烟火属性检测器 - Fire/Smoke Attribute Detection
使用YOLOv8n进行烟火细分类别识别

检测类别:
- fire: 明火
- smoke: 烟雾
- light_interference: 灯光干扰
- welding_interference: 电焊干扰
"""

from .base_detector import BaseDetector


class FireAttrDetector(BaseDetector):
    """
    二级烟火属性检测器
    
    职责:
    - 输入: fire_region crop (疑似烟火区域裁剪)
    - 输出: 火焰、烟雾、干扰源细分类别
    
    模型选型: YOLOv8n
    原因:
    - crop区域较小
    - 推理速度快
    - 易于并行部署
    
    干扰识别目的:
    - light_interference: 强灯光、反光干扰（降低误报）
    - welding_interference: 电焊光干扰（降低夜间误报）
    """
    
    # 类别定义
    CLASS_NAMES = {
        0: 'fire',
        1: 'smoke',
        2: 'light_interference',
        3: 'welding_interference'
    }
    
    def __init__(
        self,
        model_path: str = None,
        device: str = 'auto',
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45
    ):
        """
        初始化烟火属性检测器
        
        Args:
            model_path: 模型路径
            device: 运行设备
            conf_threshold: 置信度阈值
            iou_threshold: NMS IoU阈值
        """
        super().__init__(
            model_path=model_path,
            device=device,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            imgsz=320  # 二级模型使用320x320
        )
        
        self.class_names = self.CLASS_NAMES
        
        print(f"[FireAttrDetector] 二级烟火属性检测器初始化完成")
        print(f"[FireAttrDetector] 检测类别: {list(self.CLASS_NAMES.values())}")
    
    def detect(self, fire_crop):
        """
        执行烟火属性检测
        
        Args:
            fire_crop: 烟火区域裁剪图像
            
        Returns:
            属性检测结果列表
        """
        if fire_crop is None or fire_crop.size == 0:
            return []
        
        results = super().detect(fire_crop)
        
        # 过滤只保留烟火属性类别
        valid_classes = set(self.CLASS_NAMES.values())
        filtered_results = [
            r for r in results 
            if r['class'] in valid_classes
        ]
        
        return filtered_results
    
    def classify_fire_region(self, fire_crop) -> dict:
        """
        对烟火区域进行分类
        
        Args:
            fire_crop: 烟火区域裁剪
            
        Returns:
            分类结果字典
        """
        results = self.detect(fire_crop)
        
        classification = {
            'is_fire': False,
            'is_smoke': False,
            'is_light_interference': False,
            'is_welding_interference': False,
            'fire_conf': 0.0,
            'smoke_conf': 0.0,
            'is_interference': False
        }
        
        for r in results:
            class_name = r['class']
            conf = r['conf']
            
            if class_name == 'fire':
                classification['is_fire'] = True
                classification['fire_conf'] = conf
            elif class_name == 'smoke':
                classification['is_smoke'] = True
                classification['smoke_conf'] = conf
            elif class_name == 'light_interference':
                classification['is_light_interference'] = True
                classification['is_interference'] = True
            elif class_name == 'welding_interference':
                classification['is_welding_interference'] = True
                classification['is_interference'] = True
        
        return classification
    
    def is_real_fire(self, fire_crop, fire_conf_threshold: float = 0.5) -> tuple:
        """
        判断是否为真实火情（非干扰）
        
        Args:
            fire_crop: 烟火区域裁剪
            fire_conf_threshold: 火情置信度阈值
            
        Returns:
            (是否真实火情, 置信度, 详细信息)
        """
        classification = self.classify_fire_region(fire_crop)
        
        # 如果有干扰源，降低置信度
        if classification['is_interference']:
            is_real = False
            reason = "检测到干扰源"
        elif classification['is_fire'] and classification['fire_conf'] >= fire_conf_threshold:
            is_real = True
            reason = "检测到明火"
        elif classification['is_smoke'] and classification['smoke_conf'] >= fire_conf_threshold:
            is_real = True
            reason = "检测到烟雾"
        else:
            is_real = False
            reason = "置信度不足或类别不符"
        
        conf = max(classification['fire_conf'], classification['smoke_conf'])
        
        return is_real, conf, {
            'reason': reason,
            'classification': classification
        }
