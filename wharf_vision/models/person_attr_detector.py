"""
二级人员属性检测器 - Person Attribute Detection
使用YOLOv8n进行人员属性识别

检测类别:
- helmet: 安全帽
- reflective_vest: 反光衣
- life_jacket: 救生衣
- work_clothes: 工作服
- military_uniform: 军装
- suspected_cigarette: 疑似烟支
"""

from .base_detector import BaseDetector


class PersonAttrDetector(BaseDetector):
    """
    二级人员属性检测器
    
    职责:
    - 输入: person crop (人员裁剪区域)
    - 输出: PPE属性、军装属性、疑似烟支属性
    
    模型选型: YOLOv8n
    原因:
    - crop区域较小
    - 推理速度快
    - 易于并行部署
    
    注意:
    - 不直接定义"抽烟行为"（需要时序判断）
    - 只负责单帧疑似烟支检测
    """
    
    # 类别定义
    CLASS_NAMES = {
        0: 'helmet',
        1: 'reflective_vest',
        2: 'life_jacket',
        3: 'work_clothes',
        4: 'military_uniform',
        5: 'suspected_cigarette'
    }
    
    def __init__(
        self,
        model_path: str = None,
        device: str = 'auto',
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45
    ):
        """
        初始化人员属性检测器
        
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
        
        print(f"[PersonAttrDetector] 二级人员属性检测器初始化完成")
        print(f"[PersonAttrDetector] 检测类别: {list(self.CLASS_NAMES.values())}")
    
    def detect(self, person_crop):
        """
        执行人员属性检测
        
        Args:
            person_crop: 人员裁剪区域图像
            
        Returns:
            属性检测结果列表
        """
        if person_crop is None or person_crop.size == 0:
            return []
        
        results = super().detect(person_crop)
        
        # 过滤只保留人员属性类别
        valid_classes = set(self.CLASS_NAMES.values())
        filtered_results = [
            r for r in results 
            if r['class'] in valid_classes
        ]
        
        return filtered_results
    
    def check_military_uniform(self, person_crop) -> tuple:
        """
        检查是否穿着军装
        
        Args:
            person_crop: 人员裁剪区域
            
        Returns:
            (是否军人, 置信度)
        """
        results = self.detect(person_crop)
        
        for r in results:
            if r['class'] == 'military_uniform':
                return True, r['conf']
        
        return False, 0.0
    
    def check_ppe(self, person_crop) -> dict:
        """
        检查PPE穿戴情况
        
        Args:
            person_crop: 人员裁剪区域
            
        Returns:
            PPE状态字典
        """
        results = self.detect(person_crop)
        
        ppe_status = {
            'has_helmet': False,
            'has_reflective_vest': False,
            'has_life_jacket': False,
            'has_work_clothes': False,
            'suspected_cigarette': False
        }
        
        for r in results:
            class_name = r['class']
            if class_name == 'helmet':
                ppe_status['has_helmet'] = True
            elif class_name == 'reflective_vest':
                ppe_status['has_reflective_vest'] = True
            elif class_name == 'life_jacket':
                ppe_status['has_life_jacket'] = True
            elif class_name == 'work_clothes':
                ppe_status['has_work_clothes'] = True
            elif class_name == 'suspected_cigarette':
                ppe_status['suspected_cigarette'] = True
        
        return ppe_status
