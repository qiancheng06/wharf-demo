"""
基础检测器类
提供YOLOv8模型的通用封装
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path


class BaseDetector:
    """YOLOv8检测器基类"""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = 'auto',
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        imgsz: int = 640
    ):
        """
        初始化检测器
        
        Args:
            model_path: 模型文件路径 (.pt)
            device: 运行设备 ('cpu', 'cuda', 'auto')
            conf_threshold: 置信度阈值
            iou_threshold: NMS IoU阈值
            imgsz: 输入图像尺寸
        """
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        self.model = None
        self.model_path = model_path
        
        # 类别名称映射（子类需要覆盖）
        self.class_names = {}
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """
        加载YOLOv8模型
        
        Args:
            model_path: 模型文件路径
        """
        try:
            from ultralytics import YOLO
            
            self.model = YOLO(model_path)
            
            # 设置设备
            if self.device == 'auto':
                import torch
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            print(f"[BaseDetector] 模型加载成功: {model_path}")
            print(f"[BaseDetector] 使用设备: {self.device}")
            
        except ImportError:
            print("[BaseDetector] 警告: 未安装ultralytics，使用模拟模式")
            self.model = None
    
    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        执行检测
        
        Args:
            image: 输入图像 (BGR格式)
            
        Returns:
            检测结果列表，每个结果包含:
            - class: 类别名称
            - bbox: 边界框 [x1, y1, x2, y2]
            - conf: 置信度
        """
        if self.model is None:
            return self._mock_detect(image)
        
        # YOLOv8推理
        results = self.model(
            image,
            device=self.device,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.imgsz,
            verbose=False
        )
        
        # 解析结果
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            
            for box in boxes:
                cls_id = int(box.cls.item())
                conf = float(box.conf.item())
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                
                class_name = self.class_names.get(cls_id, f"class_{cls_id}")
                
                detections.append({
                    'class': class_name,
                    'bbox': xyxy.tolist(),
                    'conf': round(conf, 4)
                })
        
        return detections
    
    def _mock_detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        模拟检测（用于无模型时的测试）
        
        Args:
            image: 输入图像
            
        Returns:
            空列表或模拟结果
        """
        print(f"[BaseDetector] 模拟模式: 未加载实际模型")
        return []
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        图像预处理
        
        Args:
            image: 输入图像
            
        Returns:
            预处理后的图像
        """
        # 调整尺寸
        h, w = image.shape[:2]
        scale = self.imgsz / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        resized = cv2.resize(image, (new_w, new_h))
        
        # 填充到目标尺寸
        padded = np.full((self.imgsz, self.imgsz, 3), 114, dtype=np.uint8)
        padded[:new_h, :new_w] = resized
        
        return padded
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            模型信息字典
        """
        return {
            'model_path': self.model_path,
            'device': self.device,
            'conf_threshold': self.conf_threshold,
            'iou_threshold': self.iou_threshold,
            'imgsz': self.imgsz,
            'class_names': self.class_names
        }
