"""
数字码头单帧视觉感知模块 - 主推理引擎
支持并行二级推理架构
"""

import cv2
import numpy as np
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .models.global_detector import GlobalDetector
from .models.person_attr_detector import PersonAttrDetector
from .models.fire_attr_detector import FireAttrDetector


@dataclass
class DetectionResult:
    """检测结果数据类"""
    class_name: str
    bbox: List[int]  # [x1, y1, x2, y2]
    confidence: float
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'class': self.class_name,
            'bbox': self.bbox,
            'conf': round(self.confidence, 4),
            'attributes': self.attributes
        }


@dataclass
class FrameResult:
    """单帧处理结果"""
    frame_id: int
    timestamp: float
    detections: List[DetectionResult]
    
    def to_dict(self) -> Dict:
        return {
            'frame_id': self.frame_id,
            'timestamp': self.timestamp,
            'detections': [d.to_dict() for d in self.detections]
        }


class WharfVisionInference:
    """
    数字码头单帧视觉感知推理引擎
    
    架构:
        输入视频帧
             │
             ▼
    一级：Global Detection（全图）
             │
     ┌───────┴───────┐
     ▼               ▼
Person Crop    Fire/Smoke Crop
     │               │
     ▼               ▼
二级A:Person Attr  二级B:Fire Attr
(PPE/军装/烟支)   (火/烟/干扰)
    """
    
    def __init__(
        self,
        global_model_path: Optional[str] = None,
        person_attr_model_path: Optional[str] = None,
        fire_attr_model_path: Optional[str] = None,
        device: str = 'auto',
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        enable_parallel: bool = True,
        max_workers: int = 4
    ):
        """
        初始化推理引擎
        
        Args:
            global_model_path: 一级全局检测模型路径
            person_attr_model_path: 二级人员属性模型路径
            fire_attr_model_path: 二级烟火属性模型路径
            device: 运行设备 ('cpu', 'cuda', 'auto')
            conf_threshold: 置信度阈值
            iou_threshold: NMS IoU阈值
            enable_parallel: 是否启用并行推理
            max_workers: 并行工作线程数
        """
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.enable_parallel = enable_parallel
        self.max_workers = max_workers
        
        # 初始化一级模型
        print("[WharfVision] 初始化一级全局检测模型...")
        self.global_detector = GlobalDetector(
            model_path=global_model_path,
            device=device,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold
        )
        
        # 初始化二级模型
        print("[WharfVision] 初始化二级人员属性模型...")
        self.person_attr_detector = PersonAttrDetector(
            model_path=person_attr_model_path,
            device=device,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold
        )
        
        print("[WharfVision] 初始化二级烟火属性模型...")
        self.fire_attr_detector = FireAttrDetector(
            model_path=fire_attr_model_path,
            device=device,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold
        )
        
        self.frame_count = 0
        print("[WharfVision] 推理引擎初始化完成")
    
    def process_frame(
        self,
        frame: np.ndarray,
        frame_id: Optional[int] = None
    ) -> FrameResult:
        """
        处理单帧图像
        
        Args:
            frame: 输入图像 (BGR格式)
            frame_id: 帧ID，不指定则自动递增
            
        Returns:
            FrameResult: 处理结果
        """
        if frame_id is None:
            self.frame_count += 1
            frame_id = self.frame_count
        
        timestamp = time.time()
        
        # ========== 一级：全局检测 ==========
        global_results = self.global_detector.detect(frame)
        
        # 分离不同类型的检测结果
        person_detections = [r for r in global_results if r['class'] == 'person']
        container_detections = [r for r in global_results if r['class'] == 'container']
        fire_region_detections = [r for r in global_results if r['class'] == 'fire_region']
        obstacle_detections = [r for r in global_results if r['class'] == 'obstacle']
        
        # 准备二级推理任务
        all_detections = []
        
        # ========== 二级：属性识别 ==========
        if self.enable_parallel and (person_detections or fire_region_detections):
            # 并行推理
            all_detections = self._parallel_secondary_inference(
                frame, person_detections, fire_region_detections
            )
        else:
            # 串行推理
            # 处理人员属性
            for det in person_detections:
                person_result = self._process_person_detection(frame, det)
                all_detections.append(person_result)
            
            # 处理烟火属性
            for det in fire_region_detections:
                fire_result = self._process_fire_detection(frame, det)
                all_detections.append(fire_result)
        
        # 添加无需二级处理的检测结果
        for det in container_detections:
            all_detections.append(DetectionResult(
                class_name='container',
                bbox=det['bbox'],
                confidence=det['conf'],
                attributes={}
            ))
        
        for det in obstacle_detections:
            all_detections.append(DetectionResult(
                class_name='obstacle',
                bbox=det['bbox'],
                confidence=det['conf'],
                attributes={}
            ))
        
        return FrameResult(
            frame_id=frame_id,
            timestamp=timestamp,
            detections=all_detections
        )
    
    def _parallel_secondary_inference(
        self,
        frame: np.ndarray,
        person_detections: List[Dict],
        fire_region_detections: List[Dict]
    ) -> List[DetectionResult]:
        """并行二级推理"""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            
            # 提交人员属性检测任务
            for det in person_detections:
                future = executor.submit(self._process_person_detection, frame, det)
                futures.append(('person', future))
            
            # 提交烟火属性检测任务
            for det in fire_region_detections:
                future = executor.submit(self._process_fire_detection, frame, det)
                futures.append(('fire', future))
            
            # 收集结果
            for det_type, future in futures:
                try:
                    result = future.result(timeout=5.0)
                    results.append(result)
                except Exception as e:
                    print(f"[WharfVision] 二级推理失败 ({det_type}): {e}")
        
        return results
    
    def _process_person_detection(
        self,
        frame: np.ndarray,
        detection: Dict
    ) -> DetectionResult:
        """处理人员检测，进行属性识别"""
        bbox = detection['bbox']
        
        # 裁剪人员区域
        x1, y1, x2, y2 = bbox
        person_crop = frame[y1:y2, x1:x2]
        
        if person_crop.size == 0:
            return DetectionResult(
                class_name='person',
                bbox=bbox,
                confidence=detection['conf'],
                attributes={}
            )
        
        # 二级属性识别
        attr_results = self.person_attr_detector.detect(person_crop)
        
        # 构建属性字典
        attributes = {
            'has_helmet': False,
            'has_reflective_vest': False,
            'has_life_jacket': False,
            'has_work_clothes': False,
            'is_military': False,
            'suspected_cigarette': False,
            'detected_items': []
        }
        
        for attr in attr_results:
            attr_class = attr['class']
            attributes['detected_items'].append({
                'class': attr_class,
                'conf': attr['conf'],
                'bbox': attr['bbox']
            })
            
            if attr_class == 'helmet':
                attributes['has_helmet'] = True
            elif attr_class == 'reflective_vest':
                attributes['has_reflective_vest'] = True
            elif attr_class == 'life_jacket':
                attributes['has_life_jacket'] = True
            elif attr_class == 'work_clothes':
                attributes['has_work_clothes'] = True
            elif attr_class == 'military_uniform':
                attributes['is_military'] = True
            elif attr_class == 'suspected_cigarette':
                attributes['suspected_cigarette'] = True
        
        return DetectionResult(
            class_name='person',
            bbox=bbox,
            confidence=detection['conf'],
            attributes=attributes
        )
    
    def _process_fire_detection(
        self,
        frame: np.ndarray,
        detection: Dict
    ) -> DetectionResult:
        """处理烟火区域检测，进行细分类别识别"""
        bbox = detection['bbox']
        
        # 裁剪烟火区域
        x1, y1, x2, y2 = bbox
        fire_crop = frame[y1:y2, x1:x2]
        
        if fire_crop.size == 0:
            return DetectionResult(
                class_name='fire_region',
                bbox=bbox,
                confidence=detection['conf'],
                attributes={}
            )
        
        # 二级属性识别
        attr_results = self.fire_attr_detector.detect(fire_crop)
        
        # 构建属性字典
        attributes = {
            'is_fire': False,
            'is_smoke': False,
            'is_light_interference': False,
            'is_welding_interference': False,
            'detected_items': []
        }
        
        for attr in attr_results:
            attr_class = attr['class']
            attributes['detected_items'].append({
                'class': attr_class,
                'conf': attr['conf'],
                'bbox': attr['bbox']
            })
            
            if attr_class == 'fire':
                attributes['is_fire'] = True
            elif attr_class == 'smoke':
                attributes['is_smoke'] = True
            elif attr_class == 'light_interference':
                attributes['is_light_interference'] = True
            elif attr_class == 'welding_interference':
                attributes['is_welding_interference'] = True
        
        return DetectionResult(
            class_name='fire_region',
            bbox=bbox,
            confidence=detection['conf'],
            attributes=attributes
        )
    
    def process_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        display: bool = False
    ) -> List[FrameResult]:
        """
        处理视频文件
        
        Args:
            video_path: 视频文件路径
            output_path: 输出视频路径（可选）
            display: 是否实时显示
            
        Returns:
            List[FrameResult]: 所有帧的处理结果
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 初始化视频写入器
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        results = []
        frame_id = 0
        
        print(f"[WharfVision] 开始处理视频: {video_path}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_id += 1
            
            # 处理帧
            result = self.process_frame(frame, frame_id)
            results.append(result)
            
            # 可视化
            vis_frame = self.visualize(frame, result)
            
            if writer:
                writer.write(vis_frame)
            
            if display:
                cv2.imshow('Wharf Vision', vis_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # 进度打印
            if frame_id % 30 == 0:
                print(f"[WharfVision] 已处理 {frame_id} 帧")
        
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        print(f"[WharfVision] 视频处理完成，共 {frame_id} 帧")
        
        return results
    
    def visualize(
        self,
        frame: np.ndarray,
        result: FrameResult
    ) -> np.ndarray:
        """
        可视化检测结果
        
        Args:
            frame: 原始图像
            result: 检测结果
            
        Returns:
            可视化后的图像
        """
        vis = frame.copy()
        
        # 颜色映射
        color_map = {
            'person': (0, 255, 0),
            'container': (255, 165, 0),
            'fire_region': (0, 0, 255),
            'obstacle': (128, 128, 128)
        }
        
        for det in result.detections:
            x1, y1, x2, y2 = det.bbox
            color = color_map.get(det.class_name, (255, 255, 255))
            
            # 绘制检测框
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            
            # 构建标签
            label = f"{det.class_name}: {det.confidence:.2f}"
            
            # 添加属性信息
            if det.class_name == 'person' and det.attributes:
                attrs = []
                if det.attributes.get('is_military'):
                    attrs.append("军人")
                if det.attributes.get('has_helmet'):
                    attrs.append("安全帽")
                if det.attributes.get('has_reflective_vest'):
                    attrs.append("反光衣")
                if det.attributes.get('suspected_cigarette'):
                    attrs.append("疑似烟支")
                if attrs:
                    label += f" [{', '.join(attrs)}]"
            
            elif det.class_name == 'fire_region' and det.attributes:
                attrs = []
                if det.attributes.get('is_fire'):
                    attrs.append("明火")
                if det.attributes.get('is_smoke'):
                    attrs.append("烟雾")
                if det.attributes.get('is_light_interference'):
                    attrs.append("灯光干扰")
                if det.attributes.get('is_welding_interference'):
                    attrs.append("电焊干扰")
                if attrs:
                    label += f" [{', '.join(attrs)}]"
            
            # 绘制标签
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(vis, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(vis, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # 绘制帧信息
        info_text = f"Frame: {result.frame_id}"
        cv2.putText(vis, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return vis
