"""
测试推理引擎
"""

import sys
from pathlib import Path
import numpy as np

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestDetectionResult:
    """测试DetectionResult数据类"""

    def test_detection_result_creation(self):
        """测试创建检测结果"""
        from wharf_vision.inference import DetectionResult

        det = DetectionResult(
            class_name='person',
            bbox=[100, 100, 200, 300],
            confidence=0.95,
            attributes={'has_helmet': True}
        )

        assert det.class_name == 'person'
        assert det.bbox == [100, 100, 200, 300]
        assert det.confidence == 0.95
        assert det.attributes['has_helmet'] == True

    def test_detection_result_to_dict(self):
        """测试转换为字典"""
        from wharf_vision.inference import DetectionResult

        det = DetectionResult(
            class_name='fire',
            bbox=[50, 50, 150, 150],
            confidence=0.88
        )

        result_dict = det.to_dict()

        assert isinstance(result_dict, dict)
        assert result_dict['class'] == 'fire'
        assert result_dict['conf'] == 0.88


class TestFrameResult:
    """测试FrameResult数据类"""

    def test_frame_result_creation(self):
        """测试创建帧结果"""
        from wharf_vision.inference import FrameResult, DetectionResult

        det1 = DetectionResult(
            class_name='person',
            bbox=[100, 100, 200, 300],
            confidence=0.95
        )
        det2 = DetectionResult(
            class_name='container',
            bbox=[300, 200, 500, 400],
            confidence=0.90
        )

        frame_result = FrameResult(
            frame_id=1,
            timestamp=1234567890.123,
            detections=[det1, det2]
        )

        assert frame_result.frame_id == 1
        assert len(frame_result.detections) == 2

    def test_frame_result_to_dict(self):
        """测试帧结果转换为字典"""
        from wharf_vision.inference import FrameResult, DetectionResult

        det = DetectionResult(
            class_name='person',
            bbox=[100, 100, 200, 300],
            confidence=0.95
        )

        frame_result = FrameResult(
            frame_id=1,
            timestamp=1234567890.123,
            detections=[det]
        )

        result_dict = frame_result.to_dict()

        assert isinstance(result_dict, dict)
        assert result_dict['frame_id'] == 1
        assert len(result_dict['detections']) == 1


class TestWharfVisionInference:
    """测试WharfVisionInference类"""

    def test_inference_init(self):
        """测试推理引擎初始化"""
        from wharf_vision import WharfVisionInference

        inference = WharfVisionInference(
            global_model_path=None,
            person_attr_model_path=None,
            fire_attr_model_path=None,
            device='cpu',
            conf_threshold=0.25,
            enable_parallel=False
        )

        assert inference.device == 'cpu'
        assert inference.conf_threshold == 0.25
        assert inference.enable_parallel == False
        assert inference.frame_count == 0

    def test_process_frame_empty(self):
        """测试处理空帧"""
        from wharf_vision import WharfVisionInference

        inference = WharfVisionInference(
            global_model_path=None,
            person_attr_model_path=None,
            fire_attr_model_path=None,
            device='cpu',
            enable_parallel=False
        )

        # 创建空白图像
        test_image = np.ones((480, 640, 3), dtype=np.uint8) * 128

        result = inference.process_frame(test_image)

        assert isinstance(result, FrameResult)
        assert result.frame_id == 1
        assert isinstance(result.detections, list)

    def test_visualize(self):
        """测试可视化功能"""
        from wharf_vision import WharfVisionInference, FrameResult

        inference = WharfVisionInference(
            global_model_path=None,
            person_attr_model_path=None,
            fire_attr_model_path=None,
            device='cpu'
        )

        # 创建测试图像
        test_image = np.ones((480, 640, 3), dtype=np.uint8) * 128

        # 创建测试结果
        result = FrameResult(
            frame_id=1,
            timestamp=1234567890.123,
            detections=[]
        )

        vis_image = inference.visualize(test_image, result)

        assert vis_image.shape == test_image.shape


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])
