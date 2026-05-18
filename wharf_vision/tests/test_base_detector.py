"""
测试基础检测器
"""

import sys
from pathlib import Path
import numpy as np

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestBaseDetector:
    """测试BaseDetector类"""

    def test_base_detector_init(self):
        """测试基础检测器初始化"""
        from wharf_vision.models.base_detector import BaseDetector

        detector = BaseDetector(
            model_path=None,
            device='cpu',
            conf_threshold=0.25,
            iou_threshold=0.45
        )

        assert detector.device == 'cpu'
        assert detector.conf_threshold == 0.25
        assert detector.iou_threshold == 0.45
        assert detector.model is None

    def test_base_detector_mock_detect(self):
        """测试模拟检测"""
        from wharf_vision.models.base_detector import BaseDetector

        detector = BaseDetector()
        # 创建测试图像
        test_image = np.ones((640, 640, 3), dtype=np.uint8)

        # 无模型时应返回空列表
        results = detector._mock_detect(test_image)
        assert isinstance(results, list)
        assert len(results) == 0

    def test_preprocess(self):
        """测试图像预处理"""
        from wharf_vision.models.base_detector import BaseDetector

        detector = BaseDetector(imgsz=640)
        test_image = np.ones((480, 640, 3), dtype=np.uint8) * 128

        processed = detector.preprocess(test_image)

        assert processed.shape == (640, 640, 3)

    def test_get_model_info(self):
        """测试获取模型信息"""
        from wharf_vision.models.base_detector import BaseDetector

        detector = BaseDetector(
            model_path='test_model.pt',
            device='cuda',
            conf_threshold=0.5,
            iou_threshold=0.4,
            imgsz=320
        )

        info = detector.get_model_info()

        assert info['model_path'] == 'test_model.pt'
        assert info['device'] == 'cuda'
        assert info['conf_threshold'] == 0.5
        assert info['iou_threshold'] == 0.4
        assert info['imgsz'] == 320


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])
