"""
测试全局检测器
"""

import sys
from pathlib import Path
import numpy as np

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestGlobalDetector:
    """测试GlobalDetector类"""

    def test_global_detector_init(self):
        """测试全局检测器初始化"""
        from wharf_vision.models.global_detector import GlobalDetector

        detector = GlobalDetector(
            model_path=None,
            device='cpu',
            conf_threshold=0.25,
            iou_threshold=0.45
        )

        assert detector.device == 'cpu'
        assert detector.conf_threshold == 0.25
        assert detector.iou_threshold == 0.45
        assert detector.imgsz == 640

        # 验证类别名称
        expected_classes = {
            0: 'person',
            1: 'container',
            2: 'fire_region',
            3: 'obstacle'
        }
        assert detector.class_names == expected_classes

    def test_class_names_count(self):
        """测试类别数量"""
        from wharf_vision.models.global_detector import GlobalDetector

        detector = GlobalDetector(model_path=None)

        assert len(detector.CLASS_NAMES) == 4
        assert 'person' in detector.CLASS_NAMES.values()
        assert 'container' in detector.CLASS_NAMES.values()
        assert 'fire_region' in detector.CLASS_NAMES.values()
        assert 'obstacle' in detector.CLASS_NAMES.values()


class TestPersonAttrDetector:
    """测试PersonAttrDetector类"""

    def test_person_attr_detector_init(self):
        """测试人员属性检测器初始化"""
        from wharf_vision.models.person_attr_detector import PersonAttrDetector

        detector = PersonAttrDetector(
            model_path=None,
            device='cpu',
            conf_threshold=0.25,
            iou_threshold=0.45
        )

        assert detector.device == 'cpu'
        assert detector.conf_threshold == 0.25
        assert detector.imgsz == 320

        # 验证类别数量
        assert len(detector.CLASS_NAMES) == 6

    def test_check_military_uniform(self):
        """测试军装检查功能"""
        from wharf_vision.models.person_attr_detector import PersonAttrDetector

        detector = PersonAttrDetector(model_path=None)
        test_image = np.ones((100, 100, 3), dtype=np.uint8)

        # 无模型时应返回False
        is_military, conf = detector.check_military_uniform(test_image)
        assert is_military == False
        assert conf == 0.0

    def test_check_ppe(self):
        """测试PPE检查功能"""
        from wharf_vision.models.person_attr_detector import PersonAttrDetector

        detector = PersonAttrDetector(model_path=None)
        test_image = np.ones((100, 100, 3), dtype=np.uint8)

        ppe_status = detector.check_ppe(test_image)

        assert isinstance(ppe_status, dict)
        assert 'has_helmet' in ppe_status
        assert 'has_reflective_vest' in ppe_status
        assert 'has_life_jacket' in ppe_status
        assert 'has_work_clothes' in ppe_status
        assert 'suspected_cigarette' in ppe_status


class TestFireAttrDetector:
    """测试FireAttrDetector类"""

    def test_fire_attr_detector_init(self):
        """测试烟火属性检测器初始化"""
        from wharf_vision.models.fire_attr_detector import FireAttrDetector

        detector = FireAttrDetector(
            model_path=None,
            device='cpu',
            conf_threshold=0.25,
            iou_threshold=0.45
        )

        assert detector.device == 'cpu'
        assert detector.conf_threshold == 0.25
        assert detector.imgsz == 320

        # 验证类别数量
        assert len(detector.CLASS_NAMES) == 4

    def test_classify_fire_region(self):
        """测试烟火区域分类"""
        from wharf_vision.models.fire_attr_detector import FireAttrDetector

        detector = FireAttrDetector(model_path=None)
        test_image = np.ones((100, 100, 3), dtype=np.uint8)

        classification = detector.classify_fire_region(test_image)

        assert isinstance(classification, dict)
        assert 'is_fire' in classification
        assert 'is_smoke' in classification
        assert 'is_light_interference' in classification
        assert 'is_welding_interference' in classification

    def test_is_real_fire(self):
        """测试真实火情判断"""
        from wharf_vision.models.fire_attr_detector import FireAttrDetector

        detector = FireAttrDetector(model_path=None)
        test_image = np.ones((100, 100, 3), dtype=np.uint8)

        is_real, conf, details = detector.is_real_fire(test_image)

        assert isinstance(is_real, bool)
        assert isinstance(conf, float)
        assert isinstance(details, dict)


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])
