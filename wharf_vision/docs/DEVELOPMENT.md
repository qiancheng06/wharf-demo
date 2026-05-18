# 开发指南

## 项目结构

```
wharf_vision/
├── __init__.py              # 包初始化
├── inference.py             # 主推理引擎
├── models/                  # 模型模块
│   ├── __init__.py
│   ├── base_detector.py     # 基础检测器
│   ├── global_detector.py   # 一级全局检测器
│   ├── person_attr_detector.py  # 二级人员属性检测器
│   └── fire_attr_detector.py    # 二级烟火属性检测器
├── config/                  # 配置文件
│   ├── global.yaml
│   ├── person_attr.yaml
│   └── fire_attr.yaml
├── scripts/                 # 训练脚本
│   ├── train_global.py
│   ├── train_person_attr.py
│   └── train_fire_attr.py
├── examples/                # 使用示例
│   ├── demo_image.py
│   ├── demo_video.py
│   └── demo_camera.py
├── utils/                   # 工具函数
│   ├── __init__.py
│   ├── visualization.py
│   └── data_utils.py
├── data/                    # 数据集目录
├── tests/                   # 测试目录
│   ├── __init__.py
│   ├── test_base_detector.py
│   ├── test_detectors.py
│   └── test_inference.py
├── docs/                    # 文档目录
│   └── API.md
├── requirements.txt         # 依赖列表
└── README.md               # 项目说明
```

## 开发环境设置

### 1. 克隆项目

```bash
git clone <repository_url>
cd wharf_vision
```

### 2. 创建虚拟环境

```bash
# 使用venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
.\venv\Scripts\activate  # Windows
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 运行测试

```bash
# 运行所有测试
pytest wharf_vision/tests/ -v

# 运行特定测试
pytest wharf_vision/tests/test_inference.py -v

# 生成覆盖率报告
pytest wharf_vision/tests/ --cov=wharf_vision --cov-report=html
```

## 代码规范

### Python 编码规范

- 遵循 PEP 8 标准
- 使用 4 个空格缩进
- 行长度限制在 100 字符以内
- 使用中文注释（主要逻辑）或英文注释

### 命名规范

- 类名：`PascalCase` (如 `WharfVisionInference`)
- 函数名：`snake_case` (如 `process_frame`)
- 常量：`UPPER_CASE` (如 `MAX_WORKERS`)
- 私有成员：前缀 `_` (如 `_process_person_detection`)

### 文档字符串

所有公共类和函数应包含文档字符串：

```python
def process_frame(self, frame, frame_id=None):
    """
    处理单帧图像
    
    Args:
        frame: 输入图像 (BGR格式)
        frame_id: 帧ID，不指定则自动递增
        
    Returns:
        FrameResult: 处理结果
        
    Example:
        >>> inference = WharfVisionInference()
        >>> image = cv2.imread('test.jpg')
        >>> result = inference.process_frame(image)
        >>> print(f"检测到 {len(result.detections)} 个目标")
    """
    pass
```

## 添加新的检测器

### 1. 创建检测器类

在 `models/` 目录下创建新的检测器：

```python
# models/new_detector.py
from .base_detector import BaseDetector

class NewDetector(BaseDetector):
    """新检测器描述"""
    
    CLASS_NAMES = {
        0: 'class_1',
        1: 'class_2',
    }
    
    def __init__(self, model_path=None, device='auto', 
                 conf_threshold=0.25, iou_threshold=0.45):
        super().__init__(
            model_path=model_path,
            device=device,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            imgsz=320  # 根据需要设置
        )
        self.class_names = self.CLASS_NAMES
```

### 2. 更新模块导出

在 `models/__init__.py` 中添加：

```python
from .new_detector import NewDetector

__all__ = [
    # ... 其他导出
    'NewDetector',
]
```

### 3. 更新推理引擎

在 `inference.py` 中集成新检测器：

```python
from .models.new_detector import NewDetector

class WharfVisionInference:
    def __init__(self, ..., new_model_path=None):
        # ...
        self.new_detector = NewDetector(
            model_path=new_model_path,
            device=device,
            conf_threshold=conf_threshold
        )
```

## 数据集准备

### 数据集目录结构

```
data/
├── dataset_name/
│   ├── images/
│   │   ├── train/
│   │   │   ├── image001.jpg
│   │   │   └── ...
│   │   └── val/
│   │       ├── image100.jpg
│   │       └── ...
│   └── labels/
│       ├── train/
│       │   ├── image001.txt
│       │   └── ...
│       └── val/
│           ├── image100.txt
│           └── ...
```

### YOLO 标签格式

每行一个目标：
```
<class_id> <x_center> <y_center> <width> <height>
```

所有坐标都是相对于图像尺寸的归一化值（0-1）。

### 数据集配置

创建 `dataset.yaml`：

```yaml
path: ./data/dataset_name
train: images/train
val: images/val

names:
  0: class_name_1
  1: class_name_2
```

## 模型训练

### 本地训练

```bash
python wharf_vision/scripts/train_global.py \
    --data wharf_vision/config/global.yaml \
    --epochs 80 \
    --imgsz 640 \
    --batch 16 \
    --device cuda
```

### 分布式训练

使用Ultralytics的分布式训练功能：

```bash
# 单机多卡
python -m torch.distributed.run --nproc_per_node=2 \
    wharf_vision/scripts/train_global.py \
    --data wharf_vision/config/global.yaml \
    --epochs 80 \
    --device 0,1
```

## 模型导出

### 支持的格式

- `onnx`: ONNX 格式
- `engine`: TensorRT 引擎
- `tflite`: TensorFlow Lite
- `coreml`: CoreML (macOS/iOS)
- `saved_model`: TensorFlow SavedModel

### 导出命令

```bash
# ONNX
python wharf_vision/scripts/train_global.py --export onnx

# TensorRT
python wharf_vision/scripts/train_global.py --export engine
```

## 调试技巧

### 启用详细日志

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### 可视化中间结果

```python
# 可视化crop区域
import cv2
cv2.imwrite('person_crop.jpg', person_crop)

# 可视化检测结果
vis_image = inference.visualize(image, result)
cv2.imwrite('detection_result.jpg', vis_image)
```

### 性能分析

```python
import time

start = time.time()
result = inference.process_frame(image)
elapsed = time.time() - start

print(f"处理时间: {elapsed*1000:.2f}ms")
```

## 常见问题

### Q: 模型加载失败

确保：
1. 模型文件存在
2. 模型格式正确（.pt）
3. PyTorch版本兼容

### Q: GPU 不可用

检查 CUDA 安装：
```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
```

### Q: 内存不足

减少 batch size 或图像尺寸：
```bash
python train.py --batch 8 --imgsz 320
```

### Q: 检测结果为空

可能原因：
1. 置信度阈值过高
2. 图像中确实没有目标
3. 模型未正确加载

尝试降低阈值：
```python
inference = WharfVisionInference(conf_threshold=0.15)
```
