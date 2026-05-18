# 数字码头单帧视觉感知模块

基于YOLOv8的两级检测架构，为数字码头项目提供单帧视觉感知能力。

## 📋 目录

- [架构设计](#架构设计)
- [安装](#安装)
- [快速开始](#快速开始)
- [项目结构](#项目结构)
- [训练模型](#训练模型)
- [输出格式](#输出格式)
- [性能优化](#性能优化)
- [测试](#测试)
- [文档](#文档)
- [贡献指南](#贡献指南)

## 🏗️ 架构设计

```
                    输入视频帧
                         │
                         ▼
            ┌─────────────────────────────┐
            │   一级：Global Detection    │
            │      (YOLOv8s, 640x640)     │
            └─────────────────────────────┘
                         │
         ┌───────────────┴───────────────┐
         ▼                               ▼
    ┌──────────┐                   ┌──────────┐
    │Person Crop│                   │Fire Crop │
    └──────────┘                   └──────────┘
         │                               │
         ▼                               ▼
    ┌─────────────────────┐     ┌─────────────────────┐
    │  二级A：Person Attr  │     │  二级B：Fire Attr   │
    │ (YOLOv8n, 320x320)  │     │ (YOLOv8n, 320x320) │
    │  PPE/军装/烟支检测   │     │   火/烟/干扰检测    │
    └─────────────────────┘     └─────────────────────┘
```

### 一级模型：Global Detection (YOLOv8s)

检测类别：
- `person` - 人员
- `container` - 集装箱
- `fire_region` - 疑似烟火区域
- `obstacle` - 障碍物

### 二级模型A：Person Attribute (YOLOv8n)

检测类别：
- `helmet` - 安全帽
- `reflective_vest` - 反光衣
- `life_jacket` - 救生衣
- `work_clothes` - 工作服
- `military_uniform` - 军装（高优先级）
- `suspected_cigarette` - 疑似烟支

### 二级模型B：Fire Attribute (YOLOv8n)

检测类别：
- `fire` - 明火
- `smoke` - 烟雾
- `light_interference` - 灯光干扰（降低误报）
- `welding_interference` - 电焊干扰（降低夜间误报）

## 📦 安装

### 环境要求

- Python >= 3.8
- PyTorch >= 2.0.0
- CUDA >= 11.0 (GPU支持，可选)

### 安装步骤

```bash
# 克隆项目
git clone <repository_url>
cd wharf_vision

# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
.\venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### GPU支持（可选）

```bash
# 安装PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 安装ONNX Runtime GPU版本
pip install onnxruntime-gpu
```

## 🚀 快速开始

### 1. 单帧图像检测

```bash
python examples/demo_image.py --image path/to/image.jpg
```

### 2. 视频检测

```bash
python examples/demo_video.py --video path/to/video.mp4 --output output.mp4
```

### 3. 摄像头实时检测

```bash
python examples/demo_camera.py --source 0
```

### 4. Python API使用

```python
from wharf_vision import WharfVisionInference
import cv2

# 初始化推理引擎
inference = WharfVisionInference(
    global_model_path='path/to/global_model.pt',
    person_attr_model_path='path/to/person_attr_model.pt',
    fire_attr_model_path='path/to/fire_attr_model.pt',
    device='cuda',
    conf_threshold=0.25,
    enable_parallel=True
)

# 读取图像
image = cv2.imread('image.jpg')

# 执行检测
result = inference.process_frame(image)

# 处理结果
for det in result.detections:
    print(f"类别: {det.class_name}")
    print(f"置信度: {det.confidence}")
    print(f"边界框: {det.bbox}")
    print(f"属性: {det.attributes}")

# 可视化
vis_image = inference.visualize(image, result)
cv2.imwrite('output.jpg', vis_image)
```

## 📁 项目结构

```
wharf_vision/
├── __init__.py              # 包初始化
├── inference.py             # 主推理引擎
│
├── models/                  # 模型模块
│   ├── __init__.py
│   ├── base_detector.py     # 基础检测器基类
│   ├── global_detector.py   # 一级全局检测器
│   ├── person_attr_detector.py  # 二级人员属性检测器
│   └── fire_attr_detector.py    # 二级烟火属性检测器
│
├── config/                  # 训练配置文件
│   ├── global.yaml          # 全局检测配置
│   ├── person_attr.yaml     # 人员属性配置
│   └── fire_attr.yaml       # 烟火属性配置
│
├── data/                    # 数据集目录
│   ├── global_dataset.yaml  # 全局检测数据集配置
│   ├── person_attr_dataset.yaml  # 人员属性数据集配置
│   ├── fire_attr_dataset.yaml    # 烟火属性数据集配置
│   ├── global_dataset/      # 全局检测数据集
│   ├── person_attr_dataset/ # 人员属性数据集
│   └── fire_attr_dataset/   # 烟火属性数据集
│
├── scripts/                 # 训练脚本
│   ├── train_global.py      # 训练一级模型
│   ├── train_person_attr.py # 训练人员属性模型
│   ├── train_fire_attr.py   # 训练烟火属性模型
│   ├── download_datasets.py  # 数据集下载脚本
│   ├── create_test_datasets.py   # 创建测试数据集
│   ├── download_small_real_datasets.py
│   ├── prepare_datasets_fast.py
│   └── train_all.py         # 一键训练所有模型
│
├── examples/                # 使用示例
│   ├── demo_image.py        # 单帧图像检测
│   ├── demo_video.py        # 视频检测
│   └── demo_camera.py       # 摄像头实时检测
│
├── utils/                   # 工具函数
│   ├── __init__.py
│   ├── visualization.py     # 可视化工具
│   └── data_utils.py        # 数据处理工具
│
├── tests/                   # 测试目录
│   ├── __init__.py
│   ├── test_base_detector.py   # 基础检测器测试
│   ├── test_detectors.py       # 检测器测试
│   └── test_inference.py       # 推理引擎测试
│
├── docs/                    # 文档目录
│   ├── API.md               # API文档
│   └── DEVELOPMENT.md       # 开发指南
│
├── requirements.txt         # 依赖列表
└── README.md               # 项目说明
```

## 🎓 训练模型

### 准备数据集

按照YOLO格式组织数据集：

```
data/
├── global_dataset/          # 一级模型数据集
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   └── labels/
│       ├── train/
│       └── val/
├── person_attr_dataset/     # 二级人员属性数据集
│   └── ...
└── fire_attr_dataset/       # 二级烟火属性数据集
    └── ...
```

### 训练一级模型

```bash
python wharf_vision/scripts/train_global.py \
    --data wharf_vision/config/global.yaml \
    --epochs 80 \
    --imgsz 640 \
    --batch 16
```

### 训练二级人员属性模型

```bash
python wharf_vision/scripts/train_person_attr.py \
    --data wharf_vision/config/person_attr.yaml \
    --epochs 80 \
    --imgsz 320
```

### 训练二级烟火属性模型

```bash
python wharf_vision/scripts/train_fire_attr.py \
    --data wharf_vision/config/fire_attr.yaml \
    --epochs 100 \
    --imgsz 320
```

### 一键训练所有模型

```bash
python wharf_vision/scripts/train_all.py
```

### 验证模型

```bash
# 验证一级模型
python wharf_vision/scripts/train_global.py --val

# 导出为ONNX
python wharf_vision/scripts/train_global.py --export onnx

# 导出为TensorRT
python wharf_vision/scripts/train_global.py --export engine
```

## 📊 输出格式

### 单帧检测结果

```json
{
  "frame_id": 1,
  "timestamp": 1699999999.999,
  "detections": [
    {
      "class": "person",
      "bbox": [100, 200, 300, 500],
      "conf": 0.95,
      "attributes": {
        "has_helmet": true,
        "has_reflective_vest": false,
        "is_military": false,
        "suspected_cigarette": false,
        "detected_items": [
          {"class": "helmet", "conf": 0.92, "bbox": [120, 200, 180, 240]}
        ]
      }
    },
    {
      "class": "fire_region",
      "bbox": [400, 300, 600, 500],
      "conf": 0.88,
      "attributes": {
        "is_fire": true,
        "is_smoke": false,
        "is_light_interference": false,
        "detected_items": [
          {"class": "fire", "conf": 0.91, "bbox": [20, 20, 150, 150]}
        ]
      }
    }
  ]
}
```

## ⚡ 性能优化

### 并行推理

启用并行推理可提升多目标场景的处理速度：

```python
inference = WharfVisionInference(
    enable_parallel=True,
    max_workers=4
)
```

### 模型导出

导出为ONNX或TensorRT格式可进一步提升推理速度：

```bash
# 导出为ONNX
python scripts/train_global.py --export onnx

# 导出为TensorRT（需要GPU）
python scripts/train_global.py --export engine
```

## 🧪 测试

### 运行所有测试

```bash
pytest wharf_vision/tests/ -v
```

### 运行特定测试

```bash
# 测试推理引擎
pytest wharf_vision/tests/test_inference.py -v

# 测试检测器
pytest wharf_vision/tests/test_detectors.py -v
```

### 生成覆盖率报告

```bash
pytest wharf_vision/tests/ --cov=wharf_vision --cov-report=html
```

## 📚 文档

- [API 文档](docs/API.md) - 完整的API参考
- [开发指南](docs/DEVELOPMENT.md) - 开发规范和技巧

## 🔲 模块边界

本模块**只负责**：
- ✅ 单帧视觉检测与属性识别
- ✅ 输出结构化视觉实体
- ✅ 提供可扩展视觉原子能力

本模块**不负责**：
- ❌ 长时间停留检测（依赖时间状态）
- ❌ 超期检测（依赖计时）
- ❌ 新堆放/移走检测（依赖历史帧比较）
- ❌ 通道受阻事件（依赖ROI持续占用）
- ❌ 越界事件（依赖电子围栏规则）
- ❌ 人员跟踪（依赖Tracking）
- ❌ 业务规则判断

上述能力由后续模块（Tracking、状态机、ROI系统、规则引擎）组合实现。

## 🤝 贡献指南

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建 Pull Request

### 代码规范

- 遵循 PEP 8 标准
- 使用 4 个空格缩进
- 所有公共函数需包含文档字符串
- 运行测试确保通过 (`pytest wharf_vision/tests/ -v`)

## 📄 许可

本项目为数字码头项目专用。

## 🙏 致谢

- [Ultralytics](https://github.com/ultralytics/ultralytics) - YOLOv8框架
- [COCO Dataset](https://cocodataset.org/) - 人员检测数据集
