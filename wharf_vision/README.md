# 数字码头单帧视觉感知模块

基于YOLOv8的两级检测架构，为数字码头项目提供单帧视觉感知能力。

## 架构设计

```
                    输入视频帧
                         │
                         ▼
            一级：Global Detection（全图）
                         │
         ┌───────────────┴───────────────┐
         ▼                               ▼
    Person Crop                    Fire/Smoke Crop
         │                               │
         ▼                               ▼
    二级A：Person Attr            二级B：Fire Attr
    (PPE/军装/烟支)               (火/烟/干扰)
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

## 安装

```bash
# 克隆项目
cd wharf_vision

# 安装依赖
pip install -r requirements.txt
```

## 快速开始

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

## 模型训练

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
python scripts/train_global.py --data config/global.yaml --epochs 80
```

### 训练二级人员属性模型

```bash
python scripts/train_person_attr.py --data config/person_attr.yaml --epochs 80
```

### 训练二级烟火属性模型

```bash
python scripts/train_fire_attr.py --data config/fire_attr.yaml --epochs 100
```

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
│   ├── visualization.py     # 可视化工具
│   └── data_utils.py        # 数据处理工具
├── data/                    # 数据集目录
├── requirements.txt         # 依赖列表
└── README.md               # 项目说明
```

## 输出格式

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

## 性能优化

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

## 模块边界

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

## 许可

本项目为数字码头项目专用。
