# API 文档

## 快速开始

### 安装

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

## 核心 API

### WharfVisionInference

主推理引擎类，用于处理图像和视频中的目标检测。

#### 初始化

```python
from wharf_vision import WharfVisionInference

inference = WharfVisionInference(
    global_model_path='path/to/global_model.pt',      # 一级模型路径
    person_attr_model_path='path/to/person_model.pt', # 人员属性模型路径
    fire_attr_model_path='path/to/fire_model.pt',     # 烟火属性模型路径
    device='cuda',                                     # 设备: 'cpu', 'cuda', 'auto'
    conf_threshold=0.25,                               # 置信度阈值
    iou_threshold=0.45,                               # NMS IoU阈值
    enable_parallel=True,                              # 启用并行推理
    max_workers=4                                      # 并行工作线程数
)
```

#### 方法

##### process_frame(image, frame_id=None)

处理单帧图像。

**参数:**
- `image` (np.ndarray): 输入图像 (BGR格式)
- `frame_id` (int, optional): 帧ID，不指定则自动递增

**返回:**
- `FrameResult`: 处理结果对象

**示例:**

```python
import cv2
from wharf_vision import WharfVisionInference

inference = WharfVisionInference()
image = cv2.imread('test.jpg')
result = inference.process_frame(image)

print(f"检测到 {len(result.detections)} 个目标")
for det in result.detections:
    print(f"  - {det.class_name}: {det.confidence:.2f}")
```

##### process_video(video_path, output_path=None, display=False)

处理视频文件。

**参数:**
- `video_path` (str): 视频文件路径
- `output_path` (str, optional): 输出视频路径
- `display` (bool): 是否实时显示

**返回:**
- `List[FrameResult]`: 所有帧的处理结果

**示例:**

```python
results = inference.process_video(
    video_path='input.mp4',
    output_path='output.mp4',
    display=True
)
```

##### visualize(image, result)

可视化检测结果。

**参数:**
- `image` (np.ndarray): 原始图像
- `result` (FrameResult): 检测结果

**返回:**
- `np.ndarray`: 可视化后的图像

**示例:**

```python
vis_image = inference.visualize(image, result)
cv2.imwrite('result.jpg', vis_image)
```

### 数据类

#### DetectionResult

检测结果数据类。

**属性:**
- `class_name` (str): 类别名称
- `bbox` (List[int]): 边界框 [x1, y1, x2, y2]
- `confidence` (float): 置信度
- `attributes` (Dict): 属性字典

**方法:**
- `to_dict()`: 转换为字典格式

**示例:**

```python
det = DetectionResult(
    class_name='person',
    bbox=[100, 200, 300, 500],
    confidence=0.95,
    attributes={'has_helmet': True}
)

det_dict = det.to_dict()
# {'class': 'person', 'bbox': [100, 200, 300, 500], 'conf': 0.95, 'attributes': {...}}
```

#### FrameResult

单帧处理结果数据类。

**属性:**
- `frame_id` (int): 帧ID
- `timestamp` (float): 时间戳
- `detections` (List[DetectionResult]): 检测结果列表

**方法:**
- `to_dict()`: 转换为字典格式

**示例:**

```python
frame_result = FrameResult(
    frame_id=1,
    timestamp=1234567890.123,
    detections=[det1, det2]
)
```

## 检测器 API

### GlobalDetector

一级全局检测器。

```python
from wharf_vision.models import GlobalDetector

detector = GlobalDetector(
    model_path='yolov8s.pt',  # 可选，默认使用yolov8s.pt
    device='cuda',
    conf_threshold=0.25,
    iou_threshold=0.45
)

results = detector.detect(image)
```

**检测类别:**
- `person`: 人员
- `container`: 集装箱
- `fire_region`: 疑似烟火区域
- `obstacle`: 障碍物

### PersonAttrDetector

二级人员属性检测器。

```python
from wharf_vision.models import PersonAttrDetector

detector = PersonAttrDetector(
    model_path='person_attr.pt',
    device='cuda',
    conf_threshold=0.25
)

# 基本检测
results = detector.detect(person_crop)

# 检查军装
is_military, conf = detector.check_military_uniform(person_crop)

# 检查PPE
ppe_status = detector.check_ppe(person_crop)
# {'has_helmet': True, 'has_reflective_vest': False, ...}
```

**检测类别:**
- `helmet`: 安全帽
- `reflective_vest`: 反光衣
- `life_jacket`: 救生衣
- `work_clothes`: 工作服
- `military_uniform`: 军装
- `suspected_cigarette`: 疑似烟支

### FireAttrDetector

二级烟火属性检测器。

```python
from wharf_vision.models import FireAttrDetector

detector = FireAttrDetector(
    model_path='fire_attr.pt',
    device='cuda',
    conf_threshold=0.25
)

# 基本检测
results = detector.detect(fire_crop)

# 分类烟火区域
classification = detector.classify_fire_region(fire_crop)
# {'is_fire': True, 'is_smoke': False, ...}

# 判断是否为真实火情
is_real, conf, details = detector.is_real_fire(fire_crop)
```

**检测类别:**
- `fire`: 明火
- `smoke`: 烟雾
- `light_interference`: 灯光干扰
- `welding_interference`: 电焊干扰

## 训练 API

### 训练脚本

```bash
# 训练一级全局检测模型
python wharf_vision/scripts/train_global.py \
    --data config/global.yaml \
    --epochs 80 \
    --imgsz 640 \
    --batch 16

# 训练二级人员属性模型
python wharf_vision/scripts/train_person_attr.py \
    --data config/person_attr.yaml \
    --epochs 80 \
    --imgsz 320

# 训练二级烟火属性模型
python wharf_vision/scripts/train_fire_attr.py \
    --data config/fire_attr.yaml \
    --epochs 100
```

### 验证模型

```bash
# 验证模型
python wharf_vision/scripts/train_global.py --val --data config/global.yaml
```

### 导出模型

```bash
# 导出为ONNX
python wharf_vision/scripts/train_global.py --export onnx

# 导出为TensorRT
python wharf_vision/scripts/train_global.py --export engine
```

## 配置文件格式

### 数据集配置 (YAML)

```yaml
# 数据集路径
path: ./data/dataset_name
train: images/train
val: images/val
test: images/test

# 类别定义
names:
  0: class_name_1
  1: class_name_2
  2: class_name_3
```

### 训练配置 (YAML)

```yaml
# 数据集路径
path: ./data/fire_attr_dataset
train: images/train
val: images/val

# 类别定义
names:
  0: fire
  1: smoke
  2: light_interference
  3: welding_interference

# 训练参数
epochs: 100
imgsz: 320
batch: 32

# 模型选择
model: yolov8n.pt

# 数据增强
hsv_h: 0.02
hsv_s: 0.8
hsv_v: 0.5
degrees: 0.0
translate: 0.1
scale: 0.4
fliplr: 0.5
mosaic: 0.5

# 优化器
optimizer: AdamW
lr0: 0.001
lrf: 0.01

# 其他设置
patience: 50
save: True
cache: True
device: auto
workers: 8
```

## 输出格式

### JSON 格式输出

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
    max_workers=4  # 根据CPU核心数调整
)
```

### 模型导出

导出为ONNX或TensorRT格式可进一步提升推理速度：

```python
from ultralytics import YOLO

model = YOLO('best.pt')
model.export(format='onnx')   # ONNX格式
model.export(format='engine') # TensorRT格式（需要GPU）
```

### 批处理

对于大量图像，可使用批处理：

```python
import cv2
from glob import glob

images = glob('images/*.jpg')
batch_results = []

for img_path in images:
    image = cv2.imread(img_path)
    result = inference.process_frame(image)
    batch_results.append(result)
```
