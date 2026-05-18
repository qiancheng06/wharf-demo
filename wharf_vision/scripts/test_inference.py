"""
推理测试脚本 - 用训练好的模型对验证集图片进行推理并可视化
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "test_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def test_model(model_path, dataset_name, imgsz=640, conf=0.25):
    """测试单个模型"""
    print(f"\n{'='*50}")
    print(f"测试: {dataset_name}")
    print(f"模型: {model_path}")
    print(f"{'='*50}")

    model = YOLO(str(model_path))

    # 读取验证集图片
    val_img_dir = DATA_DIR / dataset_name / "images" / "val"
    images = sorted(val_img_dir.glob("*.jpg"))[:5]  # 取5张测试

    if not images:
        print(f"  未找到验证图片: {val_img_dir}")
        return

    out_dir = OUTPUT_DIR / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    for img_path in images:
        # 推理
        results = model(str(img_path), conf=conf, imgsz=imgsz, verbose=False)
        r = results[0]

        # 绘制结果
        img = cv2.imread(str(img_path))
        plotted = r.plot()

        # 保存
        out_path = out_dir / f"result_{img_path.name}"
        cv2.imwrite(str(out_path), plotted)

        # 打印检测信息
        boxes = r.boxes
        if boxes is not None and len(boxes) > 0:
            names = r.names
            for box in boxes:
                cls_id = int(box.cls.item())
                cls_name = names[cls_id]
                conf_val = float(box.conf.item())
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                print(f"  {img_path.name}: {cls_name} conf={conf_val:.3f} bbox={xyxy.tolist()}")
        else:
            print(f"  {img_path.name}: 未检测到目标")

    print(f"  结果已保存: {out_dir}")


def test_pipeline():
    """测试完整推理流水线（一级+二级）"""
    print(f"\n{'='*50}")
    print(f"测试完整推理流水线")
    print(f"{'='*50}")

    global_model = YOLO(str(MODELS_DIR / "global_detector.pt"))
    person_model = YOLO(str(MODELS_DIR / "person_attr_detector.pt"))
    fire_model = YOLO(str(MODELS_DIR / "fire_attr_detector.pt"))

    # 取一张全局检测验证图
    val_img_dir = DATA_DIR / "global_dataset" / "images" / "val"
    images = sorted(val_img_dir.glob("*.jpg"))[:3]

    out_dir = OUTPUT_DIR / "pipeline"
    out_dir.mkdir(parents=True, exist_ok=True)

    for img_path in images:
        img = cv2.imread(str(img_path))

        # 一级检测
        results = global_model(str(img_path), conf=0.25, imgsz=640, verbose=False)
        r = results[0]

        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            print(f"  {img_path.name}: 一级未检测到目标")
            continue

        # 对每个检测到的目标进行二级推理
        for box in boxes:
            cls_id = int(box.cls.item())
            cls_name = r.names[cls_id]
            conf_val = float(box.conf.item())
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy

            print(f"  {img_path.name}: 一级检测到 {cls_name} (conf={conf_val:.3f})")

            # 裁剪目标区域
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # 根据类别选择二级模型
            if cls_name == "person":
                sub_results = person_model(crop, conf=0.25, imgsz=320, verbose=False)
                sr = sub_results[0]
                if sr.boxes is not None and len(sr.boxes) > 0:
                    for sb in sr.boxes:
                        sub_cls = sr.names[int(sb.cls.item())]
                        sub_conf = float(sb.conf.item())
                        print(f"    -> 二级: {sub_cls} (conf={sub_conf:.3f})")
            elif cls_name == "fire_region":
                sub_results = fire_model(crop, conf=0.25, imgsz=320, verbose=False)
                sr = sub_results[0]
                if sr.boxes is not None and len(sr.boxes) > 0:
                    for sb in sr.boxes:
                        sub_cls = sr.names[int(sb.cls.item())]
                        sub_conf = float(sb.conf.item())
                        print(f"    -> 二级: {sub_cls} (conf={sub_conf:.3f})")

        # 可视化
        plotted = r.plot()
        out_path = out_dir / f"pipeline_{img_path.name}"
        cv2.imwrite(str(out_path), plotted)

    print(f"  流水线结果已保存: {out_dir}")


if __name__ == "__main__":
    print("=" * 50)
    print("数字码头 - 推理测试")
    print("=" * 50)

    # 测试三个模型
    test_model(MODELS_DIR / "global_detector.pt", "global_dataset", imgsz=640)
    test_model(MODELS_DIR / "person_attr_detector.pt", "person_attr_dataset", imgsz=320)
    test_model(MODELS_DIR / "fire_attr_detector.pt", "fire_attr_dataset", imgsz=320)

    # 测试完整流水线
    test_pipeline()

    print(f"\n{'='*50}")
    print(f"测试完成! 结果保存在: {OUTPUT_DIR}")
    print(f"{'='*50}")
