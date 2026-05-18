"""
快速创建测试数据集 - 简化版
"""

import cv2
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

def create_dataset(name, classes, samples_per_class=10):
    """创建数据集"""
    print(f"创建 {name}...")
    
    d = DATA_DIR / name
    (d / "images/train").mkdir(parents=True, exist_ok=True)
    (d / "images/val").mkdir(parents=True, exist_ok=True)
    (d / "labels/train").mkdir(parents=True, exist_ok=True)
    (d / "labels/val").mkdir(parents=True, exist_ok=True)
    
    idx = 0
    for cid, cls in enumerate(classes):
        for i in range(samples_per_class):
            # 生成图像
            img = np.ones((640, 640, 3), dtype=np.uint8) * 240
            
            if cls == "person":
                cv2.rectangle(img, (200, 200), (400, 500), (100, 100, 200), -1)
                cv2.circle(img, (300, 150), 50, (150, 150, 250), -1)
            elif cls == "container":
                cv2.rectangle(img, (150, 250), (490, 450), (200, 150, 100), -1)
            elif cls == "fire_region":
                for r in range(100, 0, -5):
                    intensity = int(255 * (100-r)/100)
                    cv2.circle(img, (320, 400), r, (0, intensity//2, intensity), -1)
            elif cls == "obstacle":
                cv2.rectangle(img, (220, 300), (420, 500), (128, 128, 128), -1)
            elif cls == "helmet":
                cv2.ellipse(img, (320, 200), (80, 60), 0, 180, 360, (0, 255, 255), -1)
                cv2.rectangle(img, (240, 250), (400, 500), (100, 100, 150), -1)
            elif cls == "reflective_vest":
                cv2.rectangle(img, (240, 250), (400, 500), (100, 100, 150), -1)
                cv2.rectangle(img, (250, 300), (390, 450), (0, 255, 255), -1)
            elif cls == "fire":
                for r in range(80, 0, -5):
                    intensity = int(255 * (80-r)/80)
                    cv2.circle(img, (320, 350), r, (0, intensity//2, intensity), -1)
            elif cls == "smoke":
                for r in range(100, 0, -10):
                    gray = int(100 + 100*(100-r)/100)
                    cv2.circle(img, (320, 300), r, (gray, gray, gray), -1)
            else:
                cv2.rectangle(img, (200, 200), (440, 440), (150, 150, 150), -1)
            
            # 生成标注
            x, y, w, h = np.random.randint(100, 400), np.random.randint(100, 400), 200, 200
            xc, yc, wn, hn = (x+w/2)/640, (y+h/2)/640, w/640, h/640
            label = f"{cid} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}"
            
            # 保存
            is_val = i % 5 == 0 and i > 0
            split = "val" if is_val else "train"
            cv2.imwrite(str(d / f"images/{split}/{name}_{idx:04d}.jpg"), img)
            with open(d / f"labels/{split}/{name}_{idx:04d}.txt", "w") as f:
                f.write(label)
            
            idx += 1
    
    # 创建yaml
    with open(DATA_DIR / f"{name}.yaml", "w") as f:
        f.write(f"path: ./data/{name}\ntrain: images/train\nval: images/val\n\nnames:\n")
        for i, c in enumerate(classes):
            f.write(f"  {i}: {c}\n")
    
    print(f"  完成: {idx} 张")

if __name__ == "__main__":
    print("="*50)
    print("创建测试数据集")
    print("="*50)
    
    create_dataset("global_dataset", ["person", "container", "fire_region", "obstacle"], 12)
    create_dataset("person_attr_dataset", ["helmet", "reflective_vest", "life_jacket", "work_clothes", "military_uniform", "suspected_cigarette"], 8)
    create_dataset("fire_attr_dataset", ["fire", "smoke", "light_interference", "welding_interference"], 8)
    
    print("\n完成!")
    print("训练命令:")
    print("  python scripts/train_global.py --data data/global_dataset.yaml --epochs 3")
    print("  python scripts/train_person_attr.py --data data/person_attr_dataset.yaml --epochs 3")
    print("  python scripts/train_fire_attr.py --data data/fire_attr_dataset.yaml --epochs 3")
