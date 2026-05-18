"""
单帧图像检测示例
Usage: python examples/demo_image.py --image path/to/image.jpg
"""

import argparse
import sys
from pathlib import Path
import cv2

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from wharf_vision import WharfVisionInference


def main():
    parser = argparse.ArgumentParser(description='单帧图像检测示例')
    parser.add_argument('--image', type=str, required=True,
                       help='输入图像路径')
    parser.add_argument('--output', type=str, default='output.jpg',
                       help='输出图像路径')
    parser.add_argument('--global-model', type=str, default=None,
                       help='一级全局检测模型路径')
    parser.add_argument('--person-model', type=str, default=None,
                       help='二级人员属性模型路径')
    parser.add_argument('--fire-model', type=str, default=None,
                       help='二级烟火属性模型路径')
    parser.add_argument('--device', type=str, default='auto',
                       help='运行设备 (cpu, cuda, auto)')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='置信度阈值')
    parser.add_argument('--no-parallel', action='store_true',
                       help='禁用并行推理')
    
    args = parser.parse_args()
    
    # 初始化推理引擎
    print("=" * 60)
    print("数字码头 - 单帧视觉感知模块")
    print("=" * 60)
    
    inference = WharfVisionInference(
        global_model_path=args.global_model,
        person_attr_model_path=args.person_model,
        fire_attr_model_path=args.fire_model,
        device=args.device,
        conf_threshold=args.conf,
        enable_parallel=not args.no_parallel,
        max_workers=4
    )
    
    # 读取图像
    image = cv2.imread(args.image)
    if image is None:
        print(f"错误: 无法读取图像 {args.image}")
        return
    
    print(f"\n处理图像: {args.image}")
    print(f"图像尺寸: {image.shape[1]}x{image.shape[0]}")
    
    # 执行检测
    result = inference.process_frame(image)
    
    # 打印结果
    print("\n检测结果:")
    print("-" * 60)
    for det in result.detections:
        print(f"类别: {det.class_name}")
        print(f"  置信度: {det.confidence:.4f}")
        print(f"  边界框: {det.bbox}")
        
        if det.attributes:
            print(f"  属性:")
            for key, value in det.attributes.items():
                if key != 'detected_items':
                    print(f"    {key}: {value}")
        print()
    
    # 可视化
    vis_image = inference.visualize(image, result)
    cv2.imwrite(args.output, vis_image)
    print(f"可视化结果已保存: {args.output}")
    
    # 显示结果（可选）
    cv2.imshow('Detection Result', vis_image)
    print("按任意键关闭窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
