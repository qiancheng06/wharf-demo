"""
视频检测示例
Usage: python examples/demo_video.py --video path/to/video.mp4
"""

import argparse
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from wharf_vision import WharfVisionInference


def main():
    parser = argparse.ArgumentParser(description='视频检测示例')
    parser.add_argument('--video', type=str, required=True,
                       help='输入视频路径')
    parser.add_argument('--output', type=str, default='output.mp4',
                       help='输出视频路径')
    parser.add_argument('--global-model', type=str, default=None,
                       help='一级全局检测模型路径')
    parser.add_argument('--person-model', type=str, default=None,
                       help='二级人员属性模型路径')
    parser.add_argument('--fire-model', type=str, default=None,
                       help='二级烟火属性模型路径')
    parser.add_argument('--device', type=str, default='auto',
                       help='运行设备')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='置信度阈值')
    parser.add_argument('--display', action='store_true',
                       help='实时显示')
    parser.add_argument('--no-parallel', action='store_true',
                       help='禁用并行推理')
    
    args = parser.parse_args()
    
    # 初始化推理引擎
    print("=" * 60)
    print("数字码头 - 视频检测示例")
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
    
    # 处理视频
    results = inference.process_video(
        video_path=args.video,
        output_path=args.output,
        display=args.display
    )
    
    # 统计信息
    print("\n" + "=" * 60)
    print("处理统计")
    print("=" * 60)
    print(f"总帧数: {len(results)}")
    
    # 统计各类别检测数量
    class_counts = {}
    for result in results:
        for det in result.detections:
            class_name = det.class_name
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    print("\n各类别检测次数:")
    for class_name, count in sorted(class_counts.items()):
        print(f"  {class_name}: {count}")
    
    print(f"\n输出视频: {args.output}")


if __name__ == '__main__':
    main()
