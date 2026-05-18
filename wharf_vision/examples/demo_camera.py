"""
摄像头实时检测示例
Usage: python examples/demo_camera.py --source 0
"""

import argparse
import sys
from pathlib import Path
import cv2
import time

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from wharf_vision import WharfVisionInference


def main():
    parser = argparse.ArgumentParser(description='摄像头实时检测示例')
    parser.add_argument('--source', type=str, default='0',
                       help='摄像头索引或RTSP地址')
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
    parser.add_argument('--no-parallel', action='store_true',
                       help='禁用并行推理')
    
    args = parser.parse_args()
    
    # 解析摄像头源
    if args.source.isdigit():
        source = int(args.source)
    else:
        source = args.source
    
    # 初始化推理引擎
    print("=" * 60)
    print("数字码头 - 摄像头实时检测")
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
    
    # 打开摄像头
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"错误: 无法打开摄像头 {source}")
        return
    
    # 获取摄像头参数
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"\n摄像头参数:")
    print(f"  分辨率: {width}x{height}")
    print(f"  FPS: {fps}")
    print("\n按 'q' 退出")
    print("=" * 60)
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("错误: 无法读取帧")
            break
        
        frame_count += 1
        
        # 执行检测
        result = inference.process_frame(frame, frame_count)
        
        # 可视化
        vis_frame = inference.visualize(frame, result)
        
        # 计算FPS
        elapsed = time.time() - start_time
        current_fps = frame_count / elapsed if elapsed > 0 else 0
        
        # 显示FPS
        cv2.putText(vis_frame, f"FPS: {current_fps:.1f}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 显示结果
        cv2.imshow('Wharf Vision - Camera', vis_frame)
        
        # 按'q'退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n" + "=" * 60)
    print(f"处理完成，共 {frame_count} 帧")
    print(f"平均FPS: {frame_count / elapsed:.1f}")
    print("=" * 60)


if __name__ == '__main__':
    main()
