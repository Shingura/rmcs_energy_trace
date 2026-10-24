#!/usr/bin/env python3
"""把 Ultralytics YOLO11 的 .pt 权重导出为 OpenVINO IR，供 energy_unit_detect 和组件使用。

用法：
    python3 scripts/export_yolo.py <best.pt> <输出目录>

导出参数固定为 imgsz=640、不含 NMS、不用半精度，与 C++ 解码器（输出 [1, 5, 8400]）一致。
输出目录下会生成 <名称>_openvino_model/，其中的 .xml 就是 model_path。
需要事先安装 ultralytics 和 openvino（本脚本不会安装依赖）；权重和导出结果不要提交到仓库。
"""

import argparse
import shutil
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a YOLO .pt model to OpenVINO IR.")
    parser.add_argument("weights", type=Path, help="Ultralytics .pt weights")
    parser.add_argument("out_dir", type=Path, help="directory to receive the OpenVINO model")
    args = parser.parse_args()

    if not args.weights.is_file():
        print(f"weights not found: {args.weights}", file=sys.stderr)
        return 1

    from ultralytics import YOLO

    exported = Path(
        YOLO(str(args.weights)).export(format="openvino", imgsz=640, half=False, nms=False)
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    destination = args.out_dir / exported.name
    if destination.exists():
        shutil.rmtree(destination)
    shutil.move(str(exported), destination)

    print(f"exported to {destination}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
