#!/usr/bin/env python3
"""把 render_dataset.py 的渲染结果转换成 Ultralytics 的实例分割数据集。

用法：
    python3 scripts/synth/make_yolo_dataset.py <渲染目录> <数据集目录> \\
        [--val 0.1] [--min-pixels 300] [--min-visible 0.2] [--preview 20]

每根能量单元的标签由它在 index 中的可见像素生成。可见像素少于 min-pixels，或者可见部分不到
完整轮廓（full 中的像素）的 min-visible，就不标注，这样的单元在图中保留，当作背景。
数据集目录下生成 images/ 和 labels/（各含 train、val，图像用硬链接）、data.yaml，
以及 preview/ 下画出标注的预览图。需要 opencv-python、numpy 和 ultralytics。
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np


def read_index(path):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"cannot read {path}")
    if image.ndim == 3:
        image = image[..., 0]
    return image.astype(np.int32)


def polygon(mask):
    """掩码的外轮廓。被遮挡分成几块时，按 Ultralytics 转换 COCO 的做法连成一个多边形"""
    from ultralytics.data.converter import merge_multi_segment

    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    parts = [
        cv2.approxPolyDP(c, 1.0, True).reshape(-1, 2) for c in contours if cv2.contourArea(c) >= 20.0
    ]
    parts = [p for p in parts if len(p) >= 3]
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    return np.concatenate(merge_multi_segment([p.astype(np.float64) for p in parts]), axis=0)


def is_val(frame, ratio):
    # 按帧号散列分配，与渲染顺序无关，补渲染的帧不会改变已有帧的划分
    return (frame * 2654435761 % 2**32) / 2**32 < ratio


def link_or_copy(src, dst):
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def draw_preview(image, polygons):
    overlay = image.copy()
    for points in polygons:
        cv2.fillPoly(overlay, [points.astype(np.int32)], (0, 200, 255))
    image = cv2.addWeighted(overlay, 0.35, image, 0.65, 0.0)
    for points in polygons:
        cv2.polylines(image, [points.astype(np.int32)], True, (0, 200, 255), 2)
        x, y, w, h = cv2.boundingRect(points.astype(np.int32))
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)
    return image


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert rendered frames to a YOLO segmentation dataset.")
    parser.add_argument("raw", type=Path, help="output directory of render_dataset.py")
    parser.add_argument("out", type=Path, help="dataset directory to create")
    parser.add_argument("--val", type=float, default=0.1, help="fraction of frames for validation")
    parser.add_argument("--min-pixels", type=int, default=300, help="minimum visible pixels per unit")
    parser.add_argument("--min-visible", type=float, default=0.2, help="minimum visible fraction per unit")
    parser.add_argument("--preview", type=int, default=20, help="number of preview images to draw")
    args = parser.parse_args()

    metas = sorted((args.raw / "meta").glob("*.json"))
    if not metas:
        print(f"no frames in {args.raw / 'meta'}", file=sys.stderr)
        return 1
    out = args.out.resolve()
    for sub in ("images/train", "images/val", "labels/train", "labels/val", "preview"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    counts = {"train": 0, "val": 0, "labeled": 0, "skipped": 0, "empty": 0}
    for n, meta_path in enumerate(metas):
        meta = json.loads(meta_path.read_text())
        name = meta_path.stem
        index = read_index(args.raw / "index" / f"{name}.png")
        full = read_index(args.raw / "full" / f"{name}.png")
        height, width = index.shape

        polygons = []
        for unit in meta["units"]:
            visible = index == unit["index"]
            pixels = int(visible.sum())
            whole = int((full == unit["index"]).sum())
            if pixels < args.min_pixels or pixels < args.min_visible * max(whole, 1):
                counts["skipped"] += pixels > 0
                continue
            points = polygon(visible)
            if points is not None:
                polygons.append(points)

        split = "val" if is_val(int(name), args.val) else "train"
        counts[split] += 1
        counts["labeled"] += len(polygons)
        counts["empty"] += not polygons
        image_path = args.raw / "images" / f"{name}.jpg"
        link_or_copy(image_path, out / "images" / split / f"{name}.jpg")
        lines = [
            "0 " + " ".join(f"{x / width:.5f} {y / height:.5f}" for x, y in points) for points in polygons
        ]
        (out / "labels" / split / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))

        if n < args.preview:
            cv2.imwrite(str(out / "preview" / f"{name}.jpg"), draw_preview(cv2.imread(str(image_path)), polygons))

    # 不写 path：Ultralytics 会以 data.yaml 所在目录为根，数据集在容器和宿主机（挂载路径不同）上都能用
    (out / "data.yaml").write_text(
        "train: images/train\nval: images/val\nnames:\n  0: energy_unit\n"
    )
    print(
        f"{len(metas)} frames ({counts['train']} train, {counts['val']} val), "
        f"{counts['labeled']} labeled units, {counts['skipped']} partly visible units left unlabeled, "
        f"{counts['empty']} frames without labels"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
