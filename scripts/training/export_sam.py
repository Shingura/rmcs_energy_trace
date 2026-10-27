#!/usr/bin/env python3
"""把 Segment Anything（SAM1）的官方权重导出为两个 ONNX 网络，供 C++ 的 segmentation::Segmenter 使用。

用法：
    python3 scripts/training/export_sam.py <sam_vit_b_01ec64.pth> <输出目录> [--model-type vit_b]
        [--check <图片> <x1,y1,x2,y2>]

输出：
    sam_<type>_encoder.onnx   输入 image [1, 3, 1024, 1024]（已缩放、归一化、补零的 RGB），
                              输出 image_embeddings [1, 256, 64, 64]
    sam_<type>_decoder.onnx   输入 image_embeddings 和 box [1, 2, 2]（检测框左上、右下角，
                              坐标在 1024 输入图中），输出 iou_predictions [1, 4] 和
                              low_res_masks [1, 4, 256, 256]（logits）

解码器只接受检测框提示，不输出放大后的掩码：放大到原图在 C++ 中完成，网络的输入输出尺寸固定。
第 0 个掩码对应官方 SamPredictor.predict(box=..., multimask_output=False) 的结果。

--check 用官方 SamPredictor 和导出的网络分别分割同一张图，打印两者的 IoU。

需要事先安装 torch、torchvision、segment-anything、onnx、onnxruntime、opencv-python（本脚本不会安装依赖）。
权重只当作数据读取（torch.load(weights_only=True)）。
"""

import argparse
import sys
from pathlib import Path

import numpy as np

IMAGE_SIZE = 1024
PIXEL_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
PIXEL_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)


def build_sam(model_type: str, checkpoint: Path):
    import torch
    from segment_anything import sam_model_registry

    sam = sam_model_registry[model_type](checkpoint=None)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    sam.load_state_dict(state)
    sam.eval()
    return sam


def make_modules(sam):
    import torch
    from segment_anything.utils.onnx import SamOnnxModel

    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.image_encoder = sam.image_encoder

        def forward(self, image):
            return self.image_encoder(image)

    class BoxDecoder(torch.nn.Module):
        # 与官方 SamOnnxModel 相同的点编码；框的两个角点标签为 2、3，没有输入掩码
        def __init__(self):
            super().__init__()
            self.prompt = SamOnnxModel(sam, return_single_mask=False)
            self.register_buffer("labels", torch.tensor([[2.0, 3.0]]))

        def forward(self, image_embeddings, box):
            sparse = self.prompt._embed_points(box, self.labels)
            dense = sam.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                1, -1, image_embeddings.shape[2], image_embeddings.shape[3]
            )
            masks, scores = sam.mask_decoder.predict_masks(
                image_embeddings=image_embeddings,
                image_pe=sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse,
                dense_prompt_embeddings=dense,
            )
            return scores, masks

    return Encoder().eval(), BoxDecoder().eval()


def export(sam, out_dir: Path, model_type: str) -> tuple[Path, Path]:
    import torch

    encoder, decoder = make_modules(sam)
    encoder_path = out_dir / f"sam_{model_type}_encoder.onnx"
    decoder_path = out_dir / f"sam_{model_type}_decoder.onnx"
    with torch.no_grad():
        image = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE)
        torch.onnx.export(
            encoder, (image,), str(encoder_path), input_names=["image"],
            output_names=["image_embeddings"], opset_version=17, do_constant_folding=True,
        )
        embeddings = encoder(image)
        box = torch.tensor([[[100.0, 100.0], [400.0, 500.0]]])
        torch.onnx.export(
            decoder, (embeddings, box), str(decoder_path),
            input_names=["image_embeddings", "box"],
            output_names=["iou_predictions", "low_res_masks"], opset_version=17,
            do_constant_folding=True,
        )
    return encoder_path, decoder_path


def preprocess(rgb: np.ndarray):
    """与官方 ResizeLongestSide + Sam.preprocess 相同：长边缩放到 1024，归一化，右下补零。"""
    import cv2

    height, width = rgb.shape[:2]
    scale = IMAGE_SIZE / max(height, width)
    new_h, new_w = int(height * scale + 0.5), int(width * scale + 0.5)
    resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    normalized = (resized - PIXEL_MEAN) / PIXEL_STD
    padded = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)
    padded[:new_h, :new_w] = normalized
    return padded.transpose(2, 0, 1)[None], (new_h, new_w), (new_w / width, new_h / height)


def onnx_mask(encoder_path: Path, decoder_path: Path, rgb: np.ndarray, box: np.ndarray):
    import cv2
    import onnxruntime

    image, (new_h, new_w), (sx, sy) = preprocess(rgb)
    encoder = onnxruntime.InferenceSession(str(encoder_path))
    decoder = onnxruntime.InferenceSession(str(decoder_path))
    (embeddings,) = encoder.run(None, {"image": image})
    corners = np.array([[[box[0] * sx, box[1] * sy], [box[2] * sx, box[3] * sy]]], np.float32)
    scores, masks = decoder.run(None, {"image_embeddings": embeddings, "box": corners})
    low_res = masks[0, 0]
    upscaled = cv2.resize(low_res, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
    cropped = upscaled[:new_h, :new_w]
    full = cv2.resize(cropped, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    return full > 0.0, float(scores[0, 0])


def check(sam, encoder_path: Path, decoder_path: Path, image_path: Path, box: np.ndarray) -> float:
    import cv2
    from segment_anything import SamPredictor

    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise SystemExit(f"cannot read image: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    predictor = SamPredictor(sam)
    predictor.set_image(rgb)
    masks, scores, _ = predictor.predict(box=box, multimask_output=False)
    reference = masks[0]
    exported, score = onnx_mask(encoder_path, decoder_path, rgb, box)
    union = np.logical_or(reference, exported).sum()
    iou = float(np.logical_and(reference, exported).sum() / union) if union > 0 else 1.0
    print(f"official mask {int(reference.sum())} px (score {scores[0]:.3f}), "
          f"exported mask {int(exported.sum())} px (score {score:.3f}), IoU {iou:.4f}")
    return iou


def main() -> int:
    parser = argparse.ArgumentParser(description="Export SAM to ONNX for the C++ segmenter.")
    parser.add_argument("checkpoint", type=Path, help="official SAM checkpoint (.pth)")
    parser.add_argument("out_dir", type=Path, help="directory to receive the ONNX files")
    parser.add_argument("--model-type", default="vit_b", choices=["vit_b", "vit_l", "vit_h"])
    parser.add_argument("--check", nargs=2, metavar=("IMAGE", "BOX"),
                        help="compare with the official predictor on IMAGE with BOX=x1,y1,x2,y2")
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        print(f"checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 1
    args.out_dir.mkdir(parents=True, exist_ok=True)

    sam = build_sam(args.model_type, args.checkpoint)
    encoder_path, decoder_path = export(sam, args.out_dir, args.model_type)
    print(f"wrote {encoder_path}")
    print(f"wrote {decoder_path}")

    if args.check:
        box = np.array([float(v) for v in args.check[1].split(",")], dtype=np.float32)
        if box.shape != (4,):
            print("BOX must be x1,y1,x2,y2", file=sys.stderr)
            return 1
        check(sam, encoder_path, decoder_path, Path(args.check[0]), box)
    return 0


if __name__ == "__main__":
    sys.exit(main())
