"""在 Blender 中渲染一段连续的 RGB-D 序列，用来离线回放整条位姿估计流程。

用法（在包目录下）：
    blender --background --factory-startup --python scripts/synth/render_sequence.py -- \\
        --out dataset/synthetic/sequences/seq_000 [--frames 200] [--seed 0] [--width 1280] [--height 720] \\
        [--samples 32] [--device CPU] [--hdri-dir dataset/synthetic/hdri]

场景用 render_dataset.py 的搭建函数，整段序列只搭建一次，中心块上至少有 3 根能量单元。
相机模拟装在机械臂上的 RGB-D 相机（水平视场角 69°），分三段运动：
    1. 前 20% 的帧：停在中心块前方约 1.1 m 处，视线左右扫过中心块；
    2. 到 75% 的帧：接近中心块上的目标单元，停在单元外侧约 0.3 m 处；
    3. 其余帧：取出目标单元，单元和相机一起沿单元轴线向外移动 12 cm。
从 50% 的帧起，两根深色的夹爪从画面下边缘伸进来。相机每帧有约 2 mm、0.3° 的抖动。
镜头效果和曝光整段相同。每帧输出：
    color/<帧号>.jpg   彩色图
    depth/<帧号>.png   深度图：16 位无符号整数，单位 mm，沿光轴的平面深度，0 表示没有深度
    index/<帧号>.png   每个像素属于第几根能量单元（0 为其他物体）
    meta/<帧号>.json   各单元的位姿：mesh 到相机，OpenCV 相机坐标系，单位 m。原点在能量单元的包围盒中心，
                       z 轴指向封闭端，与 C++ 的 geometry::load_binary_stl 相同
sequence.json 记录相机内参、目标单元的编号、帧数和用到的效果。已有 meta 的帧会跳过，中断后可以继续。
"""

import argparse
import json
import math
import random
import struct
import sys
import time
import zlib
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
import render_dataset as rd  # noqa: E402

FOV = math.radians(69.0)
MAX_DEPTH = 10.0  # m，更远的像素（包括没有击中任何物体的背景）记为没有深度


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description="Render a synthetic RGB-D sequence.")
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument("--frames", type=int, default=200, help="number of frames")
    parser.add_argument("--seed", type=int, default=0, help="scene seed")
    parser.add_argument("--samples", type=int, default=32, help="Cycles samples per pixel")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--device", choices=("CPU", "CUDA", "OPTIX"), default="CPU")
    parser.add_argument("--hdri-dir", type=Path, default=rd.PACKAGE_DIR / "dataset" / "synthetic" / "hdri")
    return parser.parse_args(argv)


def smoothstep(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


def look_at(position, target, roll):
    rotation = (target - position).to_track_quat("-Z", "Y").to_matrix().to_4x4()
    return Matrix.Translation(position) @ rotation @ Matrix.Rotation(roll, 4, "Z")


def write_png16(path, image):
    """写 16 位灰度 PNG（OpenCV 用 IMREAD_UNCHANGED 读出 CV_16U）。用标准库编码，避开 Blender 的色彩管理"""
    height, width = image.shape
    raw = b"".join(b"\x00" + row.astype(">u2").tobytes() for row in image)

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    header = struct.pack(">IIBBBBB", width, height, 16, 0, 0, 0, 0)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw, 6))
                     + chunk(b"IEND", b""))


def exr_to_depth_png(exr_path, png_path):
    image = bpy.data.images.load(str(exr_path))
    width, height = image.size
    pixels = np.empty(width * height * 4, dtype=np.float32)
    image.pixels.foreach_get(pixels)
    bpy.data.images.remove(image)
    depth = pixels[0::4].reshape(height, width)[::-1]  # Blender 的像素从下往上排
    millimeters = np.where((depth > 0.0) & (depth < MAX_DEPTH), np.rint(depth * 1000.0), 0.0)
    write_png16(png_path, millimeters.astype(np.uint16))
    exr_path.unlink()


def build_scene(args, rng, mesh):
    """搭建资源区、中心块（至少 3 根单元）、地上的单元、干扰物和灯光，返回单元列表"""
    for _ in range(100):
        rd.clear_scene()
        units = []
        half_width, back, height = rd.build_zone(rng)
        rd.build_hub(rng, mesh, units)
        if sum(u["where"] == "hub" for u in units) >= 3:
            break
    taken = []
    rd.build_floor_units(rng, mesh, units, half_width, taken)
    rd.build_distractors(rng, half_width, taken)
    rd.build_world(rng, args.hdris)
    rd.build_sources(rng, half_width, back, height)
    rd.build_lights(rng, half_width)
    return units


def build_fingers():
    """夹爪的两根手指：深色金属块，其中一根带绿色指示灯。返回物体和它们在相机坐标系下的基准位置"""
    metal = rd.principled("finger", rd.gray(0.05), 0.35, metallic=0.6)
    led = rd.principled("finger_led", (0.1, 1.0, 0.2), 0.5, emission=(0.1, 1.0, 0.2), strength=5.0)
    fingers = []
    for side in (-1.0, 1.0):
        finger = rd.add_box(f"finger_{side:+.0f}", (0.022, 0.12, 0.022), metal, Matrix.Identity(4))
        fingers.append((finger, Vector((side * 0.045, 0.0, -0.16))))
    light = rd.add_box("finger_led", (0.006, 0.02, 0.006), led, Matrix.Identity(4))
    fingers.append((light, Vector((0.045 - 0.012, 0.03, -0.16 + 0.012))))
    return fingers


class Trajectory:
    """相机、目标单元和夹爪在第 frame 帧的位姿"""

    def __init__(self, args, rng, units, center_offset):
        self.frames = args.frames
        self.center_offset = center_offset
        hub = bpy.data.objects["hub"].matrix_world.translation.copy()
        candidates = [u for u in units if u["where"] == "hub" and self.axis(u).z > -0.3]
        self.target = rng.choice(candidates or [u for u in units if u["where"] == "hub"])
        self.target_matrix = self.target["object"].matrix_world.copy()
        center = self.target_matrix @ center_offset
        self.axis_world = self.axis(self.target)

        self.hub = hub
        self.center = center
        self.start = hub + Vector((rng.uniform(-0.15, 0.15), -1.1, rng.uniform(0.0, 0.2)))
        view = (0.6 * self.axis_world + Vector((0.0, -0.8, 0.0))).normalized()
        self.final = center + 0.30 * view
        # 夹爪下沿刚好在画面下边缘时的 y（相机坐标，单位 m，深度 0.16 m 处）
        self.bottom = -math.tan(FOV / 2) * args.height / args.width * 0.16

    @staticmethod
    def axis(unit):
        return (unit["object"].matrix_world.to_3x3() @ Vector((0.0, 0.0, 1.0))).normalized()

    def at(self, frame):
        t = frame / max(self.frames - 1, 1)
        shift = Vector((0.0, 0.0, 0.0))
        if t < 0.2:
            position = self.start
            target = self.hub + Vector((0.15 * math.sin(2 * math.pi * t / 0.2), 0.0, 0.0))
        elif t < 0.75:
            s = smoothstep((t - 0.2) / 0.55)
            position = self.start.lerp(self.final, s)
            target = self.hub.lerp(self.center, s)
        else:
            shift = 0.12 * smoothstep((t - 0.75) / 0.25) * self.axis_world
            position = self.final + shift
            target = self.center + shift
        # 机械臂的抖动：每帧不同、幅度固定，与帧号一一对应，重新运行结果相同
        jitter = 0.002 * Vector((math.sin(17.3 * frame), math.sin(13.1 * frame + 1.0), math.sin(11.7 * frame + 2.0)))
        camera = look_at(position + jitter, target, math.radians(0.3) * math.sin(0.37 * frame))
        unit = Matrix.Translation(shift) @ self.target_matrix
        # 夹爪从 50% 的帧开始伸入画面，到 70% 时伸入约 3.5 cm
        intrusion = 0.035 * smoothstep((t - 0.5) / 0.2) if t >= 0.5 else -0.02
        return camera, unit, self.bottom - 0.06 + intrusion


def setup_depth_output(tree, layers, out):
    bpy.context.view_layer.use_pass_z = True
    node = tree.nodes.new("CompositorNodeOutputFile")
    node.name = "depth_output"
    node.base_path = str(out / "depth")
    node.format.file_format = "OPEN_EXR"
    node.format.color_depth = "32"
    node.format.color_mode = "BW"
    node.format.exr_codec = "ZIP"
    node.file_slots[0].path = "######"
    tree.links.new(layers.outputs["Depth"], node.inputs[0])
    return node


def main():
    args = parse_args()
    args.out = args.out.resolve()
    for sub in ("color", "depth", "index", "meta"):
        (args.out / sub).mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    mesh = rd.load_unit_mesh()
    corners = np.array([v.co for v in mesh.vertices])
    center_offset = Vector(((corners.min(axis=0) + corners.max(axis=0)) / 2).tolist())
    args.hdris = sorted(args.hdri_dir.glob("*.hdr"))
    index_output = rd.setup_render(args)
    index_output.base_path = str(args.out / "index")

    rng = random.Random(args.seed)
    scene = bpy.context.scene
    units = build_scene(args, rng, mesh)
    trajectory = Trajectory(args, rng, units, center_offset)
    fingers = build_fingers()

    data = bpy.data.cameras.new("camera")
    data.sensor_fit = "HORIZONTAL"
    data.angle = FOV
    camera = rd.link(bpy.data.objects.new("camera", data))
    scene.camera = camera

    tree = scene.node_tree
    setup_depth_output(tree, tree.nodes["Render Layers"], args.out)
    effects = rd.build_effects(rng, tree, args)
    noisy = rng.random() < 0.6
    if noisy:
        effects.append("sensor_noise")
    scene.cycles.samples = rng.randint(12, 24) if noisy else args.samples
    scene.cycles.use_adaptive_sampling = not noisy
    scene.cycles.use_denoising = not noisy
    scene.view_settings.view_transform = rng.choice(("Standard", "AgX", "Filmic"))
    scene.view_settings.exposure = rng.uniform(-1.0, 1.0)
    scene.render.image_settings.quality = 90

    fx = (args.width / 2) / math.tan(FOV / 2)
    target_index = trajectory.target["object"].pass_index
    (args.out / "sequence.json").write_text(json.dumps({
        "frames": args.frames,
        "seed": args.seed,
        "target": target_index,
        "camera": {"width": args.width, "height": args.height, "fx": fx, "fy": fx,
                   "cx": args.width / 2 - 0.5, "cy": args.height / 2 - 0.5, "depth_scale": 0.001},
        "effects": effects,
    }, indent=1))

    for frame in range(args.frames):
        if (args.out / "meta" / f"{frame:06d}.json").exists():
            continue
        begin = time.monotonic()
        camera_matrix, unit_matrix, finger_y = trajectory.at(frame)
        camera.matrix_world = camera_matrix
        trajectory.target["object"].matrix_world = unit_matrix
        for obj, base in fingers:
            obj.matrix_world = camera_matrix @ Matrix.Translation(base + Vector((0.0, finger_y, 0.0)))

        scene.frame_current = frame
        scene.cycles.seed = frame
        scene.render.filepath = str(args.out / "color" / f"{frame:06d}.jpg")
        bpy.ops.render.render(write_still=True)
        rd.exr_to_png(args.out / "index" / f"{frame:06d}.exr", args.out / "index" / f"{frame:06d}.png")
        exr_to_depth_png(args.out / "depth" / f"{frame:06d}.exr", args.out / "depth" / f"{frame:06d}.png")

        world_to_camera = rd.BLENDER_TO_OPENCV @ camera_matrix.inverted()
        meta = {
            "frame": frame,
            "units": [
                {"index": u["object"].pass_index, "where": u["where"],
                 "pose": [list(row) for row in
                          world_to_camera @ u["object"].matrix_world @ Matrix.Translation(center_offset)]}
                for u in units
            ],
        }
        (args.out / "meta" / f"{frame:06d}.json").write_text(json.dumps(meta, indent=1))
        print(f"frame {frame}: {time.monotonic() - begin:.1f} s", flush=True)


if __name__ == "__main__":
    main()
