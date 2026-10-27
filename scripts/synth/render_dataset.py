"""在 Blender 中渲染能量单元的合成训练数据。

用法（在包目录下）：
    blender --background --factory-startup --python scripts/synth/render_dataset.py -- \\
        --out dataset/synthetic/raw --count 100 [--start 0] [--seed 0] [--samples 32] [--device CPU] \\
        [--hdri-dir dataset/synthetic/hdri]

每一帧重新搭建场景：资源区（地面、墙、底座、支架、中心块）、中心块上 0～6 根能量单元、
地上的能量单元、干扰物和靠近相机的遮挡物；材质、光照和相机都随机；
背景和环境光来自随机的 HDRI（scripts/synth/download_hdri.sh 下载，目录里没有 .hdr 时用深色背景）。
彩色图再按各自的概率叠加相机和镜头的效果（眩光、脏镜头、虚焦、运动模糊、雾化、暗角、色散、
传感器噪声、曝光和白平衡），这些效果只作用于彩色图，编号图不受影响。每帧输出：
    images/<帧号>.jpg   彩色图
    index/<帧号>.png    每个像素属于第几根能量单元（0 为其他物体），即各单元的可见部分
    full/<帧号>.png     只渲染能量单元时的编号图，与 index 对比得到各单元被遮挡的比例
    meta/<帧号>.json    相机内参，各单元的位姿（mesh -> 相机，OpenCV 相机坐标系，单位 m），本帧用到的效果
已有 meta 的帧会跳过，所以中断后可以继续。标签由 scripts/synth/make_yolo_dataset.py 生成。
"""

import argparse
import colorsys
import json
import math
import random
import sys
import time
from pathlib import Path

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector

PACKAGE_DIR = Path(__file__).resolve().parents[2]
MESH_PATH = PACKAGE_DIR / "meshes" / "energy_unit.stl"

# 能量单元 mesh（STL，单位 mm）：轴线在 x = y = 47.5，z 从开口端 0 到封闭端 150
UNIT_AXIS_XY = 0.0475
UNIT_HALF_LENGTH = 0.075
UNIT_CAP_RADIUS = 0.0475
CAP, RIB, TUBE, CAVITY = range(4)

# 中心块（规则图 4-22）：6 个安装位相隔 60°，单元外端在 Ø450 的圆上，中心离地 700 mm
HUB_HEIGHT = 0.7
HUB_APOTHEM = 0.095
HUB_THICKNESS = 0.102
UNIT_CENTER_RADIUS = 0.150

# OpenCV 相机坐标系（x 右、y 下、z 前）与 Blender 相机坐标系（x 右、y 上、z 后）的转换
BLENDER_TO_OPENCV = Matrix.Diagonal((1.0, -1.0, -1.0, 1.0))


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description="Render synthetic energy unit images.")
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument("--count", type=int, required=True, help="number of frames")
    parser.add_argument("--start", type=int, default=0, help="first frame number")
    parser.add_argument("--seed", type=int, default=0, help="dataset seed")
    parser.add_argument("--samples", type=int, default=32, help="Cycles samples per pixel")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--device", choices=("CPU", "CUDA", "OPTIX"), default="CPU")
    parser.add_argument("--hdri-dir", type=Path, default=PACKAGE_DIR / "dataset" / "synthetic" / "hdri",
                        help="directory of .hdr environment maps")
    return parser.parse_args(argv)


# ---------------------------------------------------------------- 材质


def linear(rgb):
    """sRGB 颜色转成 Blender 材质使用的线性值"""
    return tuple(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb)


def hsv(h, s, v):
    return linear(colorsys.hsv_to_rgb(h, s, v))


def gray(v):
    return linear((v, v, v))


def principled(name, color, roughness, metallic=0.0, emission=None, strength=0.0):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    bsdf = material.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    if emission is not None:
        bsdf.inputs["Emission Color"].default_value = (*emission, 1.0)
        bsdf.inputs["Emission Strength"].default_value = strength
    return material


def textured(rng, name, color_a, color_b, roughness):
    """两种颜色按随机程序纹理混合的材质，用于墙面、地面和干扰物"""
    material = principled(name, color_a, roughness)
    nodes, links = material.node_tree.nodes, material.node_tree.links
    kind = rng.choice(
        ("ShaderNodeTexNoise", "ShaderNodeTexVoronoi", "ShaderNodeTexWave", "ShaderNodeTexChecker")
    )
    texture = nodes.new(kind)
    texture.inputs["Scale"].default_value = rng.uniform(2.0, 40.0)
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (*color_a, 1.0)
    ramp.color_ramp.elements[1].color = (*color_b, 1.0)
    links.new(texture.outputs[0], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], nodes["Principled BSDF"].inputs["Base Color"])
    return material


def random_material(rng, name):
    color_a = hsv(rng.random(), rng.uniform(0.0, 0.9), rng.uniform(0.05, 0.95))
    if rng.random() < 0.5:
        return principled(name, color_a, rng.uniform(0.1, 0.9), metallic=rng.choice((0.0, 0.0, 1.0)))
    color_b = hsv(rng.random(), rng.uniform(0.0, 0.9), rng.uniform(0.05, 0.95))
    return textured(rng, name, color_a, color_b, rng.uniform(0.2, 0.9))


def unit_materials(rng, index):
    """能量单元四个部分（端盖、筋、内筒、内腔）的材质。能量单元本身不发光。
    按实物的配色（规则图 4-39）：深灰端盖、黑色筋、奶白内筒，颜色和粗糙度在范围内随机"""
    cap = gray(rng.uniform(0.05, 0.15))
    rib = gray(rng.uniform(0.01, 0.05))
    tube = hsv(rng.uniform(0.08, 0.14), rng.uniform(0.08, 0.3), rng.uniform(0.7, 0.92))
    roughness = (rng.uniform(0.55, 0.9), rng.uniform(0.45, 0.85), rng.uniform(0.3, 0.6))
    cavity = gray(rng.uniform(0.02, 0.2))
    materials = [
        principled(f"unit{index}_cap", cap, roughness[0]),
        principled(f"unit{index}_rib", rib, roughness[1]),
        principled(f"unit{index}_tube", tube, roughness[2]),
        principled(f"unit{index}_cavity", cavity, 0.8),
    ]
    return materials


# ---------------------------------------------------------------- 几何


def link(obj):
    bpy.context.scene.collection.objects.link(obj)
    return obj


def mesh_object(name, bm, material, matrix=Matrix.Identity(4)):
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    mesh.materials.append(material)
    obj = link(bpy.data.objects.new(name, mesh))
    obj.matrix_world = matrix
    return obj


def add_box(name, size, material, matrix):
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.scale(bm, vec=Vector(size), verts=bm.verts)
    return mesh_object(name, bm, material, matrix)


def add_cylinder(name, radius, length, material, matrix, segments=32):
    """沿局部 z 轴、中心在原点的圆柱"""
    bm = bmesh.new()
    bmesh.ops.create_cone(
        bm, cap_ends=True, segments=segments, radius1=radius, radius2=radius, depth=length
    )
    obj = mesh_object(name, bm, material, matrix)
    obj.data.shade_smooth()
    obj.data.set_sharp_from_angle(angle=math.radians(40))
    return obj


def add_sphere(name, radius, material, matrix):
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=24, v_segments=12, radius=radius)
    obj = mesh_object(name, bm, material, matrix)
    obj.data.shade_smooth()
    return obj


def unit_part(center, normal):
    """按面的位置（mm）判断它属于能量单元的哪个部分"""
    x, y, z = center.x - 47.5, center.y - 47.5, center.z
    r = math.hypot(x, y)
    radial = (normal.x * x + normal.y * y) / max(r, 1e-6)
    if r < 21.0 and z < 101.0:
        return CAVITY
    if z <= 35.0 or z >= 110.0:
        return CAP
    # 中段：内筒外表面 Ø60，其余（凸出到 Ø65 的筋、筋两侧的槽和侧壁）都是黑色
    if 29.5 <= r < 30.6 and radial > 0.3:
        return TUBE
    return RIB


def load_unit_mesh():
    """导入 STL（mm），按部分设置材质编号，再缩放到 m。各单元共用这份 mesh，材质设在物体上"""
    bpy.ops.wm.stl_import(filepath=str(MESH_PATH))
    imported = bpy.context.selected_objects[0]
    mesh = imported.data
    mesh.name = "energy_unit"
    mesh.transform(imported.matrix_world)
    bpy.data.objects.remove(imported)
    for _ in range(4):
        mesh.materials.append(None)
    for polygon in mesh.polygons:
        polygon.material_index = unit_part(polygon.center, polygon.normal)
    mesh.transform(Matrix.Scale(0.001, 4))
    mesh.shade_smooth()
    mesh.set_sharp_from_angle(angle=math.radians(30))
    mesh.use_fake_user = True
    return mesh


def unit_matrix(center, axis, roll):
    """单元中心在 center、+z（封闭端）朝 axis 时，mesh（STL 坐标）到世界的变换"""
    rotation = axis.to_track_quat("Z", "Y").to_matrix().to_4x4() @ Matrix.Rotation(roll, 4, "Z")
    offset = Matrix.Translation((-UNIT_AXIS_XY, -UNIT_AXIS_XY, -UNIT_HALF_LENGTH))
    return Matrix.Translation(center) @ rotation @ offset


def add_unit(rng, mesh, units, matrix, where):
    index = len(units) + 1
    obj = link(bpy.data.objects.new(f"unit_{index}", mesh))
    materials = unit_materials(rng, index)
    for slot, material in zip(obj.material_slots, materials):
        slot.link = "OBJECT"
        slot.material = material
    obj.pass_index = index
    obj.matrix_world = matrix
    units.append({"object": obj, "where": where})


# ---------------------------------------------------------------- 场景


def build_zone(rng):
    """资源区：地面、U 形墙、墙上的红色灯条、中心块下方的底座和上方的支架。返回墙内半宽、后墙位置和墙高"""
    floor = textured(
        rng, "floor", gray(rng.uniform(0.15, 0.5)), gray(rng.uniform(0.1, 0.6)), rng.uniform(0.4, 0.9)
    )
    add_box("floor", (12.0, 12.0, 0.02), floor, Matrix.Translation((0.0, 0.0, -0.01)))

    half_width = rng.uniform(0.75, 1.1)
    back = rng.uniform(0.45, 0.8)
    front = -rng.uniform(0.4, 0.8)
    height = rng.uniform(1.2, 1.6)
    wall_gray = rng.uniform(0.15, 0.45)
    wall = textured(
        rng, "wall", hsv(rng.uniform(0.1, 0.3), rng.uniform(0.0, 0.25), wall_gray),
        gray(wall_gray * rng.uniform(0.7, 1.0)), rng.uniform(0.5, 0.9),
    )
    add_box("wall_back", (2 * half_width, 0.05, height), wall,
            Matrix.Translation((0.0, back, height / 2)))
    strip = principled("strip", (1.0, 0.02, 0.02), 0.5, emission=(1.0, 0.02, 0.01),
                       strength=rng.uniform(0.0, 15.0))
    for side in (-1.0, 1.0):
        x = side * half_width
        add_box(f"wall_{side:+.0f}", (0.05, back - front, height), wall,
                Matrix.Translation((x, (back + front) / 2, height / 2)))
        add_box(f"strip_{side:+.0f}", (0.02, 0.03, 0.6), strip,
                Matrix.Translation((x - side * 0.035, front + 0.02, rng.uniform(0.6, 0.9))))

    dark = principled("stand", gray(rng.uniform(0.02, 0.12)), rng.uniform(0.3, 0.8))
    base_height = rng.uniform(0.38, 0.45)
    add_box("base", (rng.uniform(0.5, 0.75), rng.uniform(0.35, 0.5), base_height), dark,
            Matrix.Translation((0.0, rng.uniform(0.0, 0.1), base_height / 2)))
    add_box("support", (0.22, 0.3, 0.5), dark, Matrix.Translation((0.0, 0.2, HUB_HEIGHT + 0.35)))
    return half_width, back, height


def build_hub(rng, mesh, units):
    """中心块：倒角的六棱柱，正面凹进一个六边形，边角有白色面板。各安装位随机装上能量单元，
    空着的安装位露出柱子。正面朝 -y（机器人一侧）"""
    matrix = (
        Matrix.Translation((0.0, 0.0, HUB_HEIGHT + rng.uniform(-0.02, 0.02)))
        @ Matrix.Rotation(math.radians(rng.uniform(-15.0, 15.0)), 4, "Z")
        @ Matrix.Rotation(math.radians(rng.uniform(-8.0, 8.0)), 4, "X")
        @ Matrix.Rotation(math.radians(rng.uniform(0.0, 60.0)), 4, "Y")
    )

    radius = HUB_APOTHEM / math.cos(math.radians(30.0))
    bm = bmesh.new()
    rings = []
    for y in (-HUB_THICKNESS / 2, HUB_THICKNESS / 2):
        rings.append([
            bm.verts.new((radius * math.cos(math.radians(60 * k)), y, radius * math.sin(math.radians(60 * k))))
            for k in range(6)
        ])
    for ring in rings:
        bm.faces.new(ring)
    for k in range(6):
        bm.faces.new((rings[0][k], rings[0][(k + 1) % 6], rings[1][(k + 1) % 6], rings[1][k]))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    cap_edges = [e for e in bm.edges if abs(e.verts[0].co.y - e.verts[1].co.y) < 1e-6]
    bevel = rng.uniform(0.012, 0.02)
    bmesh.ops.bevel(bm, geom=cap_edges, offset=bevel, segments=1, affect="EDGES")
    bm.normal_update()
    front_face = next(f for f in bm.faces if f.normal.y < -0.99)
    bmesh.ops.inset_individual(bm, faces=[front_face], thickness=rng.uniform(0.015, 0.03), depth=-0.008)
    body = principled("hub", gray(rng.uniform(0.02, 0.08)), rng.uniform(0.4, 0.8),
                      metallic=rng.uniform(0.0, 0.3))
    mesh_object("hub", bm, body, matrix)

    glow = rng.random() < 0.5
    panel = principled("hub_panel", hsv(0.6, rng.uniform(0.0, 0.2), 0.9), 0.4,
                       emission=(0.9, 0.95, 1.0), strength=rng.uniform(1.0, 8.0) if glow else 0.0)
    for k in range(6):
        angle = math.radians(60 * k)
        radial = Vector((math.cos(angle), 0.0, math.sin(angle)))
        tangent = Vector((-math.sin(angle), 0.0, math.cos(angle)))
        normal = (radial - Vector((0.0, 1.0, 0.0))).normalized()
        center = radial * (radius - 0.6 * bevel) + Vector((0.0, -HUB_THICKNESS / 2 + 0.5 * bevel, 0.0))
        rotation = Matrix((tangent, tangent.cross(normal), normal)).transposed().to_4x4()
        add_box(f"hub_panel_{k}", (0.04, 0.01, 0.003), panel,
                matrix @ Matrix.Translation(center + normal * 0.0015) @ rotation)

    if rng.random() < 0.7:
        curve = bpy.data.curves.new("logo", "FONT")
        curve.body = "ROBOMASTER"
        curve.size = 0.014
        curve.align_x = curve.align_y = "CENTER"
        logo = link(bpy.data.objects.new("logo", curve))
        logo.data.materials.append(principled("logo", gray(0.8), 0.5))
        logo.matrix_world = (
            matrix
            @ Matrix.Translation((0.0, -HUB_THICKNESS / 2 + 0.0075, -0.02))
            @ Matrix.Rotation(math.radians(90.0), 4, "X")
        )

    pillar = principled("pillar", gray(rng.uniform(0.1, 0.5)), rng.uniform(0.2, 0.6),
                        metallic=rng.uniform(0.0, 1.0))
    occupancy = rng.uniform(0.2, 1.0)
    for k in range(6):
        angle = math.radians(30 + 60 * k)
        direction = (matrix.to_3x3() @ Vector((math.cos(angle), 0.0, math.sin(angle)))).normalized()
        center = matrix.translation
        if rng.random() < occupancy:
            jitter = Vector([rng.gauss(0.0, 0.002) for _ in range(3)])
            add_unit(rng, mesh, units,
                     unit_matrix(center + direction * UNIT_CENTER_RADIUS + jitter, direction,
                                 rng.uniform(0.0, 2 * math.pi)),
                     "hub")
        else:
            rotation = direction.to_track_quat("Z", "Y").to_matrix().to_4x4()
            add_cylinder(f"pillar_{k}", 0.017, 0.08, pillar,
                         Matrix.Translation(center + direction * 0.13) @ rotation)


def place_on_floor(rng, half_width, taken, clearance):
    """在资源区前方的地面上找一个不与已有物体重叠的位置"""
    for _ in range(50):
        p = Vector((rng.uniform(-half_width + 0.15, half_width - 0.15), rng.uniform(-1.3, -0.25), 0.0))
        if all((p - q).length > clearance + r for q, r in taken):
            taken.append((p, clearance))
            return p
    return None


def build_floor_units(rng, mesh, units, half_width, taken):
    """地上的能量单元：横躺或竖立"""
    for _ in range(rng.choice((0, 0, 1, 1, 2, 3))):
        p = place_on_floor(rng, half_width, taken, 0.1)
        if p is None:
            continue
        yaw = rng.uniform(0.0, 2 * math.pi)
        if rng.random() < 0.6:
            axis = Vector((math.cos(yaw), math.sin(yaw), 0.0))
            center = p + Vector((0.0, 0.0, UNIT_CAP_RADIUS))
        else:
            axis = Vector((0.0, 0.0, rng.choice((1.0, -1.0))))
            center = p + Vector((0.0, 0.0, UNIT_HALF_LENGTH))
        add_unit(rng, mesh, units, unit_matrix(center, axis, rng.uniform(0.0, 2 * math.pi)), "floor")


def random_primitive(rng, name, matrix_at, material):
    """随机的长方体、圆柱或球。其中一部分是与能量单元大小相近的深色圆柱，作为难负样本"""
    kind = rng.choice(("box", "cylinder", "sphere", "dark_cylinder"))
    if kind == "box":
        size = [rng.uniform(0.03, 0.3) for _ in range(3)]
        return add_box(name, size, material, matrix_at(size[2] / 2))
    if kind == "sphere":
        radius = rng.uniform(0.02, 0.12)
        return add_sphere(name, radius, material, matrix_at(radius))
    if kind == "dark_cylinder":
        radius, length = rng.uniform(0.025, 0.055), rng.uniform(0.08, 0.22)
        material = principled(name, gray(rng.uniform(0.02, 0.2)), rng.uniform(0.3, 0.8))
    else:
        radius, length = rng.uniform(0.02, 0.12), rng.uniform(0.03, 0.4)
    return add_cylinder(name, radius, length, material, matrix_at(length / 2))


def build_distractors(rng, half_width, taken):
    for i in range(rng.randint(2, 7)):
        p = place_on_floor(rng, half_width, taken, 0.15)
        if p is None:
            continue
        yaw = Matrix.Rotation(rng.uniform(0.0, 2 * math.pi), 4, "Z")
        random_primitive(rng, f"distractor_{i}", lambda h: Matrix.Translation(p + Vector((0, 0, h))) @ yaw,
                         random_material(rng, f"distractor_{i}"))


def build_world(rng, hdris):
    """背景和环境光：随机的 HDRI，绕竖直轴随机旋转。场地较暗，所以强度压低；没有 HDRI 时用深色背景"""
    world = bpy.data.worlds.new("world")
    world.use_nodes = True
    nodes, links = world.node_tree.nodes, world.node_tree.links
    background = nodes["Background"]
    if hdris:
        environment = nodes.new("ShaderNodeTexEnvironment")
        environment.image = bpy.data.images.load(str(rng.choice(hdris)))
        mapping = nodes.new("ShaderNodeMapping")
        mapping.inputs["Rotation"].default_value[2] = rng.uniform(0.0, 2 * math.pi)
        links.new(mapping.outputs["Vector"], environment.inputs["Vector"])
        links.new(environment.outputs["Color"], background.inputs["Color"])
        background.inputs["Strength"].default_value = rng.uniform(0.05, 0.6)
    else:
        background.inputs["Color"].default_value = (*hsv(rng.random(), rng.uniform(0.0, 0.5), rng.uniform(0.0, 0.15)), 1.0)
        background.inputs["Strength"].default_value = 1.0
    bpy.context.scene.world = world


def emitter(name, size, color, strength, matrix):
    material = principled(name, (0.0, 0.0, 0.0), 0.5, emission=color, strength=strength)
    return add_box(name, size, material, matrix)


def build_sources(rng, half_width, back, height):
    """相机能直接看到的亮光源，让眩光有东西可作用：天花板灯具、墙上多出的灯条、中心块后面朝相机的背光"""
    if rng.random() < 0.4:
        for i in range(rng.randint(1, 3)):
            size = (rng.uniform(0.3, 1.2), rng.uniform(0.1, 0.3), 0.02)
            color = colorsys.hsv_to_rgb(rng.uniform(0.05, 0.6), rng.uniform(0.0, 0.25), 1.0)
            emitter(f"fixture_{i}", size, color, rng.uniform(15.0, 80.0),
                    Matrix.Translation((rng.uniform(-1.5, 1.5), rng.uniform(0.5, 3.5), rng.uniform(1.7, 3.0))))

    if rng.random() < 0.4:
        for i in range(rng.randint(1, 2)):
            color = colorsys.hsv_to_rgb(rng.random(), rng.choice((0.0, 0.0, 0.8)), 1.0)
            emitter(f"led_strip_{i}", (rng.uniform(0.6, 1.6) * half_width, 0.015, 0.015), color,
                    rng.uniform(5.0, 30.0),
                    Matrix.Translation((rng.uniform(-0.3, 0.3), back - 0.03, rng.uniform(0.3, height - 0.1))))

    if rng.random() < 0.2:
        # 球体提供看得见的亮点，聚光灯把光打向相机一侧，在中心块边缘形成轮廓光
        position = Vector((rng.choice((-1.0, 1.0)) * rng.uniform(0.35, 0.6), back - 0.1, rng.uniform(0.4, 1.1)))
        radius = rng.uniform(0.02, 0.05)
        sphere = principled("backlight", (0.0, 0.0, 0.0), 0.5, emission=(1.0, 0.95, 0.9),
                            strength=rng.uniform(300.0, 3000.0))
        add_sphere("backlight", radius, sphere, Matrix.Translation(position))
        light = bpy.data.lights.new("backlight", "SPOT")
        light.energy = rng.uniform(200.0, 1000.0)
        light.spot_size = math.radians(60.0)
        light.spot_blend = 0.5
        obj = link(bpy.data.objects.new("backlight", light))
        obj.location = position
        obj.rotation_euler = Vector((0.0, -1.0, 0.0)).to_track_quat("-Z", "Y").to_euler()


def build_lights(rng, half_width):
    for i in range(rng.randint(1, 3)):
        light = bpy.data.lights.new(f"ceiling_{i}", "AREA")
        light.size = rng.uniform(0.5, 2.5)
        light.energy = rng.uniform(50.0, 600.0)
        light.color = colorsys.hsv_to_rgb(rng.uniform(0.05, 0.6), rng.uniform(0.0, 0.25), 1.0)
        obj = link(bpy.data.objects.new(f"ceiling_{i}", light))
        obj.location = (rng.uniform(-1.5, 1.5), rng.uniform(-2.0, 0.5), rng.uniform(1.8, 3.0))
        obj.rotation_euler = (rng.uniform(-0.4, 0.4), rng.uniform(-0.4, 0.4), 0.0)

    # 场馆里的彩色光，照在能量单元上形成彩色反光
    palette = ((0.6, 0.2, 1.0), (0.2, 1.0, 0.3), (0.2, 0.4, 1.0), (1.0, 0.15, 0.1), (0.2, 1.0, 1.0))
    for i in range(rng.randint(0, 3)):
        light = bpy.data.lights.new(f"color_{i}", "POINT")
        light.energy = rng.uniform(5.0, 40.0)
        light.color = rng.choice(palette)
        light.shadow_soft_size = rng.uniform(0.05, 0.5)
        obj = link(bpy.data.objects.new(f"color_{i}", light))
        obj.location = (rng.uniform(-half_width, half_width), rng.uniform(-2.5, -0.5), rng.uniform(1.0, 2.5))


def build_camera(rng, units, half_width):
    """相机从机器人一侧看向中心块、某根单元或附近，返回相机物体和水平视场角"""
    hub_units = [u for u in units if u["where"] == "hub"]
    floor_units = [u for u in units if u["where"] == "floor"]
    choice = rng.random()
    if choice < 0.25 and hub_units:
        target = rng.choice(hub_units)["object"].matrix_world @ Vector((UNIT_AXIS_XY, UNIT_AXIS_XY, UNIT_HALF_LENGTH))
    elif choice < 0.35 and floor_units:
        target = rng.choice(floor_units)["object"].matrix_world @ Vector((UNIT_AXIS_XY, UNIT_AXIS_XY, UNIT_HALF_LENGTH))
    else:
        target = Vector((rng.uniform(-0.15, 0.15), 0.0, HUB_HEIGHT + rng.uniform(-0.15, 0.15)))

    for _ in range(100):
        distance = rng.uniform(0.35, 1.6)
        azimuth = math.radians(rng.uniform(-65.0, 65.0))
        elevation = math.radians(rng.uniform(-30.0, 50.0))
        direction = Vector((math.sin(azimuth) * math.cos(elevation), -math.cos(azimuth) * math.cos(elevation),
                            math.sin(elevation)))
        location = target + direction * distance
        if abs(location.x) < half_width - 0.1 and 0.15 < location.z < 1.7:
            break

    data = bpy.data.cameras.new("camera")
    data.sensor_fit = "HORIZONTAL"
    fov = math.radians(rng.uniform(55.0, 90.0))
    data.angle = fov
    if rng.random() < 0.3:
        data.dof.use_dof = True
        data.dof.focus_distance = distance * rng.uniform(0.8, 1.2)
        data.dof.aperture_fstop = rng.uniform(1.4, 8.0)
    camera = link(bpy.data.objects.new("camera", data))
    rotation = (target - location).to_track_quat("-Z", "Y").to_matrix().to_4x4()
    camera.matrix_world = Matrix.Translation(location) @ rotation @ Matrix.Rotation(
        math.radians(rng.uniform(-15.0, 15.0)), 4, "Z")
    bpy.context.scene.camera = camera
    return camera, fov


def build_occluders(rng, camera, fov, width, height):
    """靠近相机、从画面边缘伸进来的物体，模拟自己的夹爪和机械臂"""
    if rng.random() > 0.4:
        return
    tan_x = math.tan(fov / 2)
    tan_y = tan_x * height / width
    for i in range(rng.randint(1, 2)):
        depth = rng.uniform(0.12, 0.35)
        edge = rng.choices(("bottom", "left", "right", "top"), weights=(0.5, 0.2, 0.2, 0.1))[0]
        u = rng.uniform(-1.0, 1.0) if edge in ("bottom", "top") else (-1.0 if edge == "left" else 1.0)
        v = rng.uniform(-1.0, 1.0) if edge in ("left", "right") else (-1.0 if edge == "bottom" else 1.0)
        local = Vector((u * tan_x * depth, v * tan_y * depth, -depth))
        size = (rng.uniform(0.02, 0.06), rng.uniform(0.02, 0.06), rng.uniform(0.1, 0.3))
        twist = Matrix.Rotation(rng.uniform(0.0, math.pi), 4, "X") @ Matrix.Rotation(rng.uniform(0.0, math.pi), 4, "Y")
        material = principled(f"occluder_{i}", gray(rng.uniform(0.01, 0.15)), rng.uniform(0.2, 0.7),
                              metallic=rng.uniform(0.0, 1.0))
        matrix = camera.matrix_world @ Matrix.Translation(local) @ twist
        add_box(f"occluder_{i}", size, material, matrix)
        if rng.random() < 0.3:
            # 夹爪上的绿色指示灯
            led = principled(f"led_{i}", (0.1, 1.0, 0.2), 0.5, emission=(0.1, 1.0, 0.2),
                             strength=rng.uniform(2.0, 10.0))
            add_box(f"led_{i}", (0.008, 0.008, 0.02), led,
                    matrix @ Matrix.Translation((size[0] / 2, 0.0, rng.uniform(-0.3, 0.3) * size[2])))


# ---------------------------------------------------------------- 相机和镜头效果


def add_node(tree, kind, source, **inputs):
    """加一个合成器节点，source 接到第一个输入，其余输入按名字设置（下划线代替空格）。
    当前类型下不可用的输入（如 Glare 的 Size 只用于 bloom）按名字索引会找不到，所以用 find"""
    node = tree.nodes.new(kind)
    tree.links.new(source, node.inputs[0])
    for name, value in inputs.items():
        node.inputs[node.inputs.find(name.replace("_", " "))].default_value = value
    return node


def mix(tree, blend, fac, a, b):
    """MixRGB 节点，三个输入各自可以是输出口或常数"""
    node = tree.nodes.new("CompositorNodeMixRGB")
    node.blend_type = blend
    for socket, value in zip(node.inputs, (fac, a, b)):
        if isinstance(value, bpy.types.NodeSocket):
            tree.links.new(value, socket)
        else:
            socket.default_value = value
    return node.outputs[0]


def build_effects(rng, tree, args):
    """在合成器里给彩色图加相机和镜头的效果。合成器处理的是视图变换之前的线性 HDR 值，所以眩光随真实光强变化。
    实拍画面通常同时带有几种轻微的效果，所以各效果的概率较高、强度较轻。返回本帧用到的效果名"""
    image = tree.nodes["Render Layers"].outputs["Image"]
    used = []

    def chance(name, probability):
        if rng.random() < probability:
            used.append(name)
            return True
        return False

    if chance("chromatic", 0.4):
        image = add_node(tree, "CompositorNodeLensdist", image, Dispersion=rng.uniform(0.003, 0.012)).outputs[0]

    if chance("glare", 0.7):
        node = add_node(tree, "CompositorNodeGlare", image, Threshold=rng.uniform(0.5, 3.0),
                        Strength=rng.uniform(0.1, 0.6), Size=rng.uniform(0.3, 0.9),
                        Streaks=rng.randint(2, 6), Streaks_Angle=rng.uniform(0.0, math.pi / 2))
        node.glare_type = rng.choice(("BLOOM", "FOG_GLOW", "STREAKS", "GHOSTS"))
        image = node.outputs["Image"]

    if chance("dirty_lens", 0.35):
        # 低频噪声决定哪里脏：脏处混入一份模糊且略微发亮的图
        noise = tree.nodes.new("CompositorNodeTexture")
        noise.texture = bpy.data.textures["dirt"]
        noise.inputs["Offset"].default_value = [rng.uniform(-5.0, 5.0) for _ in range(3)]
        patches = tree.nodes.new("CompositorNodeValToRGB")
        patches.color_ramp.elements[0].position = 0.52
        patches.color_ramp.elements[1].position = 0.68
        patches.color_ramp.elements[1].color = (*[rng.uniform(0.2, 0.5)] * 3, 1.0)
        tree.links.new(noise.outputs["Value"], patches.inputs["Fac"])
        blurred = add_node(tree, "CompositorNodeBlur", image, Size=[rng.uniform(15.0, 40.0)] * 2).outputs[0]
        haze = rng.uniform(0.01, 0.04)
        image = mix(tree, "MIX", patches.outputs["Image"], image, mix(tree, "ADD", 1.0, blurred, (haze,) * 3 + (1.0,)))

    if chance("soft", 0.6):
        # 节点的 Size 约为高斯标准差的 3 倍
        image = add_node(tree, "CompositorNodeBlur", image, Size=[3.0 * rng.uniform(0.3, 1.5)] * 2).outputs[0]

    if chance("motion", 0.2):
        # 方向模糊只向一侧拖尾，拖尾的一半长度用平移补回，免得彩色图相对标签偏移
        length, angle = rng.uniform(2.0, 8.0), rng.uniform(0.0, math.pi)
        image = add_node(tree, "CompositorNodeDBlur", image, Samples=5, Direction=angle,
                         Amount=length / math.hypot(args.width, args.height)).outputs[0]
        image = add_node(tree, "CompositorNodeTranslate", image, X=length / 2 * math.cos(angle),
                         Y=-length / 2 * math.sin(angle)).outputs[0]

    if chance("haze", 0.5):
        tint = colorsys.hsv_to_rgb(rng.random(), rng.uniform(0.0, 0.12), 1.0)
        image = mix(tree, "MIX", rng.uniform(0.03, 0.2), image, (*[c * rng.uniform(0.05, 0.4) for c in tint], 1.0))

    if chance("vignette", 0.5):
        mask = tree.nodes.new("CompositorNodeEllipseMask")
        mask.inputs["Size"].default_value = [rng.uniform(1.1, 1.3)] * 2
        blurred = add_node(tree, "CompositorNodeBlur", mask.outputs[0], Size=[150.0] * 2).outputs[0]
        darkest = rng.uniform(0.35, 0.75)
        image = mix(tree, "MULTIPLY", 1.0, image, mix(tree, "MIX", blurred, (darkest,) * 3 + (1.0,), (1.0,) * 4))

    if chance("white_balance", 0.5):
        warm = rng.uniform(-0.12, 0.12)
        image = mix(tree, "MULTIPLY", 1.0, image, (1.0 + warm, 1.0, 1.0 - warm, 1.0))

    tree.links.new(image, tree.nodes["Composite"].inputs["Image"])
    return used


# ---------------------------------------------------------------- 渲染


def clear_scene():
    for collection in (bpy.data.objects, bpy.data.materials, bpy.data.lights, bpy.data.cameras,
                       bpy.data.curves, bpy.data.worlds, bpy.data.images):
        for item in list(collection):
            collection.remove(item)
    for mesh in list(bpy.data.meshes):
        if mesh.name != "energy_unit":
            bpy.data.meshes.remove(mesh)


def setup_render(args):
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.render.resolution_x = args.width
    scene.render.resolution_y = args.height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "JPEG"
    scene.cycles.use_denoising = True
    scene.cycles.denoiser = "OPENIMAGEDENOISE"
    scene.cycles.max_bounces = 6
    if args.device != "CPU":
        preferences = bpy.context.preferences.addons["cycles"].preferences
        preferences.compute_device_type = args.device
        preferences.get_devices()
        for device in preferences.devices:
            device.use = device.type == args.device
        scene.cycles.device = "GPU"

    bpy.context.view_layer.use_pass_object_index = True
    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()
    layers = tree.nodes.new("CompositorNodeRLayers")
    composite = tree.nodes.new("CompositorNodeComposite")
    tree.links.new(layers.outputs["Image"], composite.inputs["Image"])
    index_output = tree.nodes.new("CompositorNodeOutputFile")
    index_output.format.file_format = "OPEN_EXR"
    index_output.format.color_depth = "16"
    index_output.format.color_mode = "BW"
    index_output.format.exr_codec = "ZIP"
    index_output.file_slots[0].path = "######"
    tree.links.new(layers.outputs["IndexOB"], index_output.inputs[0])

    # 脏镜头用的低频噪声，每帧只改节点上的偏移
    dirt = bpy.data.textures.new("dirt", "CLOUDS")
    dirt.noise_scale = 0.7
    dirt.noise_depth = 2
    return index_output


def exr_to_png(exr_path, png_path):
    """合成器写出的编号图（EXR）转成 8 位 PNG，像素值就是单元编号。pip 安装的 OpenCV 不能读 EXR"""
    image = bpy.data.images.load(str(exr_path))
    width, height = image.size
    pixels = np.empty(width * height * 4, dtype=np.float32)
    image.pixels.foreach_get(pixels)
    bpy.data.images.remove(image)
    rgba = np.repeat(np.rint(pixels[0::4]).clip(0, 255) / 255.0, 4).astype(np.float32)
    rgba[3::4] = 1.0
    png = bpy.data.images.new("index", width, height)
    png.colorspace_settings.name = "Non-Color"
    png.pixels.foreach_set(rgba)
    png.filepath_raw = str(png_path)
    png.file_format = "PNG"
    png.save()
    bpy.data.images.remove(png)
    exr_path.unlink()


def render_frame(args, mesh, index_output, frame):
    rng = random.Random(args.seed * 1_000_003 + frame)
    scene = bpy.context.scene
    clear_scene()

    units = []
    half_width, back, height = build_zone(rng)
    build_hub(rng, mesh, units)
    taken = []
    build_floor_units(rng, mesh, units, half_width, taken)
    build_distractors(rng, half_width, taken)
    build_world(rng, args.hdris)
    build_sources(rng, half_width, back, height)
    build_lights(rng, half_width)
    camera, fov = build_camera(rng, units, half_width)
    build_occluders(rng, camera, fov, args.width, args.height)

    scene.frame_current = frame
    scene.cycles.seed = frame
    scene.view_settings.view_transform = rng.choice(("Standard", "AgX", "Filmic"))
    scene.view_settings.exposure = rng.uniform(-2.0, 1.5)
    scene.render.image_settings.quality = rng.randint(60, 95)
    scene.render.filepath = str(args.out / "images" / f"{frame:06d}.jpg")

    tree = scene.node_tree
    for node in list(tree.nodes):
        if node.name not in ("Render Layers", "Composite", index_output.name):
            tree.nodes.remove(node)
    effects = build_effects(rng, tree, args)

    # 第一次：正常渲染，得到彩色图和各单元的可见部分。多数帧用较少的采样且不降噪，
    # 渲染噪声就像传感器噪声
    noisy = rng.random() < 0.6
    if noisy:
        effects.append("sensor_noise")
    scene.cycles.samples = rng.randint(12, 24) if noisy else args.samples
    scene.cycles.use_adaptive_sampling = not noisy
    scene.cycles.use_denoising = not noisy
    index_output.base_path = str(args.out / "index")
    bpy.ops.render.render(write_still=True)
    exr_to_png(args.out / "index" / f"{frame:06d}.exr", args.out / "index" / f"{frame:06d}.png")

    # 第二次：只留能量单元，得到各单元不被其他物体遮挡时的轮廓；彩色图不再需要，跳过效果节点
    tree.links.new(tree.nodes["Render Layers"].outputs["Image"], tree.nodes["Composite"].inputs["Image"])
    unit_objects = {u["object"].name for u in units}
    for obj in scene.objects:
        obj.hide_render = obj.type == "MESH" and obj.name not in unit_objects or obj.type == "FONT"
    scene.cycles.samples = 1
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.use_denoising = False
    index_output.base_path = str(args.out / "full")
    bpy.ops.render.render(write_still=False)
    exr_to_png(args.out / "full" / f"{frame:06d}.exr", args.out / "full" / f"{frame:06d}.png")

    fx = (args.width / 2) / math.tan(fov / 2)
    world_to_camera = BLENDER_TO_OPENCV @ camera.matrix_world.inverted()
    meta = {
        "frame": frame,
        "seed": args.seed,
        "effects": effects,
        "camera": {"width": args.width, "height": args.height, "fx": fx, "fy": fx,
                   "cx": args.width / 2 - 0.5, "cy": args.height / 2 - 0.5},
        "units": [
            {"index": i + 1, "where": u["where"],
             "pose": [list(row) for row in world_to_camera @ u["object"].matrix_world]}
            for i, u in enumerate(units)
        ],
    }
    (args.out / "meta" / f"{frame:06d}.json").write_text(json.dumps(meta, indent=1))
    return len(units)


def main():
    args = parse_args()
    args.out = args.out.resolve()
    for sub in ("images", "index", "full", "meta"):
        (args.out / sub).mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    mesh = load_unit_mesh()
    args.hdris = sorted(args.hdri_dir.glob("*.hdr"))
    index_output = setup_render(args)
    for frame in range(args.start, args.start + args.count):
        if (args.out / "meta" / f"{frame:06d}.json").exists():
            continue
        begin = time.monotonic()
        count = render_frame(args, mesh, index_output, frame)
        print(f"frame {frame}: {count} units, {time.monotonic() - begin:.1f} s", flush=True)


if __name__ == "__main__":
    main()
