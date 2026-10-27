"""Blender 中的工程机械臂：读取 URDF、正运动学、导入各连杆的 STL，以及装在 link_3 上的相机支架。

由 render_sequence.py 导入，在 Blender 自带的 Python 中运行。URDF 和 STL 来自工作区的 arm_description 包。
关节链按 URDF 中关节出现的顺序（joint_1 到 joint_6），每个关节只有一个转轴。
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix

import render_dataset as rd

DESCRIPTION_DIR = rd.PACKAGE_DIR.parent / "arm_description"
URDF_PATH = DESCRIPTION_DIR / "urdf" / "arm_description.urdf"


def rpy_matrix(roll, pitch, yaw):
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def axis_angle(axis, angle):
    k = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    return np.eye(3) + np.sin(angle) * k + (1.0 - np.cos(angle)) * k @ k


def to_blender(matrix):
    return Matrix([list(row) for row in matrix])


class Arm:
    """机械臂的运动学和 Blender 物体。位姿都是 4×4 numpy 数组，单位 m"""

    def __init__(self, urdf_path=URDF_PATH):
        root = ET.parse(urdf_path).getroot()
        self.root_link = None
        self.joints = []  # (子连杆, 原点 4×4, 转轴)
        for joint in root.iter("joint"):
            origin = joint.find("origin")
            transform = np.eye(4)
            transform[:3, :3] = rpy_matrix(*np.array(origin.get("rpy", "0 0 0").split(), float))
            transform[:3, 3] = np.array(origin.get("xyz", "0 0 0").split(), float)
            axis = np.array(joint.find("axis").get("xyz").split(), float)
            if self.root_link is None:
                self.root_link = joint.find("parent").get("link")
            self.joints.append((joint.find("child").get("link"), transform, axis / np.linalg.norm(axis)))
        # 每个连杆的模型文件：package://arm_description/meshes/<名>.STL，对应包内 meshes/
        self.meshes = {}
        for link in root.iter("link"):
            mesh = link.find("visual/geometry/mesh")
            if mesh is not None:
                self.meshes[link.get("name")] = DESCRIPTION_DIR / "meshes" / Path(mesh.get("filename")).name
        self.objects = {}

    def forward(self, joints, world_base):
        """各连杆在世界坐标系中的位姿"""
        frames = {self.root_link: np.array(world_base, float)}
        transform = frames[self.root_link]
        for (child, origin, axis), angle in zip(self.joints, joints):
            rotation = np.eye(4)
            rotation[:3, :3] = axis_angle(axis, angle)
            transform = transform @ origin @ rotation
            frames[child] = transform
        return frames

    def build(self, material):
        """导入各连杆的 STL（单位 m），所有连杆用同一种材质"""
        for link, path in self.meshes.items():
            bpy.ops.wm.stl_import(filepath=str(path))
            obj = bpy.context.selected_objects[0]
            obj.name = f"arm_{link}"
            obj.data.materials.append(material)
            self.objects[link] = obj

    def pose(self, frames):
        for link, obj in self.objects.items():
            obj.matrix_world = to_blender(frames[link])

    def add_bracket(self, points, link, material):
        """支架：沿 points（连杆系中的折线）的方管，截面 2 cm，作为 link 的子物体随它运动"""
        parent = self.objects[link]
        for k, (a, b) in enumerate(zip(points[:-1], points[1:])):
            a, b = np.asarray(a, float), np.asarray(b, float)
            length = np.linalg.norm(b - a)
            z = (b - a) / length
            x = np.cross([0.0, 1.0, 0.0], z) if abs(z[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
            x /= np.linalg.norm(x)
            local = np.eye(4)
            local[:3, :3] = np.stack([x, np.cross(z, x), z], axis=1)
            local[:3, 3] = (a + b) / 2
            bar = rd.add_box(f"bracket_{k}", (0.02, 0.02, length), material, Matrix.Identity(4))
            bar.parent = parent
            bar.matrix_parent_inverse = Matrix.Identity(4)
            bar.matrix_basis = to_blender(local)
