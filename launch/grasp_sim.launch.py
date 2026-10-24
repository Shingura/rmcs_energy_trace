import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    GroupAction,
    IncludeLaunchDescription,
    OpaqueFunction,
    RegisterEventHandler,
)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder

# 可通过启动参数覆盖的场景参数 (未提供则使用 yaml 中的值)
OVERRIDES = {
    "object_x": ("object.x", float),
    "object_y": ("object.y", float),
    "object_z": ("object.z", float),
    "object_roll": ("object.roll", float),
    "object_pitch": ("object.pitch", float),
    "object_yaw": ("object.yaw", float),
    "playback_speed": ("playback_speed", float),
    "allow_link_contact": ("grasp.allow_link_contact", lambda v: v.lower() in ("1", "true")),
    "keep_alive": ("keep_alive", lambda v: v.lower() in ("1", "true")),
    "loop": ("loop", lambda v: v.lower() in ("1", "true")),
}


def launch_setup(context):
    share = get_package_share_directory("rmcs_energy_trace")
    moveit_share = get_package_share_directory("arm_moveit_config")
    # 在包含 demo.launch.py 之前读出，否则会被它的 use_rviz:=false 覆盖
    use_rviz = LaunchConfiguration("use_rviz").perform(context).lower() in ("1", "true")

    # 仅复用 move_group / robot_state_publisher / world->base_link TF，
    # 不启动任何 /joint_states 发布者 (由仿真节点独占)。
    # scoped=True：demo.launch.py 的参数不泄漏到本文件
    moveit = GroupAction(
        [
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(moveit_share, "launch", "demo.launch.py")
                ),
                launch_arguments={
                    "use_rviz": "false",
                }.items(),
            )
        ],
        scoped=True,
    )

    moveit_config = (
        MoveItConfigsBuilder("arm_description", package_name="arm_moveit_config")
        .robot_description(file_path="config/arm_description.urdf.xacro")
        .robot_description_semantic(file_path="config/arm_description.srdf")
        .robot_description_kinematics(file_path="config/kinematics.yaml")
        .joint_limits(file_path="config/joint_limits.yaml")
        .planning_pipelines(pipelines=["ompl"], default_planning_pipeline="ompl")
        .to_moveit_configs()
    )

    # 节点内的 MoveIt 机器人模型需要 kinematics 配置才能做 IK
    parameters = [
        os.path.join(share, "config", "grasp_sim.yaml"),
        moveit_config.robot_description_kinematics,
    ]
    overrides = {}
    for argument, (name, convert) in OVERRIDES.items():
        value = LaunchConfiguration(argument).perform(context)
        if value != "":
            overrides[name] = convert(value)
    # 使用 RViz 时等待窗口就绪再运动，结束后保持打开（按 Ctrl+C 退出）
    if use_rviz:
        overrides["wait_for_viewer"] = True
        overrides.setdefault("keep_alive", True)
        overrides.setdefault("loop", True)
    if overrides:
        parameters.append(overrides)

    sim = Node(
        package="rmcs_energy_trace",
        executable="grasp_sim",
        name="grasp_sim",
        output="screen",
        parameters=parameters,
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", LaunchConfiguration("rviz_config")],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.planning_pipelines,
            moveit_config.joint_limits,
        ],
    )

    # 仿真节点结束后关闭整个 launch (keep_alive 时节点不会退出)
    shutdown = RegisterEventHandler(
        OnProcessExit(target_action=sim, on_exit=[EmitEvent(event=Shutdown())])
    )
    return [moveit, sim, shutdown] + ([rviz] if use_rviz else [])


def generate_launch_description():
    share = get_package_share_directory("rmcs_energy_trace")
    arguments = [
        DeclareLaunchArgument("use_rviz", default_value="true"),
        DeclareLaunchArgument(
            "rviz_config", default_value=os.path.join(share, "rviz", "grasp_sim.rviz")
        ),
    ]
    arguments += [DeclareLaunchArgument(name, default_value="") for name in OVERRIDES]
    return LaunchDescription(arguments + [OpaqueFunction(function=launch_setup)])
