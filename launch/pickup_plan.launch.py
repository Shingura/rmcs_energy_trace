import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, OpaqueFunction, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def launch_setup(context):
    share = get_package_share_directory("rmcs_energy_trace")
    moveit_share = get_package_share_directory("arm_moveit_config")
    use_rviz = LaunchConfiguration("use_rviz").perform(context).lower() in ("1", "true")

    moveit_config = (
        MoveItConfigsBuilder("arm_description", package_name="arm_moveit_config")
        .robot_description(file_path="config/arm_description.urdf.xacro")
        .robot_description_semantic(file_path="config/arm_description.srdf")
        .robot_description_kinematics(file_path="config/kinematics.yaml")
        .joint_limits(file_path="config/joint_limits.yaml")
        .planning_pipelines(pipelines=["ompl"], default_planning_pipeline="ompl")
        .planning_scene_monitor(
            publish_robot_description=True, publish_robot_description_semantic=True
        )
        .trajectory_execution(file_path="config/moveit_controllers.yaml")
        .to_moveit_configs()
    )

    # OMPL 的结果由规划工具自己做 TOTG 时间参数化：去掉 move_group 自带的时间参数化，
    # 否则它按默认的路径容差削角，可能让路径穿过障碍
    move_group_parameters = moveit_config.to_dict()
    move_group_parameters["ompl"]["response_adapters"] = [
        adapter
        for adapter in move_group_parameters["ompl"]["response_adapters"]
        if "AddTimeOptimalParameterization" not in adapter
    ]
    move_group = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            move_group_parameters,
            {"robot_description_planning.default_robot_padding": 0.0},
        ],
    )
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="log",
        parameters=[moveit_config.robot_description],
    )
    static_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        output="log",
        arguments=["--frame-id", "world", "--child-frame-id", "base_link"],
    )

    # 节点内的 MoveIt 机器人模型需要 kinematics 配置才能做 IK，需要关节速度、加速度限制才能做 TOTG
    parameters = [
        os.path.join(share, "config", "pickup_plan.yaml"),
        moveit_config.robot_description_kinematics,
        moveit_config.joint_limits,
    ]
    output_dir = LaunchConfiguration("output_dir").perform(context)
    if output_dir:
        parameters.append({"output_dir": output_dir})

    planner = Node(
        package="rmcs_energy_trace",
        executable="pickup_plan",
        name="pickup_plan",
        output="screen",
        parameters=parameters,
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", os.path.join(moveit_share, "config", "moveit.rviz")],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.planning_pipelines,
            moveit_config.joint_limits,
        ],
    )

    # 规划工具结束后关闭整个 launch
    shutdown = RegisterEventHandler(
        OnProcessExit(target_action=planner, on_exit=[EmitEvent(event=Shutdown())])
    )
    return [static_tf, robot_state_publisher, move_group, planner, shutdown] + (
        [rviz] if use_rviz else []
    )


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("use_rviz", default_value="false"),
            # 输出目录，留空时用 config/pickup_plan.yaml 中的值
            DeclareLaunchArgument("output_dir", default_value=""),
            OpaqueFunction(function=launch_setup),
        ]
    )
