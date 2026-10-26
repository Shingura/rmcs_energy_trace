// 能量单元视觉 + 抓取运动学仿真：渲染深度 -> 点云配准 -> 生成抓取候选 -> MoveIt 规划 -> 回放
// /joint_states
#include "grasp_candidates.hpp"
#include "joint_state_player.hpp"

#include "geometry/camera.hpp"
#include "geometry/depth_cloud.hpp"
#include "geometry/mesh.hpp"
#include "registration/registration.hpp"
#include "synthetic/depth_renderer.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <format>
#include <map>
#include <memory>
#include <mutex>
#include <numbers>
#include <optional>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <moveit/collision_detection/collision_matrix.hpp>
#include <moveit/move_group_interface/move_group_interface.hpp>
#include <moveit/planning_scene_interface/planning_scene_interface.hpp>
#include <moveit/robot_state/conversions.hpp>
#include <moveit/robot_state/robot_state.hpp>
#include <moveit/robot_trajectory/robot_trajectory.hpp>
#include <moveit/trajectory_processing/time_optimal_trajectory_generation.hpp>
#include <moveit_msgs/msg/attached_collision_object.hpp>
#include <moveit_msgs/msg/collision_object.hpp>
#include <moveit_msgs/msg/planning_scene.hpp>
#include <moveit_msgs/srv/get_planning_scene.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <shape_msgs/msg/mesh.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>
#include <tf2_eigen/tf2_eigen.hpp>
#include <tf2_ros/static_transform_broadcaster.h>
#include <visualization_msgs/msg/marker_array.hpp>

using namespace rmcs_energy_trace;
using namespace rmcs_energy_trace::grasp_sim;
using Isometry = Eigen::Isometry3d;

namespace {

constexpr auto kGroup = "alliance_arm";
constexpr auto kLink = "link_6";
constexpr auto kFrame = "base_link";
constexpr auto kObjectId = "energy_unit";
constexpr auto kTableId = "table";
constexpr auto kHalfLength = 0.075;
constexpr double kRadToDeg = 180. / std::numbers::pi;

template <typename T>
auto param(rclcpp::Node& node, const std::string& name, const T& fallback) -> T {
    if (!node.has_parameter(name))
        node.declare_parameter<T>(name, fallback);
    return node.get_parameter(name).get_value<T>();
}

auto to_msg(const Isometry& pose) -> geometry_msgs::msg::Pose { return tf2::toMsg(pose); }

auto vec3(const std::vector<double>& v) -> Eigen::Vector3d {
    return v.size() >= 3 ? Eigen::Vector3d{v[0], v[1], v[2]} : Eigen::Vector3d::Zero();
}

auto rotation_rpy(double roll, double pitch, double yaw) -> Eigen::Matrix3d {
    return (Eigen::AngleAxisd{yaw, Eigen::Vector3d::UnitZ()}
            * Eigen::AngleAxisd{pitch, Eigen::Vector3d::UnitY()}
            * Eigen::AngleAxisd{roll, Eigen::Vector3d::UnitX()})
        .toRotationMatrix();
}

auto angle_deg(const Eigen::Vector3d& a, const Eigen::Vector3d& b) -> double {
    return std::acos(std::clamp(a.normalized().dot(b.normalized()), -1., 1.)) * kRadToDeg;
}

auto to_mesh_msg(const geometry::Mesh& mesh) -> shape_msgs::msg::Mesh {
    auto message = shape_msgs::msg::Mesh{};
    for (const auto& triangle : mesh.triangles) {
        auto face = shape_msgs::msg::MeshTriangle{};
        for (std::size_t i = 0; i < 3; i++) {
            face.vertex_indices[i] = static_cast<std::uint32_t>(message.vertices.size());
            message.vertices.push_back(
                geometry_msgs::msg::Point()
                    .set__x(triangle[i].x())
                    .set__y(triangle[i].y())
                    .set__z(triangle[i].z()));
        }
        message.triangles.push_back(face);
    }
    return message;
}

auto millis(std::chrono::steady_clock::time_point since) -> double {
    return std::chrono::duration<double, std::milli>{std::chrono::steady_clock::now() - since}
        .count();
}

struct Stage {
    bool ok = false;
    trajectory_msgs::msg::JointTrajectory trajectory;
    std::vector<double> end;
    double ms = 0.;
    double fraction = 0.;
};

struct Summary {
    double center_error_mm = 0.;
    double axis_error_deg = 0.;
    double inlier_ratio = 0.;
    registration::Registration::Timing timing;
    std::size_t tried = 0;
    std::size_t ik_rejected = 0;
    std::optional<std::size_t> chosen;
    double plan_ms[3] = {};
    double final_tcp_error_mm = 0.;
    double final_axis_error_deg = 0.;
};

class GraspSim {
public:
    explicit GraspSim(const rclcpp::Node::SharedPtr& node)
        : node_{node}
        , logger_{node->get_logger()} {}

    // 返回进程退出码：0 成功，1 感知失败，2 无可行抓取
    auto run() -> int {
        if (!load_parameters())
            return 1;
        build_scene();
        if (!perceive())
            return 1;
        // estimated_pose_ 只在 perceive 中写入，之后定时器线程才开始读取
        republish_timer_ =
            node_->create_wall_timer(std::chrono::seconds{1}, [this] { publish_visuals(); });
        publish_visuals();

        wait_for_move_group();
        move_group_ =
            std::make_unique<moveit::planning_interface::MoveGroupInterface>(node_, kGroup);
        move_group_->setPlanningTime(planning_time_);
        move_group_->setMaxVelocityScalingFactor(velocity_scaling_);
        move_group_->setMaxAccelerationScalingFactor(velocity_scaling_);
        move_group_->setGoalPositionTolerance(0.002);
        move_group_->setGoalOrientationTolerance(0.02);
        joint_names_ = move_group_->getJointNames();
        model_ = move_group_->getRobotModel();

        const auto initial = move_group_->getNamedTargetValues("initial");
        auto q0 = std::vector<double>{};
        for (const auto& name : joint_names_)
            q0.push_back(initial.at(name));
        player_ = std::make_unique<JointStatePlayer>(*node_, joint_names_, q0, playback_speed_);
        scene_interface_ = std::make_unique<moveit::planning_interface::PlanningSceneInterface>();
        std::this_thread::sleep_for(std::chrono::seconds{1});

        add_world_objects();
        update_acm(false);
        if (wait_for_viewer_)
            wait_for_viewer();

        const auto code = plan_and_execute(q0);
        report(code == 0);
        if (keep_alive_ && loop_ && code == 0) {
            while (rclcpp::ok()) {
                std::this_thread::sleep_for(std::chrono::duration<double>{loop_pause_});
                if (!rclcpp::ok())
                    break;
                RCLCPP_INFO(logger_, "replaying ...");
                replay(q0);
            }
        }
        if (keep_alive_) {
            while (rclcpp::ok())
                std::this_thread::sleep_for(std::chrono::milliseconds{200});
        }
        return code;
    }

private:
    // ---- 参数与场景 ----
    auto load_parameters() -> bool {
        auto& n = *node_;
        const auto default_mesh = ament_index_cpp::get_package_share_directory("rmcs_energy_trace")
                                + "/meshes/energy_unit.stl";
        auto mesh_path = param<std::string>(n, "mesh_path", "");
        if (mesh_path.empty())
            mesh_path = default_mesh;
        auto mesh = geometry::load_binary_stl(mesh_path);
        if (!mesh.has_value()) {
            RCLCPP_ERROR(logger_, "%s", mesh.error().c_str());
            return false;
        }
        mesh_ = std::move(*mesh);
        if (auto ok = registration_.initialize(mesh_path); !ok.has_value()) {
            RCLCPP_ERROR(logger_, "%s", ok.error().c_str());
            return false;
        }

        // 真值位姿
        const auto rotation = rotation_rpy(
            param(n, "object.roll", 0.), param(n, "object.pitch", 0.), param(n, "object.yaw", 0.));
        true_pose_ = Isometry::Identity();
        true_pose_.linear() = rotation;
        true_pose_.translation() = Eigen::Vector3d{
            param(n, "object.x", 0.40), param(n, "object.y", 0.0), param(n, "object.z", 0.075)};

        // 桌面：物体最低点所在高度
        table_top_ = std::numeric_limits<double>::max();
        for (const auto& triangle : mesh_.triangles) {
            for (const auto& vertex : triangle) {
                const Eigen::Vector3d p = true_pose_ * vertex.cast<double>();
                table_top_ = std::min(table_top_, p.z());
            }
        }
        table_size_ = vec3(param(n, "table.size", std::vector<double>{0.6, 0.8, 0.05}));
        table_center_xy_ =
            Eigen::Vector2d{true_pose_.translation().x(), true_pose_.translation().y()};

        // 相机：位置 + 注视点，光学系 x 右 y 下 z 前
        const Eigen::Vector3d eye =
            vec3(param(n, "camera.position", std::vector<double>{0.0, 0.0, 0.5}));
        // 未给注视点时对准物体中心
        const auto target_param = param(n, "camera.target", std::vector<double>{});
        const Eigen::Vector3d target =
            target_param.size() >= 3 ? vec3(target_param) : true_pose_.translation();
        const Eigen::Vector3d forward = (target - eye).normalized();
        const Eigen::Vector3d right = forward.cross(Eigen::Vector3d::UnitZ()).normalized();
        const Eigen::Vector3d down = forward.cross(right);
        camera_pose_ = Isometry::Identity();
        camera_pose_.linear().col(0) = right;
        camera_pose_.linear().col(1) = down;
        camera_pose_.linear().col(2) = forward;
        camera_pose_.translation() = eye;

        auto& camera = render_config_.camera;
        camera.width = param(n, "camera.width", 640);
        camera.height = param(n, "camera.height", 480);
        camera.fx = param(n, "camera.fx", 600.);
        camera.fy = param(n, "camera.fy", 600.);
        camera.cx = param(n, "camera.cx", camera.width / 2.);
        camera.cy = param(n, "camera.cy", camera.height / 2.);
        render_config_.noise_sigma_mm = static_cast<float>(param(n, "camera.noise_sigma_mm", 1.));
        seed_ = static_cast<std::uint32_t>(param(n, "seed", 2024));

        grasp_config_.tcp_offset =
            vec3(param(n, "grasp.tcp_offset", std::vector<double>{0., 0., 0.}));
        grasp_config_.pre_distance = param(n, "grasp.pre_distance", 0.10);
        grasp_config_.azimuth_samples = param(n, "grasp.azimuth_samples", 24);
        lift_distance_ = param(n, "grasp.lift_distance", 0.10);
        eef_step_ = param(n, "grasp.eef_step", 0.005);
        min_fraction_ = param(n, "grasp.min_fraction", 0.99);
        planning_time_ = param(n, "planning_time", 2.0);
        velocity_scaling_ = param(n, "velocity_scaling", 0.5);
        playback_speed_ = param(n, "playback_speed", 1.0);
        keep_alive_ = param(n, "keep_alive", false);
        wait_for_viewer_ = param(n, "wait_for_viewer", false);
        viewer_delay_ = param(n, "viewer_delay", 3.0);
        loop_ = param(n, "loop", false);
        loop_pause_ = param(n, "loop_pause", 2.0);
        ik_precheck_ = param(n, "grasp.ik_precheck", true);
        allow_link_contact_ = param(n, "grasp.allow_link_contact", true);
        return true;
    }

    void build_scene() {
        marker_pub_ = node_->create_publisher<visualization_msgs::msg::MarkerArray>(
            "/grasp_sim/markers", rclcpp::QoS{1}.transient_local());
        cloud_pub_ = node_->create_publisher<sensor_msgs::msg::PointCloud2>(
            "/grasp_sim/scene_cloud", rclcpp::QoS{1}.transient_local());
        tf_broadcaster_ = std::make_unique<tf2_ros::StaticTransformBroadcaster>(node_);

        auto transform = tf2::eigenToTransform(camera_pose_);
        transform.header.stamp = node_->now();
        transform.header.frame_id = kFrame;
        transform.child_frame_id = kCameraFrame;
        tf_broadcaster_->sendTransform(transform);
    }

    // ---- 感知 ----
    auto perceive() -> bool {
        const Isometry camera_from_base = camera_pose_.inverse();
        const Isometry true_in_camera = camera_from_base * true_pose_;

        auto renderer = std::make_unique<synthetic::DepthRenderer>();
        renderer->initialize(mesh_, render_config_);
        const Eigen::Vector3d normal = camera_from_base.linear() * Eigen::Vector3d::UnitZ();
        const Eigen::Vector3d origin = camera_from_base * Eigen::Vector3d{0., 0., table_top_};

        auto engine = std::mt19937{seed_};
        const auto frame = renderer->render(
            true_in_camera, engine, synthetic::DepthRenderer::Plane{normal, origin});
        RCLCPP_INFO(
            logger_, "rendered depth, roi=(%d,%d,%d,%d)", frame.roi.x, frame.roi.y, frame.roi.width,
            frame.roi.height);

        build_cloud_message(frame.depth);

        const auto result =
            registration_.register_depth(frame.depth, render_config_.camera, frame.roi);
        if (!result.has_value()) {
            RCLCPP_ERROR(logger_, "registration failed or rejected by quality check");
            set_text("registration failed");
            return false;
        }
        estimated_pose_ = camera_pose_ * result->pose;

        summary_.center_error_mm =
            (estimated_pose_.translation() - true_pose_.translation()).norm() * 1e3;
        const Eigen::Vector3d true_axis = true_pose_.linear().col(2);
        const Eigen::Vector3d est_axis = estimated_pose_.linear().col(2);
        // 长轴无方向，两端认反不计入误差
        summary_.axis_error_deg =
            std::min(angle_deg(true_axis, est_axis), angle_deg(true_axis, -est_axis));
        summary_.inlier_ratio = result->inlier_ratio;
        summary_.timing = result->timing;

        set_text(
            std::format(
                "center err {:.2f} mm, axis err {:.2f} deg", summary_.center_error_mm,
                summary_.axis_error_deg));
        RCLCPP_INFO(
            logger_, "registration: center err %.2f mm, axis err %.2f deg, inlier %.2f, %.0f ms",
            summary_.center_error_mm, summary_.axis_error_deg, summary_.inlier_ratio,
            result->timing.total_ms);
        return true;
    }

    // 整幅深度图反投影，裁到桌面范围以便在 RViz 中观察
    void build_cloud_message(const cv::Mat& depth) {
        const auto full = cv::Rect2i{0, 0, depth.cols, depth.rows};
        const auto cloud = geometry::depth_to_cloud(depth, render_config_.camera, full);
        auto cropped = pcl::PointCloud<pcl::PointXYZ>{};
        for (const auto& point : *cloud) {
            const Eigen::Vector3d p = camera_pose_ * point.getVector3fMap().cast<double>();
            if (std::abs(p.x() - table_center_xy_.x()) > table_size_.x() / 2.
                || std::abs(p.y() - table_center_xy_.y()) > table_size_.y() / 2.
                || p.z() < table_top_ - 0.01)
                continue;
            cropped.push_back(point);
        }
        const auto lock = std::scoped_lock{visual_mutex_};
        pcl::toROSMsg(cropped, cloud_message_);
        cloud_message_.header.frame_id = kCameraFrame;
    }

    // ---- RViz 可视化 ----
    auto mesh_marker(
        int id, const std::string& ns, const Isometry& pose, std::array<float, 4> color) const
        -> visualization_msgs::msg::Marker {
        auto marker = visualization_msgs::msg::Marker{};
        marker.header.frame_id = kFrame;
        marker.ns = ns;
        marker.id = id;
        marker.type = visualization_msgs::msg::Marker::TRIANGLE_LIST;
        marker.pose = to_msg(pose);
        marker.scale.x = marker.scale.y = marker.scale.z = 1.;
        marker.color.r = color[0];
        marker.color.g = color[1];
        marker.color.b = color[2];
        marker.color.a = color[3];
        for (const auto& triangle : mesh_.triangles) {
            for (const auto& vertex : triangle) {
                marker.points.push_back(
                    geometry_msgs::msg::Point()
                        .set__x(vertex.x())
                        .set__y(vertex.y())
                        .set__z(vertex.z()));
            }
        }
        return marker;
    }

    void set_text(const std::string& text) {
        const auto lock = std::scoped_lock{visual_mutex_};
        text_ = text;
    }

    void publish_visuals() {
        auto array = visualization_msgs::msg::MarkerArray{};
        array.markers.push_back(mesh_marker(0, "true", true_pose_, {0.1f, 0.8f, 0.2f, 0.6f}));
        array.markers.push_back(
            mesh_marker(1, "estimated", estimated_pose_, {1.f, 0.55f, 0.f, 0.6f}));

        auto text = visualization_msgs::msg::Marker{};
        text.header.frame_id = kFrame;
        text.ns = "text";
        text.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
        text.pose.position.x = true_pose_.translation().x();
        text.pose.position.y = true_pose_.translation().y();
        text.pose.position.z = true_pose_.translation().z() + 0.2;
        text.pose.orientation.w = 1.;
        text.scale.z = 0.03;
        text.color.r = text.color.g = text.color.b = text.color.a = 1.;
        {
            const auto lock = std::scoped_lock{visual_mutex_};
            text.text = text_;
            if (!cloud_message_.data.empty()) {
                cloud_message_.header.stamp = node_->now();
                cloud_pub_->publish(cloud_message_);
            }
        }
        array.markers.push_back(text);
        marker_pub_->publish(array);
    }

    // ---- MoveIt 场景 ----
    void wait_for_move_group() {
        auto client = node_->create_client<moveit_msgs::srv::GetPlanningScene>("/get_planning_"
                                                                               "scene");
        while (rclcpp::ok() && !client->wait_for_service(std::chrono::seconds{1})) {
            RCLCPP_INFO(logger_, "waiting for move_group ...");
        }
        acm_client_ = client;
    }

    auto object_message(const Isometry& pose) const -> moveit_msgs::msg::CollisionObject {
        auto object = moveit_msgs::msg::CollisionObject{};
        object.header.frame_id = kFrame;
        object.id = kObjectId;
        object.meshes.push_back(to_mesh_msg(mesh_));
        object.mesh_poses.push_back(to_msg(pose));
        object.operation = moveit_msgs::msg::CollisionObject::ADD;
        return object;
    }

    void add_world_objects() {
        auto table = moveit_msgs::msg::CollisionObject{};
        table.header.frame_id = kFrame;
        table.id = kTableId;
        auto box = shape_msgs::msg::SolidPrimitive{};
        box.type = shape_msgs::msg::SolidPrimitive::BOX;
        box.dimensions = {table_size_.x(), table_size_.y(), table_size_.z()};
        table.primitives.push_back(box);
        auto pose = Isometry::Identity();
        pose.translation() = Eigen::Vector3d{
            table_center_xy_.x(), table_center_xy_.y(), table_top_ - table_size_.z() / 2.};
        table.primitive_poses.push_back(to_msg(pose));
        table.operation = moveit_msgs::msg::CollisionObject::ADD;
        scene_interface_->applyCollisionObjects({table, object_message(estimated_pose_)});
    }

    void remove_object() {
        auto object = moveit_msgs::msg::CollisionObject{};
        object.header.frame_id = kFrame;
        object.id = kObjectId;
        object.operation = moveit_msgs::msg::CollisionObject::REMOVE;
        scene_interface_->applyCollisionObject(object);
    }

    // 物体与桌面始终允许接触；allow_links 时额外允许夹爪 link_6 与物体接触（接近/抬升阶段）
    // link_5 是手腕，不允许接触，以免掩盖真实碰撞
    void update_acm(bool allow_links) {
        auto request = std::make_shared<moveit_msgs::srv::GetPlanningScene::Request>();
        request->components.components =
            moveit_msgs::msg::PlanningSceneComponents::ALLOWED_COLLISION_MATRIX;
        auto future = acm_client_->async_send_request(request);
        if (future.wait_for(std::chrono::seconds{5}) != std::future_status::ready) {
            RCLCPP_WARN(logger_, "get_planning_scene timed out");
            return;
        }
        auto acm = collision_detection::AllowedCollisionMatrix{
            future.get()->scene.allowed_collision_matrix};
        acm.setEntry(kObjectId, kTableId, true);
        acm.setEntry(kObjectId, "link_5", false);
        acm.setEntry(kObjectId, kLink, allow_links);

        auto diff = moveit_msgs::msg::PlanningScene{};
        diff.is_diff = true;
        acm.getMessage(diff.allowed_collision_matrix);
        scene_interface_->applyPlanningScene(diff);
    }

    // ---- 规划 ----
    auto state_message(const std::vector<double>& q, bool attached) const
        -> moveit_msgs::msg::RobotState {
        auto state = moveit_msgs::msg::RobotState{};
        state.joint_state.name = joint_names_;
        state.joint_state.position = q;
        if (attached) {
            // 物体在抓取时相对 link_6 固定：以抓取位姿换算
            auto aco = moveit_msgs::msg::AttachedCollisionObject{};
            aco.link_name = kLink;
            aco.touch_links = {kLink};
            aco.object = object_message(grasp_relative_);
            aco.object.header.frame_id = kLink;
            state.attached_collision_objects.push_back(aco);
        }
        return state;
    }

    auto plan_to_pose(const std::vector<double>& start, const Isometry& target) -> Stage {
        auto stage = Stage();
        const auto begin = std::chrono::steady_clock::now();
        move_group_->setStartState(state_message(start, false));
        move_group_->setPoseTarget(to_msg(target), kLink);
        auto plan = moveit::planning_interface::MoveGroupInterface::Plan();
        const auto code = move_group_->plan(plan);
        move_group_->clearPoseTargets();
        stage.ms = millis(begin);
        if (code != moveit::core::MoveItErrorCode::SUCCESS)
            return stage;
        stage.trajectory = plan.trajectory.joint_trajectory;
        stage.end = stage.trajectory.points.back().positions;
        stage.ok = true;
        return stage;
    }

    auto plan_cartesian(const std::vector<double>& start, bool attached, const Isometry& target)
        -> Stage {
        auto stage = Stage();
        const auto begin = std::chrono::steady_clock::now();
        const auto state = state_message(start, attached);
        move_group_->setStartState(state);

        auto trajectory = moveit_msgs::msg::RobotTrajectory{};
        auto error = moveit_msgs::msg::MoveItErrorCodes{};
        stage.fraction = move_group_->computeCartesianPath(
            {to_msg(target)}, eef_step_, trajectory, true, &error);
        if (stage.fraction >= min_fraction_ && !trajectory.joint_trajectory.points.empty()) {
            if (!has_timing(trajectory.joint_trajectory))
                time_parameterize(state, trajectory);
            stage.trajectory = trajectory.joint_trajectory;
            stage.end = stage.trajectory.points.back().positions;
            stage.ok = stage.trajectory.points.size() > 1;
        }
        stage.ms = millis(begin);
        return stage;
    }

    static auto has_timing(const trajectory_msgs::msg::JointTrajectory& trajectory) -> bool {
        const auto& last = trajectory.points.back().time_from_start;
        return last.sec > 0 || last.nanosec > 0;
    }

    void time_parameterize(
        const moveit_msgs::msg::RobotState& start, moveit_msgs::msg::RobotTrajectory& trajectory) {
        auto state = moveit::core::RobotState{model_};
        moveit::core::robotStateMsgToRobotState(start, state);
        auto robot_trajectory = robot_trajectory::RobotTrajectory{model_, kGroup};
        robot_trajectory.setRobotTrajectoryMsg(state, trajectory);
        auto totg = trajectory_processing::TimeOptimalTrajectoryGeneration{};
        if (totg.computeTimeStamps(robot_trajectory, velocity_scaling_, velocity_scaling_)) {
            robot_trajectory.getRobotTrajectoryMsg(trajectory);
        }
    }

    // 仅检查抓取与预抓取位姿的 IK 可达性（不含碰撞），快速剔除不可达候选
    auto ik_reachable(const std::vector<double>& seed, const Isometry& pose) const -> bool {
        auto state = moveit::core::RobotState{model_};
        const auto* group = model_->getJointModelGroup(kGroup);
        state.setJointGroupPositions(group, seed);
        return state.setFromIK(group, pose, kLink, 0.05);
    }

    auto end_effector(const std::vector<double>& q) const -> Isometry {
        auto state = moveit::core::RobotState{model_};
        state.setJointGroupPositions(model_->getJointModelGroup(kGroup), q);
        state.update();
        return state.getGlobalLinkTransform(kLink);
    }

    auto plan_and_execute(const std::vector<double>& q0) -> int {
        const auto candidates = generate_grasps(
            estimated_pose_.translation(), estimated_pose_.linear().col(2), Eigen::Vector3d::Zero(),
            grasp_config_);
        RCLCPP_INFO(logger_, "%zu grasp candidates", candidates.size());

        for (std::size_t i = 0; i < candidates.size() && rclcpp::ok(); i++) {
            const auto& candidate = candidates[i];
            const auto tag = std::format(
                "#{} (x6 {}axis, az {:+.0f} deg)", i, candidate.sign > 0 ? "+" : "-",
                candidate.azimuth * kRadToDeg);
            summary_.tried++;

            if (ik_precheck_
                && (!ik_reachable(q0, candidate.grasp) || !ik_reachable(q0, candidate.pre_grasp))) {
                summary_.ik_rejected++;
                RCLCPP_INFO(logger_, "candidate %s: IK unreachable", tag.c_str());
                continue;
            }

            update_acm(false);
            const auto pre = plan_to_pose(q0, candidate.pre_grasp);
            if (!pre.ok) {
                RCLCPP_INFO(
                    logger_, "candidate %s: pre-grasp plan failed (%.0f ms)", tag.c_str(), pre.ms);
                continue;
            }

            update_acm(allow_link_contact_);
            const auto approach = plan_cartesian(pre.end, false, candidate.grasp);
            if (!approach.ok) {
                RCLCPP_INFO(
                    logger_, "candidate %s: approach failed (%.0f%%, %.0f ms)", tag.c_str(),
                    approach.fraction * 100., approach.ms);
                continue;
            }

            // 物体相对 link_6 的位姿在抓取后固定；规划抬升时世界中的物体已转为附着体
            const Isometry link6_at_grasp = end_effector(approach.end);
            grasp_relative_ = link6_at_grasp.inverse() * estimated_pose_;
            auto lift_target = link6_at_grasp;
            lift_target.translation() += lift_distance_ * Eigen::Vector3d::UnitZ();

            remove_object();
            const auto lift = plan_cartesian(approach.end, true, lift_target);
            if (!lift.ok) {
                RCLCPP_INFO(
                    logger_, "candidate %s: lift failed (%.0f%%, %.0f ms)", tag.c_str(),
                    lift.fraction * 100., lift.ms);
                scene_interface_->applyCollisionObject(object_message(estimated_pose_));
                continue;
            }

            RCLCPP_INFO(
                logger_, "candidate %s: success (pre %.0f ms, approach %.0f ms, lift %.0f ms)",
                tag.c_str(), pre.ms, approach.ms, lift.ms);
            for (const auto* stage : {&pre, &approach, &lift}) {
                RCLCPP_INFO(
                    logger_, "  stage: %zu points, duration %.2f s",
                    stage->trajectory.points.size(),
                    static_cast<double>(stage->trajectory.points.back().time_from_start.sec)
                        + 1e-9 * stage->trajectory.points.back().time_from_start.nanosec);
            }
            summary_.chosen = i;
            summary_.plan_ms[0] = pre.ms;
            summary_.plan_ms[1] = approach.ms;
            summary_.plan_ms[2] = lift.ms;

            const Isometry tcp = link6_at_grasp * Eigen::Translation3d{grasp_config_.tcp_offset};
            summary_.final_tcp_error_mm =
                (tcp.translation() - true_pose_.translation()).norm() * 1e3;
            summary_.final_axis_error_deg = std::min(
                angle_deg(link6_at_grasp.linear().col(0), true_pose_.linear().col(2)),
                angle_deg(link6_at_grasp.linear().col(0), -true_pose_.linear().col(2)));

            stages_ = {pre, approach, lift};
            execute(pre, approach, lift);
            return 0;
        }
        return 2;
    }

    // 复位到初始关节位置，把能量单元放回估计位姿，再完整播放一次
    void replay(const std::vector<double>& q0) {
        auto detach = moveit_msgs::msg::AttachedCollisionObject{};
        detach.link_name = kLink;
        detach.object.id = kObjectId;
        detach.object.header.frame_id = kLink;
        detach.object.operation = moveit_msgs::msg::CollisionObject::REMOVE;
        scene_interface_->applyAttachedCollisionObject(detach);
        player_->reset(q0);
        set_text("replay");
        publish_visuals();
        std::this_thread::sleep_for(std::chrono::seconds{1});
        execute(stages_[0], stages_[1], stages_[2]);
    }

    // RViz 订阅标记话题后再等 viewer_delay 秒，让它加载完机器人模型，避免错过运动过程
    void wait_for_viewer() {
        RCLCPP_INFO(logger_, "waiting for RViz ...");
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{60};
        while (rclcpp::ok() && marker_pub_->get_subscription_count() == 0) {
            if (std::chrono::steady_clock::now() > deadline) {
                RCLCPP_WARN(logger_, "no RViz after 60 s, continue without it");
                return;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds{200});
        }
        std::this_thread::sleep_for(std::chrono::duration<double>{viewer_delay_});
    }

    // 依次播放三段轨迹。ROS 关闭（例如 Ctrl-C）时停止
    void execute(const Stage& pre, const Stage& approach, const Stage& lift) {
        scene_interface_->applyCollisionObject(object_message(estimated_pose_));
        RCLCPP_INFO(logger_, "executing pre-grasp ...");
        if (!player_->play(pre.trajectory))
            return;
        RCLCPP_INFO(logger_, "executing approach ...");
        if (!player_->play(approach.trajectory))
            return;

        // 附着到 link_6：先移出世界，再以附着体加入
        remove_object();
        auto aco = state_message(approach.end, true).attached_collision_objects.front();
        scene_interface_->applyAttachedCollisionObject(aco);
        RCLCPP_INFO(logger_, "attached, executing lift ...");
        if (!player_->play(lift.trajectory))
            return;
        set_text(
            std::format(
                "grasped: center err {:.2f} mm, axis err {:.2f} "
                "deg\nTCP-to-true-center {:.2f} mm",
                summary_.center_error_mm, summary_.axis_error_deg, summary_.final_tcp_error_mm));
        publish_visuals();
    }

    void report(bool success) const {
        const auto& s = summary_;
        RCLCPP_INFO(logger_, "================ grasp sim summary ================");
        RCLCPP_INFO(
            logger_, "true centre (base_link): %.4f %.4f %.4f", true_pose_.translation().x(),
            true_pose_.translation().y(), true_pose_.translation().z());
        RCLCPP_INFO(
            logger_, "registration: centre error %.2f mm, axis error %.2f deg, inlier %.2f",
            s.center_error_mm, s.axis_error_deg, s.inlier_ratio);
        RCLCPP_INFO(
            logger_,
            "registration timing (ms): pre %.0f normal %.0f coarse %.0f icp %.0f quality %.0f "
            "total %.0f",
            s.timing.preprocess_ms, s.timing.normal_ms, s.timing.coarse_ms, s.timing.icp_ms,
            s.timing.quality_ms, s.timing.total_ms);
        RCLCPP_INFO(logger_, "candidates tried %zu (IK rejected %zu)", s.tried, s.ik_rejected);
        if (success) {
            RCLCPP_INFO(
                logger_,
                "chosen candidate #%zu; planning ms: pre-grasp %.0f, approach %.0f, lift %.0f",
                *s.chosen, s.plan_ms[0], s.plan_ms[1], s.plan_ms[2]);
            RCLCPP_INFO(
                logger_,
                "final TCP-to-true-centre error %.2f mm, gripper-x to true-axis error %.2f deg",
                s.final_tcp_error_mm, s.final_axis_error_deg);
        } else {
            RCLCPP_WARN(logger_, "no feasible grasp found");
        }
    }

    static constexpr auto kCameraFrame = "camera_optical_frame";

    rclcpp::Node::SharedPtr node_;
    rclcpp::Logger logger_;

    geometry::Mesh mesh_;
    registration::Registration registration_;
    synthetic::DepthRenderer::Config render_config_;
    std::uint32_t seed_ = 2024;

    Isometry true_pose_ = Isometry::Identity();
    Isometry estimated_pose_ = Isometry::Identity();
    Isometry camera_pose_ = Isometry::Identity();
    Isometry grasp_relative_ = Isometry::Identity();
    double table_top_ = 0.;
    Eigen::Vector3d table_size_ = Eigen::Vector3d::Zero();
    Eigen::Vector2d table_center_xy_ = Eigen::Vector2d::Zero();

    GraspConfig grasp_config_;
    double lift_distance_ = 0.10;
    double eef_step_ = 0.005;
    double min_fraction_ = 0.99;
    double planning_time_ = 2.0;
    double velocity_scaling_ = 0.5;
    double playback_speed_ = 1.0;
    bool keep_alive_ = false;
    bool wait_for_viewer_ = false;
    double viewer_delay_ = 3.0;
    bool loop_ = false;
    double loop_pause_ = 2.0;
    std::array<Stage, 3> stages_; // 预抓取、接近、抬起，供重播使用
    bool ik_precheck_ = true;
    bool allow_link_contact_ = true;

    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_pub_;
    std::unique_ptr<tf2_ros::StaticTransformBroadcaster> tf_broadcaster_;
    rclcpp::TimerBase::SharedPtr republish_timer_;
    rclcpp::Client<moveit_msgs::srv::GetPlanningScene>::SharedPtr acm_client_;
    std::mutex visual_mutex_;
    sensor_msgs::msg::PointCloud2 cloud_message_;
    std::string text_;

    std::unique_ptr<moveit::planning_interface::MoveGroupInterface> move_group_;
    std::unique_ptr<moveit::planning_interface::PlanningSceneInterface> scene_interface_;
    moveit::core::RobotModelConstPtr model_;
    std::vector<std::string> joint_names_;
    std::unique_ptr<JointStatePlayer> player_;

    Summary summary_;
};

} // namespace

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>(
        "grasp_sim", rclcpp::NodeOptions{}.automatically_declare_parameters_from_overrides(true));

    auto executor = rclcpp::executors::MultiThreadedExecutor{};
    executor.add_node(node);
    auto spin_thread = std::thread{[&] { executor.spin(); }};

    auto sim = GraspSim{node};
    const auto code = sim.run();

    executor.cancel();
    spin_thread.join();
    rclcpp::shutdown();
    return code;
}
