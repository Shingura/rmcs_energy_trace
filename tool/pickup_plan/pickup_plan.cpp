// 工程机械臂取矿规划：在中心块场景里用 MoveIt 规划 观察 -> 接近 -> 抓取 -> 拉出 的整套动作，
// 按固定帧率导出规划文件（JSON），供 scripts/synth/render_sequence.py --plan 渲染 RGB-D 序列。
// 规划文件的格式见 render_sequence.py 开头的说明。
#include "grasp_candidates.hpp"

#include "geometry/mesh.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <format>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <numbers>
#include <optional>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <moveit/collision_detection/collision_matrix.hpp>
#include <moveit/move_group_interface/move_group_interface.hpp>
#include <moveit/planning_scene/planning_scene.hpp>
#include <moveit/planning_scene_interface/planning_scene_interface.hpp>
#include <moveit/robot_state/conversions.hpp>
#include <moveit/robot_state/robot_state.hpp>
#include <moveit/robot_trajectory/robot_trajectory.hpp>
#include <moveit/trajectory_processing/time_optimal_trajectory_generation.hpp>
#include <moveit_msgs/msg/attached_collision_object.hpp>
#include <moveit_msgs/msg/collision_object.hpp>
#include <moveit_msgs/msg/planning_scene.hpp>
#include <moveit_msgs/srv/get_planning_scene.hpp>
#include <rclcpp/rclcpp.hpp>
#include <shape_msgs/msg/mesh.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>
#include <tf2_eigen/tf2_eigen.hpp>

using namespace rmcs_energy_trace;
using Isometry = Eigen::Isometry3d;
using Joints = std::vector<double>;
using Vector3 = Eigen::Vector3d;

namespace {

constexpr auto kGroup = "alliance_arm";
constexpr auto kLink = "link_6";
constexpr auto kSightId = "sight";
constexpr auto kFrame = "base_link";
constexpr double kRadToDeg = 180. / std::numbers::pi;
constexpr std::size_t kSlots = 6;

template <typename T>
auto param(rclcpp::Node& node, const std::string& name, const T& fallback) -> T {
    if (!node.has_parameter(name))
        node.declare_parameter<T>(name, fallback);
    return node.get_parameter(name).get_value<T>();
}

auto vec3(const std::vector<double>& v) -> Vector3 {
    return v.size() >= 3 ? Vector3{v[0], v[1], v[2]} : Vector3::Zero();
}

auto to_msg(const Isometry& pose) -> geometry_msgs::msg::Pose { return tf2::toMsg(pose); }

auto millis(std::chrono::steady_clock::time_point since) -> double {
    return std::chrono::duration<double, std::milli>{std::chrono::steady_clock::now() - since}
        .count();
}

auto yaw_pose(double x, double y, double z, double yaw) -> Isometry {
    auto pose = Isometry::Identity();
    pose.linear() = Eigen::AngleAxisd{yaw, Vector3::UnitZ()}.toRotationMatrix();
    pose.translation() = Vector3{x, y, z};
    return pose;
}

auto rotation_angle(const Eigen::Matrix3d& rotation) -> double {
    return Eigen::AngleAxisd{rotation}.angle();
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

auto matrix_json(const Isometry& pose) -> std::string {
    auto text = std::string{"["};
    for (int row = 0; row < 4; row++) {
        text += row == 0 ? "[" : ", [";
        for (int col = 0; col < 4; col++)
            text += std::format("{}{:.9g}", col == 0 ? "" : ", ", pose.matrix()(row, col));
        text += "]";
    }
    return text + "]";
}

struct Config {
    std::vector<std::int64_t> seeds;
    std::vector<std::int64_t> target_slots;
    double slot_probability = 0.7;
    std::string output_dir;
    std::string mesh_path;

    double fps = 30.;
    double hold_time = 2.;
    double grasp_time = 0.3;
    double velocity_scaling = 0.3;
    double acceleration_scaling = 0.3;
    double planning_time = 5.;
    int planning_attempts = 5;
    double path_tolerance = 0.1;

    Vector3 hub_center{0., 0., 0.7};
    double hub_apothem = 0.095;
    double hub_thickness = 0.102;
    double unit_offset = 0.15;

    double base_height = 0.4;
    std::vector<double> q4_candidates{0., 1.5707963, -1.5707963, 3.1415926};
    std::vector<double> q5_candidates{-1.57, -1.2, 1.2, 1.57};
    double sight_radius = 0.04;
    double sight_shorten = 0.08;
    double aim_horizontal = 0.03;
    double aim_vertical = 0.02;
    double distance_min = 0.3;
    double distance_max = 1.1;
    double distance_step = 0.05;

    grasp_sim::GraspConfig grasp;
    double extract_distance = 0.12;
    double eef_step = 0.005;
    double min_fraction = 0.999;
    double max_approach_angle = 60.;
    double ik_timeout = 0.02;
    double jump_threshold = 0.5;
    int solutions_per_distance = 3;

    std::string camera_link = "link_3";
    Vector3 camera_position{0.13, 0.03, -0.10};
    Vector3 camera_target{0.03, 0.75, 0.035};
    Vector3 camera_down{-1., 0., 0.};
    int camera_width = 1280;
    int camera_height = 720;
    double camera_hfov = 69.;
    std::vector<double> camera_bracket;
};

// 一份规划要解决的问题：安装位上有没有单元、目标和对准点
struct Task {
    int seed = 0;
    int slot = 0;
    std::array<bool, kSlots> present{};
    std::array<Isometry, kSlots> unit_pose; // 世界系，z 轴指向封闭端
    Vector3 aim = Vector3::Zero();          // 世界系，光轴要穿过的点

    auto target() const -> const Isometry& { return unit_pose[static_cast<std::size_t>(slot)]; }
    auto axis() const -> Vector3 { return target().linear().col(2); }
};

// 底盘放置：底座在目标正前方（-y 一侧）distance 处，朝向 yaw
struct Layout {
    double distance = 0.;
    double yaw = 0.;
    Isometry world_from_base = Isometry::Identity();
    Joints observe;
    Joints pre;
    Joints grasp_joints;
    grasp_sim::GraspCandidate candidate; // base_link 系
};

// 一段带时间的关节轨迹
struct Timed {
    std::vector<double> times;
    std::vector<Joints> points;
};

struct Segment {
    std::string phase;
    bool attached = false;
    double start = 0.;
    Timed path;
    auto end() const -> double { return start + path.times.back(); }
};

struct Frame {
    Joints joints;
    std::string phase;
    bool attached = false;
};

class PickupPlanner {
public:
    explicit PickupPlanner(const rclcpp::Node::SharedPtr& node)
        : node_{node}
        , logger_{node->get_logger()} {}

    // 返回进程退出码：0 全部成功，2 有规划失败
    auto run() -> int {
        if (!load_parameters())
            return 1;
        wait_for_move_group();
        move_group_ =
            std::make_unique<moveit::planning_interface::MoveGroupInterface>(node_, kGroup);
        move_group_->setPlanningTime(config_.planning_time);
        move_group_->setNumPlanningAttempts(config_.planning_attempts);
        move_group_->setMaxVelocityScalingFactor(config_.velocity_scaling);
        move_group_->setMaxAccelerationScalingFactor(config_.acceleration_scaling);
        move_group_->setGoalJointTolerance(1e-4);
        joint_names_ = move_group_->getJointNames();
        model_ = move_group_->getRobotModel();
        group_ = model_->getJointModelGroup(kGroup);
        state_ = std::make_unique<moveit::core::RobotState>(model_);
        scene_ = std::make_shared<planning_scene::PlanningScene>(model_);
        scene_interface_ = std::make_unique<moveit::planning_interface::PlanningSceneInterface>();
        std::this_thread::sleep_for(std::chrono::seconds{1});

        auto failed = 0;
        for (std::size_t i = 0; i < config_.seeds.size() && rclcpp::ok(); i++) {
            const auto slot = static_cast<int>(config_.target_slots[i]);
            const auto seed = static_cast<int>(config_.seeds[i]);
            if (!make_plan(make_task(seed, slot)))
                failed++;
        }
        return failed == 0 ? 0 : 2;
    }

private:
    // ---- 参数 ----
    auto load_parameters() -> bool {
        auto& n = *node_;
        auto& c = config_;
        const auto share = ament_index_cpp::get_package_share_directory("rmcs_energy_trace");
        c.mesh_path = param<std::string>(n, "mesh_path", "");
        if (c.mesh_path.empty())
            c.mesh_path = share + "/meshes/energy_unit.stl";
        auto mesh = geometry::load_binary_stl(c.mesh_path);
        if (!mesh.has_value()) {
            RCLCPP_ERROR(logger_, "%s", mesh.error().c_str());
            return false;
        }
        unit_mesh_ = to_mesh_msg(*mesh);

        c.seeds = param(n, "seeds", std::vector<std::int64_t>{0, 1, 2});
        c.target_slots = param(n, "target_slots", std::vector<std::int64_t>{1, 0, 4});
        if (c.seeds.size() != c.target_slots.size()) {
            RCLCPP_ERROR(logger_, "seeds and target_slots must have the same length");
            return false;
        }
        for (auto slot : c.target_slots) {
            if (slot < 0 || slot >= static_cast<std::int64_t>(kSlots)) {
                RCLCPP_ERROR(logger_, "target slot %ld out of range", slot);
                return false;
            }
        }
        c.slot_probability = param(n, "slot_probability", 0.7);
        c.output_dir = param<std::string>(n, "output_dir", "");
        if (c.output_dir.empty())
            c.output_dir = std::string{RMCS_ENERGY_TRACE_SOURCE_DIR} + "/dataset/synthetic/plans";

        c.fps = param(n, "fps", 30.);
        c.hold_time = param(n, "hold_time", 2.0);
        c.grasp_time = param(n, "grasp_time", 0.3);
        c.velocity_scaling = param(n, "velocity_scaling", 0.3);
        c.acceleration_scaling = param(n, "acceleration_scaling", 0.3);
        c.planning_time = param(n, "planning_time", 5.0);
        c.planning_attempts = param(n, "planning_attempts", 5);
        c.path_tolerance = param(n, "path_tolerance", 0.1);

        c.hub_center = vec3(param(n, "hub.center", std::vector<double>{0., 0., 0.7}));
        c.hub_apothem = param(n, "hub.apothem", 0.095);
        c.hub_thickness = param(n, "hub.thickness", 0.102);
        c.unit_offset = param(n, "hub.unit_offset", 0.15);

        c.base_height = param(n, "base.height", 0.4);
        c.q4_candidates = param(n, "observe.q4_candidates", c.q4_candidates);
        c.q5_candidates = param(n, "observe.q5_candidates", c.q5_candidates);
        c.sight_radius = param(n, "observe.sight_radius", 0.04);
        c.sight_shorten = param(n, "observe.sight_shorten", 0.08);
        c.aim_horizontal = param(n, "observe.aim_horizontal", 0.03);
        c.aim_vertical = param(n, "observe.aim_vertical", 0.02);
        c.distance_min = param(n, "distance.min", 0.3);
        c.distance_max = param(n, "distance.max", 1.1);
        c.distance_step = param(n, "distance.step", 0.05);

        c.grasp.tcp_offset = vec3(param(n, "grasp.tcp_offset", std::vector<double>{0., 0., 0.}));
        c.grasp.pre_distance = param(n, "grasp.pre_distance", 0.10);
        c.grasp.azimuth_samples = param(n, "grasp.azimuth_samples", 24);
        c.extract_distance = param(n, "grasp.extract_distance", 0.12);
        c.eef_step = param(n, "grasp.eef_step", 0.005);
        c.min_fraction = param(n, "grasp.min_fraction", 0.999);
        c.max_approach_angle = param(n, "grasp.max_approach_angle_deg", 60.);
        c.ik_timeout = param(n, "grasp.ik_timeout", 0.02);
        c.jump_threshold = param(n, "grasp.jump_threshold", 0.5);
        c.solutions_per_distance = param(n, "grasp.solutions_per_distance", 3);
        allow_link_contact_ = param(n, "grasp.allow_link_contact", true);

        c.camera_link = param<std::string>(n, "camera.link", "link_3");
        c.camera_position =
            vec3(param(n, "camera.position", std::vector<double>{0.13, 0.03, -0.1}));
        c.camera_target = vec3(param(n, "camera.target", std::vector<double>{0.03, 0.75, 0.035}));
        c.camera_down = vec3(param(n, "camera.down", std::vector<double>{-1., 0., 0.}));
        c.camera_width = param(n, "camera.width", 1280);
        c.camera_height = param(n, "camera.height", 720);
        c.camera_hfov = param(n, "camera.hfov_deg", 69.);
        c.camera_bracket = param(n, "camera.bracket", std::vector<double>{});

        // 相机（OpenCV 系：x 右，y 下，z 前）在相机所在连杆中的位姿
        const Vector3 forward = (c.camera_target - c.camera_position).normalized();
        const Vector3 down = (c.camera_down - c.camera_down.dot(forward) * forward).normalized();
        camera_in_link_ = Isometry::Identity();
        camera_in_link_.linear().col(0) = down.cross(forward);
        camera_in_link_.linear().col(1) = down;
        camera_in_link_.linear().col(2) = forward;
        camera_in_link_.translation() = c.camera_position;
        return true;
    }

    // ---- 任务 ----
    // 第 slot 个安装位的方向（中心块系，也是世界系：中心块不旋转）
    static auto slot_direction(std::size_t k) -> Vector3 {
        const auto angle = (30. + 60. * static_cast<double>(k)) * std::numbers::pi / 180.;
        return Vector3{std::cos(angle), 0., std::sin(angle)};
    }

    auto make_task(int seed, int slot) const -> Task {
        auto task = Task{};
        task.seed = seed;
        task.slot = slot;
        auto engine = std::mt19937{static_cast<std::uint32_t>(seed)};
        auto uniform = [&](double low, double high) {
            return std::uniform_real_distribution<double>{low, high}(engine);
        };
        for (std::size_t k = 0; k < kSlots; k++) {
            const auto draw = uniform(0., 1.);
            task.present[k] = static_cast<int>(k) == slot || draw < config_.slot_probability;
            const Vector3 axis = slot_direction(k);
            const Vector3 x = Vector3::UnitY().cross(axis).normalized();
            auto pose = Isometry::Identity();
            pose.linear().col(0) = x;
            pose.linear().col(1) = axis.cross(x);
            pose.linear().col(2) = axis;
            pose.translation() = config_.hub_center + config_.unit_offset * axis;
            task.unit_pose[k] = pose;
        }
        // 对准点：目标中心加随机偏移，水平方向取世界 x，竖直方向取世界 z
        const auto dx = uniform(-config_.aim_horizontal, config_.aim_horizontal);
        const auto dz = uniform(-config_.aim_vertical, config_.aim_vertical);
        task.aim = task.target().translation() + Vector3{dx, 0., dz};
        return task;
    }

    auto base_pose(const Task& task, double distance, double yaw) const -> Isometry {
        const Vector3 target = task.target().translation();
        return yaw_pose(target.x(), target.y() - distance, config_.base_height, yaw);
    }

    // ---- 场景 ----
    static auto unit_id(std::size_t k) -> std::string { return std::format("unit_{}", k); }

    auto hub_mesh() const -> shape_msgs::msg::Mesh {
        // 六棱柱：轴沿 y，正六边形在 xz 平面，顶点角度 60°·k，对边距 2·apothem
        const auto radius = config_.hub_apothem / std::cos(std::numbers::pi / 6.);
        const auto half = config_.hub_thickness / 2.;
        auto mesh = shape_msgs::msg::Mesh{};
        auto point = [](double x, double y, double z) {
            return geometry_msgs::msg::Point().set__x(x).set__y(y).set__z(z);
        };
        for (const auto y : {-half, half}) {
            for (int k = 0; k < 6; k++) {
                const auto a = std::numbers::pi / 3. * k;
                mesh.vertices.push_back(point(radius * std::cos(a), y, radius * std::sin(a)));
            }
        }
        auto triangle = [&](std::uint32_t a, std::uint32_t b, std::uint32_t c) {
            auto face = shape_msgs::msg::MeshTriangle{};
            face.vertex_indices = {a, b, c};
            mesh.triangles.push_back(face);
        };
        for (std::uint32_t k = 1; k + 1 < 6; k++) {
            triangle(0, k, k + 1);
            triangle(6, 6 + k, 6 + k + 1);
        }
        for (std::uint32_t k = 0; k < 6; k++) {
            const auto next = (k + 1) % 6;
            triangle(k, next, 6 + next);
            triangle(k, 6 + next, 6 + k);
        }
        return mesh;
    }

    static auto mesh_object(
        const std::string& id, const shape_msgs::msg::Mesh& mesh, const Isometry& pose,
        const std::string& frame) -> moveit_msgs::msg::CollisionObject {
        auto object = moveit_msgs::msg::CollisionObject{};
        object.header.frame_id = frame;
        object.id = id;
        object.meshes.push_back(mesh);
        object.mesh_poses.push_back(to_msg(pose));
        object.operation = moveit_msgs::msg::CollisionObject::ADD;
        return object;
    }

    // 场景中的物体（base_link 系）：地面、中心块和各安装位上的单元
    auto world_objects(const Task& task, const Isometry& world_from_base) const
        -> std::vector<moveit_msgs::msg::CollisionObject> {
        const Isometry base_from_world = world_from_base.inverse();
        auto objects = std::vector<moveit_msgs::msg::CollisionObject>{};

        auto ground = moveit_msgs::msg::CollisionObject{};
        ground.header.frame_id = kFrame;
        ground.id = "ground";
        auto box = shape_msgs::msg::SolidPrimitive{};
        box.type = shape_msgs::msg::SolidPrimitive::BOX;
        box.dimensions = {8., 8., 0.05};
        ground.primitives.push_back(box);
        auto ground_pose = Isometry::Identity();
        ground_pose.translation() = Vector3{0., 0., -world_from_base.translation().z() - 0.025};
        ground.primitive_poses.push_back(to_msg(ground_pose));
        ground.operation = moveit_msgs::msg::CollisionObject::ADD;
        objects.push_back(ground);

        auto hub_pose = Isometry::Identity();
        hub_pose.translation() = config_.hub_center;
        objects.push_back(mesh_object("hub", hub_mesh(), base_from_world * hub_pose, kFrame));
        for (std::size_t k = 0; k < kSlots; k++) {
            if (task.present[k]) {
                objects.push_back(mesh_object(
                    unit_id(k), unit_mesh_, base_from_world * task.unit_pose[k], kFrame));
            }
        }
        return objects;
    }

    static auto world_ids(const Task& task) -> std::vector<std::string> {
        auto ids = std::vector<std::string>{"ground", "hub"};
        for (std::size_t k = 0; k < kSlots; k++) {
            if (task.present[k])
                ids.push_back(unit_id(k));
        }
        return ids;
    }

    // 场景物体之间互相不检查碰撞；allow_target_link 时允许夹爪 link_6 接触目标单元，
    // 手腕 link_5 不允许
    static void set_world_acm(
        collision_detection::AllowedCollisionMatrix& acm, const Task& task,
        bool allow_target_link) {
        const auto ids = world_ids(task);
        for (const auto& a : ids) {
            for (const auto& b : ids)
                acm.setEntry(a, b, true);
        }
        const auto target = unit_id(static_cast<std::size_t>(task.slot));
        acm.setEntry(target, "link_5", false);
        acm.setEntry(target, kLink, allow_target_link);
    }

    // 本地规划场景（只用于搜索底座距离和观察姿态，不经过 move_group）
    void build_local_world(const Task& task, const Isometry& world_from_base) {
        scene_->getWorldNonConst()->clearObjects();
        for (const auto& object : world_objects(task, world_from_base))
            scene_->processCollisionObjectMsg(object);
        acm_strict_ = scene_->getAllowedCollisionMatrix();
        set_world_acm(acm_strict_, task, false);
        acm_contact_ = scene_->getAllowedCollisionMatrix();
        set_world_acm(acm_contact_, task, allow_link_contact_);
        // 视线检查只关心圆柱与 link_4、link_5、link_6 的碰撞，其余配对全部允许
        auto names = model_->getLinkModelNames();
        for (const auto& id : world_ids(task))
            names.push_back(id);
        names.push_back(kSightId);
        acm_sight_ = collision_detection::AllowedCollisionMatrix{};
        acm_sight_.setEntry(names, names, true);
        for (const auto* link : {"link_4", "link_5", "link_6"})
            acm_sight_.setEntry(kSightId, link, false);
        task_ = &task;
        world_from_base_ = world_from_base;
    }

    void set_target_in_local_world(bool present) {
        const auto k = static_cast<std::size_t>(task_->slot);
        if (present) {
            scene_->processCollisionObjectMsg(mesh_object(
                unit_id(k), unit_mesh_, world_from_base_.inverse() * task_->unit_pose[k], kFrame));
        } else {
            scene_->getWorldNonConst()->removeObject(unit_id(k));
        }
    }

    // 相机视线是否被机械臂挡住：从光心到目标中心画一个圆柱（靠目标一端缩短，不与目标本身相交），
    // 与 link_4、link_5、link_6 碰撞就是被挡住。圆柱只在检查期间存在于本地场景
    auto sight_blocked(const moveit::core::RobotState& state) const -> bool {
        const Isometry camera = state.getGlobalLinkTransform(config_.camera_link) * camera_in_link_;
        const Vector3 from = camera.translation();
        const Vector3 to = world_from_base_.inverse() * task_->target().translation();
        const auto length = (to - from).norm() - config_.sight_shorten;
        if (length <= 0.)
            return false;
        const Vector3 direction = (to - from).normalized();
        auto pose = Isometry::Identity();
        pose.linear() = Eigen::Quaterniond::FromTwoVectors(Vector3::UnitZ(), direction)
                            .toRotationMatrix();
        pose.translation() = from + direction * length / 2.;
        auto object = moveit_msgs::msg::CollisionObject{};
        object.header.frame_id = kFrame;
        object.id = kSightId;
        auto cylinder = shape_msgs::msg::SolidPrimitive{};
        cylinder.type = shape_msgs::msg::SolidPrimitive::CYLINDER;
        cylinder.dimensions = {length, config_.sight_radius};
        object.primitives.push_back(cylinder);
        object.primitive_poses.push_back(to_msg(pose));
        object.operation = moveit_msgs::msg::CollisionObject::ADD;
        scene_->processCollisionObjectMsg(object);

        auto request = collision_detection::CollisionRequest{};
        auto result = collision_detection::CollisionResult{};
        scene_->checkCollision(request, result, state, acm_sight_);
        scene_->getWorldNonConst()->removeObject(kSightId);
        return result.collision;
    }

    auto collides(moveit::core::RobotState& state, bool contact) const -> bool {
        auto request = collision_detection::CollisionRequest{};
        auto result = collision_detection::CollisionResult{};
        scene_->checkCollision(request, result, state, contact ? acm_contact_ : acm_strict_);
        return result.collision;
    }

    // ---- 机器人 ----
    void set_joints(moveit::core::RobotState& state, const Joints& q) const {
        state.setJointGroupPositions(group_, q);
        state.update();
    }

    auto joints_of(const moveit::core::RobotState& state) const -> Joints {
        auto q = Joints{};
        state.copyJointGroupPositions(group_, q);
        return q;
    }

    auto link_pose(const Joints& q, const std::string& link) const -> Isometry {
        set_joints(*state_, q);
        return state_->getGlobalLinkTransform(link);
    }

    auto state_message(const Joints& q, const std::optional<Isometry>& attached) const
        -> moveit_msgs::msg::RobotState {
        auto state = moveit_msgs::msg::RobotState{};
        state.joint_state.name = joint_names_;
        state.joint_state.position = q;
        if (attached.has_value()) {
            auto aco = moveit_msgs::msg::AttachedCollisionObject{};
            aco.link_name = kLink;
            aco.touch_links = {kLink};
            aco.object = mesh_object(unit_id(target_slot_), unit_mesh_, *attached, kLink);
            state.attached_collision_objects.push_back(aco);
        }
        return state;
    }

    // ---- 观察姿态 ----
    struct Observation {
        Joints q;
        double yaw = 0.;
    };

    // 相机（OpenCV 系）在世界系中的位姿
    auto camera_in_world(const Isometry& world_from_base, double q2, double q3) const -> Isometry {
        return world_from_base * link_pose({0., q2, q3, 0., 0., 0.}, config_.camera_link)
             * camera_in_link_;
    }

    // 光轴相对对准点的偏差：水平夹角和仰角之差
    auto aim_error(const Task& task, double x, double y, double q2, double q3, double yaw) const
        -> std::array<double, 2> {
        const auto camera = camera_in_world(yaw_pose(x, y, config_.base_height, yaw), q2, q3);
        const Vector3 axis = camera.linear().col(2);
        const Vector3 to_aim = task.aim - camera.translation();
        const auto horizontal = std::atan2(
            axis.x() * to_aim.y() - axis.y() * to_aim.x(),
            axis.x() * to_aim.x() + axis.y() * to_aim.y());
        const auto vertical = std::atan2(axis.z(), std::hypot(axis.x(), axis.y()))
                            - std::atan2(to_aim.z(), std::hypot(to_aim.x(), to_aim.y()));
        return {horizontal, vertical};
    }

    // 固定 q2，解 q3 和底盘朝向 yaw 使光轴穿过对准点；fixed_yaw 时只解 q3（只满足竖直对准）。
    // 返回 (q3, yaw)
    auto solve_aim(
        const Task& task, double x, double y, double q2, double q3_guess, double yaw_guess,
        bool fixed_yaw) const -> std::optional<std::pair<double, double>> {
        auto q3 = q3_guess;
        auto yaw = yaw_guess;
        constexpr auto kDelta = 1e-6;
        for (int iteration = 0; iteration < 40; iteration++) {
            const auto e = aim_error(task, x, y, q2, q3, yaw);
            const auto reached = fixed_yaw ? std::abs(e[1]) : std::hypot(e[0], e[1]);
            if (reached < 1e-10)
                return std::pair{q3, std::remainder(yaw, 2. * std::numbers::pi)};
            const auto e3 = aim_error(task, x, y, q2, q3 + kDelta, yaw);
            const auto ey = aim_error(task, x, y, q2, q3, yaw + kDelta);
            // 雅可比 [[a, b], [c, d]]：行为 (水平, 竖直)，列为 (q3, yaw)
            const auto a = (e3[0] - e[0]) / kDelta;
            const auto b = (ey[0] - e[0]) / kDelta;
            const auto c = (e3[1] - e[1]) / kDelta;
            const auto d = (ey[1] - e[1]) / kDelta;
            auto step_q3 = 0.;
            auto step_yaw = 0.;
            if (fixed_yaw) {
                step_q3 = -e[1] / c;
            } else {
                const auto det = a * d - b * c;
                if (std::abs(det) < 1e-9)
                    return std::nullopt;
                step_q3 = (-d * e[0] + b * e[1]) / det;
                step_yaw = (c * e[0] - a * e[1]) / det;
            }
            if (!std::isfinite(step_q3) || !std::isfinite(step_yaw))
                return std::nullopt;
            q3 += std::clamp(step_q3, -0.2, 0.2);
            yaw += std::clamp(step_yaw, -0.2, 0.2);
        }
        return std::nullopt;
    }

    // 观察姿态：q1 = q6 = 0；q2 在可行范围内扫描，q3 和底盘朝向由对准条件确定（与 q4、q5 无关）；
    // q4、q5 在候选集合里逐个试。在对准有解、无碰撞、视线不被挡住的解里取离 reference
    // （预抓取姿态）最近的。yaw_guess 是底盘朝向的迭代初值
    auto observe(
        const Task& task, double x, double y, const Joints& reference, double yaw_guess,
        std::map<std::string, int>& reasons) const -> std::optional<Observation> {
        const auto* joint2 = model_->getJointModel("joint_2");
        const auto lower = joint2->getVariableBounds()[0].min_position_;
        const auto upper = joint2->getVariableBounds()[0].max_position_;
        const auto* joint3 = model_->getJointModel("joint_3");
        const auto lower3 = joint3->getVariableBounds()[0].min_position_;
        const auto upper3 = joint3->getVariableBounds()[0].max_position_;

        auto best = std::optional<Observation>{};
        auto best_distance = std::numeric_limits<double>::max();
        auto evaluate = [&](double q2, double& q3_guess, double& yaw_start) {
            // 先用上一个 q2 的解作初值，失败或超出关节范围时换几个典型初值
            auto solved = std::optional<std::pair<double, double>>{};
            for (const auto guess : {q3_guess, -1.0, -0.3, 0.3}) {
                solved = solve_aim(task, x, y, q2, guess, yaw_start, false);
                if (solved.has_value() && solved->first >= lower3 && solved->first <= upper3)
                    break;
            }
            if (!solved.has_value()) {
                reasons["observe: aim unsolvable"]++;
                return;
            }
            q3_guess = solved->first;
            yaw_start = solved->second;
            for (const auto q4 : config_.q4_candidates) {
                for (const auto q5 : config_.q5_candidates) {
                    const auto observation =
                        Observation{{0., q2, solved->first, q4, q5, 0.}, solved->second};
                    set_joints(*state_, observation.q);
                    if (!state_->satisfiesBounds(group_)) {
                        reasons["observe: joint limit"]++;
                        continue;
                    }
                    auto distance = 0.;
                    for (std::size_t i = 0; i < 6; i++)
                        distance += std::pow(observation.q[i] - reference[i], 2);
                    if (distance >= best_distance)
                        continue;
                    if (collides(*state_, false)) {
                        reasons["observe: collision"]++;
                        continue;
                    }
                    if (sight_blocked(*state_)) {
                        reasons["observe: line of sight blocked"]++;
                        continue;
                    }
                    best = observation;
                    best_distance = distance;
                }
            }
        };

        auto q3_guess = -0.5;
        auto yaw_start = yaw_guess;
        constexpr auto kCoarse = 0.02;
        for (int i = 0; lower + i * kCoarse <= upper; i++)
            evaluate(lower + i * kCoarse, q3_guess, yaw_start);
        if (!best.has_value())
            return std::nullopt;
        // 在最优解附近细扫
        const auto center = best->q[1];
        q3_guess = best->q[2];
        yaw_start = best->yaw;
        for (int i = -20; i <= 20; i++) {
            const auto q2 = center + i * kCoarse / 20.;
            if (q2 >= lower && q2 <= upper)
                evaluate(q2, q3_guess, yaw_start);
        }
        return best;
    }

    // ---- 抓取链：抓取、预抓取、拉出的 IK 与碰撞 ----
    struct Chain {
        grasp_sim::GraspCandidate candidate;
        Joints pre;
        Joints grasp;
        Joints extract;
    };

    auto solve_ik(moveit::core::RobotState& state, const Isometry& pose, const Joints& seed) const
        -> bool {
        state.setJointGroupPositions(group_, seed);
        return state.setFromIK(group_, pose, kLink, config_.ik_timeout);
    }

    // 沿 direction 直线移动 distance，逐步 IK 并检查关节跳变和碰撞；成功时返回终点关节角
    auto linear_path(
        moveit::core::RobotState& state, const Isometry& start_pose, const Joints& start,
        const Vector3& direction, double distance, bool contact, const char* ik_reason,
        const char* collision_reason, std::map<std::string, int>& reasons) const
        -> std::optional<Joints> {
        auto previous = start;
        const auto steps = static_cast<int>(std::ceil(distance / config_.eef_step - 1e-9));
        for (int i = 1; i <= steps; i++) {
            auto pose = start_pose;
            pose.translation() += std::min(i * config_.eef_step, distance) * direction;
            if (!solve_ik(state, pose, previous)) {
                reasons[ik_reason]++;
                return std::nullopt;
            }
            state.update();
            const auto q = joints_of(state);
            auto jump = 0.;
            for (std::size_t j = 0; j < q.size(); j++)
                jump = std::max(jump, std::abs(q[j] - previous[j]));
            if (jump > config_.jump_threshold || !state.satisfiesBounds(group_)) {
                reasons[ik_reason]++;
                return std::nullopt;
            }
            if (collides(state, contact)) {
                reasons[collision_reason]++;
                return std::nullopt;
            }
            previous = q;
        }
        return previous;
    }

    auto build_chain(
        const Task& task, const grasp_sim::GraspCandidate& candidate, const Joints& seed,
        std::map<std::string, int>& reasons) -> std::optional<Chain> {
        auto state = moveit::core::RobotState{model_};
        if (!solve_ik(state, candidate.grasp, seed)) {
            reasons["grasp: IK no solution"]++;
            return std::nullopt;
        }
        state.update();
        if (!state.satisfiesBounds(group_)) {
            reasons["grasp: IK joint limit"]++;
            return std::nullopt;
        }
        if (collides(state, allow_link_contact_)) {
            reasons["grasp: collision"]++;
            return std::nullopt;
        }
        auto chain = Chain{};
        chain.candidate = candidate;
        chain.grasp = joints_of(state);

        const Vector3 back = candidate.grasp.linear().col(2);
        const auto pre = linear_path(
            state, candidate.grasp, chain.grasp, back, config_.grasp.pre_distance,
            allow_link_contact_, "pre-grasp: IK/jump failed", "pre-grasp: collision", reasons);
        if (!pre.has_value())
            return std::nullopt;
        chain.pre = *pre;

        // 拉出：目标单元附着在 link_6 上，世界里的目标单元移除
        const Isometry unit_in_base = world_from_base_.inverse() * task.target();
        const Isometry relative = candidate.grasp.inverse() * unit_in_base;
        auto attached = moveit::core::RobotState{model_};
        moveit::core::robotStateMsgToRobotState(state_message(chain.grasp, relative), attached);
        const Vector3 outward = world_from_base_.linear().transpose() * task.axis();
        set_target_in_local_world(false);
        const auto extract = linear_path(
            attached, candidate.grasp, chain.grasp, outward, config_.extract_distance, true,
            "extract: IK/jump failed", "extract: collision", reasons);
        set_target_in_local_world(true);
        if (!extract.has_value())
            return std::nullopt;
        chain.extract = *extract;
        return chain;
    }

    // 作为 IK 初值的关节角：覆盖肩、肘、腕的几种典型姿态
    static auto ik_seeds() -> std::vector<Joints> {
        auto seeds = std::vector<Joints>{};
        for (const auto q2 : {-0.8, 0.0, 0.7}) {
            for (const auto q3 : {-1.0, -0.4, 0.3}) {
                for (const auto q5 : {-0.9, 0.9})
                    seeds.push_back({0., q2, q3, 0., q5, 0.});
            }
        }
        return seeds;
    }

    // 按代价从小到大尝试抓取候选，每个候选取第一个可行的 IK 初值，最多 limit 个解
    auto find_chains(const Task& task, std::size_t limit, std::map<std::string, int>& reasons)
        -> std::vector<Chain> {
        const Vector3 center = world_from_base_.inverse() * task.target().translation();
        const Vector3 axis = world_from_base_.linear().transpose() * task.axis();
        const Vector3 radial = Vector3{center.x(), center.y(), 0.}.normalized();
        auto chains = std::vector<Chain>{};
        const auto candidates =
            grasp_sim::generate_grasps(center, axis, Vector3::Zero(), config_.grasp);
        for (const auto& candidate : candidates) {
            const Vector3 approach = -candidate.grasp.linear().col(2);
            const auto angle = std::acos(std::clamp(approach.dot(radial), -1., 1.)) * kRadToDeg;
            if (angle > config_.max_approach_angle)
                continue;
            for (const auto& seed : ik_seeds()) {
                auto chain = build_chain(task, candidate, seed, reasons);
                if (chain.has_value()) {
                    chains.push_back(*chain);
                    break;
                }
            }
            if (chains.size() >= limit)
                break;
        }
        return chains;
    }

    // ---- 底座距离 ----
    // 在 distance 处求完整的布局：抓取链 + 观察姿态。底盘朝向取决于观察姿态，而抓取链取决于
    // 底盘朝向，所以迭代到朝向不再变化
    auto evaluate_distance(const Task& task, double distance, std::map<std::string, int>& reasons)
        -> std::optional<Layout> {
        const Vector3 target = task.target().translation();
        const auto x = target.x();
        const auto y = target.y() - distance;

        // 朝向初值：用一个典型的 q2 解出
        auto yaw = 0.;
        {
            // 机械臂在 q1 = 0 时沿底座 x 轴伸出，朝向中心块（+y）时底盘朝向约 90°
            const auto nominal = solve_aim(task, x, y, -0.3, -0.5, std::numbers::pi / 2., false);
            if (!nominal.has_value()) {
                reasons["observe: aim unsolvable"]++;
                return std::nullopt;
            }
            yaw = nominal->second;
        }

        auto chains = std::vector<Chain>{};
        for (int iteration = 0; iteration < 8; iteration++) {
            const auto world = base_pose(task, distance, yaw);
            build_local_world(task, world);
            if (iteration == 0) {
                chains = find_chains(
                    task, static_cast<std::size_t>(config_.solutions_per_distance), reasons);
            } else {
                // 朝向变了：以上一次的解为初值重新求链
                auto refined = std::vector<Chain>{};
                for (const auto& chain : chains) {
                    const Vector3 center = world.inverse() * task.target().translation();
                    const Vector3 axis = world.linear().transpose() * task.axis();
                    auto candidate = chain.candidate;
                    for (const auto& c :
                         grasp_sim::generate_grasps(center, axis, Vector3::Zero(), config_.grasp)) {
                        // 与原候选同一个接近方位
                        if (c.sign == chain.candidate.sign
                            && std::abs(c.azimuth - chain.candidate.azimuth) < 1e-9) {
                            candidate = c;
                            break;
                        }
                    }
                    if (auto again = build_chain(task, candidate, chain.grasp, reasons);
                        again.has_value())
                        refined.push_back(*again);
                }
                chains = refined;
            }
            if (chains.empty())
                return std::nullopt;

            // 每条链配一个观察姿态，选接近运动最短的
            auto best = std::optional<std::pair<Chain, Observation>>{};
            auto best_distance = std::numeric_limits<double>::max();
            for (const auto& chain : chains) {
                const auto observation = observe(task, x, y, chain.pre, yaw, reasons);
                if (!observation.has_value())
                    continue;
                auto sum = 0.;
                for (std::size_t i = 0; i < 6; i++)
                    sum += std::pow(observation->q[i] - chain.pre[i], 2);
                if (sum < best_distance) {
                    best_distance = sum;
                    best = std::pair{chain, *observation};
                }
            }
            if (!best.has_value())
                return std::nullopt;

            const auto change = std::abs(best->second.yaw - yaw);
            if (change < 1e-5 || iteration == 7) {
                // 底盘朝向取抓取链用的 yaw，q3 按它重新解竖直对准
                const auto fixed =
                    solve_aim(task, x, y, best->second.q[1], best->second.q[2], yaw, true);
                if (!fixed.has_value())
                    return std::nullopt;
                auto layout = Layout{};
                layout.distance = distance;
                layout.yaw = yaw;
                layout.world_from_base = base_pose(task, distance, yaw);
                layout.observe = best->second.q;
                layout.observe[2] = fixed->first;
                layout.pre = best->first.pre;
                layout.grasp_joints = best->first.grasp;
                layout.candidate = best->first.candidate;
                set_joints(*state_, layout.observe);
                if (collides(*state_, false)) {
                    reasons["observe: collision"]++;
                    return std::nullopt;
                }
                if (sight_blocked(*state_)) {
                    reasons["observe: line of sight blocked"]++;
                    return std::nullopt;
                }
                return layout;
            }
            yaw = best->second.yaw;
            chains = {best->first};
        }
        return std::nullopt;
    }

    // ---- MoveIt 规划 ----
    struct Stage {
        bool ok = false;
        Timed timed;
        Joints end;
        double fraction = 0.;
        double ms = 0.;
        int code = 0;
    };

    void wait_for_move_group() {
        acm_client_ =
            node_->create_client<moveit_msgs::srv::GetPlanningScene>("/get_planning_scene");
        while (rclcpp::ok() && !acm_client_->wait_for_service(std::chrono::seconds{1}))
            RCLCPP_INFO(logger_, "waiting for move_group ...");
    }

    // 在 move_group 的场景里换成这份规划的物体（base_link 系）
    void apply_world(const Task& task, const Isometry& world_from_base) {
        scene_interface_->removeCollisionObjects(scene_ids_);
        const auto objects = world_objects(task, world_from_base);
        scene_ids_.clear();
        for (const auto& object : objects)
            scene_ids_.push_back(object.id);
        scene_interface_->applyCollisionObjects(objects);
    }

    void update_acm(const Task& task, bool allow_target_link) {
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
        set_world_acm(acm, task, allow_target_link);
        auto diff = moveit_msgs::msg::PlanningScene{};
        diff.is_diff = true;
        acm.getMessage(diff.allowed_collision_matrix);
        scene_interface_->applyPlanningScene(diff);
    }

    // TOTG 时间参数化，采样间隔与帧间隔一致
    auto time_parameterize(
        const moveit_msgs::msg::RobotState& start,
        moveit_msgs::msg::RobotTrajectory trajectory) const -> std::optional<Timed> {
        auto state = moveit::core::RobotState{model_};
        moveit::core::robotStateMsgToRobotState(start, state);
        auto robot_trajectory = robot_trajectory::RobotTrajectory{model_, kGroup};
        robot_trajectory.setRobotTrajectoryMsg(state, trajectory);
        auto totg = trajectory_processing::TimeOptimalTrajectoryGeneration{
            config_.path_tolerance, 1. / config_.fps};
        if (!totg.computeTimeStamps(
                robot_trajectory, config_.velocity_scaling, config_.acceleration_scaling))
            return std::nullopt;
        auto timed = Timed{};
        for (std::size_t i = 0; i < robot_trajectory.getWayPointCount(); i++) {
            timed.times.push_back(robot_trajectory.getWayPointDurationFromStart(i));
            timed.points.push_back(joints_of(robot_trajectory.getWayPoint(i)));
        }
        return timed;
    }

    auto plan_joint(const Joints& start, const Joints& goal) -> Stage {
        auto stage = Stage{};
        const auto begin = std::chrono::steady_clock::now();
        const auto start_message = state_message(start, std::nullopt);
        move_group_->setStartState(start_message);
        move_group_->setJointValueTarget(goal);
        auto plan = moveit::planning_interface::MoveGroupInterface::Plan();
        const auto code = move_group_->plan(plan);
        stage.ms = millis(begin);
        stage.code = code.val;
        if (code != moveit::core::MoveItErrorCode::SUCCESS)
            return stage;
        if (auto timed = time_parameterize(start_message, plan.trajectory); timed.has_value()) {
            stage.timed = *timed;
            stage.end = stage.timed.points.back();
            stage.ok = true;
        }
        return stage;
    }

    auto plan_cartesian(
        const Joints& start, const std::optional<Isometry>& attached, const Isometry& target)
        -> Stage {
        auto stage = Stage{};
        const auto begin = std::chrono::steady_clock::now();
        const auto start_message = state_message(start, attached);
        move_group_->setStartState(start_message);
        auto trajectory = moveit_msgs::msg::RobotTrajectory{};
        auto error = moveit_msgs::msg::MoveItErrorCodes{};
        stage.fraction = move_group_->computeCartesianPath(
            {to_msg(target)}, config_.eef_step, trajectory, true, &error);
        if (stage.fraction >= config_.min_fraction
            && trajectory.joint_trajectory.points.size() > 1) {
            // computeCartesianPath 不带时间；清零后由 TOTG 重新参数化
            if (auto timed = time_parameterize(start_message, trajectory); timed.has_value()) {
                stage.timed = *timed;
                stage.end = stage.timed.points.back();
                stage.ok = true;
            }
        }
        stage.ms = millis(begin);
        return stage;
    }

    // 在本地场景里逐点检查时间参数化之后的路径（TOTG 会按路径容差削角）。
    // attached 给定时目标单元附着在 link_6 上（相对位姿为 attached），世界里的目标单元移除
    auto path_collision_free(
        const Timed& path, bool contact, const std::optional<Isometry>& attached) -> bool {
        auto state = moveit::core::RobotState{model_};
        if (attached.has_value()) {
            moveit::core::robotStateMsgToRobotState(
                state_message(path.points.front(), attached), state);
            set_target_in_local_world(false);
        }
        auto free = true;
        for (std::size_t i = 0; i < path.points.size(); i++) {
            set_joints(state, path.points[i]);
            auto request = collision_detection::CollisionRequest{};
            request.contacts = true;
            auto result = collision_detection::CollisionResult{};
            scene_->checkCollision(request, result, state, contact ? acm_contact_ : acm_strict_);
            if (result.collision) {
                for (const auto& contact_pair : result.contacts)
                    RCLCPP_INFO(
                        logger_, "  path collision at waypoint %zu/%zu: %s - %s", i,
                        path.points.size(), contact_pair.first.first.c_str(),
                        contact_pair.first.second.c_str());
                free = false;
                break;
            }
        }
        if (attached.has_value())
            set_target_in_local_world(true);
        return free;
    }

    struct Motion {
        Timed approach;                                 // 观察姿态 -> 预抓取 -> 抓取
        Timed extract;
        Isometry link6_at_grasp = Isometry::Identity(); // base_link 系
    };

    auto plan_motion(const Task& task, const Layout& layout, Motion& motion, std::string& error)
        -> bool {
        apply_world(task, layout.world_from_base);
        update_acm(task, false);

        // OMPL 每次的结果不同；时间参数化后路径可能削到障碍，多试几次
        auto pre = Stage{};
        for (int attempt = 0; attempt < 5 && !pre.ok; attempt++) {
            pre = plan_joint(layout.observe, layout.pre);
            if (pre.ok && !path_collision_free(pre.timed, false, std::nullopt)) {
                pre.ok = false;
                pre.code = 0;
            }
        }
        if (!pre.ok) {
            error = std::format(
                "OMPL: no collision-free plan from the observation pose to the pre-grasp pose "
                "(error code {})",
                pre.code);
            return false;
        }
        update_acm(task, allow_link_contact_);
        const auto approach = plan_cartesian(pre.end, std::nullopt, layout.candidate.grasp);
        if (!approach.ok) {
            error = std::format(
                "Cartesian path to the grasp pose incomplete ({:.1f}%)", approach.fraction * 100.);
            return false;
        }
        if (!path_collision_free(approach.timed, allow_link_contact_, std::nullopt)) {
            error = "straight approach path collides after time parameterization";
            return false;
        }
        motion.link6_at_grasp = link_pose(approach.end, kLink);
        const Isometry unit_in_base = layout.world_from_base.inverse() * task.target();
        const Isometry relative = motion.link6_at_grasp.inverse() * unit_in_base;

        // 拉出：世界里的目标单元移除，以附着体随 link_6 运动
        scene_interface_->removeCollisionObjects({unit_id(static_cast<std::size_t>(task.slot))});
        // 移除物体时 MoveIt 会清掉它在 ACM 中的条目，附着后需要重新设置（单元与中心块有重叠）
        update_acm(task, true);
        auto target = motion.link6_at_grasp;
        target.translation() += config_.extract_distance
                              * Vector3{layout.world_from_base.linear().transpose() * task.axis()};
        const auto extract = plan_cartesian(approach.end, relative, target);
        if (!extract.ok) {
            error =
                std::format("Cartesian extract path incomplete ({:.1f}%)", extract.fraction * 100.);
            return false;
        }
        if (!path_collision_free(extract.timed, true, relative)) {
            error = "extract path collides after time parameterization";
            return false;
        }
        RCLCPP_INFO(
            logger_,
            "  MoveIt: pre-grasp %.0f ms (%.2f s), approach %.0f ms (%.2f s), extract %.0f ms "
            "(%.2f s)",
            pre.ms, pre.timed.times.back(), approach.ms, approach.timed.times.back(), extract.ms,
            extract.timed.times.back());

        motion.approach = pre.timed;
        const auto offset = pre.timed.times.back();
        for (std::size_t i = 1; i < approach.timed.times.size(); i++) {
            motion.approach.times.push_back(offset + approach.timed.times[i]);
            motion.approach.points.push_back(approach.timed.points[i]);
        }
        motion.extract = extract.timed;
        return true;
    }

    // ---- 总流程 ----
    auto make_plan(const Task& task) -> bool {
        target_slot_ = static_cast<std::size_t>(task.slot);
        const auto name = std::format("plan_slot{}_seed{}", task.slot, task.seed);
        RCLCPP_INFO(logger_, "==== %s ====", name.c_str());
        auto slots = std::string{};
        for (std::size_t k = 0; k < kSlots; k++)
            slots += task.present[k] ? '1' : '0';
        RCLCPP_INFO(logger_, "  slots with a unit (k = 0..5): %s", slots.c_str());
        const auto begin = std::chrono::steady_clock::now();

        // 扫描底座距离：记录每个距离能否取到
        auto reasons = std::map<std::string, int>{};
        auto feasible = std::vector<double>{};
        const auto distance_count =
            static_cast<int>(std::floor(
                (config_.distance_max - config_.distance_min) / config_.distance_step + 1e-9))
            + 1;
        for (int i = 0; i < distance_count; i++) {
            const auto d = config_.distance_min + i * config_.distance_step;
            auto local = std::map<std::string, int>{};
            if (evaluate_distance(task, d, local).has_value()) {
                feasible.push_back(d);
                RCLCPP_INFO(logger_, "  distance %.3f m: feasible", d);
            } else {
                auto text = std::string{};
                for (const auto& [reason, count] : local)
                    text += std::format(" [{} x{}]", reason, count);
                RCLCPP_INFO(logger_, "  distance %.3f m: infeasible%s", d, text.c_str());
                for (const auto& [reason, count] : local)
                    reasons[reason] += count;
            }
        }
        if (feasible.empty()) {
            RCLCPP_ERROR(
                logger_, "%s: no feasible base distance in [%.2f, %.2f]", name.c_str(),
                config_.distance_min, config_.distance_max);
            for (const auto& [reason, count] : reasons)
                RCLCPP_ERROR(logger_, "  %s x%d", reason.c_str(), count);
            return false;
        }

        // 取最长连续可行区间的中点；区间内其余距离按离中点由近到远作为备选
        auto best_begin = std::size_t{0};
        auto best_length = std::size_t{0};
        for (std::size_t i = 0; i < feasible.size();) {
            auto j = i;
            while (j + 1 < feasible.size()
                   && feasible[j + 1] - feasible[j] < 1.5 * config_.distance_step)
                j++;
            if (j - i + 1 > best_length) {
                best_begin = i;
                best_length = j - i + 1;
            }
            i = j + 1;
        }
        const auto low = feasible[best_begin];
        const auto high = feasible[best_begin + best_length - 1];
        const auto middle = (low + high) / 2.;
        RCLCPP_INFO(
            logger_, "  feasible distance range [%.3f, %.3f] m, middle %.3f m", low, high, middle);
        auto order = std::vector<double>{middle};
        auto rest = std::vector<double>(
            feasible.begin() + static_cast<std::ptrdiff_t>(best_begin),
            feasible.begin() + static_cast<std::ptrdiff_t>(best_begin + best_length));
        std::ranges::sort(
            rest, [&](double a, double b) { return std::abs(a - middle) < std::abs(b - middle); });
        for (const auto distance : rest) {
            if (std::abs(distance - middle) > 1e-9)
                order.push_back(distance);
        }

        auto motion = Motion{};
        auto layout = std::optional<Layout>{};
        auto error = std::string{};
        for (const auto distance : order) {
            auto local = std::map<std::string, int>{};
            auto candidate = evaluate_distance(task, distance, local);
            if (!candidate.has_value())
                continue;
            build_local_world(task, candidate->world_from_base);
            RCLCPP_INFO(logger_, "  planning at distance %.3f m ...", distance);
            if (plan_motion(task, *candidate, motion, error)) {
                layout = candidate;
                break;
            }
            RCLCPP_WARN(logger_, "  distance %.3f m: %s", distance, error.c_str());
        }
        if (!layout.has_value()) {
            RCLCPP_ERROR(
                logger_, "%s: MoveIt planning failed at every feasible distance (last: %s)",
                name.c_str(), error.c_str());
            return false;
        }
        build_local_world(task, layout->world_from_base);
        write_plan(task, *layout, motion, name);
        RCLCPP_INFO(logger_, "  total %.1f s", millis(begin) / 1e3);
        return true;
    }

    // ---- 输出 ----
    static auto interpolate(const Timed& path, double t) -> Joints {
        if (t <= path.times.front())
            return path.points.front();
        if (t >= path.times.back())
            return path.points.back();
        const auto upper =
            static_cast<std::size_t>(std::ranges::upper_bound(path.times, t) - path.times.begin());
        const auto lower = upper - 1;
        const auto span = path.times[upper] - path.times[lower];
        const auto s = span > 1e-12 ? (t - path.times[lower]) / span : 1.;
        auto q = path.points[lower];
        for (std::size_t i = 0; i < q.size(); i++)
            q[i] += s * (path.points[upper][i] - q[i]);
        return q;
    }

    static auto still(const Joints& q, double duration) -> Timed {
        return Timed{{0., duration}, {q, q}};
    }

    void write_plan(
        const Task& task, const Layout& layout, const Motion& motion, const std::string& name) {
        // 各段按时间拼接
        auto segments = std::vector<Segment>{};
        auto add = [&](const std::string& phase, bool attached, Timed path) {
            const auto start = segments.empty() ? 0. : segments.back().end();
            segments.push_back({phase, attached, start, std::move(path)});
        };
        add("hold", false, still(layout.observe, config_.hold_time));
        add("approach", false, motion.approach);
        add("grasp", true, still(motion.approach.points.back(), config_.grasp_time));
        add("extract", true, motion.extract);
        const auto total = segments.back().end();

        const auto count = static_cast<std::size_t>(std::ceil(total * config_.fps - 1e-9)) + 1;
        auto frames = std::vector<Frame>{};
        for (std::size_t i = 0; i < count; i++) {
            const auto t = std::min(static_cast<double>(i) / config_.fps, total);
            const Segment* segment = &segments.back();
            for (const auto& s : segments) {
                if (t < s.end() - 1e-9) {
                    segment = &s;
                    break;
                }
            }
            frames.push_back(
                {interpolate(segment->path, t - segment->start), segment->phase,
                 segment->attached});
        }

        // JSON
        auto text = std::string{"{\n"};
        text += std::format("  \"fps\": {},\n", config_.fps);
        text += std::format("  \"base\": {},\n", matrix_json(layout.world_from_base));
        auto hub_pose = Isometry::Identity();
        hub_pose.translation() = config_.hub_center;
        text += std::format(R"(  "hub": {{"pose": {}, "slots": [)", matrix_json(hub_pose));
        for (std::size_t k = 0; k < kSlots; k++)
            text += std::format("{}{}", k == 0 ? "" : ", ", task.present[k] ? "true" : "false");
        text += std::format("], \"target\": {}}},\n", task.slot);
        text += std::format(
            "  \"camera\": {{\"link\": \"{}\", \"pose\": {}, \"width\": {}, \"height\": {}, "
            "\"hfov_deg\": {}",
            config_.camera_link, matrix_json(camera_in_link_), config_.camera_width,
            config_.camera_height, config_.camera_hfov);
        if (config_.camera_bracket.size() >= 6) {
            text += ", \"bracket\": [";
            for (std::size_t i = 0; i + 2 < config_.camera_bracket.size(); i += 3) {
                text += std::format(
                    "{}[{}, {}, {}]", i == 0 ? "" : ", ", config_.camera_bracket[i],
                    config_.camera_bracket[i + 1], config_.camera_bracket[i + 2]);
            }
            text += "]";
        }
        text += "},\n  \"frames\": [\n";
        for (std::size_t i = 0; i < frames.size(); i++) {
            text += "    {\"joints\": [";
            for (std::size_t j = 0; j < 6; j++)
                text += std::format("{}{:.9g}", j == 0 ? "" : ", ", frames[i].joints[j]);
            text += std::format(
                "], \"phase\": \"{}\", \"attached\": {}}}{}\n", frames[i].phase,
                frames[i].attached ? "true" : "false", i + 1 == frames.size() ? "" : ",");
        }
        text += "  ]\n}\n";

        std::filesystem::create_directories(config_.output_dir);
        const auto path = std::format("{}/{}.json", config_.output_dir, name);
        std::ofstream{path} << text;
        RCLCPP_INFO(logger_, "  wrote %s (%zu frames)", path.c_str(), frames.size());

        report(task, layout, motion, frames, segments);
    }

    // ---- 报告 ----
    void report(
        const Task& task, const Layout& layout, const Motion& motion,
        const std::vector<Frame>& frames, const std::vector<Segment>& segments) {
        const Isometry& base = layout.world_from_base;
        const Vector3 center = task.target().translation();
        const Vector3 base_position = base.translation();
        const auto yaw = std::atan2(base.linear()(1, 0), base.linear()(0, 0));
        RCLCPP_INFO(
            logger_,
            "  base: position (%.4f, %.4f, %.4f) m, yaw %.3f deg; horizontal distance to target "
            "%.4f m",
            base_position.x(), base_position.y(), base_position.z(), yaw * kRadToDeg,
            std::hypot(center.x() - base_position.x(), center.y() - base_position.y()));
        RCLCPP_INFO(
            logger_, "  observe joints: %.4f %.4f %.4f %.4f %.4f %.4f", layout.observe[0],
            layout.observe[1], layout.observe[2], layout.observe[3], layout.observe[4],
            layout.observe[5]);

        const auto camera = camera_in_world_full(base, layout.observe);
        const Vector3 in_camera = camera.inverse() * center;
        const auto fx = config_.camera_width / 2.
                      / std::tan(config_.camera_hfov / 2. * std::numbers::pi / 180.);
        RCLCPP_INFO(
            logger_,
            "  camera-to-target distance %.4f m, target pixel (%.1f, %.1f) of %dx%d, aim point "
            "offset from target (%.3f, %.3f, %.3f) m",
            in_camera.norm(), fx * in_camera.x() / in_camera.z() + config_.camera_width / 2.,
            fx * in_camera.y() / in_camera.z() + config_.camera_height / 2., config_.camera_width,
            config_.camera_height, task.aim.x() - center.x(), task.aim.y() - center.y(),
            task.aim.z() - center.z());

        for (const auto& segment : segments) {
            const auto n = std::ranges::count_if(
                frames, [&](const Frame& f) { return f.phase == segment.phase; });
            RCLCPP_INFO(
                logger_, "  phase %-8s %.3f s, %ld frames", segment.phase.c_str(),
                segment.path.times.back(), n);
        }

        // 相机速度，以及目标单元在相机系中的帧间运动
        auto speed = 0.;
        auto move = std::map<std::string, double>{};
        auto turn = std::map<std::string, double>{};
        auto previous_camera = Isometry::Identity();
        auto previous_target = Isometry::Identity();
        const Isometry unit_in_base = base.inverse() * task.target();
        const Isometry relative = motion.link6_at_grasp.inverse() * unit_in_base;
        auto extract_end = Isometry::Identity();
        auto grasp_error = std::optional<double>{};
        for (std::size_t i = 0; i < frames.size(); i++) {
            const auto camera_pose = camera_in_world_full(base, frames[i].joints);
            const Isometry link6 = base * link_pose(frames[i].joints, kLink);
            const Isometry target_pose =
                frames[i].attached ? Isometry{link6 * relative} : task.target();
            const Isometry target_in_camera = camera_pose.inverse() * target_pose;
            if (i > 0) {
                speed = std::max(
                    speed, (camera_pose.translation() - previous_camera.translation()).norm()
                               * config_.fps);
                const Isometry before = previous_camera.inverse() * previous_target;
                const auto distance =
                    (target_in_camera.translation() - before.translation()).norm();
                const auto angle =
                    rotation_angle(before.linear().transpose() * target_in_camera.linear());
                move[frames[i].phase] = std::max(move[frames[i].phase], distance * 1e3);
                turn[frames[i].phase] = std::max(turn[frames[i].phase], angle * kRadToDeg);
            }
            if (frames[i].phase == "grasp" && !grasp_error.has_value()) {
                grasp_error = (link6.translation() - task.target().translation()).norm() * 1e3;
            }
            previous_camera = camera_pose;
            previous_target = target_pose;
            extract_end = target_pose;
        }
        RCLCPP_INFO(logger_, "  max camera speed %.4f m/s", speed);
        for (const auto* phase : {"approach", "extract"}) {
            RCLCPP_INFO(
                logger_,
                "  target in camera, %s: max per-frame move %.3f mm, max per-frame turn %.4f deg",
                phase, move[phase], turn[phase]);
        }
        RCLCPP_INFO(
            logger_,
            "  first grasp frame: link_6 to unit centre %.3f mm; last frame: unit moved %.3f mm "
            "(expect %.1f)",
            grasp_error.value_or(-1.),
            (extract_end.translation() - task.target().translation()).norm() * 1e3,
            config_.extract_distance * 1e3);
    }

    // 相机在世界系中的位姿，给定底座位姿和六个关节角
    auto camera_in_world_full(const Isometry& world_from_base, const Joints& q) const -> Isometry {
        return world_from_base * link_pose(q, config_.camera_link) * camera_in_link_;
    }

    rclcpp::Node::SharedPtr node_;
    rclcpp::Logger logger_;
    Config config_;
    bool allow_link_contact_ = true;
    shape_msgs::msg::Mesh unit_mesh_;
    Isometry camera_in_link_ = Isometry::Identity();

    std::unique_ptr<moveit::planning_interface::MoveGroupInterface> move_group_;
    std::unique_ptr<moveit::planning_interface::PlanningSceneInterface> scene_interface_;
    rclcpp::Client<moveit_msgs::srv::GetPlanningScene>::SharedPtr acm_client_;
    moveit::core::RobotModelConstPtr model_;
    const moveit::core::JointModelGroup* group_ = nullptr;
    std::vector<std::string> joint_names_;
    std::vector<std::string> scene_ids_; // move_group 场景里当前由本工具添加的物体
    std::unique_ptr<moveit::core::RobotState> state_;

    // 本地规划场景及其当前对应的任务和底座位姿
    std::shared_ptr<planning_scene::PlanningScene> scene_;
    collision_detection::AllowedCollisionMatrix acm_strict_;
    collision_detection::AllowedCollisionMatrix acm_contact_;
    collision_detection::AllowedCollisionMatrix acm_sight_;
    const Task* task_ = nullptr;
    Isometry world_from_base_ = Isometry::Identity();
    std::size_t target_slot_ = 0;
};

} // namespace

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>(
        "pickup_plan", rclcpp::NodeOptions{}.automatically_declare_parameters_from_overrides(true));

    auto executor = rclcpp::executors::MultiThreadedExecutor{};
    executor.add_node(node);
    auto spin_thread = std::thread{[&] { executor.spin(); }};

    auto planner = PickupPlanner{node};
    const auto code = planner.run();

    executor.cancel();
    spin_thread.join();
    rclcpp::shutdown();
    return code;
}
