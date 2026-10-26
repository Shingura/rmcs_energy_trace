#include <string>

#include <eigen3/Eigen/Geometry>
#include <pluginlib/class_list_macros.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rmcs_executor/component.hpp>

#include "geometry/camera.hpp"

namespace rmcs_energy_trace {

// 能量单元位姿估计组件。
// 输出相机坐标系下能量单元的位姿，单位 m：原点在几何中心，z 轴沿长轴指向 Ø95 封闭头。
// 能量单元绕长轴近似 3 重对称，绕 z 轴的转角无法可靠观测。
// x、y 轴的方向没有意义，使用方只能用原点和 z 轴。
class EnergyTraceComponent final
    : public rmcs_executor::Component
    , public rclcpp::Node {
public:
    EnergyTraceComponent()
        : Node(
              get_component_name(),
              rclcpp::NodeOptions{}.automatically_declare_parameters_from_overrides(true)) {
        register_output("/energy_trace/is_pose_valid", is_pose_valid_, false);
        register_output("/energy_trace/pose", pose_, Eigen::Isometry3d::Identity());

        mesh_path_ = get_parameter_or<std::string>("mesh_path", "");
        // 相机参数没有默认值：缺少时构造失败，避免用错误的内参静默运行
        camera_ = geometry::Camera{
            .width = static_cast<int>(get_parameter("camera.width").as_int()),
            .height = static_cast<int>(get_parameter("camera.height").as_int()),
            .fx = get_parameter("camera.fx").as_double(),
            .fy = get_parameter("camera.fy").as_double(),
            .cx = get_parameter("camera.cx").as_double(),
            .cy = get_parameter("camera.cy").as_double(),
            .depth_scale = get_parameter("camera.depth_scale").as_double(),
        };
        RCLCPP_INFO(
            get_logger(), "energy trace component created, mesh_path='%s'", mesh_path_.c_str());
    }

    void update() override {
        // 检测、配准和跟踪尚未接入，暂时始终输出无效
        *is_pose_valid_ = false;
    }

private:
    OutputInterface<bool> is_pose_valid_;
    OutputInterface<Eigen::Isometry3d> pose_;

    std::string mesh_path_;
    geometry::Camera camera_;
};

} // namespace rmcs_energy_trace

PLUGINLIB_EXPORT_CLASS(rmcs_energy_trace::EnergyTraceComponent, rmcs_executor::Component)
