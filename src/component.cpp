#include <string>

#include <eigen3/Eigen/Geometry>
#include <pluginlib/class_list_macros.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rmcs_executor/component.hpp>

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
        register_output("/energy_trace/pose_valid", pose_valid_, false);
        register_output("/energy_trace/pose", pose_, Eigen::Isometry3d::Identity());

        mesh_path_ = get_parameter_or<std::string>("mesh_path", "");
        RCLCPP_INFO(
            get_logger(), "energy trace component created, mesh_path='%s'", mesh_path_.c_str());
    }

    void update() override {
        // 检测、配准和跟踪尚未接入，暂时始终输出无效
        *pose_valid_ = false;
    }

private:
    OutputInterface<bool> pose_valid_;
    OutputInterface<Eigen::Isometry3d> pose_;

    std::string mesh_path_;
};

} // namespace rmcs_energy_trace

PLUGINLIB_EXPORT_CLASS(rmcs_energy_trace::EnergyTraceComponent, rmcs_executor::Component)
