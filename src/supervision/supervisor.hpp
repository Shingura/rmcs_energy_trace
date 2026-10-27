#pragma once

#include <cstddef>
#include <optional>
#include <string_view>

#include <eigen3/Eigen/Geometry>

#include "registration/depth_check.hpp"

namespace rmcs_energy_trace::supervision {

// 跟踪监督：每帧根据深度检查、位姿跳变和检测框判断跟踪结果是否可信，决定何时重新初始化。
// 只做判断，不做渲染；深度检查由调用方用 registration::DepthCheck 完成。
// 设计和阈值参考 RCIA_Benmao_Vision 的 m3t_tracking_health_gate 与 simple_tracking_lifecycle
// （文章许可证 CC BY-NC-SA 4.0）。各模块在同一进程内同步调用，所以省去了其中等待异步节点的状态
class Supervisor {
public:
    struct Config {
        // 未被遮挡的像素中，空闲空间违例（mesh 处看到了更远的背景）的比例超过该值判为不健康。
        // 遮挡不算失败：抓取时夹爪会挡住能量单元
        double max_free_space_ratio = 0.15;
        // 未被遮挡的有效像素少于该值时，这一帧无法判断（例如几乎全被挡住，或离开视野）
        std::size_t min_unoccluded_pixels = 200;
        // mesh 两端中较小的画面内面积比例低于该值时，这一帧无法判断。
        // 轴向位置只能靠两端约束，一端出了画面，跟踪器会沿轴线滑动；
        // 深度检查只统计画面内的像素，看不出来，遮挡检查有意不看支持率，也拦不住
        double min_end_in_image_ratio = 0.8;
        // 相邻两帧的位姿跳变上限，沿用 Benmao 的 8 cm、30°。只看中心和长轴，绕长轴的转角没有意义
        double max_jump_translation = 0.08;
        double max_jump_axis_deg = 30.;
        // 有检测框时，mesh 投影框与检测框的 IoU 下限，沿用 Benmao 的 0.2
        double min_detection_iou = 0.2;
        // 自上一个健康帧起，不健康的帧数达到该值时重新初始化，沿用 Benmao 的 5
        int bad_frames = 5;
        // 自上一个健康帧起，没有健康帧（不健康或无法判断）的帧数达到该值时也重新初始化，
        // 约 1 s（30 帧/s），处理目标离开视野。只有健康帧会清零这两个计数
        int no_good_frames = 30;
    };

    enum class State { kSearching, kTracking };
    enum class Health { kGood, kBad, kUnknown };

    struct Observation {
        std::optional<Eigen::Isometry3d> pose; // 跟踪器的结果 (mesh -> camera)，跟踪失败时为空
        // pose 处的深度检查。ROI 用 DepthCheck::projected_box，不用掩码
        registration::DepthCheck::Quality depth;
        // 这一帧运行了检测时，mesh 投影框与检测框的 IoU
        std::optional<double> detection_iou;
        // pose 处 mesh 两端中较小的画面内面积比例 (DepthCheck::end_in_image_ratio)，不给则不检查
        std::optional<double> end_in_image_ratio;
    };

    struct Decision {
        State state;
        Health health;
        bool trusted;            // 这一帧的位姿可以交给机械臂
        bool reinitialize;       // 需要重新初始化：调用方运行检测、分割、配准后调用 start
        std::string_view reason; // 不健康或无法判断的原因，健康时为空
    };

    Supervisor() noexcept = default;
    explicit Supervisor(const Config& config) noexcept;

    // 配准通过后调用，进入跟踪状态
    void start(const Eigen::Isometry3d& pose) noexcept;
    // 放弃当前目标，回到搜索状态
    void reset() noexcept;
    // 每帧跟踪后调用一次
    Decision update(const Observation& observation) noexcept;

    [[nodiscard]] State state() const noexcept { return state_; }

private:
    // jumped：位姿相对上一帧跳变。跳变帧的位姿不能作为下一帧的比较基准，否则跳变后的第二帧会被当作正常
    Health judge(
        const Observation& observation, std::string_view& reason, bool& jumped) const noexcept;

    Config config_;
    State state_ = State::kSearching;
    Eigen::Isometry3d last_pose_ = Eigen::Isometry3d::Identity();
    int bad_frames_ = 0;     // 自上一个健康帧起不健康的帧数
    int no_good_frames_ = 0; // 自上一个健康帧起没有健康帧的帧数
};

} // namespace rmcs_energy_trace::supervision
