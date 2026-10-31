#pragma once

#include <optional>

#include <eigen3/Eigen/Geometry>

#include "pipeline/pipeline.hpp"

namespace rmcs_energy_trace::pipeline {

// 参考位姿监视：取矿流程里，目标在夹住之前静止在中心块上。第一次 stable 时把位姿换到底座系，
// 存为参考位姿，机械臂按它运动；之后每个可信帧与参考比较，连续多帧不一致就判为目标被移动。
// 判为被移动后保持 kMoved，直到上层调用 reset()：上层要在 kMoved 时停臂、原地重新初始化，
// 不能因为下一帧自动建立了新参考而漏掉这个事件。
// 夹爪合拢后（update 的 grasped 为 true）单元偏离参考是预期结果，不再比较、不会判为被移动，
// 只输出相对参考的位移，用来确认单元随夹爪被拉出。
//
// 接近末段夹爪会遮挡目标，M3T 可能漂移、重新配准也会失败，所以不能“跟丢就停下”：
// 不可信的帧是弱证据，不参与比较，不计入也不清零不一致计数。
// 流程内部重新配准后的可信位姿照样参与比较。
//
// 另外记录目标最近一次可信的位姿（target()），上层在流程重新初始化时用它按位置选检测框：
// 机械臂接近、抓取、拉出的过程中相机不再对准目标，离画面中心最近的可能是相邻单元。
// 用最近一次可信的位姿而不用参考位姿：夹住之后单元随夹爪被拉出，参考位姿处已经没有单元了。
//
// 只做判断，不调用其他模块；只用到 Pipeline::Result 的数据，每帧调用一次 update
class ReferenceMonitor {
public:
    struct Config {
        // 与参考位姿比较：中心距离（m）不超过该值，且长轴夹角不超过 max_axis_deg 为一致。
        // 参考和当前位姿各有误差：单元静止时，差值的 99% 分位约为中心 3.3 mm、长轴 4.3°（合成序列回放）
        double max_center_distance = 0.005;
        double max_axis_deg = 6.;
        // 连续这么多个可信帧不一致，判为目标被移动
        int inconsistent_frames = 5;
    };

    enum class State {
        kNoReference, // 还没有建立参考（等待第一次 stable），或目标被移动后已清除
        kEstablished, // 已建立参考
        kMoved,       // 判为目标被移动，参考已清除。保持到 reset()，期间不建立参考、不比较
        kGrasped,     // 已夹住：有参考，只输出位移
    };

    // 这一帧可信位姿与参考的比较
    struct Comparison {
        double center_distance = 0.; // m
        double axis_deg = 0.;        // 长轴夹角，0°～180°
        bool consistent = false;
    };

    // 可信位姿相对参考的位移（底座系）
    struct Displacement {
        double total = 0.;      // m，中心距离
        double along_axis = 0.; // m，沿参考长轴的分量，沿 +z（指向封闭端）为正
    };

    struct Output {
        State state = State::kNoReference;
        // 底座系下的参考位姿 (mesh -> base)；kNoReference 和 kMoved 时为空
        std::optional<Eigen::Isometry3d> reference;
        // 这一帧参与了比较时给出。建立参考的那一帧、不可信的帧、已夹住和没有参考时为空；
        // 处于 kMoved 时一直是判为被移动的那一次比较
        std::optional<Comparison> comparison;
        // 自上一个一致的可信帧起，连续不一致的可信帧数。处于 kMoved 时为 inconsistent_frames
        int inconsistent_count = 0;
        // 有参考的可信帧给出，建立参考的那一帧为 0。不可信的帧和 kMoved 时为空
        std::optional<Displacement> displacement;
    };

    ReferenceMonitor() noexcept = default;
    explicit ReferenceMonitor(const Config& config) noexcept;

    // 清除参考、计数和 kMoved，回到 kNoReference。上层处理完目标被移动、重新初始化时调用
    void reset() noexcept;

    // 每帧调用一次。camera_to_base 为相机（OpenCV 坐标系）到底座的位姿；
    // grasped 为夹爪已合拢（抓取开始之后）。没有参考时 grasped 的帧不建立参考
    Output update(
        const Pipeline::Result& result, const Eigen::Isometry3d& camera_to_base,
        bool grasped = false) noexcept;

    // 最近一次 update 的状态（reset 后为 kNoReference）
    [[nodiscard]] State state() const noexcept { return state_; }

    // 有参考时，目标最近一次可信的底座系位姿 (mesh -> base)，不可信的帧不改变它；
    // 没有参考、判为被移动和 reset 后为空
    [[nodiscard]] const std::optional<Eigen::Isometry3d>& target() const noexcept {
        return target_;
    }

private:
    Config config_;
    State state_ = State::kNoReference;
    std::optional<Eigen::Isometry3d> reference_;
    std::optional<Eigen::Isometry3d> target_; // 最近一次可信的位姿
    std::optional<Comparison> moved_comparison_; // 判为被移动的那一次比较
    int inconsistent_count_ = 0;
};

} // namespace rmcs_energy_trace::pipeline
