#pragma once

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>

namespace rmcs_energy_trace::grasp_sim {

// 没有 ros2_control 时，由本节点独占 /joint_states：保持当前关节角并按轨迹时间插值回放
class JointStatePlayer {
public:
    JointStatePlayer(
        rclcpp::Node& node, std::vector<std::string> names, std::vector<double> initial,
        double speed)
        : names_{std::move(names)}
        , positions_{std::move(initial)}
        , velocities_(names_.size(), 0.)
        , speed_{speed}
        , publisher_{node.create_publisher<sensor_msgs::msg::JointState>("/joint_states", 10)}
        , clock_{node.get_clock()} {
        timer_ = node.create_wall_timer(std::chrono::milliseconds{20}, [this] { tick(); });
    }

    auto positions() const -> std::vector<double> {
        const auto lock = std::scoped_lock{mutex_};
        return positions_;
    }

    // 直接跳到给定关节位置（用于重播前复位）
    void reset(const std::vector<double>& positions) {
        const auto lock = std::scoped_lock{mutex_};
        positions_ = positions;
        std::ranges::fill(velocities_, 0.);
    }

    // 阻塞回放，直到轨迹结束
    void play(const trajectory_msgs::msg::JointTrajectory& trajectory) {
        if (trajectory.points.empty())
            return;
        auto lock = std::unique_lock{mutex_};
        active_ = Active{trajectory, Clock::now(), map_joints(trajectory)};
        done_.wait(lock, [this] { return !active_.has_value(); });
    }

private:
    using Clock = std::chrono::steady_clock;

    struct Active {
        trajectory_msgs::msg::JointTrajectory trajectory;
        Clock::time_point start;
        std::vector<std::size_t> index; // 轨迹关节 -> names_ 下标
    };

    auto map_joints(const trajectory_msgs::msg::JointTrajectory& trajectory) const
        -> std::vector<std::size_t> {
        auto index = std::vector<std::size_t>{};
        for (const auto& name : trajectory.joint_names) {
            index.push_back(
                static_cast<std::size_t>(std::ranges::find(names_, name) - names_.begin()));
        }
        return index;
    }

    static auto seconds(const builtin_interfaces::msg::Duration& d) -> double {
        return static_cast<double>(d.sec) + static_cast<double>(d.nanosec) * 1e-9;
    }

    void tick() {
        auto message = sensor_msgs::msg::JointState{};
        {
            const auto lock = std::scoped_lock{mutex_};
            if (active_.has_value())
                advance();
            message.name = names_;
            message.position = positions_;
            message.velocity = velocities_;
        }
        message.header.stamp = clock_->now();
        publisher_->publish(message);
    }

    // 调用时已持有锁
    void advance() {
        const auto& points = active_->trajectory.points;
        const auto elapsed =
            std::chrono::duration<double>{Clock::now() - active_->start}.count() * speed_;

        auto apply = [&](std::size_t i, double blend, std::size_t j) {
            for (std::size_t n = 0; n < active_->index.size(); n++) {
                const auto k = active_->index[n];
                if (k >= positions_.size())
                    continue;
                positions_[k] =
                    (1. - blend) * points[i].positions[n] + blend * points[j].positions[n];
            }
        };

        if (elapsed >= seconds(points.back().time_from_start)) {
            apply(points.size() - 1, 0., points.size() - 1);
            std::ranges::fill(velocities_, 0.);
            active_.reset();
            done_.notify_all();
            return;
        }
        auto j = std::size_t{1};
        while (j < points.size() - 1 && seconds(points[j].time_from_start) < elapsed)
            j++;
        const auto t0 = seconds(points[j - 1].time_from_start);
        const auto t1 = seconds(points[j].time_from_start);
        const auto blend = t1 > t0 ? std::clamp((elapsed - t0) / (t1 - t0), 0., 1.) : 1.;
        apply(j - 1, blend, j);
        for (std::size_t n = 0; n < active_->index.size(); n++) {
            const auto k = active_->index[n];
            if (k >= velocities_.size() || t1 <= t0)
                continue;
            velocities_[k] =
                (points[j].positions[n] - points[j - 1].positions[n]) / (t1 - t0) * speed_;
        }
    }

    std::vector<std::string> names_;
    std::vector<double> positions_;
    std::vector<double> velocities_;
    double speed_;
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr publisher_;
    rclcpp::Clock::SharedPtr clock_;
    rclcpp::TimerBase::SharedPtr timer_;

    mutable std::mutex mutex_;
    std::condition_variable done_;
    std::optional<Active> active_;
};

} // namespace rmcs_energy_trace::grasp_sim
