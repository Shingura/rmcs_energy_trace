#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <limits>
#include <numbers>
#include <optional>
#include <print>
#include <random>
#include <string>
#include <vector>

#include <eigen3/Eigen/Geometry>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "geometry/mesh.hpp"
#include "pipeline/pipeline.hpp"
#include "pipeline/reference_monitor.hpp"

#if !defined(RMCS_ENERGY_UNIT_MESH) || !defined(RMCS_ENERGY_TRACE_ASSET_DIR)
# error "RMCS_ENERGY_UNIT_MESH and RMCS_ENERGY_TRACE_ASSET_DIR must be defined"
#endif

// 离线回放：在 render_sequence.py 渲染的连续 RGB-D 序列上运行整条位姿估计流程，
// 输出叠加画面的视频、逐帧 CSV 和相对真值的误差汇总

namespace fs = std::filesystem;
using rmcs_energy_trace::geometry::Camera;
using rmcs_energy_trace::pipeline::Pipeline;
using rmcs_energy_trace::pipeline::ReferenceMonitor;
using Clock = std::chrono::steady_clock;

namespace {

constexpr double kRadToDeg = 180. / std::numbers::pi;
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
// 深度图随机置为无效的像素比例
constexpr double kDropoutRatio = 0.005;
constexpr std::uint32_t kNoiseSeed = 20261027;
// 错误接受：可信但误差超过该阈值
constexpr double kWrongCenterMm = 10.;
constexpr double kWrongAxisDeg = 5.;

struct Options {
    fs::path sequence_dir;
    fs::path output_dir;
    std::string network;
    std::string mask = "network";
    std::string sam_encoder, sam_decoder;
    double fps = 30.;
    double depth_noise = 1.5;
    int frames = 0;   // 0 表示全部
    bool video = true;
};

struct TruthUnit {
    int index;
    Eigen::Isometry3d pose;
};

struct Sequence {
    Camera camera;
    int target = 0;
    int frames = 0;
    double fps = 30.; // 序列的帧率，没有时按 30
};

// 一帧的记录，对应 frames.csv 的一行
struct Row {
    int frame = 0;
    bool processed = false;
    std::string state;
    bool trusted = false;
    bool stable = false;
    std::string phase;           // 序列的阶段（hold、approach、grasp、extract），没有时为空
    bool has_pose = false;
    int unit = -1;
    bool is_target = false;
    double center_mm = kNaN, axis_deg = kNaN;
    double mask_iou = kNaN;      // 选中的掩码与匹配单元的真值可见区域
    double mask_iou_best = kNaN; // 与重合最多的单元，没有位姿的帧也记录
    std::size_t detections = 0;
    bool attempted = false, registered = false, tracker_ok = false;
    std::string health, reason;
    double detection_iou = kNaN, support = kNaN, free_space = kNaN;
    double end_in_image = kNaN;
    Pipeline::Timing timing;
    // 参考位姿监视器，meta 里没有 camera 时不运行，monitor 为空
    std::string monitor;        // no_reference、established、grasped、moved
    bool attached = false;      // meta 的 attached：目标已被夹住
    bool reference_set = false; // 这一帧建立了参考
    bool compared = false;      // 这一帧参与了比较
    bool consistent = false;
    int inconsistent_count = 0;
    double ref_center_mm = kNaN, ref_axis_deg = kNaN;
    double displacement_mm = kNaN, along_axis_mm = kNaN;
};

auto median(std::vector<double> values) -> double {
    if (values.empty())
        return kNaN;
    std::ranges::sort(values);
    return values[values.size() / 2];
}

auto percentile(std::vector<double> values, double p) -> double {
    if (values.empty())
        return kNaN;
    std::ranges::sort(values);
    return values[static_cast<std::size_t>(
        std::lround(p * static_cast<double>(values.size() - 1)))];
}

auto max_of(const std::vector<double>& values) -> double {
    return values.empty() ? kNaN : *std::ranges::max_element(values);
}

auto frame_name(int frame) -> std::string { return std::format("{:06d}", frame); }

auto cell(double value) -> std::string {
    return std::isnan(value) ? "" : std::format("{:.3f}", value);
}

auto parse_pose(const cv::FileNode& node) -> Eigen::Isometry3d {
    auto pose = Eigen::Isometry3d::Identity();
    auto r = 0;
    for (const auto& row : node) {
        auto c = 0;
        for (const auto& value : row)
            pose.matrix()(r, c++) = static_cast<double>(value.real());
        r++;
    }
    return pose;
}

auto load_sequence(const fs::path& dir) -> std::optional<Sequence> {
    auto storage = cv::FileStorage{(dir / "sequence.json").string(), cv::FileStorage::READ};
    if (!storage.isOpened())
        return std::nullopt;
    auto sequence = Sequence{};
    const auto camera = storage["camera"];
    sequence.camera = {
        .width = static_cast<int>(camera["width"].real()),
        .height = static_cast<int>(camera["height"].real()),
        .fx = camera["fx"].real(),
        .fy = camera["fy"].real(),
        .cx = camera["cx"].real(),
        .cy = camera["cy"].real(),
        .depth_scale = camera["depth_scale"].real(),
    };
    sequence.target = static_cast<int>(storage["target"].real());
    sequence.frames = static_cast<int>(storage["frames"].real());
    if (const auto fps = storage["fps"]; !fps.empty() && fps.real() > 0.)
        sequence.fps = fps.real();
    return sequence;
}

auto load_truth(const fs::path& dir, int frame) -> std::vector<TruthUnit> {
    auto units = std::vector<TruthUnit>{};
    auto storage = cv::FileStorage{
        (dir / "meta" / (frame_name(frame) + ".json")).string(), cv::FileStorage::READ};
    if (!storage.isOpened())
        return units;
    for (const auto& unit : storage["units"]) {
        units.push_back({
            .index = static_cast<int>(unit["index"].real()),
            .pose = parse_pose(unit["pose"]),
        });
    }
    return units;
}

// meta 里的相机（OpenCV 坐标系）到世界的位姿，没有时返回空
auto load_camera_pose(const fs::path& dir, int frame) -> std::optional<Eigen::Isometry3d> {
    auto storage = cv::FileStorage{
        (dir / "meta" / (frame_name(frame) + ".json")).string(), cv::FileStorage::READ};
    if (!storage.isOpened())
        return std::nullopt;
    const auto camera = storage["camera"];
    if (!camera.isSeq() || camera.size() != 4)
        return std::nullopt;
    return parse_pose(camera);
}

// meta 里的 attached（目标已被夹住），没有时视为未夹住
auto load_attached(const fs::path& dir, int frame) -> bool {
    auto storage = cv::FileStorage{
        (dir / "meta" / (frame_name(frame) + ".json")).string(), cv::FileStorage::READ};
    if (!storage.isOpened())
        return false;
    const auto attached = storage["attached"];
    return !attached.empty() && static_cast<int>(attached.real()) != 0;
}

// 按规划渲染的序列，meta 里有阶段；内置轨迹的序列没有，返回空字符串
auto load_phase(const fs::path& dir, int frame) -> std::string {
    auto storage = cv::FileStorage{
        (dir / "meta" / (frame_name(frame) + ".json")).string(), cv::FileStorage::READ};
    if (!storage.isOpened())
        return {};
    const auto phase = storage["phase"];
    return phase.isString() ? phase.string() : std::string{};
}

// 深度噪声：标准差 = noise_mm * z^2（mm，z 单位 m），另有一部分像素置为无效。
// 种子由帧号决定，同一帧的噪声不随跳帧而变
auto add_depth_noise(const cv::Mat& depth, const Camera& camera, double noise_mm, int frame)
    -> cv::Mat {
    auto noisy = depth.clone();
    if (noise_mm <= 0.)
        return noisy;
    auto engine = std::mt19937{kNoiseSeed + static_cast<std::uint32_t>(frame)};
    auto normal = std::normal_distribution<double>{};
    auto dropout = std::bernoulli_distribution{kDropoutRatio};
    const auto to_unit = 1e-3 / camera.depth_scale; // mm -> 深度图单位
    for (int v = 0; v < noisy.rows; v++) {
        auto* row = noisy.ptr<std::uint16_t>(v);
        for (int u = 0; u < noisy.cols; u++) {
            if (row[u] == 0)
                continue;
            if (dropout(engine)) {
                row[u] = 0;
                continue;
            }
            const auto z = row[u] * camera.depth_scale;
            const auto value = row[u] + normal(engine) * noise_mm * z * z * to_unit;
            row[u] = static_cast<std::uint16_t>(std::clamp(std::lround(value), 1L, 65535L));
        }
    }
    return noisy;
}

auto axis_error_deg(const Eigen::Isometry3d& a, const Eigen::Isometry3d& b) -> double {
    return std::acos(std::clamp(a.linear().col(2).dot(b.linear().col(2)), -1., 1.)) * kRadToDeg;
}

auto mask_iou(const cv::Mat& mask, const cv::Mat& visible) -> double {
    auto both = cv::Mat{}, either = cv::Mat{};
    cv::bitwise_and(mask, visible, both);
    cv::bitwise_or(mask, visible, either);
    const auto total = cv::countNonZero(either);
    return total == 0 ? kNaN : static_cast<double>(cv::countNonZero(both)) / total;
}

// mesh 按位姿投影后所有顶点的凸包，作为轮廓
auto outline(
    const std::vector<Eigen::Vector3f>& vertices, const Camera& camera,
    const Eigen::Isometry3d& pose) -> std::vector<cv::Point> {
    const Eigen::Matrix3f rotation = pose.linear().cast<float>();
    const Eigen::Vector3f translation = pose.translation().cast<float>();
    auto points = std::vector<cv::Point2f>{};
    points.reserve(vertices.size());
    for (const auto& vertex : vertices) {
        const Eigen::Vector3f p = rotation * vertex + translation;
        if (p.z() <= 0.01f)
            continue;
        points.emplace_back(
            static_cast<float>(camera.fx) * p.x() / p.z() + static_cast<float>(camera.cx),
            static_cast<float>(camera.fy) * p.y() / p.z() + static_cast<float>(camera.cy));
    }
    auto hull = std::vector<cv::Point2f>{};
    if (!points.empty())
        cv::convexHull(points, hull);
    return {hull.begin(), hull.end()};
}

void put_text(cv::Mat& image, const std::string& text, int line) {
    const auto origin = cv::Point{6, 16 + 16 * line};
    cv::putText(image, text, origin, cv::FONT_HERSHEY_SIMPLEX, 0.5, {0, 0, 0}, 3, cv::LINE_AA);
    cv::putText(
        image, text, origin, cv::FONT_HERSHEY_SIMPLEX, 0.5, {255, 255, 255}, 1, cv::LINE_AA);
}

auto draw_overlay(
    const cv::Mat& color, const Pipeline::Result& result, const Row& row,
    const std::vector<Eigen::Vector3f>& vertices, const Camera& camera,
    const std::optional<Eigen::Isometry3d>& truth) -> cv::Mat {
    auto image = color.clone();
    const auto& diag = result.diagnostics;
    if (!diag.mask.empty()) {
        auto layer = image.clone();
        layer.setTo(cv::Scalar{255, 0, 255}, diag.mask);
        cv::addWeighted(layer, 0.4, image, 0.6, 0., image);
    }
    for (std::size_t i = 0; i < diag.detections.size(); i++) {
        const auto selected = diag.selected && *diag.selected == i;
        cv::rectangle(
            image, cv::Rect2i{diag.detections[i].box},
            selected ? cv::Scalar{0, 140, 255} : cv::Scalar{255, 200, 0}, selected ? 3 : 1);
    }
    if (truth) {
        const auto hull = outline(vertices, camera, *truth);
        cv::polylines(
            image, std::vector<std::vector<cv::Point>>{hull}, true, {255, 255, 255}, 1,
            cv::LINE_AA);
    }
    if (result.pose) {
        const auto hull = outline(vertices, camera, *result.pose);
        cv::polylines(
            image, std::vector<std::vector<cv::Point>>{hull}, true,
            result.trusted ? cv::Scalar{0, 220, 0} : cv::Scalar{0, 255, 255}, 2, cv::LINE_AA);
    }
    put_text(
        image,
        std::format(
            "#{} {} {} {}", row.frame,
            result.state == Pipeline::State::kTracking ? "tracking" : "searching",
            result.trusted ? "trusted" : "untrusted", result.stable ? "stable" : "unstable"),
        0);
    if (row.has_pose) {
        put_text(
            image,
            std::format(
                "{} err {:.1f} mm {:.1f} deg", row.is_target ? "target" : "other unit",
                row.center_mm, row.axis_deg),
            1);
    }
    put_text(image, std::format("{:.0f} ms", row.timing.total_ms), 2);
    if (!row.phase.empty())
        put_text(image, std::format("phase {}", row.phase), 3);
    if (!row.monitor.empty()) {
        auto text = std::format("ref {}", row.monitor);
        if (row.compared) {
            text += std::format(
                " {} {:.1f} mm {:.1f} deg n={}", row.consistent ? "ok" : "off", row.ref_center_mm,
                row.ref_axis_deg, row.inconsistent_count);
        }
        if (!std::isnan(row.displacement_mm)) {
            text += std::format(
                " disp {:.1f} mm axis {:.1f} mm", row.displacement_mm, row.along_axis_mm);
        }
        put_text(image, text, 4);
    }
    return image;
}

void write_csv(const fs::path& path, const std::vector<Row>& rows) {
    auto out = std::ofstream{path};
    out << "frame,processed,state,trusted,stable,phase,unit,is_target,center_mm,axis_deg,mask_iou,"
           "mask_iou_best,"
           "detections,attempted,registered,tracker_ok,health,reason,detection_iou,support,"
           "free_space,end_in_image,detect_ms,segment_ms,register_ms,track_ms,check_ms,total_ms,"
           "monitor,attached,reference_set,compared,consistent,inconsistent_count,ref_center_mm,"
           "ref_axis_deg,displacement_mm,along_axis_mm\n";
    for (const auto& r : rows) {
        out << r.frame << ',' << r.processed << ',' << r.state << ',' << r.trusted << ','
            << r.stable << ',' << r.phase << ',' << (r.unit >= 0 ? std::to_string(r.unit) : "")
            << ',' << (r.has_pose ? r.is_target : 0) << ',' << cell(r.center_mm) << ','
            << cell(r.axis_deg) << ',' << cell(r.mask_iou) << ',' << cell(r.mask_iou_best) << ','
            << r.detections << ',' << r.attempted << ',' << r.registered << ',' << r.tracker_ok
            << ',' << r.health << ',' << r.reason << ',' << cell(r.detection_iou) << ','
            << cell(r.support) << ',' << cell(r.free_space) << ',' << cell(r.end_in_image) << ','
            << cell(r.timing.detect_ms) << ',' << cell(r.timing.segment_ms) << ','
            << cell(r.timing.register_ms) << ',' << cell(r.timing.track_ms) << ','
            << cell(r.timing.check_ms) << ',' << cell(r.timing.total_ms) << ',' << r.monitor << ','
            << r.attached << ',' << r.reference_set << ',' << r.compared << ',' << r.consistent
            << ',' << (r.monitor.empty() ? "" : std::to_string(r.inconsistent_count)) << ','
            << cell(r.ref_center_mm) << ',' << cell(r.ref_axis_deg) << ','
            << cell(r.displacement_mm) << ',' << cell(r.along_axis_mm) << '\n';
    }
}

// 一个阶段内的统计
auto phase_summary(const std::string& phase, const std::vector<Row>& rows) -> std::string {
    auto processed = 0, trusted = 0, unknown = 0, wrong = 0;
    auto centers = std::vector<double>{}, axes = std::vector<double>{};
    for (const auto& r : rows) {
        if (!r.processed || r.phase != phase)
            continue;
        processed++;
        trusted += r.trusted;
        unknown += r.health == "unknown";
        if (r.trusted && r.has_pose) {
            centers.push_back(r.center_mm);
            axes.push_back(r.axis_deg);
            wrong += r.center_mm > kWrongCenterMm || r.axis_deg > kWrongAxisDeg;
        }
    }
    return std::format(
        "phase {}: processed {} trusted {} ({:.1f}%) unknown {} | trusted center mm median {:.2f} "
        "max {:.2f} | axis deg median {:.2f} max {:.2f} | wrong accepts {}\n",
        phase, processed, trusted, processed > 0 ? 100. * trusted / processed : 0., unknown,
        median(centers), max_of(centers), median(axes), max_of(axes), wrong);
}

// 参考位姿监视器和盲区的统计。meta 里有 camera 才有监视器；盲区只需要阶段
auto monitor_summary(const Sequence& sequence, const std::vector<Row>& rows) -> std::string {
    auto text = std::string{};
    const auto has_monitor =
        std::ranges::any_of(rows, [](const Row& r) { return !r.monitor.empty(); });
    auto phases = std::vector<std::string>{};
    auto grasp_start = -1, last_trusted = -1, last_trusted_before_grasp = -1;
    for (const auto& r : rows) {
        if (!r.phase.empty() && std::ranges::find(phases, r.phase) == phases.end())
            phases.push_back(r.phase);
        if (r.phase == "grasp" && grasp_start < 0)
            grasp_start = r.frame;
    }
    for (const auto& r : rows) {
        if (!r.processed || !r.trusted)
            continue;
        last_trusted = r.frame;
        if (grasp_start >= 0 && r.frame < grasp_start)
            last_trusted_before_grasp = r.frame;
    }

    if (has_monitor) {
        auto established = std::string{}, moved = std::string{};
        for (const auto& r : rows) {
            if (r.reference_set)
                established += std::format("{}{}", established.empty() ? "" : ", ", r.frame);
            if (r.monitor == "moved")
                moved += std::format("{}{}", moved.empty() ? "" : ", ", r.frame);
        }
        text += std::format(
            "reference established at frame {}\n", established.empty() ? "none" : established);
        if (moved.empty()) {
            text += "target moved: no\n";
        } else {
            text += std::format(
                "target moved at frame {} (the replay calls reset() after each, so the reference "
                "is re-established at the next stable frame)\n",
                moved);
        }
        for (const auto& phase : phases) {
            auto compared = 0, inconsistent = 0;
            for (const auto& r : rows) {
                if (r.phase != phase || !r.compared)
                    continue;
                compared++;
                inconsistent += !r.consistent;
            }
            text += std::format(
                "phase {}: monitor compared {} inconsistent {}\n", phase, compared, inconsistent);
        }
    } else {
        text += "reference monitor not run (no camera pose in meta)\n";
    }

    if (grasp_start >= 0) {
        if (last_trusted_before_grasp >= 0) {
            text += std::format(
                "last trusted frame before grasp {} is {:.2f} s before grasp starts (frame {})\n",
                last_trusted_before_grasp, (grasp_start - last_trusted_before_grasp) / sequence.fps,
                grasp_start);
        } else {
            text += std::format("no trusted frame before grasp starts (frame {})\n", grasp_start);
        }
        if (last_trusted >= grasp_start) {
            text += std::format(
                "last trusted frame overall {} is {:.2f} s after grasp starts\n", last_trusted,
                (last_trusted - grasp_start) / sequence.fps);
        }
    }

    // extract 阶段最后一个可信帧相对参考的位移
    if (has_monitor && std::ranges::find(phases, "extract") != phases.end()) {
        const Row* last = nullptr;
        for (const auto& r : rows) {
            if (r.processed && r.trusted && r.phase == "extract")
                last = &r;
        }
        if (last == nullptr) {
            text += "extract: no trusted frame\n";
        } else if (std::isnan(last->displacement_mm)) {
            text += std::format(
                "extract: last trusted frame {} has no reference to compare\n", last->frame);
        } else {
            text += std::format(
                "extract: last trusted frame {} displacement {:.1f} mm, along axis {:.1f} mm\n",
                last->frame, last->displacement_mm, last->along_axis_mm);
        }
    }
    return text;
}

auto summarize(const Options& options, const Sequence& sequence, const std::vector<Row>& rows)
    -> std::string {
    auto processed = 0, skipped = 0, attempts = 0, successes = 0, reinitialize = 0;
    auto first_trusted = -1, first_stable = -1, motion_start = -1;
    auto phases = std::vector<std::string>{}; // 按出现顺序
    auto after = 0, after_trusted = 0, wrong = 0, trusted_other = 0;
    auto centers = std::vector<double>{}, axes = std::vector<double>{};
    auto sum = Pipeline::Timing{};
    for (const auto& r : rows) {
        if (!r.phase.empty()) {
            if (std::ranges::find(phases, r.phase) == phases.end())
                phases.push_back(r.phase);
            if (r.phase != "hold" && motion_start < 0)
                motion_start = r.frame;
        }
        if (!r.processed) {
            skipped++;
            continue;
        }
        processed++;
        attempts += r.attempted;
        successes += r.registered;
        reinitialize += r.health == "reinitialize";
        sum.detect_ms += r.timing.detect_ms;
        sum.segment_ms += r.timing.segment_ms;
        sum.register_ms += r.timing.register_ms;
        sum.track_ms += r.timing.track_ms;
        sum.check_ms += r.timing.check_ms;
        sum.total_ms += r.timing.total_ms;
        if (r.trusted && first_trusted < 0)
            first_trusted = r.frame;
        if (r.stable && first_stable < 0)
            first_stable = r.frame;
        if (first_trusted >= 0) {
            after++;
            after_trusted += r.trusted;
        }
        if (r.trusted && r.has_pose) {
            centers.push_back(r.center_mm);
            axes.push_back(r.axis_deg);
            wrong += r.center_mm > kWrongCenterMm || r.axis_deg > kWrongAxisDeg;
            trusted_other += !r.is_target;
        }
    }
    const auto n = std::max(processed, 1);
    auto text = std::format(
        "sequence {} | mask {} | fps {} | depth noise {} mm/m^2\n", options.sequence_dir.string(),
        options.mask, options.fps, options.depth_noise);
    text += std::format("frames processed {} skipped {}\n", processed, skipped);
    text += std::format(
        "first trusted frame {}\n", first_trusted >= 0 ? std::to_string(first_trusted) : "none");
    if (first_stable >= 0) {
        text += std::format(
            "first stable frame {} ({:.2f} s at {} fps)\n", first_stable,
            first_stable / sequence.fps, sequence.fps);
    } else {
        text += "first stable frame none\n";
    }
    if (!phases.empty()) {
        if (motion_start >= 0) {
            text += std::format(
                "motion starts at frame {} ({:.2f} s)\n", motion_start,
                motion_start / sequence.fps);
            if (first_stable >= 0) {
                const auto lead = (motion_start - first_stable) / sequence.fps;
                text += std::format(
                    "stable {} motion start by {:.2f} s\n", lead >= 0. ? "before" : "after",
                    std::abs(lead));
            } else {
                text += "stable never reached before the sequence ended\n";
            }
        } else {
            text += "motion never starts\n";
        }
    }
    text += std::format("registration attempts {} successes {}\n", attempts, successes);
    text += std::format("reinitializations {}\n", reinitialize);
    text += std::format(
        "trusted after first trusted {}/{} ({:.1f}%)\n", after_trusted, after,
        after > 0 ? 100. * after_trusted / after : 0.);
    text += std::format(
        "trusted center error mm: median {:.2f} p90 {:.2f} max {:.2f}\n", median(centers),
        percentile(centers, 0.9), max_of(centers));
    text += std::format(
        "trusted axis error deg: median {:.2f} p90 {:.2f} max {:.2f}\n", median(axes),
        percentile(axes, 0.9), max_of(axes));
    text += std::format(
        "wrong accepts (trusted, center > {:.0f} mm or axis > {:.0f} deg) {}; trusted frames on a "
        "non-target unit {}\n",
        kWrongCenterMm, kWrongAxisDeg, wrong, trusted_other);
    text += std::format(
        "mean ms per processed frame: detect {:.1f} segment {:.1f} register {:.1f} track {:.1f} "
        "check {:.1f} total {:.1f}\n",
        sum.detect_ms / n, sum.segment_ms / n, sum.register_ms / n, sum.track_ms / n,
        sum.check_ms / n, sum.total_ms / n);
    for (const auto& phase : phases)
        text += phase_summary(phase, rows);
    text += monitor_summary(sequence, rows);
    return text;
}

auto parse_options(int argc, char* argv[]) -> std::optional<Options> {
    auto options = Options{};
    auto positional = std::vector<std::string>{};
    for (int i = 1; i < argc; i++) {
        const auto arg = std::string{argv[i]};
        if (arg == "--network" && i + 1 < argc) {
            options.network = argv[++i];
        } else if (arg == "--mask" && i + 1 < argc) {
            options.mask = argv[++i];
        } else if (arg == "--sam" && i + 2 < argc) {
            options.sam_encoder = argv[++i];
            options.sam_decoder = argv[++i];
        } else if (arg == "--fps" && i + 1 < argc) {
            options.fps = std::strtod(argv[++i], nullptr);
        } else if (arg == "--depth-noise" && i + 1 < argc) {
            options.depth_noise = std::strtod(argv[++i], nullptr);
        } else if (arg == "--frames" && i + 1 < argc) {
            options.frames = static_cast<int>(std::strtol(argv[++i], nullptr, 10));
        } else if (arg == "--no-video") {
            options.video = false;
        } else {
            positional.push_back(arg);
        }
    }
    if (positional.size() != 2 || options.network.empty()
        || (options.mask != "network" && options.mask != "sam" && options.mask != "none")
        || (options.mask == "sam" && options.sam_encoder.empty())) {
        return std::nullopt;
    }
    options.sequence_dir = positional[0];
    options.output_dir = positional[1];
    return options;
}

} // namespace

auto main(int argc, char* argv[]) -> int {
    const auto options = parse_options(argc, argv);
    if (!options) {
        std::println(
            "Usage: sequence_replay <sequence_dir> <output_dir> --network <network.xml> "
            "[--mask network|sam|none] [--sam <encoder.onnx> <decoder.onnx>] [--fps 30] "
            "[--depth-noise 1.5] [--frames N] [--no-video]");
        return EXIT_FAILURE;
    }
    const auto sequence = load_sequence(options->sequence_dir);
    if (!sequence) {
        std::println(stderr, "Cannot read {}/sequence.json", options->sequence_dir.string());
        return EXIT_FAILURE;
    }
    const auto total =
        options->frames > 0 ? std::min(options->frames, sequence->frames) : sequence->frames;

    auto config = Pipeline::Config{};
    config.mesh_path = RMCS_ENERGY_UNIT_MESH;
    config.camera = sequence->camera;
    config.detector.network_path = options->network;
    config.segmenter = {.encoder_path = options->sam_encoder, .decoder_path = options->sam_decoder};
    config.tracker.model_dir = RMCS_ENERGY_TRACE_ASSET_DIR;
    config.mask_source = options->mask == "sam"  ? Pipeline::MaskSource::kSam
                       : options->mask == "none" ? Pipeline::MaskSource::kNone
                                                 : Pipeline::MaskSource::kNetwork;
    auto pipeline = Pipeline{};
    if (auto initialized = pipeline.initialize(config); !initialized) {
        std::println(stderr, "{}", initialized.error());
        return EXIT_FAILURE;
    }

    // 叠加画面用的 mesh 顶点
    auto mesh = rmcs_energy_trace::geometry::load_binary_stl(RMCS_ENERGY_UNIT_MESH);
    if (!mesh) {
        std::println(stderr, "{}", mesh.error());
        return EXIT_FAILURE;
    }
    auto vertices = std::vector<Eigen::Vector3f>{};
    for (const auto& triangle : mesh->triangles)
        vertices.insert(vertices.end(), triangle.begin(), triangle.end());

    fs::create_directories(options->output_dir);
    auto writer = cv::VideoWriter{};
    auto frames_dir = fs::path{};
    if (options->video) {
        const auto video_path = (options->output_dir / "replay.mp4").string();
        writer.open(
            video_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 30.,
            {sequence->camera.width, sequence->camera.height});
        if (!writer.isOpened()) {
            frames_dir = options->output_dir / "frames";
            fs::create_directories(frames_dir);
            std::println(
                "Cannot open {} for writing, saving JPEGs to {}", video_path, frames_dir.string());
        }
    }

    auto monitor = ReferenceMonitor{};
    auto rows = std::vector<Row>{};
    for (int frame = 0; frame < total;) {
        auto row = Row{};
        row.frame = frame;
        const auto color =
            cv::imread((options->sequence_dir / "color" / (frame_name(frame) + ".jpg")).string());
        const auto depth = cv::imread(
            (options->sequence_dir / "depth" / (frame_name(frame) + ".png")).string(),
            cv::IMREAD_UNCHANGED);
        const auto index = cv::imread(
            (options->sequence_dir / "index" / (frame_name(frame) + ".png")).string(),
            cv::IMREAD_GRAYSCALE); // 渲染脚本写的是 3 通道，三个通道相同
        if (color.empty() || depth.empty() || index.empty()) {
            std::println(stderr, "Cannot read frame {}", frame);
            return EXIT_FAILURE;
        }
        const auto noisy = add_depth_noise(depth, sequence->camera, options->depth_noise, frame);
        const auto truth = load_truth(options->sequence_dir, frame);

        // 相机位姿；已知目标位置时，重新初始化按它选检测框
        const auto camera = load_camera_pose(options->sequence_dir, frame);
        auto target = std::optional<Eigen::Isometry3d>{};
        if (camera && monitor.target())
            target = camera->inverse() * *monitor.target();

        const auto begin = Clock::now();
        const auto processed = pipeline.process(color, noisy, target);
        const auto seconds = std::chrono::duration<double>{Clock::now() - begin}.count();
        if (!processed) {
            std::println(stderr, "Frame {}: {}", frame, processed.error());
            return EXIT_FAILURE;
        }
        const auto& result = *processed;
        const auto& diag = result.diagnostics;

        row.processed = true;
        row.state = result.state == Pipeline::State::kTracking ? "tracking" : "searching";
        row.trusted = result.trusted;
        row.stable = result.stable;
        row.phase = load_phase(options->sequence_dir, frame);
        row.detections = diag.detections.size();
        row.attempted = diag.registration_attempted;
        row.registered = diag.registration.has_value();
        row.tracker_ok = diag.tracker_ok;
        row.timing = diag.timing;
        if (diag.decision) {
            const auto health = diag.decision->health;
            row.health =
                diag.decision->reinitialize ? "reinitialize"
                : health == rmcs_energy_trace::supervision::Supervisor::Health::kGood ? "good"
                : health == rmcs_energy_trace::supervision::Supervisor::Health::kBad  ? "bad"
                                                                                      : "unknown";
            row.reason = std::string{diag.decision->reason};
        }
        if (diag.observation) {
            if (diag.observation->detection_iou)
                row.detection_iou = *diag.observation->detection_iou;
            row.support = diag.observation->depth.support_ratio;
            row.free_space = diag.observation->depth.free_space_ratio;
            if (diag.observation->end_in_image_ratio)
                row.end_in_image = *diag.observation->end_in_image_ratio;
        }

        // 参考位姿监视：以世界系作底座系，meta 里的相机位姿就是相机到底座
        if (camera) {
            row.attached = load_attached(options->sequence_dir, frame);
            const auto output = monitor.update(result, *camera, row.attached);
            using MonitorState = ReferenceMonitor::State;
            row.monitor = output.state == MonitorState::kEstablished ? "established"
                        : output.state == MonitorState::kGrasped     ? "grasped"
                        : output.state == MonitorState::kMoved       ? "moved"
                                                                     : "no_reference";
            row.reference_set = output.state == MonitorState::kEstablished && !output.comparison
                             && output.displacement;
            row.inconsistent_count = output.inconsistent_count;
            if (output.comparison) {
                row.compared = true;
                row.consistent = output.comparison->consistent;
                row.ref_center_mm = output.comparison->center_distance * 1e3;
                row.ref_axis_deg = output.comparison->axis_deg;
            }
            if (output.displacement) {
                row.displacement_mm = output.displacement->total * 1e3;
                row.along_axis_mm = output.displacement->along_axis * 1e3;
            }
            // 目标被移动：模拟上层停臂、原地重新初始化，之后在下一个 stable 的帧重新建立参考
            if (output.state == MonitorState::kMoved)
                monitor.reset();
        }

        // 与真值比较：中心最近的真值单元
        auto target_pose = std::optional<Eigen::Isometry3d>{};
        for (const auto& unit : truth) {
            if (unit.index == sequence->target)
                target_pose = unit.pose;
        }
        if (result.pose && !truth.empty()) {
            const auto nearest = std::ranges::min_element(truth, {}, [&](const TruthUnit& unit) {
                return (unit.pose.translation() - result.pose->translation()).norm();
            });
            row.has_pose = true;
            row.unit = nearest->index;
            row.is_target = nearest->index == sequence->target;
            row.center_mm = (nearest->pose.translation() - result.pose->translation()).norm() * 1e3;
            row.axis_deg = axis_error_deg(nearest->pose, *result.pose);
            if (!diag.mask.empty())
                row.mask_iou = mask_iou(diag.mask, index == nearest->index);
        }
        if (!diag.mask.empty()) {
            // 与掩码重合像素最多的单元
            auto best = 0.;
            for (const auto& unit : truth) {
                const auto value = mask_iou(diag.mask, index == unit.index);
                if (!std::isnan(value) && value > best)
                    best = value;
            }
            row.mask_iou_best = best;
        }

        if (options->video) {
            const auto image =
                draw_overlay(color, result, row, vertices, sequence->camera, target_pose);
            if (writer.isOpened())
                writer.write(image);
            else
                cv::imwrite((frames_dir / (frame_name(frame) + ".jpg")).string(), image);
        }
        std::println(
            "#{:03} {:<9} {:<9} {} {:7.1f} ms", frame, row.state,
            row.trusted ? "trusted" : "untrusted",
            row.has_pose ? std::format("{:6.1f} mm {:5.1f} deg", row.center_mm, row.axis_deg)
                         : std::string{"      no pose       "},
            seconds * 1e3);
        rows.push_back(row);

        // 处理一帧用了 seconds 秒，相机在这段时间里继续出帧，下一帧取最新的一帧
        const auto step = options->fps > 0.
                            ? std::max(1, static_cast<int>(std::ceil(seconds * options->fps)))
                            : 1;
        for (int skip = 1; skip < step && frame + skip < total; skip++) {
            auto& skipped = rows.emplace_back();
            skipped.frame = frame + skip;
            skipped.state = "skipped";
            skipped.phase = load_phase(options->sequence_dir, frame + skip);
        }
        frame += step;
    }

    write_csv(options->output_dir / "frames.csv", rows);
    const auto summary = summarize(*options, *sequence, rows);
    std::ofstream{options->output_dir / "summary.txt"} << summary;
    std::print("\n{}", summary);
    return EXIT_SUCCESS;
}
