#include "registration/registration.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <numbers>
#include <tuple>

#include <eigen3/Eigen/SVD>
#include <opencv2/imgproc.hpp>
#include <opencv2/surface_matching.hpp>

#include "geometry/depth_cloud.hpp"
#include "geometry/mesh.hpp"
#include "registration/point_to_plane.hpp"

#include <pcl/ModelCoefficients.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/registration/icp.h>
#include <pcl/segmentation/sac_segmentation.h>

namespace rmcs_energy_trace::registration {

namespace {

using Clock = std::chrono::steady_clock;
using Cloud = pcl::PointCloud<pcl::PointXYZ>;
using NormalCloud = pcl::PointCloud<pcl::PointNormal>;

double elapsed_ms(Clock::time_point since) noexcept {
    return std::chrono::duration<double, std::milli>{Clock::now() - since}.count();
}

Cloud::Ptr voxel_downsample(const Cloud::Ptr& input, float leaf) noexcept {
    auto output = std::make_shared<Cloud>();
    auto filter = pcl::VoxelGrid<pcl::PointXYZ>{};
    filter.setInputCloud(input);
    filter.setLeafSize(leaf, leaf, leaf);
    filter.filter(*output);
    return output;
}

// 场景点加上法线（朝向相机），丢掉估计不出法线的点
NormalCloud::Ptr with_normals(const Cloud::Ptr& cloud, float radius) noexcept {
    auto normals = pcl::PointCloud<pcl::Normal>{};
    auto estimation = pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal>{};
    estimation.setInputCloud(cloud);
    estimation.setSearchMethod(std::make_shared<pcl::search::KdTree<pcl::PointXYZ>>());
    estimation.setRadiusSearch(radius);
    estimation.setViewPoint(0.f, 0.f, 0.f);
    estimation.compute(normals);

    auto output = std::make_shared<NormalCloud>();
    for (std::size_t i = 0; i < cloud->size(); i++) {
        const Eigen::Vector3f normal = normals[i].getNormalVector3fMap();
        if (!normal.allFinite())
            continue;
        auto point = pcl::PointNormal{};
        point.getVector3fMap() = (*cloud)[i].getVector3fMap();
        point.getNormalVector3fMap() = normal;
        output->push_back(point);
    }
    return output;
}

// Möller–Trumbore：从 origin 沿 direction 的射线是否在 epsilon 之外与三角形相交
bool ray_hits(
    const Eigen::Vector3f& origin, const Eigen::Vector3f& direction,
    const geometry::Mesh::Triangle& triangle, float epsilon) noexcept {
    const Eigen::Vector3f edge1 = triangle[1] - triangle[0];
    const Eigen::Vector3f edge2 = triangle[2] - triangle[0];
    const Eigen::Vector3f h = direction.cross(edge2);
    const auto det = edge1.dot(h);
    if (std::abs(det) < 1e-12f)
        return false;
    const auto inverse = 1.f / det;
    const Eigen::Vector3f s = origin - triangle[0];
    const auto u = inverse * s.dot(h);
    if (u < 0.f || u > 1.f)
        return false;
    const Eigen::Vector3f q = s.cross(edge1);
    const auto v = inverse * direction.dot(q);
    if (v < 0.f || u + v > 1.f)
        return false;
    return inverse * edge2.dot(q) > epsilon;
}

// 只保留外表面的点：沿法线向外的射线再碰到 mesh 的点在内腔里。
// 开口端的孔壁斜着也能看到，但取矿时开口端套在柱子上，孔壁看不到；
// 而且孔深而窄，即使从开口端看，大部分孔壁也被挡住，留着只会干扰粗配准
NormalCloud::Ptr exterior_only(const geometry::Mesh& mesh, const NormalCloud& cloud) noexcept {
    constexpr auto kEpsilon = 1e-4f;
    auto keep = std::vector<std::uint8_t>(cloud.size(), 1);
#pragma omp parallel for default(none) shared(mesh, cloud, keep, kEpsilon) schedule(dynamic, 64)
    for (std::size_t i = 0; i < cloud.size(); i++) {
        const Eigen::Vector3f normal = cloud[i].getNormalVector3fMap();
        const Eigen::Vector3f origin = cloud[i].getVector3fMap() + kEpsilon * normal;
        for (const auto& triangle : mesh.triangles) {
            if (ray_hits(origin, normal, triangle, kEpsilon)) {
                keep[i] = 0;
                break;
            }
        }
    }
    auto output = std::make_shared<NormalCloud>();
    for (std::size_t i = 0; i < cloud.size(); i++) {
        if (keep[i] != 0)
            output->push_back(cloud[i]);
    }
    return output;
}

// OpenCV PPF 的点云格式：每行 x y z nx ny nz
cv::Mat to_ppf_mat(const NormalCloud& cloud) noexcept {
    auto mat = cv::Mat(static_cast<int>(cloud.size()), 6, CV_32F);
    for (std::size_t i = 0; i < cloud.size(); i++) {
        auto* row = mat.ptr<float>(static_cast<int>(i));
        const auto& point = cloud[i];
        std::ranges::copy(
            std::array{point.x, point.y, point.z, point.normal_x, point.normal_y, point.normal_z},
            row);
    }
    return mat;
}

constexpr float kNoDepth = std::numeric_limits<float>::max();

// 评价候选位姿所需的实测深度
struct DepthView {
    const cv::Mat& depth;
    const geometry::Camera& camera;
    cv::Rect2i roi;      // 已与图像求交
    const cv::Mat& mask; // 空表示不限制
};

// 把 mesh 的三角形按位姿 (mesh -> camera) 光栅化为 roi 内的 z-buffer (m)，
// 无覆盖处为 kNoDepth。像素 (u, v) 的中心位于投影坐标 (u, v)，与 depth_to_cloud 一致
cv::Mat rasterize_depth(
    const geometry::Mesh& mesh, const Eigen::Isometry3d& pose, const geometry::Camera& camera,
    const cv::Rect2i& roi) noexcept {
    auto buffer = cv::Mat{roi.height, roi.width, CV_32FC1, cv::Scalar{kNoDepth}};
    const Eigen::Matrix3f rotation = pose.linear().cast<float>();
    const Eigen::Vector3f translation = pose.translation().cast<float>();
    const auto fx = static_cast<float>(camera.fx);
    const auto fy = static_cast<float>(camera.fy);
    const auto cx = static_cast<float>(camera.cx);
    const auto cy = static_cast<float>(camera.cy);
    constexpr float kNearPlane = 0.01f;
    constexpr float kEdgeTolerance = 1e-4f;

    const auto edge = [](const Eigen::Vector2f& a, const Eigen::Vector2f& b,
                         const Eigen::Vector2f& c) {
        return (b.x() - a.x()) * (c.y() - a.y()) - (b.y() - a.y()) * (c.x() - a.x());
    };

    for (const auto& triangle : mesh.triangles) {
        auto vertices = std::array<Eigen::Vector3f, 3>{};
        auto projected = std::array<Eigen::Vector2f, 3>{};
        auto visible = true;
        for (std::size_t k = 0; k < 3; k++) {
            vertices[k] = rotation * triangle[k] + translation;
            if (vertices[k].z() < kNearPlane) {
                visible = false;
                break;
            }
            projected[k] = {
                fx * vertices[k].x() / vertices[k].z() + cx,
                fy * vertices[k].y() / vertices[k].z() + cy};
        }
        if (!visible)
            continue;

        const auto area = edge(projected[0], projected[1], projected[2]);
        if (std::abs(area) < 1e-9f)
            continue;

        const auto lo_x = std::min({projected[0].x(), projected[1].x(), projected[2].x()});
        const auto hi_x = std::max({projected[0].x(), projected[1].x(), projected[2].x()});
        const auto lo_y = std::min({projected[0].y(), projected[1].y(), projected[2].y()});
        const auto hi_y = std::max({projected[0].y(), projected[1].y(), projected[2].y()});
        const auto x0 = std::max(roi.x, static_cast<int>(std::ceil(lo_x)));
        const auto x1 = std::min(roi.x + roi.width - 1, static_cast<int>(std::floor(hi_x)));
        const auto y0 = std::max(roi.y, static_cast<int>(std::ceil(lo_y)));
        const auto y1 = std::min(roi.y + roi.height - 1, static_cast<int>(std::floor(hi_y)));

        for (int v = y0; v <= y1; v++) {
            auto* row = buffer.ptr<float>(v - roi.y);
            for (int u = x0; u <= x1; u++) {
                const Eigen::Vector2f pixel{static_cast<float>(u), static_cast<float>(v)};
                const auto w0 = edge(projected[1], projected[2], pixel) / area;
                const auto w1 = edge(projected[2], projected[0], pixel) / area;
                const auto w2 = 1.f - w0 - w1;
                if (w0 < -kEdgeTolerance || w1 < -kEdgeTolerance || w2 < -kEdgeTolerance)
                    continue;
                // 透视下 1/z 在屏幕空间线性插值
                const auto inverse_z =
                    w0 / vertices[0].z() + w1 / vertices[1].z() + w2 / vertices[2].z();
                const auto z = 1.f / inverse_z;
                auto& cell = row[u - roi.x];
                cell = std::min(cell, z);
            }
        }
    }
    return buffer;
}

} // namespace

struct Registration::Impl {
    Config config;
    bool initialized = false;

    geometry::Mesh mesh;
    // 精配准与内点率使用的外表面点云（2 mm 间距）
    NormalCloud::Ptr refine_model;
    pcl::KdTreeFLANN<pcl::PointXYZ> refine_tree;
    // OpenCV 的 match() 不是 const，通过指针在 const 成员函数中调用
    std::unique_ptr<cv::ppf_match_3d::PPF3DDetector> detector;

    std::expected<void, std::string>
        initialize(const std::string& stl_path, const Config& new_config) noexcept {
        config = new_config;
        initialized = false;

        auto loaded = geometry::load_binary_stl(stl_path);
        if (!loaded.has_value())
            return std::unexpected{loaded.error()};
        mesh = std::move(*loaded);

        const auto sampled = geometry::sample_surface(mesh, config.model_sample_count, 42);
        // 体素下采样 PointNormal，体素内法线取平均后归一化，再去掉内腔的点
        const auto exterior = [&](float leaf) {
            auto output = NormalCloud{};
            auto voxel = pcl::VoxelGrid<pcl::PointNormal>{};
            voxel.setInputCloud(sampled);
            voxel.setLeafSize(leaf, leaf, leaf);
            voxel.filter(output);
            for (auto& point : output)
                point.getNormalVector3fMap().normalize();
            return exterior_only(mesh, output);
        };
        const auto coarse_model = exterior(config.model_voxel_size);
        refine_model = exterior(config.refine_voxel_size);
        if (coarse_model->empty() || refine_model->empty())
            return std::unexpected{"Mesh has no exterior surface: " + stl_path};

        auto refine_xyz = std::make_shared<Cloud>();
        for (const auto& point : *refine_model)
            refine_xyz->emplace_back(point.x, point.y, point.z);
        refine_tree.setInputCloud(refine_xyz);

        try {
            detector = std::make_unique<cv::ppf_match_3d::PPF3DDetector>(
                config.ppf_sampling_step, config.ppf_distance_step, config.ppf_angle_count);
            detector->trainModel(to_ppf_mat(*coarse_model));
        } catch (const cv::Exception& error) {
            return std::unexpected{std::string{"PPF training failed: "} + error.what()};
        }

        initialized = true;
        return {};
    }

    Cloud::Ptr remove_outliers(const Cloud::Ptr& cloud) const noexcept {
        auto filtered = std::make_shared<Cloud>();
        auto outlier = pcl::StatisticalOutlierRemoval<pcl::PointXYZ>{};
        outlier.setInputCloud(cloud);
        outlier.setMeanK(config.outlier_mean_k);
        outlier.setStddevMulThresh(config.outlier_stddev_multiplier);
        outlier.filter(*filtered);
        return filtered;
    }

    // 返回待尝试的场景点云。去平面是为了去掉框里的地面、墙等背景：
    // 有掩码时背景已经去掉，点云里的大平面只能是能量单元自己的端面，不去平面；
    // 只用框时先试去掉主平面的，再试原始的（主平面也可能是端面）
    std::vector<Cloud::Ptr> preprocess(const Cloud::Ptr& scene, bool masked) const noexcept {
        auto cloud = voxel_downsample(scene, config.scene_voxel_size);
        if (cloud->size() < config.min_scene_points)
            return {};
        const auto whole = remove_outliers(cloud);
        if (masked)
            return {whole};

        auto coefficients = std::make_shared<pcl::ModelCoefficients>();
        auto inliers = std::make_shared<pcl::PointIndices>();
        auto segmentation = pcl::SACSegmentation<pcl::PointXYZ>{};
        segmentation.setModelType(pcl::SACMODEL_PLANE);
        segmentation.setMethodType(pcl::SAC_RANSAC);
        segmentation.setDistanceThreshold(config.plane_distance_threshold);
        segmentation.setMaxIterations(config.plane_max_iterations);
        segmentation.setInputCloud(cloud);
        segmentation.segment(*inliers, *coefficients);
        if (static_cast<float>(inliers->indices.size())
            <= config.plane_min_inlier_ratio * static_cast<float>(cloud->size()))
            return {whole};

        auto remaining = std::make_shared<Cloud>();
        auto extract = pcl::ExtractIndices<pcl::PointXYZ>{};
        extract.setInputCloud(cloud);
        extract.setIndices(inliers);
        extract.setNegative(true);
        extract.filter(*remaining);
        if (remaining->size() < config.min_scene_points)
            return {whole};
        return {remove_outliers(remaining), whole};
    }

    // PPF 粗配准：按票数排序的前 ppf_candidates 个位姿 (mesh -> camera)
    std::vector<Eigen::Isometry3d> coarse_align(const NormalCloud& scene) const noexcept {
        auto matches = std::vector<cv::ppf_match_3d::Pose3DPtr>{};
        try {
            detector->match(
                to_ppf_mat(scene), matches, config.ppf_scene_sample_step,
                config.ppf_scene_distance);
        } catch (const cv::Exception&) {
            return {};
        }

        auto poses = std::vector<Eigen::Isometry3d>{};
        for (const auto& match : matches) {
            if (poses.size() >= config.ppf_candidates)
                break;
            auto pose = Eigen::Isometry3d::Identity();
            for (int r = 0; r < 4; r++) {
                for (int c = 0; c < 4; c++)
                    pose.matrix()(r, c) = match->pose(r, c);
            }
            if (!pose.matrix().allFinite())
                continue;
            // 位姿聚类时对旋转取平均，结果不严格正交，投影回最近的旋转矩阵
            const auto svd = Eigen::JacobiSVD<Eigen::Matrix3d>{
                pose.linear(), Eigen::ComputeFullU | Eigen::ComputeFullV};
            const Eigen::Matrix3d rotation = svd.matrixU() * svd.matrixV().transpose();
            if (rotation.determinant() < 0.)
                continue;
            pose.linear() = rotation;
            poses.push_back(pose);
        }
        return poses;
    }

    // 对应距离逐级收紧的 point-to-plane ICP。场景点先按 pose 变换到 mesh 坐标系，
    // ICP 求出把它们贴到 mesh 上的修正量，返回修正后的位姿 (mesh -> camera)
    Eigen::Isometry3d refine(
        const Cloud& scene, const Eigen::Isometry3d& pose,
        const std::vector<float>& max_correspondences) const noexcept {
        const Eigen::Matrix4f camera_to_mesh = pose.inverse().matrix().cast<float>();
        auto source = std::make_shared<NormalCloud>();
        for (const auto& point : scene) {
            auto item = pcl::PointNormal{};
            item.getVector3fMap() = (camera_to_mesh * point.getVector4fMap()).head<3>();
            source->push_back(item);
        }

        Eigen::Matrix4f correction = Eigen::Matrix4f::Identity();
        for (const auto distance : max_correspondences) {
            auto icp = pcl::IterativeClosestPoint<pcl::PointNormal, pcl::PointNormal>{};
            icp.setTransformationEstimation(std::make_shared<PointToPlaneEstimation>());
            icp.setInputSource(source);
            icp.setInputTarget(refine_model);
            icp.setMaximumIterations(config.icp_max_iterations);
            icp.setMaxCorrespondenceDistance(distance);
            icp.setTransformationEpsilon(1e-9);
            auto aligned = NormalCloud{};
            icp.align(aligned);
            if (!icp.hasConverged())
                continue;
            correction = icp.getFinalTransformation() * correction;
            source = std::make_shared<NormalCloud>(aligned);
        }
        const Eigen::Matrix4f mesh_to_camera = (correction * camera_to_mesh).inverse();
        return Eigen::Isometry3d{Eigen::Matrix4d{mesh_to_camera.cast<double>()}};
    }

    // 场景点中离 mesh 小于 inlier_distance 的比例
    double inlier_ratio(const Cloud& scene, const Eigen::Isometry3d& pose) const noexcept {
        const Eigen::Matrix4f camera_to_mesh = pose.inverse().matrix().cast<float>();
        auto inliers = std::size_t{};
        auto indices = std::vector<int>(1);
        auto distances = std::vector<float>(1);
        for (const auto& point : scene) {
            const Eigen::Vector3f p = (camera_to_mesh * point.getVector4fMap()).head<3>();
            refine_tree.nearestKSearch(pcl::PointXYZ{p.x(), p.y(), p.z()}, 1, indices, distances);
            if (distances[0] < config.inlier_distance * config.inlier_distance)
                inliers++;
        }
        return static_cast<double>(inliers) / static_cast<double>(scene.size());
    }

    // 把 mesh 渲染到候选位姿，与实测深度逐像素比较。
    // 掩码只决定一个像素能否算作支持：掩码外是别的东西，深度相同或更近都说明不了位姿对错，
    // 不统计；但掩码外看到比 mesh 更远的背景，说明 mesh 所在处是空的，仍记为违例
    Quality assess(const DepthView& view, const Eigen::Isometry3d& pose) const noexcept {
        auto quality = Quality{};
        const auto buffer = rasterize_depth(mesh, pose, view.camera, view.roi);
        auto silhouette = cv::Mat{};
        cv::compare(buffer, kNoDepth, silhouette, cv::CMP_LT);
        if (config.silhouette_erosion_px > 0) {
            const auto size = 2 * config.silhouette_erosion_px + 1;
            cv::erode(
                silhouette, silhouette,
                cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size{size, size}));
        }

        auto valid = std::size_t{}, supported = std::size_t{}, free_space = std::size_t{},
             occluded = std::size_t{}, missing = std::size_t{};
        for (int v = 0; v < view.roi.height; v++) {
            const auto* model_row = buffer.ptr<float>(v);
            const auto* silhouette_row = silhouette.ptr<std::uint8_t>(v);
            const auto* depth_row = view.depth.ptr<std::uint16_t>(v + view.roi.y);
            const auto* mask_row =
                view.mask.empty() ? nullptr : view.mask.ptr<std::uint8_t>(v + view.roi.y);
            for (int u = 0; u < view.roi.width; u++) {
                if (silhouette_row[u] == 0)
                    continue;
                const auto in_mask = mask_row == nullptr || mask_row[u + view.roi.x] != 0;
                const auto measured = depth_row[u + view.roi.x];
                if (in_mask) {
                    quality.model_pixels++;
                    if (measured == 0)
                        missing++;
                }
                if (measured == 0)
                    continue;
                const auto difference =
                    static_cast<float>(measured * view.camera.depth_scale) - model_row[u];
                if (difference > config.depth_tolerance) {
                    free_space++;
                    valid++;
                } else if (in_mask) {
                    valid++;
                    difference < -config.depth_tolerance ? occluded++ : supported++;
                }
            }
        }
        if (quality.model_pixels > 0) {
            quality.missing_ratio =
                static_cast<double>(missing) / static_cast<double>(quality.model_pixels);
        }
        if (valid > 0) {
            const auto count = static_cast<double>(valid);
            quality.support_ratio = static_cast<double>(supported) / count;
            quality.free_space_ratio = static_cast<double>(free_space) / count;
            quality.occluded_ratio = static_cast<double>(occluded) / count;
        }
        return quality;
    }

    // 绕自身长轴转一个周期（120°），取深度检查最好的转角。中心和长轴不变
    Eigen::Isometry3d
        best_roll(const DepthView& view, const Eigen::Isometry3d& pose) const noexcept {
        auto best = pose;
        auto best_score = -std::numeric_limits<double>::infinity();
        for (int i = 0; i < config.roll_samples; i++) {
            const auto angle =
                2. * std::numbers::pi / 3. * static_cast<double>(i) / config.roll_samples;
            auto candidate = pose;
            candidate.linear() =
                pose.linear()
                * Eigen::AngleAxisd{angle, Eigen::Vector3d::UnitZ()}.toRotationMatrix();
            const auto quality = assess(view, candidate);
            const auto score = quality.support_ratio - 2. * quality.free_space_ratio;
            if (score > best_score) {
                best_score = score;
                best = candidate;
            }
        }
        return best;
    }

    bool acceptable(const Result& candidate) const noexcept {
        const auto& quality = candidate.quality;
        return candidate.inlier_ratio >= config.min_inlier_ratio
            && quality.model_pixels >= config.min_model_pixels
            && quality.support_ratio >= config.min_support_ratio
            && quality.free_space_ratio <= config.max_free_space_ratio;
    }

    // 候选之间的比较：先看是否通过验收，再比分数。
    // 分数：场景被 mesh 解释得越多、mesh 被场景支持得越多、空闲空间违例越少越好
    bool better(const Result& a, const Result& b) const noexcept {
        const auto score = [](const Result& r) {
            return r.inlier_ratio + r.quality.support_ratio - 2. * r.quality.free_space_ratio;
        };
        return std::tuple{acceptable(a), score(a)} > std::tuple{acceptable(b), score(b)};
    }

    // 每个 PPF 位姿依次做 ICP、绕长轴扫描、再做 ICP；三个阶段的位姿都参与比较
    std::optional<Result>
        align(const Cloud::Ptr& scene, const DepthView& view, Timing& timing) const noexcept {
        auto stage = Clock::now();
        const auto scene_normals = with_normals(scene, config.normal_radius);
        timing.normal_ms += elapsed_ms(stage);

        stage = Clock::now();
        const auto poses = coarse_align(*scene_normals);
        timing.coarse_ms += elapsed_ms(stage);

        auto best = std::optional<Result>{};
        const auto consider = [&](const Eigen::Isometry3d& pose) {
            const auto begin = Clock::now();
            const auto candidate = Result{
                .pose = pose,
                .inlier_ratio = inlier_ratio(*scene, pose),
                .quality = assess(view, pose),
                .timing = {}};
            timing.quality_ms += elapsed_ms(begin);
            if (!best.has_value() || better(candidate, *best))
                best = candidate;
        };
        for (const auto& coarse : poses) {
            stage = Clock::now();
            const auto refined = refine(*scene, coarse, config.icp_max_correspondences);
            const auto rolled =
                refine(*scene, best_roll(view, refined), config.roll_icp_max_correspondences);
            timing.icp_ms += elapsed_ms(stage);
            for (const auto& pose : {coarse, refined, rolled})
                consider(pose);
        }
        if (!best.has_value() || !acceptable(*best))
            return std::nullopt;
        return best;
    }

    std::optional<Result> register_depth(
        const cv::Mat& depth, const geometry::Camera& camera, const cv::Rect2i& roi,
        const cv::Mat& mask) const noexcept {
        if (!initialized || depth.type() != CV_16UC1)
            return std::nullopt;
        const auto rect = roi & cv::Rect2i{0, 0, depth.cols, depth.rows};
        if (rect.empty())
            return std::nullopt;

        const auto begin = Clock::now();
        auto timing = Timing{};
        // 掩码类型或尺寸不符时点云为空，preprocess 会因点数不足而拒绝
        const auto cloud =
            geometry::depth_to_cloud(depth, camera, roi, mask, config.mask_erosion_px);
        const auto candidates = preprocess(cloud, !mask.empty());
        timing.preprocess_ms = elapsed_ms(begin);

        const auto view = DepthView{depth, camera, rect, mask};
        for (const auto& candidate : candidates) {
            auto result = align(candidate, view, timing);
            if (result.has_value()) {
                timing.total_ms = elapsed_ms(begin);
                result->timing = timing;
                return result;
            }
        }
        return std::nullopt;
    }
};

std::expected<void, std::string>
    Registration::initialize(const std::string& stl_path, const Config& config) noexcept {
    return pimpl_->initialize(stl_path, config);
}

std::expected<void, std::string> Registration::initialize(const std::string& stl_path) noexcept {
    return pimpl_->initialize(stl_path, Config{});
}

std::optional<Registration::Result> Registration::register_depth(
    const cv::Mat& depth, const geometry::Camera& camera, const cv::Rect2i& roi,
    const cv::Mat& mask) const noexcept {
    return pimpl_->register_depth(depth, camera, roi, mask);
}

Registration::Registration() noexcept
    : pimpl_{std::make_unique<Impl>()} {}

Registration::~Registration() noexcept = default;

} // namespace rmcs_energy_trace::registration
