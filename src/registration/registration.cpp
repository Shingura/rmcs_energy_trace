#include "registration/registration.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>

#include <opencv2/imgproc.hpp>

#include "geometry/mesh.hpp"

#include <pcl/ModelCoefficients.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/sample_consensus_prerejective.h>
#include <pcl/registration/transformation_estimation_point_to_plane_lls.h>
#include <pcl/segmentation/sac_segmentation.h>

namespace rmcs_energy_trace::registration {

namespace {

using Clock = std::chrono::steady_clock;
using Cloud = Registration::Cloud;
using NormalCloud = pcl::PointCloud<pcl::PointNormal>;
using Fpfh = pcl::PointCloud<pcl::FPFHSignature33>;

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

Fpfh::Ptr compute_fpfh(
    const Cloud::Ptr& cloud, const pcl::PointCloud<pcl::Normal>::Ptr& normals,
    float radius) noexcept {
    auto features = std::make_shared<Fpfh>();
    auto fpfh = pcl::FPFHEstimationOMP<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33>{};
    fpfh.setInputCloud(cloud);
    fpfh.setInputNormals(normals);
    fpfh.setSearchMethod(std::make_shared<pcl::search::KdTree<pcl::PointXYZ>>());
    fpfh.setRadiusSearch(radius);
    fpfh.compute(*features);
    return features;
}

pcl::PointCloud<pcl::Normal>::Ptr compute_normals(const Cloud::Ptr& cloud, float radius) noexcept {
    auto normals = std::make_shared<pcl::PointCloud<pcl::Normal>>();
    auto nest = pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal>{};
    nest.setInputCloud(cloud);
    nest.setSearchMethod(std::make_shared<pcl::search::KdTree<pcl::PointXYZ>>());
    nest.setRadiusSearch(radius);
    // 视点为相机原点，法线朝向相机
    nest.setViewPoint(0.f, 0.f, 0.f);
    nest.compute(*normals);
    return normals;
}

} // namespace

struct Registration::Impl {
    Config config;
    bool initialized = false;

    Cloud::Ptr model_cloud;
    pcl::PointCloud<pcl::Normal>::Ptr model_normals;
    Fpfh::Ptr model_features;
    // 精配准与验收使用更密的模型点云
    NormalCloud::Ptr refine_model;
    pcl::KdTreeFLANN<pcl::PointXYZ> refine_tree;

    std::expected<void, std::string>
        initialize(const std::string& stl_path, const Config& new_config) noexcept {
        config = new_config;

        auto mesh = geometry::load_binary_stl(stl_path);
        if (!mesh.has_value())
            return std::unexpected{mesh.error()};

        const auto sampled = geometry::sample_surface(*mesh, config.model_sample_count, 42);

        // 体素下采样 PointNormal，体素内法线取平均后归一化
        const auto downsample = [&](float leaf) {
            auto output = std::make_shared<NormalCloud>();
            auto voxel = pcl::VoxelGrid<pcl::PointNormal>{};
            voxel.setInputCloud(sampled);
            voxel.setLeafSize(leaf, leaf, leaf);
            voxel.filter(*output);
            for (auto& point : *output) {
                point.getNormalVector3fMap().normalize();
            }
            return output;
        };
        const auto downsampled = downsample(config.model_voxel_size);
        refine_model = downsample(config.refine_voxel_size);

        model_cloud = std::make_shared<Cloud>();
        model_normals = std::make_shared<pcl::PointCloud<pcl::Normal>>();
        for (const auto& point : *downsampled) {
            const Eigen::Vector3f normal = point.getNormalVector3fMap().normalized();
            model_cloud->emplace_back(point.x, point.y, point.z);
            model_normals->emplace_back(normal.x(), normal.y(), normal.z());
        }

        model_features = compute_fpfh(model_cloud, model_normals, config.feature_radius);
        auto refine_xyz = std::make_shared<Cloud>();
        for (const auto& point : *refine_model) {
            refine_xyz->emplace_back(point.x, point.y, point.z);
        }
        refine_tree.setInputCloud(refine_xyz);

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

    // 返回待尝试的场景候选：先是未去平面的，若存在主平面则再追加去平面后的。
    // 物体自身的大平面也可能被 RANSAC 当作主平面，所以不能无条件去除。
    std::vector<Cloud::Ptr> preprocess(const Cloud& scene) const noexcept {
        auto candidates = std::vector<Cloud::Ptr>{};

        auto cloud = voxel_downsample(std::make_shared<Cloud>(scene), config.scene_voxel_size);
        if (cloud->size() < config.min_scene_points)
            return candidates;
        candidates.push_back(remove_outliers(cloud));

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
            > config.plane_min_inlier_ratio * static_cast<float>(cloud->size())) {
            auto remaining = std::make_shared<Cloud>();
            auto extract = pcl::ExtractIndices<pcl::PointXYZ>{};
            extract.setInputCloud(cloud);
            extract.setIndices(inliers);
            extract.setNegative(true);
            extract.filter(*remaining);
            if (remaining->size() >= config.min_scene_points)
                candidates.push_back(remove_outliers(remaining));
        }
        return candidates;
    }

    // FPFH + SampleConsensusPrerejective，返回 scene -> model
    std::optional<Eigen::Matrix4f>
        coarse_align(const Cloud::Ptr& scene, const Fpfh::Ptr& features) const noexcept {
        auto coarse =
            pcl::SampleConsensusPrerejective<pcl::PointXYZ, pcl::PointXYZ, pcl::FPFHSignature33>{};
        coarse.setInputSource(scene);
        coarse.setSourceFeatures(features);
        coarse.setInputTarget(model_cloud);
        coarse.setTargetFeatures(model_features);
        coarse.setMaximumIterations(config.ransac_iterations);
        coarse.setNumberOfSamples(3);
        coarse.setCorrespondenceRandomness(config.ransac_correspondence_k);
        coarse.setSimilarityThreshold(config.ransac_similarity_threshold);
        coarse.setMaxCorrespondenceDistance(config.ransac_inlier_threshold);
        coarse.setInlierFraction(config.ransac_inlier_fraction);
        auto aligned = Cloud{};
        coarse.align(aligned);
        if (!coarse.hasConverged())
            return std::nullopt;
        return Eigen::Matrix4f{coarse.getFinalTransformation()};
    }

    // 对应距离逐级收紧的 point-to-plane ICP，返回 scene -> model
    Eigen::Matrix4f refine(const Cloud::Ptr& scene, const Eigen::Matrix4f& initial) const noexcept {
        auto source = std::make_shared<NormalCloud>();
        for (const auto& point : *scene) {
            auto item = pcl::PointNormal{};
            item.getVector3fMap() = (initial * point.getVector4fMap()).head<3>();
            source->push_back(item);
        }

        Eigen::Matrix4f scene_to_model = initial;
        for (const auto distance : config.icp_max_correspondences) {
            auto icp = pcl::IterativeClosestPoint<pcl::PointNormal, pcl::PointNormal>{};
            icp.setTransformationEstimation(
                std::make_shared<pcl::registration::TransformationEstimationPointToPlaneLLS<
                    pcl::PointNormal, pcl::PointNormal>>());
            icp.setInputSource(source);
            icp.setInputTarget(refine_model);
            icp.setMaximumIterations(config.icp_max_iterations);
            icp.setMaxCorrespondenceDistance(distance);
            icp.setTransformationEpsilon(1e-9);
            auto aligned = NormalCloud{};
            icp.align(aligned);
            if (!icp.hasConverged())
                continue;
            scene_to_model = icp.getFinalTransformation() * scene_to_model;
            source = std::make_shared<NormalCloud>(aligned);
        }
        return scene_to_model;
    }

    // 场景点变换到模型系后到模型最近点的距离统计
    void evaluate(const Cloud::Ptr& scene, const Eigen::Matrix4f& scene_to_model, Result& result)
        const noexcept {
        auto inliers = std::size_t{};
        auto sum_sq = 0.;
        auto indices = std::vector<int>(1);
        auto dists = std::vector<float>(1);
        for (const auto& point : *scene) {
            const Eigen::Vector3f p = (scene_to_model * point.getVector4fMap()).head<3>();
            refine_tree.nearestKSearch(pcl::PointXYZ{p.x(), p.y(), p.z()}, 1, indices, dists);
            if (dists[0] < config.inlier_distance * config.inlier_distance) {
                inliers++;
                sum_sq += dists[0];
            }
        }
        result.inlier_ratio = static_cast<double>(inliers) / static_cast<double>(scene->size());
        result.fitness = inliers == 0 ? 1e9 : sum_sq / static_cast<double>(inliers);

        const Eigen::Matrix4f model_to_scene = scene_to_model.inverse();
        result.pose = Eigen::Isometry3d{Eigen::Matrix4d{model_to_scene.cast<double>()}};
    }

    // 以场景为 source、模型为 target：场景点必然能在完整模型上找到对应。
    // 对称/近对称物体的单次 RANSAC 可能落入错误盆地，故多次尝试取内点率最高者。
    bool align(const Cloud::Ptr& scene, Result& result) const noexcept {
        auto stage = Clock::now();
        const auto normals = compute_normals(scene, config.normal_radius);
        const auto features = compute_fpfh(scene, normals, config.feature_radius);
        result.timing.feature_ms += elapsed_ms(stage);

        auto best = std::optional<Result>{};
        auto failures = 0;
        for (int attempt = 0; attempt < config.coarse_attempts; attempt++) {
            stage = Clock::now();
            const auto coarse = coarse_align(scene, features);
            result.timing.coarse_ms += elapsed_ms(stage);
            if (!coarse.has_value()) {
                // 连续无法找到满足内点比例的假设，说明该场景候选不含完整物体（如混有地面）
                if (++failures >= config.coarse_max_failures)
                    break;
                continue;
            }

            stage = Clock::now();
            const auto refined = refine(scene, *coarse);
            result.timing.icp_ms += elapsed_ms(stage);

            auto candidate = Result{};
            evaluate(scene, refined, candidate);
            if (!best.has_value() || candidate.inlier_ratio > best->inlier_ratio)
                best = candidate;
            if (best->inlier_ratio >= config.early_stop_inlier_ratio)
                break;
        }
        if (!best.has_value())
            return false;
        if (best->inlier_ratio < config.min_inlier_ratio)
            return false;
        if (best->fitness > config.max_fitness)
            return false;

        result.pose = best->pose;
        result.inlier_ratio = best->inlier_ratio;
        result.fitness = best->fitness;
        return true;
    }

    std::optional<Result> register_cloud(const Cloud& scene) const noexcept {
        if (!initialized)
            return std::nullopt;

        auto result = Result{};
        const auto begin = Clock::now();

        const auto candidates = preprocess(scene);
        result.timing.preprocess_ms = elapsed_ms(begin);

        for (const auto& candidate : candidates) {
            if (align(candidate, result)) {
                result.timing.total_ms = elapsed_ms(begin);
                return result;
            }
        }
        return std::nullopt;
    }
};

Registration::Cloud::Ptr Registration::depth_to_cloud(
    const cv::Mat& depth, const geometry::CameraIntrinsics& intrinsics,
    const cv::Rect2i& roi) noexcept {
    return depth_to_cloud(depth, intrinsics, roi, cv::Mat{}, Config{});
}

Registration::Cloud::Ptr Registration::depth_to_cloud(
    const cv::Mat& depth, const geometry::CameraIntrinsics& intrinsics, const cv::Rect2i& roi,
    const cv::Mat& mask) noexcept {
    return depth_to_cloud(depth, intrinsics, roi, mask, Config{});
}

Registration::Cloud::Ptr Registration::depth_to_cloud(
    const cv::Mat& depth, const geometry::CameraIntrinsics& intrinsics, const cv::Rect2i& roi,
    const cv::Mat& mask, const Config& config) noexcept {
    auto cloud = std::make_shared<Cloud>();
    if (depth.type() != CV_16UC1)
        return cloud;

    auto eroded = cv::Mat{};
    if (!mask.empty()) {
        if (mask.type() != CV_8UC1 || mask.size() != depth.size())
            return cloud;
        if (config.mask_erosion_px > 0) {
            const auto size = 2 * config.mask_erosion_px + 1;
            cv::erode(
                mask, eroded, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size{size, size}));
        } else {
            eroded = mask;
        }
    }

    const auto rect = roi & cv::Rect2i{0, 0, depth.cols, depth.rows};
    cloud->reserve(static_cast<std::size_t>(rect.area()));
    for (int v = rect.y; v < rect.y + rect.height; v++) {
        const auto* row = depth.ptr<std::uint16_t>(v);
        const auto* mask_row = eroded.empty() ? nullptr : eroded.ptr<std::uint8_t>(v);
        for (int u = rect.x; u < rect.x + rect.width; u++) {
            if (row[u] == 0)
                continue;
            if (mask_row != nullptr && mask_row[u] == 0)
                continue;
            const auto z = static_cast<double>(row[u]) * 1e-3;
            cloud->emplace_back(
                static_cast<float>((u - intrinsics.cx) * z / intrinsics.fx),
                static_cast<float>((v - intrinsics.cy) * z / intrinsics.fy), static_cast<float>(z));
        }
    }
    return cloud;
}

std::expected<void, std::string>
    Registration::initialize(const std::string& stl_path, const Config& config) noexcept {
    return pimpl_->initialize(stl_path, config);
}

std::expected<void, std::string> Registration::initialize(const std::string& stl_path) noexcept {
    return pimpl_->initialize(stl_path, Config{});
}

std::optional<Registration::Result>
    Registration::register_cloud(const Cloud& scene) const noexcept {
    return pimpl_->register_cloud(scene);
}

Registration::Registration() noexcept
    : pimpl_{std::make_unique<Impl>()} {}

Registration::~Registration() noexcept = default;

} // namespace rmcs_energy_trace::registration
