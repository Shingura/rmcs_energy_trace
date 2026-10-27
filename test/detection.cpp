#include <cstddef>
#include <optional>
#include <vector>

#include <gtest/gtest.h>
#include <opencv2/core.hpp>

#include "detection/decode.hpp"
#include "detection/yolo11.hpp"

using rmcs_energy_trace::detection::decode_detections;
using rmcs_energy_trace::detection::Letterbox;
using rmcs_energy_trace::detection::Predictions;
using rmcs_energy_trace::detection::Prototypes;
using rmcs_energy_trace::detection::Yolo11;

namespace {

constexpr auto kCount = std::size_t{8400};
constexpr auto kScore = 0.25f;
constexpr auto kNms = 0.7f;

// letterbox 坐标下的框，cx, cy, w, h
struct Candidate {
    float cx, cy, w, h, score;
    float coefficient0 = 0.f; // 第 0 个掩码系数，其余为 0
};

// 按属性主序拼出输出张量，属性数 5 或 5 + 32，未列出的候选得分为 0
auto make_predictions(const std::vector<Candidate>& candidates, std::size_t attributes)
    -> std::vector<float> {
    auto data = std::vector<float>(attributes * kCount, 0.f);
    for (std::size_t i = 0; i < candidates.size(); i++) {
        const auto& c = candidates[i];
        data[0 * kCount + i] = c.cx;
        data[1 * kCount + i] = c.cy;
        data[2 * kCount + i] = c.w;
        data[3 * kCount + i] = c.h;
        data[4 * kCount + i] = c.score;
        if (attributes > Yolo11::kAttributes)
            data[5 * kCount + i] = c.coefficient0;
    }
    return data;
}

auto detect(
    const std::vector<float>& data, std::size_t attributes, const Letterbox& lb, cv::Size size,
    const std::optional<Prototypes>& prototypes = std::nullopt) {
    return decode_detections(
        Predictions{.data = data.data(), .attributes = attributes, .count = kCount}, prototypes, lb,
        size, kScore, kNms);
}

// 宽图 1280 x 720：缩放 0.5，上下各填充 140
constexpr auto kWide = Letterbox{.scale = 0.5f, .pad_left = 0, .pad_top = 140};
// 高图 720 x 1280：缩放 0.5，左右各填充 140
constexpr auto kTall = Letterbox{.scale = 0.5f, .pad_left = 140, .pad_top = 0};

} // namespace

TEST(Detection, DetectionNetworkLayout) {
    const auto data =
        make_predictions({{.cx = 240, .cy = 300, .w = 180, .h = 180, .score = 0.9f}}, 5);
    const auto detections = detect(data, 5, kWide, {1280, 720});

    ASSERT_EQ(detections.size(), 1u);
    EXPECT_FLOAT_EQ(detections[0].score, 0.9f);
    EXPECT_TRUE(detections[0].mask.empty());
}

TEST(Detection, BelowScoreThresholdIsDropped) {
    const auto data =
        make_predictions({{.cx = 240, .cy = 300, .w = 180, .h = 180, .score = 0.2f}}, 5);
    EXPECT_TRUE(detect(data, 5, kWide, {1280, 720}).empty());
}

TEST(Detection, BoxMapsBackWithHorizontalImage) {
    // letterbox 中 (150, 210) - (330, 390) 对应原图 (300, 140) - (660, 500)
    const auto data =
        make_predictions({{.cx = 240, .cy = 300, .w = 180, .h = 180, .score = 0.9f}}, 5);
    const auto detections = detect(data, 5, kWide, {1280, 720});

    ASSERT_EQ(detections.size(), 1u);
    const auto& box = detections[0].box;
    EXPECT_NEAR(box.x, 300.f, 1e-3);
    EXPECT_NEAR(box.y, 140.f, 1e-3);
    EXPECT_NEAR(box.width, 360.f, 1e-3);
    EXPECT_NEAR(box.height, 360.f, 1e-3);
}

TEST(Detection, BoxMapsBackWithVerticalImage) {
    // letterbox 中 (290, 70) - (470, 250) 对应原图 (300, 140) - (660, 500)
    const auto data =
        make_predictions({{.cx = 380, .cy = 160, .w = 180, .h = 180, .score = 0.9f}}, 5);
    const auto detections = detect(data, 5, kTall, {720, 1280});

    ASSERT_EQ(detections.size(), 1u);
    const auto& box = detections[0].box;
    EXPECT_NEAR(box.x, 300.f, 1e-3);
    EXPECT_NEAR(box.y, 140.f, 1e-3);
    EXPECT_NEAR(box.width, 360.f, 1e-3);
    EXPECT_NEAR(box.height, 360.f, 1e-3);
}

TEST(Detection, BoxIsClampedToImage) {
    // 框伸进填充区，映射后裁剪到图像内
    const auto data =
        make_predictions({{.cx = 320, .cy = 150, .w = 200, .h = 200, .score = 0.9f}}, 5);
    const auto detections = detect(data, 5, kWide, {1280, 720});

    ASSERT_EQ(detections.size(), 1u);
    const auto& box = detections[0].box;
    EXPECT_FLOAT_EQ(box.y, 0.f);
    EXPECT_NEAR(box.y + box.height, (250.f - 140.f) * 2.f, 1e-3);
}

TEST(Detection, OverlappingDuplicatesAreSuppressed) {
    const auto data = make_predictions(
        {
            {.cx = 240, .cy = 300, .w = 180, .h = 180, .score = 0.8f},
            {.cx = 244, .cy = 302, .w = 180, .h = 180, .score = 0.9f},
            {.cx = 500, .cy = 500, .w = 60, .h = 60, .score = 0.6f},
        },
        5);
    const auto detections = detect(data, 5, kWide, {1280, 720});

    ASSERT_EQ(detections.size(), 2u);
    EXPECT_FLOAT_EQ(detections[0].score, 0.9f);
    EXPECT_FLOAT_EQ(detections[1].score, 0.6f);
}

TEST(Detection, MismatchedAttributesAreRejected) {
    const auto data =
        make_predictions({{.cx = 240, .cy = 300, .w = 180, .h = 180, .score = 0.9f}}, 5);
    // 分割网络需要原型，检测网络不能带原型
    EXPECT_TRUE(detect(data, 5, kWide, {1280, 720}, Prototypes{data.data(), 32, 160, 160}).empty());
    EXPECT_TRUE(detect(data, 4, kWide, {1280, 720}).empty());
}

namespace {

// 原型第 0 通道在 [col0, col1) x [row0, row1) 内为 +1，其余为 -1；其余通道为 0
auto make_prototypes(int col0, int col1, int row0, int row1) -> std::vector<float> {
    auto data =
        std::vector<float>(Yolo11::kMaskChannels * Yolo11::kMaskHeight * Yolo11::kMaskWidth, 0.f);
    for (int row = 0; row < static_cast<int>(Yolo11::kMaskHeight); row++) {
        for (int col = 0; col < static_cast<int>(Yolo11::kMaskWidth); col++) {
            const auto inside = col >= col0 && col < col1 && row >= row0 && row < row1;
            data
                [static_cast<std::size_t>(row) * Yolo11::kMaskWidth
                 + static_cast<std::size_t>(col)] = inside ? 1.f : -1.f;
        }
    }
    return data;
}

auto iou(const cv::Mat& a, const cv::Mat& b) -> double {
    const auto inter = cv::countNonZero(a & b);
    const auto uni = cv::countNonZero(a | b);
    return uni == 0 ? 1. : static_cast<double>(inter) / uni;
}

} // namespace

TEST(Detection, MaskMatchesKnownRegionHorizontalImage) {
    // 原型中列 40..80、行 55..95 为目标。原型一格对应输入 4 px、原图 8 px，
    // 宽图在原型中占行 35..125，所以目标是原图 x 320..640、y 160..480
    const auto prototypes = make_prototypes(40, 80, 55, 95);
    const auto data = make_predictions(
        {{.cx = 240, .cy = 300, .w = 180, .h = 180, .score = 0.9f, .coefficient0 = 1.f}},
        Yolo11::kAttributes + Yolo11::kMaskChannels);
    const auto detections = detect(
        data, Yolo11::kAttributes + Yolo11::kMaskChannels, kWide, {1280, 720},
        Prototypes{
            prototypes.data(), Yolo11::kMaskChannels, Yolo11::kMaskHeight, Yolo11::kMaskWidth});

    ASSERT_EQ(detections.size(), 1u);
    const auto& mask = detections[0].mask;
    ASSERT_EQ(mask.type(), CV_8UC1);
    ASSERT_EQ(mask.size(), cv::Size(1280, 720));

    cv::Mat expected = cv::Mat::zeros(720, 1280, CV_8UC1);
    expected(cv::Rect{320, 160, 320, 320}).setTo(255);
    EXPECT_GT(iou(mask, expected), 0.98);
    EXPECT_EQ(mask.at<uchar>(320, 480), 255);
    EXPECT_EQ(mask.at<uchar>(100, 100), 0);

    // 目标在检测框 (300, 140) - (660, 500) 内，框外为 0
    auto outside = mask.clone();
    outside(cv::Rect{300, 140, 360, 360}).setTo(0);
    EXPECT_EQ(cv::countNonZero(outside), 0);
}

TEST(Detection, MaskMatchesKnownRegionVerticalImage) {
    // 高图在原型中占列 35..125；目标是原图 x 160..480、y 320..640
    const auto prototypes = make_prototypes(55, 95, 40, 80);
    const auto data = make_predictions(
        {{.cx = 380, .cy = 160, .w = 180, .h = 180, .score = 0.9f, .coefficient0 = 1.f}},
        Yolo11::kAttributes + Yolo11::kMaskChannels);
    const auto detections = detect(
        data, Yolo11::kAttributes + Yolo11::kMaskChannels, kTall, {720, 1280},
        Prototypes{
            prototypes.data(), Yolo11::kMaskChannels, Yolo11::kMaskHeight, Yolo11::kMaskWidth});

    ASSERT_EQ(detections.size(), 1u);
    const auto& mask = detections[0].mask;
    ASSERT_EQ(mask.size(), cv::Size(720, 1280));

    // 检测框 (300, 140) - (660, 500) 与目标 (160, 320) - (480, 640) 相交于 300..480 x 320..500，
    // 掩码被检测框裁剪
    cv::Mat expected = cv::Mat::zeros(1280, 720, CV_8UC1);
    expected(cv::Rect{300, 320, 180, 180}).setTo(255);
    EXPECT_GT(iou(mask, expected), 0.98);
    EXPECT_EQ(mask.at<uchar>(400, 400), 255);
    EXPECT_EQ(mask.at<uchar>(560, 400), 0);
}

TEST(Detection, MaskCoefficientSignSelectsComplement) {
    // 系数取负时，目标内是负 logit、目标外是正 logit，掩码是目标区域在框内的补集
    const auto prototypes = make_prototypes(40, 80, 55, 95);
    const auto data = make_predictions(
        {{.cx = 240, .cy = 300, .w = 180, .h = 180, .score = 0.9f, .coefficient0 = -1.f}},
        Yolo11::kAttributes + Yolo11::kMaskChannels);
    const auto detections = detect(
        data, Yolo11::kAttributes + Yolo11::kMaskChannels, kWide, {1280, 720},
        Prototypes{
            prototypes.data(), Yolo11::kMaskChannels, Yolo11::kMaskHeight, Yolo11::kMaskWidth});

    ASSERT_EQ(detections.size(), 1u);
    const auto& mask = detections[0].mask;
    EXPECT_EQ(mask.at<uchar>(320, 480), 0);   // 目标中心
    EXPECT_EQ(mask.at<uchar>(150, 310), 255); // 框内、目标外
    EXPECT_EQ(mask.at<uchar>(100, 100), 0);   // 框外
}
