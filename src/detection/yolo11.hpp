#pragma once

#include <cstddef>
#include <string>

#include <openvino/core/preprocess/pre_post_process.hpp>
#include <openvino/runtime/compiled_model.hpp>
#include <openvino/runtime/core.hpp>

namespace rmcs_energy_trace::detection {

// 能量单元 YOLO11n 单类别网络的输入输出约定，检测和分割两种网络共用。
//
// 输入：u8 BGR NHWC（letterbox 之后），由 OpenVINO 预处理转为 f32 RGB NCHW 并除以 255。
// 输出没有张量名，按下标访问。坐标在 letterbox 之后的输入中，每个属性占一行：
// - 检测网络：只有 output 0，[1, 5, 8400]，属性依次为 cx, cy, w, h, score。
// - 分割网络：output 0 为 [1, 37, 8400]，属性为 cx, cy, w, h, score 和 32 个掩码系数；
//   output 1 为 [1, 32, 160, 160] 的掩码原型。
struct Yolo11 {
    static constexpr std::size_t kInputWidth = 640;
    static constexpr std::size_t kInputHeight = 640;
    static constexpr std::size_t kAttributes = 5;
    static constexpr std::size_t kMaskChannels = 32;
    static constexpr std::size_t kMaskHeight = 160;
    static constexpr std::size_t kMaskWidth = 160;

    static ov::Shape input_shape() { return ov::Shape{1, kInputHeight, kInputWidth, 3}; }

    static ov::CompiledModel
        compile(ov::Core& core, const std::string& location, const std::string& device) {
        auto model = core.read_model(location);
        auto ppp = ov::preprocess::PrePostProcessor{model};

        auto& input = ppp.input();
        input.tensor()
            .set_element_type(ov::element::u8)
            .set_shape(input_shape())
            .set_layout("NHWC")
            .set_color_format(ov::preprocess::ColorFormat::BGR);
        input.preprocess()
            .convert_element_type(ov::element::f32)
            .convert_color(ov::preprocess::ColorFormat::RGB)
            .scale(255.0);
        input.model().set_layout("NCHW");

        // 固定 f32：CPU 插件默认在支持的 CPU 上用 bf16，结果会随 CPU 是否支持 bf16 而变
        return core.compile_model(
            ppp.build(), device, ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY),
            ov::hint::inference_precision(ov::element::f32));
    }
};

} // namespace rmcs_energy_trace::detection
