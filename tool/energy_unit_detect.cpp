#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <print>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "detection/detector.hpp"
#include "segmentation/segmenter.hpp"

namespace fs = std::filesystem;

auto main(int argc, char* argv[]) -> int {
    auto positional = std::vector<std::string>{};
    auto config = rmcs_energy_trace::detection::Detector::Config{};
    auto json_path = std::string{};
    auto sam_config = rmcs_energy_trace::segmentation::Segmenter::Config{};

    for (int i = 1; i < argc; i++) {
        const auto arg = std::string{argv[i]};
        if (arg == "--conf" && i + 1 < argc) {
            config.score_threshold = std::strtof(argv[++i], nullptr);
        } else if (arg == "--iou" && i + 1 < argc) {
            config.nms_threshold = std::strtof(argv[++i], nullptr);
        } else if (arg == "--device" && i + 1 < argc) {
            config.infer_device = argv[++i];
        } else if (arg == "--json" && i + 1 < argc) {
            json_path = argv[++i];
        } else if (arg == "--sam" && i + 2 < argc) {
            sam_config.encoder_path = argv[++i];
            sam_config.decoder_path = argv[++i];
        } else {
            positional.push_back(arg);
        }
    }
    if (positional.size() != 3) {
        std::println(
            "Usage: energy_unit_detect <network.xml> <image_or_dir> <out_dir> [--conf 0.25] "
            "[--iou 0.7] [--device CPU] [--json out.json] [--sam <encoder.onnx> <decoder.onnx>]");
        return EXIT_FAILURE;
    }

    config.network_path = positional[0];
    const auto input = fs::path{positional[1]};
    const auto out_dir = fs::path{positional[2]};

    auto files = std::vector<fs::path>{};
    if (fs::is_directory(input)) {
        for (const auto& entry : fs::directory_iterator{input}) {
            const auto ext = entry.path().extension().string();
            if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp")
                files.push_back(entry.path());
        }
        std::ranges::sort(files);
    } else {
        files.push_back(input);
    }
    fs::create_directories(out_dir);

    auto detector = rmcs_energy_trace::detection::Detector{};
    if (auto result = detector.initialize(config); !result.has_value()) {
        std::println("{}", result.error());
        return EXIT_FAILURE;
    }

    // 给出 SAM 网络时，对每个检测框再做一次分割，把掩码叠加在输出图上
    auto segmenter = rmcs_energy_trace::segmentation::Segmenter{};
    const auto use_sam = !sam_config.encoder_path.empty();
    if (use_sam) {
        sam_config.infer_device = config.infer_device;
        if (auto result = segmenter.initialize(sam_config); !result.has_value()) {
            std::println("{}", result.error());
            return EXIT_FAILURE;
        }
    }

    auto json = std::string{"{"};
    for (const auto& file : files) {
        auto image = cv::imread(file.string());
        if (image.empty()) {
            std::println("{}: failed to read", file.filename().string());
            continue;
        }

        // 每个框都在未画任何标注的原图上分割，标注画在 image 上
        const auto clean = image.clone();
        const auto detections = detector.detect(image);
        const auto timing = detector.last_timing();

        std::println(
            "{}: {} box(es) | pre {:.2f} ms, infer {:.2f} ms, post {:.2f} ms",
            file.filename().string(), detections.size(), timing.preprocess, timing.infer,
            timing.postprocess);

        json += std::format(R"({}"{}":[)", json.size() > 1 ? "," : "", file.stem().string());
        auto first = true;
        for (const auto& d : detections) {
            std::println(
                "  [{:.1f}, {:.1f}, {:.1f}, {:.1f}] score {:.4f}", d.box.x, d.box.y,
                d.box.x + d.box.width, d.box.y + d.box.height, d.score);
            json += std::format(
                R"({}{{"box":[{},{},{},{}],"score":{}}})", first ? "" : ",", d.box.x, d.box.y,
                d.box.x + d.box.width, d.box.y + d.box.height, d.score);
            first = false;

            if (use_sam) {
                const auto segmented = segmenter.segment(clean, d.box);
                if (segmented.has_value()) {
                    const auto& t = segmented->timing;
                    std::println(
                        "    SAM mask {} px, score {:.3f} | pre {:.1f} ms, encoder {:.1f} ms, "
                        "decoder {:.1f} ms, post {:.1f} ms",
                        cv::countNonZero(segmented->mask), segmented->score, t.preprocess,
                        t.encoder, t.decoder, t.postprocess);
                    auto tinted = image.clone();
                    tinted.setTo(cv::Scalar{255, 0, 255}, segmented->mask);
                    cv::addWeighted(image, 0.5, tinted, 0.5, 0., image);
                } else {
                    std::println("    SAM failed");
                }
            }

            cv::rectangle(image, d.box, {0, 255, 0}, 2);
            cv::putText(
                image, std::format("{:.3f}", d.score),
                {static_cast<int>(d.box.x), std::max(static_cast<int>(d.box.y) - 6, 12)},
                cv::FONT_HERSHEY_SIMPLEX, 0.6, {0, 255, 0}, 2);
        }
        json += "]";

        cv::imwrite((out_dir / (file.stem().string() + ".png")).string(), image);
    }
    json += "}";

    if (!json_path.empty())
        std::ofstream{json_path} << json;
    return EXIT_SUCCESS;
}
