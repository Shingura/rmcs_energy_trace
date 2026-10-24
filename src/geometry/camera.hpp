#pragma once

namespace rmcs_energy_trace::geometry {

// 针孔相机内参，单位为像素
struct CameraIntrinsics {
    double fx, fy, cx, cy;
};

} // namespace rmcs_energy_trace::geometry
