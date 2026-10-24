// [RMCS] M3T 的可视化调试代码引用了 cv::imshow / cv::waitKey。
// 运行时库不链接 OpenCV highgui（它会引入 GTK / X11），这里提供空实现。
// 符号隐藏，不会影响进程中其他库对 highgui 的使用。
#include <opencv2/core.hpp>

namespace cv {

__attribute__((visibility("hidden"))) void imshow(const String&, InputArray) {}

__attribute__((visibility("hidden"))) int waitKey(int) { return -1; }

} // namespace cv
