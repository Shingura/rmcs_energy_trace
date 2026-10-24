# M3T

DLR 的 M3T（Multi-Body Multi-Modality Multi-Camera 3D Tracker），用于能量单元的 6D 位姿跟踪。许可证见 `LICENSE`（MIT）。

- 上游：<https://github.com/DLR-RM/3DObjectTracking>，子目录 `M3T/`
- 提交：`f02106186a1e756450226b3b793e309506817c58`

## 保留的内容

`include/m3t/`、`src/` 和 `third_party/` 只保留区域线索、深度线索和跟踪器需要的文件。

## 删掉的内容

- `texture_modality`：依赖 OpenCV 的 `xfeatures2d`，Ubuntu 24.04 的 OpenCV 没有这个模块，我们也不用纹理线索。
- `azure_kinect_camera`、`realsense_camera`、`loader_camera`：相机由我们自己的代码提供图像。
- `image_viewer`、`normal_viewer`、`manual_detector`、`generator.h`：可视化和手动初始化，不需要。
- 上游的 `CMakeLists.txt`、`examples/`、`test/`、`data/`、`doc/`：构建在上一级 `CMakeLists.txt` 中完成。

## 第三方头文件

`third_party/` 下的三个头文件随上游一起带入。

- `nanoflann`：BSD 许可证，全文在头文件开头。
- `tiny_obj_loader`：MIT 许可证，全文在头文件开头。
- `filesystem/filesystem.h`：作者 BrainStone，头文件里只有版权行，没有许可证文本。

## 本地修改

M3T 的源文件没有改动。运行时库的处理如下。

- `stubs/gl/`：`GL/glew.h` 和 `GLFW/glfw3.h` 的占位版本。运行时库（`m3t_runtime`）用它们编译渲染器源文件，所有 GL 函数是空实现，`glfwInit()` 返回失败并打印错误。因此运行时不链接 OpenGL、GLEW、GLFW 和 X11，也不能渲染。生成区域和深度模型要用 `m3t_full`（见 `tool/m3t_generate.cpp`）。
- `stubs/highgui_stubs.cpp`：`cv::imshow` 和 `cv::waitKey` 的空实现。M3T 的调试可视化会引用它们，我们不链接 OpenCV highgui（它依赖 GTK 和 X11）。

## 已知限制

区域和深度模型文件里记录了 `energy_unit.obj` 的路径，M3T 加载时用 `std::filesystem::equivalent` 比较。把模型目录移到别处后，需要重新运行 `m3t_generate`。
