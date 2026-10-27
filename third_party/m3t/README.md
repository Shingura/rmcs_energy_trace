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

源文件只改了一处：`src/model.cpp` 的 `Model::LoadBodyData`。上游检查模型文件里记录的 OBJ 路径与当前 OBJ 是否为同一个文件（`Equivalent`），模型目录换了位置（例如部署到机器人）就无法加载；而重新生成需要 OpenGL 和显示器，机器人上做不到。现在只比较文件名，其余几何参数（单位、绕序、裁剪、最大直径、几何到物体的位姿）照常比较。文件内容是否一致，由 `scripts/download_assets.sh` 对每个文件的 SHA256 校验保证。

运行时库的处理如下。

- `stubs/gl/`：`GL/glew.h` 和 `GLFW/glfw3.h` 的占位版本。运行时库（`m3t_runtime`）用它们编译渲染器源文件，所有 GL 函数是空实现，`glfwInit()` 返回失败并打印错误。因此运行时不链接 OpenGL、GLEW、GLFW 和 X11，也不能渲染。生成区域和深度模型要用 `m3t_full`（见 `tool/m3t_generate.cpp`）。
- `stubs/highgui_stubs.cpp`：`cv::imshow` 和 `cv::waitKey` 的空实现。M3T 的调试可视化会引用它们，我们不链接 OpenCV highgui（它依赖 GTK 和 X11）。

## 已知限制

区域和深度模型文件里仍记录着生成时 `energy_unit.obj` 的路径，但加载时只比较文件名（见上一节），模型目录可以移动。改了 mesh 或生成参数后，需要重新运行 `m3t_generate`，并更新 `scripts/assets.sha256` 中的校验值。
