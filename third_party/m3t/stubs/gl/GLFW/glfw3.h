// [RMCS] GLFW 的占位头文件，见 GL/glew.h。
// glfwInit 始终失败，因此 RendererGeometry::SetUp 返回 false，渲染相关代码不会被执行。
#ifndef RMCS_M3T_STUB_GLFW3_H
#define RMCS_M3T_STUB_GLFW3_H

#include <cstdio>

struct GLFWwindow;
struct GLFWmonitor;

#define GLFW_CONTEXT_VERSION_MAJOR 0x00022002
#define GLFW_CONTEXT_VERSION_MINOR 0x00022003
#define GLFW_OPENGL_PROFILE 0x00022008
#define GLFW_OPENGL_CORE_PROFILE 0x00032001
#define GLFW_RESIZABLE 0x00020003
#define GLFW_VISIBLE 0x00020004

inline int glfwInit() {
    std::fputs(
        "[m3t_runtime] Rendering is not available: this build has no OpenGL. "
        "Generate the model files with m3t_generate (RMCS_ENERGY_TRACE_BUILD_M3T_TOOLS=ON).\n",
        stderr);
    return 0;
}
inline void glfwTerminate() {}
inline void glfwWindowHint(int, int) {}
inline GLFWwindow* glfwCreateWindow(int, int, const char*, GLFWmonitor*, GLFWwindow*) {
    return nullptr;
}
inline void glfwDestroyWindow(GLFWwindow*) {}
inline void glfwMakeContextCurrent(GLFWwindow*) {}

#endif
