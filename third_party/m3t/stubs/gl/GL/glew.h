// [RMCS] OpenGL / GLEW 的占位头文件，只用于编译 m3t_runtime。
// M3T 的渲染器源文件原样编译，但所有 GL 函数都是空实现（调用即视为失败），
// 因此运行时库不链接 libGL / libGLEW。渲染器入口 glfwInit 会返回失败并打印错误，
// 见 GLFW/glfw3.h。需要真实渲染（生成区域/深度模型）时使用 m3t_full。
#ifndef RMCS_M3T_STUB_GLEW_H
#define RMCS_M3T_STUB_GLEW_H

#include <cstddef>

using GLenum = unsigned int;
using GLboolean = unsigned char;
using GLbitfield = unsigned int;
using GLint = int;
using GLuint = unsigned int;
using GLsizei = int;
using GLfloat = float;
using GLchar = char;
using GLsizeiptr = std::ptrdiff_t;

#define GL_FALSE 0
#define GL_TRUE 1
#define GLEW_OK 0
#define GL_TRIANGLES 0x0004
#define GL_DEPTH_BUFFER_BIT 0x00000100
#define GL_COLOR_BUFFER_BIT 0x00004000
#define GL_UNSIGNED_BYTE 0x1401
#define GL_UNSIGNED_SHORT 0x1403
#define GL_FLOAT 0x1406
#define GL_CCW 0x0901
#define GL_FRONT 0x0404
#define GL_CULL_FACE 0x0B44
#define GL_DEPTH_TEST 0x0B71
#define GL_PACK_ROW_LENGTH 0x0D02
#define GL_PACK_ALIGNMENT 0x0D05
#define GL_DEPTH_COMPONENT 0x1902
#define GL_RED 0x1903
#define GL_BGRA 0x80E1
#define GL_FRAGMENT_SHADER 0x8B30
#define GL_VERTEX_SHADER 0x8B31
#define GL_COMPILE_STATUS 0x8B81
#define GL_LINK_STATUS 0x8B82
#define GL_ARRAY_BUFFER 0x8892
#define GL_STATIC_DRAW 0x88E4
#define GL_R8 0x8229
#define GL_RGBA8 0x8058
#define GL_DEPTH_COMPONENT16 0x81A5
#define GL_FRAMEBUFFER 0x8D40
#define GL_RENDERBUFFER 0x8D41
#define GL_COLOR_ATTACHMENT0 0x8CE0
#define GL_DEPTH_ATTACHMENT 0x8D00

inline GLboolean glewExperimental = GL_FALSE;
inline GLenum glewInit() { return 1; }

inline void glAttachShader(GLuint, GLuint) {}
inline void glBindBuffer(GLenum, GLuint) {}
inline void glBindFramebuffer(GLenum, GLuint) {}
inline void glBindRenderbuffer(GLenum, GLuint) {}
inline void glBindVertexArray(GLuint) {}
inline void glBufferData(GLenum, GLsizeiptr, const void*, GLenum) {}
inline void glClear(GLbitfield) {}
inline void glClearColor(GLfloat, GLfloat, GLfloat, GLfloat) {}
inline void glCompileShader(GLuint) {}
inline GLuint glCreateProgram() { return 0; }
inline GLuint glCreateShader(GLenum) { return 0; }
inline void glCullFace(GLenum) {}
inline void glDeleteBuffers(GLsizei, const GLuint*) {}
inline void glDeleteFramebuffers(GLsizei, const GLuint*) {}
inline void glDeleteRenderbuffers(GLsizei, const GLuint*) {}
inline void glDeleteShader(GLuint) {}
inline void glDeleteVertexArrays(GLsizei, const GLuint*) {}
inline void glDisable(GLenum) {}
inline void glDrawArrays(GLenum, GLint, GLsizei) {}
inline void glEnable(GLenum) {}
inline void glEnableVertexAttribArray(GLuint) {}
inline void glFramebufferRenderbuffer(GLenum, GLenum, GLenum, GLuint) {}
inline void glFrontFace(GLenum) {}
inline void glGenBuffers(GLsizei, GLuint*) {}
inline void glGenFramebuffers(GLsizei, GLuint*) {}
inline void glGenRenderbuffers(GLsizei, GLuint*) {}
inline void glGenVertexArrays(GLsizei, GLuint*) {}
inline void glGetProgramInfoLog(GLuint, GLsizei, GLsizei*, GLchar*) {}
inline void glGetProgramiv(GLuint, GLenum, GLint*) {}
inline void glGetShaderInfoLog(GLuint, GLsizei, GLsizei*, GLchar*) {}
inline void glGetShaderiv(GLuint, GLenum, GLint*) {}
inline GLint glGetUniformLocation(GLuint, const GLchar*) { return -1; }
inline void glLinkProgram(GLuint) {}
inline void glPixelStorei(GLenum, GLint) {}
inline void glReadPixels(GLint, GLint, GLsizei, GLsizei, GLenum, GLenum, void*) {}
inline void glRenderbufferStorage(GLenum, GLenum, GLsizei, GLsizei) {}
inline void glShaderSource(GLuint, GLsizei, const GLchar* const*, const GLint*) {}
inline void glUniform1f(GLint, GLfloat) {}
inline void glUniformMatrix3fv(GLint, GLsizei, GLboolean, const GLfloat*) {}
inline void glUniformMatrix4fv(GLint, GLsizei, GLboolean, const GLfloat*) {}
inline void glUseProgram(GLuint) {}
inline void glVertexAttribPointer(GLuint, GLint, GLenum, GLboolean, GLsizei, const void*) {}
inline void glViewport(GLint, GLint, GLsizei, GLsizei) {}

#endif
