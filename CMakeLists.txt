cmake_minimum_required(VERSION 3.28)
project(rmcs_energy_trace)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

find_package(OpenCV REQUIRED)
find_package(OpenVINO REQUIRED)
find_package(PCL 1.14 REQUIRED COMPONENTS
  common features filters kdtree registration search sample_consensus segmentation
)

# ---- M3T（third_party/m3t）----
find_package(OpenMP REQUIRED)
find_package(Eigen3 REQUIRED)
set(M3T_ROOT ${PROJECT_SOURCE_DIR}/third_party/m3t)

# 与 OpenGL 无关的源文件
set(M3T_CORE_NAMES
  body camera color_histograms common constraint depth_modality depth_model detector link
  model modality optimizer publisher refiner region_modality region_model soft_constraint
  static_detector subscriber tracker viewer
)
# 依赖 OpenGL 的渲染器
set(M3T_RENDERER_NAMES
  renderer_geometry renderer normal_renderer silhouette_renderer basic_depth_renderer
)
list(TRANSFORM M3T_CORE_NAMES PREPEND ${M3T_ROOT}/src/ OUTPUT_VARIABLE M3T_SOURCES)
list(TRANSFORM M3T_RENDERER_NAMES PREPEND ${M3T_ROOT}/src/ OUTPUT_VARIABLE M3T_RENDERER_SOURCES)
list(TRANSFORM M3T_SOURCES APPEND .cpp)
list(TRANSFORM M3T_RENDERER_SOURCES APPEND .cpp)
list(APPEND M3T_SOURCES ${M3T_RENDERER_SOURCES} ${M3T_ROOT}/stubs/highgui_stubs.cpp)

# 构建一个 M3T 静态库。
# m3t_runtime 用 stubs/gl 下的占位头文件编译渲染器，不链接 OpenGL / GLEW / GLFW / X11；
# m3t_full 使用真实的 OpenGL，只用于离线生成区域 / 深度模型。
# 第三方代码不检查警告，include 目录按 SYSTEM 处理；符号隐藏，两个库可同时出现在一个进程中。
function(m3t_add_library target)
  cmake_parse_arguments(ARG "WITH_OPENGL" "" "" ${ARGN})
  add_library(${target} STATIC ${M3T_SOURCES})
  target_include_directories(${target} SYSTEM PUBLIC ${M3T_ROOT}/include ${M3T_ROOT}/third_party)
  if(NOT ARG_WITH_OPENGL)
    # 头文件 renderer_geometry.h 会包含 GL 头文件，使用方也需要看到占位版本
    target_include_directories(${target} SYSTEM BEFORE PUBLIC ${M3T_ROOT}/stubs/gl)
  endif()
  target_link_libraries(${target} PUBLIC OpenMP::OpenMP_CXX Eigen3::Eigen opencv_core opencv_imgproc opencv_imgcodecs)
  target_compile_options(${target} PRIVATE -w)
  set_target_properties(${target} PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    CXX_STANDARD 17
  )
endfunction()

m3t_add_library(m3t_runtime)

# 可选：生成 M3T 区域 / 深度模型的离线工具。需要 OpenGL、GLEW 和 X11 开发包，GLFW 从源码构建
option(RMCS_ENERGY_TRACE_BUILD_M3T_TOOLS
  "Build m3t_generate, which renders with OpenGL (needs OpenGL, GLEW and a display)" OFF)
if(RMCS_ENERGY_TRACE_BUILD_M3T_TOOLS)
  include(FetchContent)
  set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
  set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
  set(BUILD_SHARED_LIBS_BACKUP ${BUILD_SHARED_LIBS})
  set(BUILD_SHARED_LIBS OFF)
  FetchContent_Declare(glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG 7b6aead9fb88b3623e3b3725ebb42670cbe4c579 # 3.4
  )
  FetchContent_MakeAvailable(glfw)
  # 第三方代码不检查警告
  target_compile_options(glfw PRIVATE -w)
  set(BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS_BACKUP})

  set(OpenGL_GL_PREFERENCE LEGACY)
  find_package(OpenGL REQUIRED)
  find_package(GLEW REQUIRED)

  m3t_add_library(m3t_full WITH_OPENGL)
  target_link_libraries(m3t_full PUBLIC OpenGL::GL GLEW::GLEW glfw)
endif()

file(GLOB_RECURSE PROJECT_SOURCE CONFIGURE_DEPENDS
  ${PROJECT_SOURCE_DIR}/src/*.cpp
)

ament_auto_add_library(
  ${PROJECT_NAME} SHARED
  ${PROJECT_SOURCE}
)
target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/src)
target_include_directories(${PROJECT_NAME} SYSTEM PUBLIC ${PCL_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME}
  ${OpenCV_LIBS}
  openvino::runtime
  ${PCL_LIBRARIES}
  m3t_runtime
)

# GCC 14 对 PCL ICP 内部 SVD 估计中的 Eigen 表达式误报 -Warray-bounds（用到 pcl::IterativeClosestPoint 的文件）
set_source_files_properties(
  ${PROJECT_SOURCE_DIR}/src/registration/registration.cpp
  ${PROJECT_SOURCE_DIR}/test/point_to_plane.cpp
  PROPERTIES COMPILE_OPTIONS -Wno-array-bounds
)

# 离线检测工具：对图片或目录运行能量单元检测并保存可视化结果
add_executable(energy_unit_detect tool/energy_unit_detect.cpp)
target_link_libraries(energy_unit_detect PRIVATE ${PROJECT_NAME})
install(TARGETS energy_unit_detect DESTINATION lib/${PROJECT_NAME})

# 离线消融工具：比较只用检测框与加实例掩码时的点云配准效果
add_executable(mask_ablation tool/mask_ablation.cpp)
target_link_libraries(mask_ablation PRIVATE ${PROJECT_NAME})
target_compile_definitions(mask_ablation PRIVATE
  RMCS_ENERGY_UNIT_MESH="${PROJECT_SOURCE_DIR}/meshes/energy_unit.stl"
)
install(TARGETS mask_ablation DESTINATION lib/${PROJECT_NAME})

if(RMCS_ENERGY_TRACE_BUILD_M3T_TOOLS)
  # 不链接 ${PROJECT_NAME}：它含有 m3t_runtime（OpenGL 占位头文件），与 m3t_full 的真实 OpenGL 冲突。
  # 工具只需要读 mesh 和写 OBJ，直接编译这两个源文件
  add_executable(m3t_generate
    tool/m3t_generate.cpp src/geometry/mesh.cpp src/tracking/mesh_obj.cpp
  )
  target_include_directories(m3t_generate PRIVATE ${PROJECT_SOURCE_DIR}/src)
  target_include_directories(m3t_generate SYSTEM PRIVATE ${PCL_INCLUDE_DIRS})
  target_link_libraries(m3t_generate PRIVATE m3t_full ${PCL_LIBRARIES})
  target_compile_definitions(m3t_generate PRIVATE
    RMCS_ENERGY_UNIT_MESH="${PROJECT_SOURCE_DIR}/meshes/energy_unit.stl"
    RMCS_ENERGY_TRACE_ASSET_DIR="${PROJECT_SOURCE_DIR}/assets"
  )
  install(TARGETS m3t_generate DESTINATION lib/${PROJECT_NAME})
endif()

pluginlib_export_plugin_description_file(rmcs_executor plugins.xml)

install(DIRECTORY config meshes launch rviz DESTINATION share/${PROJECT_NAME})
# assets/ 由 scripts/download_assets.sh 下载，不在仓库中；还没下载时跳过
install(DIRECTORY assets DESTINATION share/${PROJECT_NAME} OPTIONAL)

# 可选：抓取运动学仿真（需要 MoveIt，缺失时跳过，不影响其余目标）
find_package(moveit_ros_planning_interface QUIET)
find_package(pcl_conversions QUIET)
find_package(tf2_eigen QUIET)
find_package(tf2_ros QUIET)
find_package(sensor_msgs QUIET)
find_package(shape_msgs QUIET)
find_package(moveit_msgs QUIET)
find_package(visualization_msgs QUIET)
find_package(geometry_msgs QUIET)
find_package(trajectory_msgs QUIET)
find_package(ament_index_cpp QUIET)
if(moveit_ros_planning_interface_FOUND AND pcl_conversions_FOUND AND tf2_eigen_FOUND
   AND tf2_ros_FOUND AND sensor_msgs_FOUND AND shape_msgs_FOUND AND moveit_msgs_FOUND
   AND visualization_msgs_FOUND AND geometry_msgs_FOUND AND trajectory_msgs_FOUND
   AND ament_index_cpp_FOUND)
  add_executable(grasp_sim tool/grasp_sim/grasp_sim.cpp)
  target_include_directories(grasp_sim PRIVATE ${PROJECT_SOURCE_DIR}/tool/grasp_sim)
  target_link_libraries(grasp_sim PRIVATE ${PROJECT_NAME})
  ament_target_dependencies(grasp_sim PUBLIC
    moveit_ros_planning_interface
    pcl_conversions
    tf2_eigen
    tf2_ros
    sensor_msgs
    shape_msgs
    moveit_msgs
    visualization_msgs
    geometry_msgs
    trajectory_msgs
    ament_index_cpp
    rclcpp
  )
  install(TARGETS grasp_sim DESTINATION lib/${PROJECT_NAME})
else()
  message(STATUS "MoveIt not found, skip grasp_sim")
endif()

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)

  ament_add_gtest(test_registration test/registration.cpp)
  target_link_libraries(test_registration ${PROJECT_NAME})
  target_compile_definitions(test_registration PRIVATE
    RMCS_ENERGY_UNIT_MESH="${PROJECT_SOURCE_DIR}/meshes/energy_unit.stl"
  )
  set_tests_properties(test_registration PROPERTIES TIMEOUT 900)

  ament_add_gtest(test_tracking test/tracking.cpp)
  target_link_libraries(test_tracking ${PROJECT_NAME})
  target_compile_definitions(test_tracking PRIVATE
    RMCS_ENERGY_UNIT_MESH="${PROJECT_SOURCE_DIR}/meshes/energy_unit.stl"
    RMCS_ENERGY_TRACE_ASSET_DIR="${PROJECT_SOURCE_DIR}/assets"
  )
  set_tests_properties(test_tracking PROPERTIES TIMEOUT 300)

  ament_add_gtest(test_depth_mask test/depth_mask.cpp)
  target_link_libraries(test_depth_mask ${PROJECT_NAME})
  set_tests_properties(test_depth_mask PROPERTIES TIMEOUT 10)

  ament_add_gtest(test_supervision test/supervision.cpp)
  target_link_libraries(test_supervision ${PROJECT_NAME})
  set_tests_properties(test_supervision PROPERTIES TIMEOUT 10)

  ament_add_gtest(test_point_to_plane test/point_to_plane.cpp)
  target_link_libraries(test_point_to_plane ${PROJECT_NAME})
  set_tests_properties(test_point_to_plane PROPERTIES TIMEOUT 10)
endif()

ament_auto_package()
