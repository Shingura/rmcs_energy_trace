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
)

# GCC 14 对 PCL ICP 内部 SVD 估计中的 Eigen 表达式误报 -Warray-bounds
set_source_files_properties(
  ${PROJECT_SOURCE_DIR}/src/registration/registration.cpp
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

pluginlib_export_plugin_description_file(rmcs_executor plugins.xml)

install(DIRECTORY config meshes DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)

  ament_add_gtest(test_registration test/registration.cpp)
  target_link_libraries(test_registration ${PROJECT_NAME})
  target_compile_definitions(test_registration PRIVATE
    RMCS_ENERGY_UNIT_MESH="${PROJECT_SOURCE_DIR}/meshes/energy_unit.stl"
  )
  set_tests_properties(test_registration PROPERTIES TIMEOUT 300)

  ament_add_gtest(test_depth_mask test/depth_mask.cpp)
  target_link_libraries(test_depth_mask ${PROJECT_NAME})
  set_tests_properties(test_depth_mask PROPERTIES TIMEOUT 10)
endif()

ament_auto_package()
