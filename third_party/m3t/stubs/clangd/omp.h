// [RMCS] 仅供 clangd / clang-tidy 使用，编译时不会用到（见 .clangd）。
// M3T 的头文件无条件包含 <omp.h>，而分析时已去掉 -fopenmp，Clang 自带的头文件中没有它。
#pragma once
