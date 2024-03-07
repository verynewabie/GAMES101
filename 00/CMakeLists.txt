cmake_minimum_required (VERSION 2.8.11)
project (Transformation)
# 使用ubuntu的apt安装Eigen库后不用加这两行
# find_package(Eigen3 REQUIRED) 
# include_directories(EIGEN3_INCLUDE_DIR)

add_executable (Transformation main.cpp)
