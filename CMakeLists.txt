cmake_minimum_required(VERSION 3.10)
project(mobile_pose)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找 OpenCV 包
find_package(OpenCV REQUIRED)

# 设置 ONNX Runtime 路径
set(ONNXRUNTIME_ROOT_PATH "/home/deepseavision/workspace/thirdparts/onnxruntime-linux-x64-1.16.3")

# 添加头文件路径
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_ROOT_PATH}/include
    include
)

# 添加可执行文件
add_executable(mobile_pose src/main.cpp src/mobile_pose.cpp)

# 链接库
target_link_libraries(mobile_pose 
    ${OpenCV_LIBS}
    ${ONNXRUNTIME_ROOT_PATH}/lib/libonnxruntime.so
)