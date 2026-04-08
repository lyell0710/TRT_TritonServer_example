# 设置项目
cmake_minimum_required(VERSION 3.12)
cmake_policy(SET CMP0091 NEW)
project(custom_plugins LANGUAGES CXX CUDA)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 添加依赖项
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.18)
    find_package(CUDAToolkit REQUIRED)
    set(CUDA_INCLUDE_DIRS ${CUDAToolkit_INCLUDE_DIRS})
    set(CUDA_LIBRARIES ${CUDAToolkit_LIBRARIES})
else()
    find_package(CUDA REQUIRED)
endif()
set(CMAKE_CUDA_ARCHITECTURES 89;86;75;70;61)  # CUDA 架构版本 8.9, 8.6, 7.5, 7.0, 6.1

# 添加编译选项
option(TENSORRT_PATH "TensorRT Path. Example: /usr/local/tensorrt" "")
if(NOT TENSORRT_PATH)
    message(FATAL_ERROR "TensorRT path is not set. Please specify the TensorRT path.")
endif()

# 配置CUDA和TensorRT的函数
function(configure_cuda_trt target)
    target_compile_definitions(${target} PRIVATE ${CUDA_DEFINITIONS})
    target_include_directories(${target} PRIVATE ${CUDA_INCLUDE_DIRS})
    target_link_libraries(${target} PRIVATE ${CUDA_cudart_LIBRARY})

    target_include_directories(${target} PRIVATE ${TENSORRT_PATH}/include)
    target_link_directories(${target} PRIVATE ${TENSORRT_PATH}/lib)
    if(MSVC AND EXISTS ${TENSORRT_PATH}/lib/nvinfer_10.dll)
        target_link_libraries(${target} PRIVATE nvinfer_10 nvinfer_plugin_10 nvonnxparser_10)
    else()
        target_link_libraries(${target} PRIVATE nvinfer nvinfer_plugin nvonnxparser)
    endif()
endfunction()

# 设置编译选项的函数
function(set_compile_options target)
    if(MSVC)
        target_compile_options(${target} PRIVATE $<$<CONFIG:Release>:-O2>)
        set_property(TARGET ${target} PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    else()
        target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3 -flto=auto>)
        target_link_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3 -flto=auto>)
    endif()
endfunction()

# 定义目标 custom_plugins
add_library(custom_plugins SHARED)

# 添加头文件搜索路径
target_include_directories(custom_plugins PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)

# 使用file(GLOB_RECURSE ...)递归搜索指定目录下的所有.cpp和.cu文件
file(GLOB_RECURSE SOURCES
    ${CMAKE_SOURCE_DIR}/src/common/*.cpp
    ${CMAKE_SOURCE_DIR}/src/common/*.cu
    ${CMAKE_SOURCE_DIR}/src/efficientRotatedNMSPlugin/*.cpp
    ${CMAKE_SOURCE_DIR}/src/efficientRotatedNMSPlugin/*.cu
    ${CMAKE_SOURCE_DIR}/src/efficientIdxNMSPlugin/*.cpp
    ${CMAKE_SOURCE_DIR}/src/efficientIdxNMSPlugin/*.cu
)

# 将搜索到的文件添加到目标中
target_sources(custom_plugins PRIVATE ${SOURCES})

configure_cuda_trt(custom_plugins)
set_compile_options(custom_plugins)
set_target_properties(custom_plugins PROPERTIES OUTPUT_NAME custom_plugins)
if(MSVC)
    set_target_properties(custom_plugins PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR}/libs) # 设置Release模式下输出目录
    set_target_properties(custom_plugins PROPERTIES RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_SOURCE_DIR}/libs)   # 设置Debug模式下输出目录
else()
    set_target_properties(custom_plugins PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/libs)
endif()