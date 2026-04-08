# 设置项目
cmake_minimum_required(VERSION 3.12)
project(TensorRTAPI VERSION 1.0.0 LANGUAGES CXX CUDA)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 生成编译数据库，便于代码分析工具使用
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

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

# 定义目标 infer_demo
add_executable(infer_demo "${PROJECT_SOURCE_DIR}/src/infer.cpp")
configure_cuda_trt(infer_demo)
set_compile_options(infer_demo)
set_target_properties(infer_demo PROPERTIES OUTPUT_NAME infer_demo)

# 定义目标 build_demo
add_executable(build_demo "${PROJECT_SOURCE_DIR}/src/build.cpp")
configure_cuda_trt(build_demo)
set_compile_options(build_demo)
set_target_properties(build_demo PROPERTIES OUTPUT_NAME build_demo)

# 设置输出目录
if(MSVC)
    foreach(target infer_demo build_demo)
        set_target_properties(${target} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR}/workspace
            ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR}/workspace
            RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_SOURCE_DIR}/workspace
            ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${CMAKE_SOURCE_DIR}/workspace
        )
    endforeach()
else()
    foreach(target infer_demo build_demo)
        set_target_properties(${target} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/workspace)
    endforeach()
endif()