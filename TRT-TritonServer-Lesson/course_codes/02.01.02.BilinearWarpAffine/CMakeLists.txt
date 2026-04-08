cmake_minimum_required(VERSION 3.12)
cmake_policy(SET CMP0091 NEW)
project(WarpAffine LANGUAGES CXX CUDA)

# 设置 CUDA 和 OpenCV 的路径
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.18)
    find_package(CUDAToolkit REQUIRED)
    set(CUDA_INCLUDE_DIRS ${CUDAToolkit_INCLUDE_DIRS})
    set(CUDA_LIBRARIES ${CUDAToolkit_LIBRARIES})
else()
    find_package(CUDA REQUIRED)
endif()
set(CMAKE_CUDA_ARCHITECTURES 89;86;75;70;61)  # CUDA 架构版本 8.9, 8.6, 7.5, 7.0, 6.1

find_package(OpenCV REQUIRED)

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

# 添加可执行文件
add_executable(bilinear_warp_affine  "${CMAKE_SOURCE_DIR}/src/bilinear_warp_affine.cu")
target_link_libraries(bilinear_warp_affine ${CUDA_LIBRARIES} ${OpenCV_LIBS})
set_compile_options(bilinear_warp_affine)
if(MSVC)
    set_target_properties(bilinear_warp_affine PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/workspace"
        RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_SOURCE_DIR}/workspace"
    )
else()
    set_target_properties(bilinear_warp_affine PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/workspace"
    )
endif()