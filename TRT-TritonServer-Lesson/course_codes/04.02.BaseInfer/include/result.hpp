/**
 * @file result.hpp
 * @author laugh12321 (laugh12321@vip.qq.com)
 * @brief 定义与模型推理结果相关的数据结构
 * @version 0.1
 * @date 2025-02-23
 *
 * @copyright Copyright (c) 2025
 *
 */

#pragma once

#include <cstdint>
#include <iostream>
#include <optional>
#include <vector>

// 定义宏 DEPLOYAPI，用于在 Windows 和其他平台上导出类和函数
#ifdef _MSC_VER
#define DEPLOYAPI __declspec(dllexport)
#else
#define DEPLOYAPI __attribute__((visibility("default")))
#endif

// 用于生成错误信息，包含消息、文件名、行号和函数名
std::string make_error_message(const std::string& msg, const std::string& file, int line, const std::string& function);

// 定义宏 MAKE_ERROR_MESSAGE，用于简化错误信息的生成
#define MAKE_ERROR_MESSAGE(msg) make_error_message(msg, __FILE__, __LINE__, __FUNCTION__)

namespace deploy {

/**
 * @brief 图像结构体，用于存储图像数据及其尺寸信息
 */
struct DEPLOYAPI Image {
    void* ptr;         // 图像数据指针
    int   width  = 0;  // 图像宽度
    int   height = 0;  // 图像高度

    // 构造函数，初始化图像数据和尺寸
    Image(void* data, int width, int height);

    // 重载输出流运算符，用于打印图像信息
    friend std::ostream& operator<<(std::ostream& os, const Image& img);
};

/**
 * @brief 掩码结构体，用于存储掩码数据及其尺寸信息
 */
struct DEPLOYAPI Mask {
    std::vector<uint8_t> data;        // 掩码数据
    int                  width  = 0;  // 掩码宽度
    int                  height = 0;  // 掩码高度

    // 默认构造函数
    Mask() = default;

    // 构造函数，初始化掩码尺寸
    Mask(int width, int height);

    // 重载输出流运算符，用于打印掩码信息
    friend std::ostream& operator<<(std::ostream& os, const Mask& mask);

    // 默认拷贝构造函数和拷贝赋值运算符
    Mask(const Mask& other)            = default;
    Mask& operator=(const Mask& other) = default;

    // 默认移动构造函数和移动赋值运算符
    Mask(Mask&& other) noexcept            = default;
    Mask& operator=(Mask&& other) noexcept = default;
};

/**
 * @brief 关键点结构体，用于存储关键点的坐标和置信度
 */
struct DEPLOYAPI KeyPoint {
    float                x;     // 关键点的 x 坐标
    float                y;     // 关键点的 y 坐标
    std::optional<float> conf;  // 关键点的置信度，可选

    // 默认构造函数
    KeyPoint() = default;

    // 构造函数，初始化关键点坐标和置信度
    KeyPoint(float x, float y, std::optional<float> conf = std::nullopt);

    // 重载输出流运算符，用于打印关键点信息
    friend std::ostream& operator<<(std::ostream& os, const KeyPoint& kp);

    // 默认拷贝构造函数和拷贝赋值运算符
    KeyPoint(const KeyPoint& other)            = default;
    KeyPoint& operator=(const KeyPoint& other) = default;

    // 默认移动构造函数和移动赋值运算符
    KeyPoint(KeyPoint&& other) noexcept            = default;
    KeyPoint& operator=(KeyPoint&& other) noexcept = default;
};

/**
 * @brief 矩形框结构体，用于存储矩形框的坐标信息
 */
struct DEPLOYAPI Box {
    float left;    // 矩形框的左边界坐标
    float top;     // 矩形框的上边界坐标
    float right;   // 矩形框的右边界坐标
    float bottom;  // 矩形框的下边界坐标

    // 默认构造函数
    Box() = default;

    // 构造函数，初始化矩形框坐标
    Box(float left, float top, float right, float bottom);

    // 重载输出流运算符，用于打印矩形框信息
    friend std::ostream& operator<<(std::ostream& os, const Box& box);

    // 默认拷贝构造函数和拷贝赋值运算符
    Box(const Box& other)            = default;
    Box& operator=(const Box& other) = default;

    // 默认移动构造函数和移动赋值运算符
    Box(Box&& other) noexcept            = default;
    Box& operator=(Box&& other) noexcept = default;
};

/**
 * @brief 旋转矩形框结构体，继承自矩形框结构体，增加旋转角度信息
 */
struct DEPLOYAPI RotatedBox : public Box {
    float theta;  // 旋转矩形框的旋转角度

    // 默认构造函数
    RotatedBox() = default;

    // 构造函数，初始化旋转矩形框坐标和角度
    RotatedBox(float left, float top, float right, float bottom, float theta);

    // 重载输出流运算符，用于打印旋转矩形框信息
    friend std::ostream& operator<<(std::ostream& os, const RotatedBox& rbox);

    // 默认拷贝构造函数和拷贝赋值运算符
    RotatedBox(const RotatedBox& other)            = default;
    RotatedBox& operator=(const RotatedBox& other) = default;

    // 默认移动构造函数和移动赋值运算符
    RotatedBox(RotatedBox&& other) noexcept            = default;
    RotatedBox& operator=(RotatedBox&& other) noexcept = default;
};

/**
 * @brief 基础结果结构体，用于存储检测结果的基础信息，如数量、类别和得分
 */
struct DEPLOYAPI BaseRes {
    int                num = 0;  // 检测结果的数量
    std::vector<int>   classes;  // 检测结果的类别
    std::vector<float> scores;   // 检测结果的得分

    // 默认构造函数
    BaseRes() = default;

    // 构造函数，初始化检测结果的数量、类别和得分
    BaseRes(int num, const std::vector<int>& classes, const std::vector<float>& scores);

    // 默认拷贝构造函数和拷贝赋值运算符
    BaseRes(const BaseRes& other)            = default;
    BaseRes& operator=(const BaseRes& other) = default;

    // 默认移动构造函数和移动赋值运算符
    BaseRes(BaseRes&& other) noexcept            = default;
    BaseRes& operator=(BaseRes&& other) noexcept = default;
};

/**
 * @brief 检测结果结构体，继承自基础结果结构体，增加矩形框信息
 */
struct DEPLOYAPI DetectRes : public BaseRes {
    std::vector<Box> boxes;  // 检测结果的矩形框

    // 默认构造函数
    DetectRes() = default;

    // 构造函数，初始化检测结果的数量、类别、得分和矩形框
    DetectRes(int num, const std::vector<int>& classes, const std::vector<float>& scores, const std::vector<Box>& boxes);

    // 重载输出流运算符，用于打印检测结果信息
    friend std::ostream& operator<<(std::ostream& os, const DetectRes& res);

    // 默认拷贝构造函数和拷贝赋值运算符
    DetectRes(const DetectRes& other)            = default;
    DetectRes& operator=(const DetectRes& other) = default;

    // 默认移动构造函数和移动赋值运算符
    DetectRes(DetectRes&& other) noexcept            = default;
    DetectRes& operator=(DetectRes&& other) noexcept = default;
};

/**
 * @brief 旋转矩形框检测结果结构体，继承自基础结果结构体，增加旋转矩形框信息
 */
struct DEPLOYAPI OBBRes : public BaseRes {
    std::vector<RotatedBox> boxes;  // 检测结果的旋转矩形框

    // 默认构造函数
    OBBRes() = default;

    // 构造函数，初始化检测结果的数量、类别、得分和旋转矩形框
    OBBRes(int num, const std::vector<int>& classes, const std::vector<float>& scores, const std::vector<RotatedBox>& boxes);

    // 重载输出流运算符，用于打印旋转矩形框检测结果信息
    friend std::ostream& operator<<(std::ostream& os, const OBBRes& res);

    // 默认拷贝构造函数和拷贝赋值运算符
    OBBRes(const OBBRes& other)            = default;
    OBBRes& operator=(const OBBRes& other) = default;

    // 默认移动构造函数和移动赋值运算符
    OBBRes(OBBRes&& other) noexcept            = default;
    OBBRes& operator=(OBBRes&& other) noexcept = default;
};

/**
 * @brief 分割结果结构体，继承自基础结果结构体，增加矩形框和掩码信息
 */
struct DEPLOYAPI SegmentRes : public BaseRes {
    std::vector<Box>  boxes;  // 分割结果的矩形框
    std::vector<Mask> masks;  // 分割结果的掩码

    // 默认构造函数
    SegmentRes() = default;

    // 构造函数，初始化分割结果的数量、类别、得分、矩形框和掩码
    SegmentRes(int num, const std::vector<int>& classes, const std::vector<float>& scores, const std::vector<Box>& boxes, const std::vector<Mask>& masks);

    // 重载输出流运算符，用于打印分割结果信息
    friend std::ostream& operator<<(std::ostream& os, const SegmentRes& res);

    // 默认拷贝构造函数和拷贝赋值运算符
    SegmentRes(const SegmentRes& other)            = default;
    SegmentRes& operator=(const SegmentRes& other) = default;

    // 默认移动构造函数和移动赋值运算符
    SegmentRes(SegmentRes&& other) noexcept            = default;
    SegmentRes& operator=(SegmentRes&& other) noexcept = default;
};

/**
 * @brief 姿态估计结果结构体，继承自基础结果结构体，增加矩形框和关键点信息
 */
struct DEPLOYAPI PoseRes : public BaseRes {
    std::vector<Box>                   boxes;  // 姿态估计结果的矩形框
    std::vector<std::vector<KeyPoint>> kpts;   // 姿态估计结果的关键点

    // 默认构造函数
    PoseRes() = default;

    // 构造函数，初始化姿态估计结果的数量、类别、得分、矩形框和关键点
    PoseRes(int num, const std::vector<int>& classes, const std::vector<float>& scores, const std::vector<Box>& boxes, const std::vector<std::vector<KeyPoint>>& kpts);

    // 重载输出流运算符，用于打印姿态估计结果信息
    friend std::ostream& operator<<(std::ostream& os, const PoseRes& res);

    // 默认拷贝构造函数和拷贝赋值运算符
    PoseRes(const PoseRes& other)            = default;
    PoseRes& operator=(const PoseRes& other) = default;

    // 默认移动构造函数和移动赋值运算符
    PoseRes(PoseRes&& other) noexcept            = default;
    PoseRes& operator=(PoseRes&& other) noexcept = default;
};

}  // namespace deploy