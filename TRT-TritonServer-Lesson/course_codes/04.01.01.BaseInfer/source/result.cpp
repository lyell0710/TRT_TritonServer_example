/**
 * @file result.cpp
 * @author laugh12321 (laugh12321@vip.qq.com)
 * @brief 模型推理结果相关的数据结构实现
 * @version 0.1
 * @date 2025-02-23
 *
 * @copyright Copyright (c) 2025
 *
 */

#include <sstream>
#include <stdexcept>

#include "result.hpp"

std::string make_error_message(const std::string& msg, const std::string& file, int line, const std::string& function) {
    std::stringstream ss;
    ss << "Error in " << file << ":" << line << " (" << function << "): " << msg;
    return ss.str();
}

namespace deploy {

// 图像结构体的构造函数，初始化图像数据指针和尺寸，并检查宽度和高度是否为正
Image::Image(void* data, int width, int height) : ptr(data), width(width), height(height) {
    if (width <= 0 || height <= 0) {
        throw std::invalid_argument(MAKE_ERROR_MESSAGE("Image: width and height must be positive"));
    }
}

std::ostream& operator<<(std::ostream& os, const Image& img) {
    os << "Image(width=" << img.width << ", height=" << img.height << ", ptr=" << img.ptr << ")";
    return os;
}

// 掩码结构体的构造函数，初始化掩码尺寸，并检查宽度和高度是否为正。同时分配掩码数据的内存
Mask::Mask(int width, int height) : width(width), height(height) {
    if (width < 0 || height < 0) {
        throw std::invalid_argument(MAKE_ERROR_MESSAGE("Mask: width and height must be positive"));
    }
    data.resize(width * height);
}

std::ostream& operator<<(std::ostream& os, const Mask& mask) {
    os << "Mask(width=" << mask.width << ", height=" << mask.height << ", data size=" << mask.data.size() << ")";
    return os;
}

// 关键点结构体的构造函数，初始化关键点的坐标和置信度
KeyPoint::KeyPoint(float x, float y, std::optional<float> conf) : x(x), y(y), conf(conf) {}

std::ostream& operator<<(std::ostream& os, const KeyPoint& kp) {
    os << "KeyPoint(x=" << kp.x << ", y=" << kp.y;
    if (kp.conf) {
        os << ", conf=" << *kp.conf;
    }
    os << ")";
    return os;
}

// 矩形框结构体的构造函数，初始化矩形框的坐标
Box::Box(float left, float top, float right, float bottom) : left(left), top(top), right(right), bottom(bottom) {}

std::ostream& operator<<(std::ostream& os, const Box& box) {
    os << "Box(left=" << box.left << ", top=" << box.top << ", right=" << box.right << ", bottom=" << box.bottom << ")";
    return os;
}

// 旋转矩形框结构体的构造函数，初始化矩形框的坐标和旋转角度
RotatedBox::RotatedBox(float left, float top, float right, float bottom, float theta) : Box(left, top, right, bottom), theta(theta) {}

std::ostream& operator<<(std::ostream& os, const RotatedBox& rbox) {
    os << "RotatedBox(left=" << rbox.left << ", top=" << rbox.top << ", right=" << rbox.right << ", bottom=" << rbox.bottom << ", theta=" << rbox.theta << ")";
    return os;
}

// 基础结果结构体的构造函数，初始化检测结果的数量、类别和得分
BaseRes::BaseRes(int num, const std::vector<int>& classes, const std::vector<float>& scores) : num(num), classes(classes), scores(scores) {}

// 检测结果结构体的构造函数，初始化检测结果的数量、类别、得分和矩形框
DetectRes::DetectRes(int num, const std::vector<int>& classes, const std::vector<float>& scores, const std::vector<Box>& boxes)
    : BaseRes(num, classes, scores), boxes(boxes) {}

std::ostream& operator<<(std::ostream& os, const DetectRes& res) {
    os << "DetectRes(\n    num=" << res.num << ",\n    classes=[";
    for (const auto& c : res.classes) os << c << ", ";
    os << "],\n    scores=[";
    for (const auto& s : res.scores) os << s << ", ";
    os << "],\n    boxes=[\n";
    for (const auto& box : res.boxes) os << "        " << box << ",\n";
    os << "    ]\n)";
    return os;
}

// 旋转矩形框检测结果结构体的构造函数，初始化检测结果的数量、类别、得分和旋转矩形框
OBBRes::OBBRes(int num, const std::vector<int>& classes, const std::vector<float>& scores, const std::vector<RotatedBox>& boxes)
    : BaseRes(num, classes, scores), boxes(boxes) {}

std::ostream& operator<<(std::ostream& os, const OBBRes& res) {
    os << "OBBRes(\n    num=" << res.num << ",\n    classes=[";
    for (const auto& c : res.classes) os << c << ", ";
    os << "],\n    scores=[";
    for (const auto& s : res.scores) os << s << ", ";
    os << "],\n    boxes=[\n";
    for (const auto& box : res.boxes) os << "        " << box << ",\n";
    os << "    ]\n)";
    return os;
}

// 分割结果结构体的构造函数，初始化分割结果的数量、类别、得分、矩形框和掩码
SegmentRes::SegmentRes(int num, const std::vector<int>& classes, const std::vector<float>& scores, const std::vector<Box>& boxes, const std::vector<Mask>& masks)
    : BaseRes(num, classes, scores), boxes(boxes), masks(masks) {}

std::ostream& operator<<(std::ostream& os, const SegmentRes& res) {
    os << "SegmentRes(\n    num=" << res.num << ",\n    classes=[";
    for (const auto& c : res.classes) os << c << ", ";
    os << "],\n    scores=[";
    for (const auto& s : res.scores) os << s << ", ";
    os << "],\n    boxes: [\n";
    for (const auto& box : res.boxes) os << "        " << box << ",\n";
    os << "],\n    masks: [\n";
    for (const auto& mask : res.masks) os << "        " << mask << "\n";
    os << "    ]\n)";
    return os;
}

// 姿态估计结果结构体的构造函数，初始化姿态估计结果的数量、类别、得分、矩形框和关键点
PoseRes::PoseRes(int num, const std::vector<int>& classes, const std::vector<float>& scores, const std::vector<Box>& boxes, const std::vector<std::vector<KeyPoint>>& kpts)
    : BaseRes(num, classes, scores), boxes(boxes), kpts(kpts) {}

std::ostream& operator<<(std::ostream& os, const PoseRes& res) {
    os << "PoseRes(\n    num=" << res.num << ",\n    classes=[";
    for (const auto& c : res.classes) os << c << ", ";
    os << "],\n    scores=[";
    for (const auto& s : res.scores) os << s << ", ";
    os << "],\n    boxes=[\n";
    for (const auto& box : res.boxes) os << "        " << box << "\n";
    os << "],\n    kpts=[\n";
    for (const auto& kp_list : res.kpts) {
        os << "        [ ";
        for (const auto& kp : kp_list) os << "            " << kp << ", ";
        os << "        ],\n";
    }
    os << "    ]\n)";
    return os;
}

}  // namespace deploy