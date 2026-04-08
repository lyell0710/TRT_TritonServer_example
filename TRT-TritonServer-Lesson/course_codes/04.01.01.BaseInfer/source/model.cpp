/**
 * @file model.cpp
 * @author laugh12321 (laugh12321@vip.qq.com)
 * @brief 模型推理的基类模板 `BaseModel` 实现
 * @version 0.1
 * @date 2025-02-23
 *
 * @copyright Copyright (c) 2025
 *
 */

#include <cstring>

#include "backend.hpp"
#include "model.hpp"
#include "result.hpp"

namespace deploy {

// 构造函数，初始化推理后端对象
template <typename ResultType>
BaseModel<ResultType>::BaseModel(const std::string& trt_engine_file)
    : backend_(std::make_unique<InferBackend>(trt_engine_file)) {}

// 默认析构函数，释放资源
template <typename ResultType>
BaseModel<ResultType>::~BaseModel() = default;

// * 私有的无参构造函数，仅在 clone 方法中使用
template <typename ResultType>
BaseModel<ResultType>::BaseModel() = default;

// * 克隆后的 BaseModel 对象的智能指针
template <typename ResultType>
std::unique_ptr<BaseModel<ResultType>> BaseModel<ResultType>::clone() const {
    auto clone_model      = std::make_unique<BaseModel<ResultType>>();
    clone_model->backend_ = backend_->clone();  // < 克隆 TrtBackend
    return clone_model;
}

// 单张图像推理的实现：将单张图像包装为一个大小为1的向量，调用批量推理方法，并返回第一个结果
template <typename ResultType>
ResultType BaseModel<ResultType>::predict(const Image& image) {
    return predict(std::vector<Image>{image}).front();
}

// 批量图像推理的实现：
// 1. 调用后端推理方法 `infer`，执行 TensorRT 引擎的推理
// 2. 遍历每张图像的索引，调用 `postProcess` 方法进行后处理
// 3. 返回推理结果的向量
template <typename ResultType>
std::vector<ResultType> BaseModel<ResultType>::predict(const std::vector<Image>& images) {
    backend_->infer(images);  // 调用推理方法

    // 预分配结果空间
    std::vector<ResultType> results(images.size());
    for (auto idx = 0u; idx < images.size(); ++idx) {
        results[idx] = postProcess(idx);
    }

    return results;
}

// 获取模型的最大批量大小，通过后端的 `max_shape` 属性获取
template <typename ResultType>
int BaseModel<ResultType>::batch_size() const {
    return backend_->max_shape.x;
}

// 特化模板函数：专门用于处理 `DetectRes` 类型的结果
template <>
DetectRes BaseModel<DetectRes>::postProcess(int idx) {
    // 获取推理引擎输出的张量信息
    auto& num_tensor   = backend_->tensor_infos[1];  // 检测数量的张量
    auto& box_tensor   = backend_->tensor_infos[2];  // 检测框的张量
    auto& score_tensor = backend_->tensor_infos[3];  // 检测得分的张量
    auto& class_tensor = backend_->tensor_infos[4];  // 检测类别的张量

    // 提取当前图像的检测数量
    int    num     = static_cast<int*>(num_tensor.buffer->host())[idx];
    // 提取当前图像的检测框数据（偏移量为 idx * box_tensor.shape.d[1] * box_tensor.shape.d[2]）
    float* boxes   = static_cast<float*>(box_tensor.buffer->host()) + idx * box_tensor.shape.d[1] * box_tensor.shape.d[2];
    // 提取当前图像的检测得分数据（偏移量为 idx * score_tensor.shape.d[1]）
    float* scores  = static_cast<float*>(score_tensor.buffer->host()) + idx * score_tensor.shape.d[1];
    // 提取当前图像的检测类别数据（偏移量为 idx * class_tensor.shape.d[1]）
    int*   classes = static_cast<int*>(class_tensor.buffer->host()) + idx * class_tensor.shape.d[1];

    // 初始化检测结果对象
    DetectRes result;
    result.num   = num;                    // 设置检测数量
    int box_size = box_tensor.shape.d[2];  // 获取每个检测框的大小（通常为4，表示左、上、右、下四个坐标）

    // 获取当前图像的仿射变换对象，用于调整检测框的坐标
    auto& affine_transform = backend_->affine_transforms[idx];

    // 预分配存储空间
    result.boxes.reserve(num);
    result.scores.reserve(num);
    result.classes.reserve(num);

    // 遍历每个检测结果
    for (int i = 0; i < num; ++i) {
        // 计算当前检测框的起始索引
        int   base_index = i * box_size;
        // 提取检测框的左、上、右、下坐标
        float left = boxes[base_index], top = boxes[base_index + 1];
        float right = boxes[base_index + 2], bottom = boxes[base_index + 3];

        // 应用仿射变换，调整检测框的坐标
        affine_transform.applyTransform(left, top, &left, &top);
        affine_transform.applyTransform(right, bottom, &right, &bottom);

        // 存储到结果中
        result.boxes.emplace_back(Box{left, top, right, bottom});
        result.scores.push_back(scores[i]);
        result.classes.push_back(classes[i]);
    }

    return result;
}

// 特化模板函数：专门用于处理 `OBBRes` 类型的结果
template <>
OBBRes BaseModel<OBBRes>::postProcess(int idx) {
    // 获取推理引擎输出的张量信息
    auto& num_tensor   = backend_->tensor_infos[1];  // 检测数量的张量
    auto& box_tensor   = backend_->tensor_infos[2];  // 检测框的张量
    auto& score_tensor = backend_->tensor_infos[3];  // 检测得分的张量
    auto& class_tensor = backend_->tensor_infos[4];  // 检测类别的张量

    // 提取当前图像的检测数量
    int    num     = static_cast<int*>(num_tensor.buffer->host())[idx];
    // 提取当前图像的检测框数据（偏移量为 idx * box_tensor.shape.d[1] * box_tensor.shape.d[2]）
    float* boxes   = static_cast<float*>(box_tensor.buffer->host()) + idx * box_tensor.shape.d[1] * box_tensor.shape.d[2];
    // 提取当前图像的检测得分数据（偏移量为 idx * score_tensor.shape.d[1]）
    float* scores  = static_cast<float*>(score_tensor.buffer->host()) + idx * score_tensor.shape.d[1];
    // 提取当前图像的检测类别数据（偏移量为 idx * class_tensor.shape.d[1]）
    int*   classes = static_cast<int*>(class_tensor.buffer->host()) + idx * class_tensor.shape.d[1];

    // 初始化检测结果对象
    OBBRes result;
    result.num   = num;                    // 设置检测数量
    int box_size = box_tensor.shape.d[2];  // 获取每个检测框的大小（通常为5，表示左、上、右、下四个坐标和旋转角度）

    // 获取当前图像的仿射变换对象，用于调整检测框的坐标
    auto& affine_transform = backend_->affine_transforms[idx];

    // 预分配存储空间
    result.boxes.reserve(num);
    result.scores.reserve(num);
    result.classes.reserve(num);

    // 遍历每个检测结果
    for (int i = 0; i < num; ++i) {
        // 计算当前检测框的起始索引
        int   base_index = i * box_size;
        // 提取检测框的左、上、右、下坐标和旋转角度
        float left = boxes[base_index], top = boxes[base_index + 1];
        float right = boxes[base_index + 2], bottom = boxes[base_index + 3];
        float theta = boxes[base_index + 4];  // 旋转角度

        // 应用仿射变换，调整检测框的坐标
        affine_transform.applyTransform(left, top, &left, &top);
        affine_transform.applyTransform(right, bottom, &right, &bottom);

        // 存储到结果中
        result.boxes.emplace_back(RotatedBox{left, top, right, bottom, theta});
        result.scores.push_back(scores[i]);
        result.classes.push_back(classes[i]);
    }

    return result;
}

// 特化模板函数：专门用于处理 `SegmentRes` 类型的结果
template <>
SegmentRes BaseModel<SegmentRes>::postProcess(int idx) {
    // 获取推理引擎输出的张量信息
    auto& num_tensor   = backend_->tensor_infos[1];  // 检测数量的张量
    auto& box_tensor   = backend_->tensor_infos[2];  // 检测框的张量
    auto& score_tensor = backend_->tensor_infos[3];  // 检测得分的张量
    auto& class_tensor = backend_->tensor_infos[4];  // 检测类别的张量
    auto& mask_tensor  = backend_->tensor_infos[5];  // 分割掩码的张量
    int   mask_height  = mask_tensor.shape.d[2];     // 掩码的高度
    int   mask_width   = mask_tensor.shape.d[3];     // 掩码的宽度

    // 提取当前图像的检测数量
    int      num     = static_cast<int*>(num_tensor.buffer->host())[idx];
    // 提取当前图像的检测框数据（偏移量为 idx * box_tensor.shape.d[1] * box_tensor.shape.d[2]）
    float*   boxes   = static_cast<float*>(box_tensor.buffer->host()) + idx * box_tensor.shape.d[1] * box_tensor.shape.d[2];
    // 提取当前图像的检测得分数据（偏移量为 idx * score_tensor.shape.d[1]）
    float*   scores  = static_cast<float*>(score_tensor.buffer->host()) + idx * score_tensor.shape.d[1];
    // 提取当前图像的检测类别数据（偏移量为 idx * class_tensor.shape.d[1]）
    int*     classes = static_cast<int*>(class_tensor.buffer->host()) + idx * class_tensor.shape.d[1];
    // 提取当前图像的分割掩码数据（偏移量为 idx * mask_tensor.shape.d[1] * mask_height * mask_width）
    uint8_t* masks   = static_cast<uint8_t*>(mask_tensor.buffer->host()) + idx * mask_tensor.shape.d[1] * mask_height * mask_width;

    // 初始化检测结果对象
    SegmentRes result;
    result.num   = num;                    // 设置检测数量
    int box_size = box_tensor.shape.d[2];  // 获取每个检测框的大小（通常为4，表示左、上、右、下四个坐标）

    // 获取当前图像的仿射变换对象，用于调整检测框的坐标
    auto& affine_transform = backend_->affine_transforms[idx];

    // 预分配存储空间
    result.boxes.reserve(num);
    result.scores.reserve(num);
    result.classes.reserve(num);
    result.masks.reserve(num);

    // 遍历每个检测结果
    for (int i = 0; i < num; ++i) {
        // 计算当前检测框的起始索引
        int   base_index = i * box_size;
        // 提取检测框的左、上、右、下坐标
        float left = boxes[base_index], top = boxes[base_index + 1];
        float right = boxes[base_index + 2], bottom = boxes[base_index + 3];

        // 应用仿射变换，调整检测框的坐标
        affine_transform.applyTransform(left, top, &left, &top);
        affine_transform.applyTransform(right, bottom, &right, &bottom);

        // 存储到结果中
        result.boxes.emplace_back(Box{left, top, right, bottom});
        result.scores.push_back(scores[i]);
        result.classes.push_back(classes[i]);

        // 创建掩码对象，并调整掩码的大小（去除边缘区域）
        Mask mask(mask_width - 2 * affine_transform.dst_offset_x, mask_height - 2 * affine_transform.dst_offset_y);

        // 裁剪掩码的边缘区域，应用偏移量调整位置
        int start_idx = i * mask_height * mask_width;
        int src_idx   = start_idx + affine_transform.dst_offset_y * mask_width + affine_transform.dst_offset_x;
        for (int y = 0; y < mask.height; ++y) {
            std::memcpy(&mask.data[y * mask.width], masks + src_idx, mask.width);
            src_idx += mask_width;
        }

        // 存储掩码到结果中
        result.masks.emplace_back(std::move(mask));
    }

    return result;
}

// 特化模板函数：专门用于处理 `PoseRes` 类型的结果
template <>
PoseRes BaseModel<PoseRes>::postProcess(int idx) {
    // 获取推理引擎输出的张量信息
    auto& num_tensor   = backend_->tensor_infos[1];  // 检测数量的张量
    auto& box_tensor   = backend_->tensor_infos[2];  // 检测框的张量
    auto& score_tensor = backend_->tensor_infos[3];  // 检测得分的张量
    auto& class_tensor = backend_->tensor_infos[4];  // 检测类别的张量
    auto& kpt_tensor   = backend_->tensor_infos[5];  // 关键点的张量
    int   nkpt         = kpt_tensor.shape.d[2];      // 关键点的数量
    int   ndim         = kpt_tensor.shape.d[3];      // 每个关键点的维度（2或3，表示x,y或x,y,score）

    // 提取当前图像的检测数量
    int    num     = static_cast<int*>(num_tensor.buffer->host())[idx];
    // 提取当前图像的检测框数据（偏移量为 idx * box_tensor.shape.d[1] * box_tensor.shape.d[2]）
    float* boxes   = static_cast<float*>(box_tensor.buffer->host()) + idx * box_tensor.shape.d[1] * box_tensor.shape.d[2];
    // 提取当前图像的检测得分数据（偏移量为 idx * score_tensor.shape.d[1]）
    float* scores  = static_cast<float*>(score_tensor.buffer->host()) + idx * score_tensor.shape.d[1];
    // 提取当前图像的检测类别数据（偏移量为 idx * class_tensor.shape.d[1]）
    int*   classes = static_cast<int*>(class_tensor.buffer->host()) + idx * class_tensor.shape.d[1];
    // 提取当前图像的关键点数据（偏移量为 idx * kpt_tensor.shape.d[1] * nkpt * ndim）
    float* kpts    = static_cast<float*>(kpt_tensor.buffer->host()) + idx * kpt_tensor.shape.d[1] * nkpt * ndim;

    // 初始化检测结果对象
    PoseRes result;
    result.num   = num;                    // 设置检测数量
    int box_size = box_tensor.shape.d[2];  // 获取每个检测框的大小（通常为4，表示左、上、右、下四个坐标）

    // 获取当前图像的仿射变换对象，用于调整检测框的坐标
    auto& affine_transform = backend_->affine_transforms[idx];

    // 预分配存储空间
    result.boxes.reserve(num);
    result.scores.reserve(num);
    result.classes.reserve(num);
    result.kpts.reserve(num);

    // 遍历每个检测结果
    for (int i = 0; i < num; ++i) {
        // 计算当前检测框的起始索引
        int   base_index = i * box_size;
        // 提取检测框的左、上、右、下坐标
        float left = boxes[base_index], top = boxes[base_index + 1];
        float right = boxes[base_index + 2], bottom = boxes[base_index + 3];

        // 应用仿射变换，调整检测框的坐标
        affine_transform.applyTransform(left, top, &left, &top);
        affine_transform.applyTransform(right, bottom, &right, &bottom);

        // 存储到结果中
        result.boxes.emplace_back(Box{left, top, right, bottom});
        result.scores.push_back(scores[i]);
        result.classes.push_back(classes[i]);

        // 提取关键点数据
        std::vector<KeyPoint> keypoints;
        for (int j = 0; j < nkpt; ++j) {
            // 提取关键点的x, y坐标
            float x = kpts[i * nkpt * ndim + j * ndim];
            float y = kpts[i * nkpt * ndim + j * ndim + 1];
            // 应用仿射变换，调整关键点的坐标
            affine_transform.applyTransform(x, y, &x, &y);
            // 如果关键点有置信度分数（ndim == 3），则存储x, y, score，否则只存储x, y
            keypoints.emplace_back((ndim == 2) ? KeyPoint(x, y) : KeyPoint(x, y, kpts[i * nkpt * ndim + j * ndim + 2]));
        }
        // 存储关键点到结果中
        result.kpts.emplace_back(std::move(keypoints));
    }

    return result;
}

// 显式实例化 `BaseModel` 模板类
template class BaseModel<DetectRes>;
template class BaseModel<OBBRes>;
template class BaseModel<SegmentRes>;
template class BaseModel<PoseRes>;

}  // namespace deploy