/**
 * @file model.hpp
 * @author laugh12321 (laugh12321@vip.qq.com)
 * @brief 定义了模型推理的基类模板 `BaseModel`，用于封装模型的推理逻辑
 * @version 0.1
 * @date 2025-02-23
 *
 * @copyright Copyright (c) 2025
 *
 */

#pragma once

#include <memory>

#include "result.hpp"

namespace deploy {

class InferBackend;  // 前向声明 InferBackend 类，表示推理后端

/**
 * @brief 基类模板，用于定义模型的基本结构和接口
 *
 * 该模板类封装了模型的推理逻辑，适用于多种类型的模型
 * 通过模板参数 `ResultType`，可以灵活地支持不同类型的推理结果
 *
 * @tparam ResultType 推理结果的类型，例如 `DetectRes、`OBBRes` 等
 */
template <typename ResultType>
class DEPLOYAPI BaseModel {
public:
    // 构造函数，初始化模型，需要传入推理引擎文件路径
    explicit BaseModel(const std::string& trt_engine_file);

    // * 私有的无参构造函数，仅在 clone 方法中使用
    BaseModel();

    // 析构函数，释放模型资源
    ~BaseModel();

    // * 克隆后的 BaseModel 对象的智能指针
    std::unique_ptr<BaseModel<ResultType>> clone() const;

    // 接收单张图像的推理接口，返回单个推理结果
    ResultType predict(const Image& image);

    // 接收多张图像的推理接口，返回多个推理结果的集合
    std::vector<ResultType> predict(const std::vector<Image>& images);

    // 获取模型的推理批量大小
    int batch_size() const;

protected:
    // 子类需要实现的后处理逻辑，`idx` 表示当前处理的索引
    ResultType postProcess(int idx);

    // 推理后端对象，使用智能指针管理资源
    std::unique_ptr<InferBackend> backend_;
};

// 定义别名
using DetectModel  = DEPLOYAPI  BaseModel<DetectRes>;
using OBBModel     = DEPLOYAPI     BaseModel<OBBRes>;
using SegmentModel = DEPLOYAPI BaseModel<SegmentRes>;
using PoseModel    = DEPLOYAPI    BaseModel<PoseRes>;

}  // namespace deploy