/**
 * @file manager.hpp
 * @author laugh12321 (laugh12321@vip.qq.com)
 * @brief TensorRT 管理器定义
 * @version 0.1
 * @date 2025-02-23
 *
 * @copyright Copyright (c) 2025
 *
 */

#pragma once

#include <NvInferPlugin.h>

#include <map>
#include <memory>

namespace deploy {

/**
 * @class TRTLogger
 * @brief TensorRT 日志记录器类，继承自 nvinfer1::ILogger。
 * 用于管理 TensorRT 的日志记录行为，支持设置日志级别并重写日志记录方法。
 */
class TRTLogger : public nvinfer1::ILogger {
public:
    /**
     * @brief 构造函数，允许设置日志级别，默认为 INFO。
     * @param severity 日志级别，默认为 nvinfer1::ILogger::Severity::kINFO。
     */
    explicit TRTLogger(nvinfer1::ILogger::Severity severity = nvinfer1::ILogger::Severity::kINFO);

    // 移动构造函数
    TRTLogger(TRTLogger&& other) noexcept;

    // 移动赋值运算符
    TRTLogger& operator=(TRTLogger&& other) noexcept;

    // 禁用拷贝构造函数和拷贝赋值运算符
    TRTLogger(const TRTLogger&)            = delete;
    TRTLogger& operator=(const TRTLogger&) = delete;

    /**
     * @brief 重写 TensorRT 的日志记录方法。
     * @param severity 日志级别。
     * @param msg 日志消息。
     */
    void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override;

private:
    nvinfer1::ILogger::Severity severity_;  // < 当前日志级别

    /**
     * @brief 日志级别与前缀的映射表。
     * 用于将日志级别转换为可读的字符串前缀。
     */
    static const std::map<nvinfer1::ILogger::Severity, std::string> severity_map_;
};

/**
 * @class TRTManager
 * @brief TensorRT 管理器类，用于管理 TensorRT 的上下文、引擎和运行时。
 * 提供初始化、克隆、设置张量地址、设置输入形状、执行推理等方法。
 */
class TRTManager {
public:
    /**
     * @brief 默认构造函数。
     */
    TRTManager();

    /**
     * @brief 析构函数。
     */
    ~TRTManager();

    /**
     * @brief 移动构造函数。
     */
    TRTManager(TRTManager&& other) noexcept;

    /**
     * @brief 移动赋值运算符。
     */
    TRTManager& operator=(TRTManager&& other) noexcept;

    // 禁用拷贝构造函数和拷贝赋值运算符
    TRTManager(const TRTManager&)            = delete;
    TRTManager& operator=(const TRTManager&) = delete;

    /**
     * @brief 初始化方法，用于加载 TensorRT 引擎。
     * @param blob 包含引擎数据的指针。
     * @param size 引擎数据的大小。
     */
    void initialize(void const* blob, std::size_t size);

    /**
     * @brief 克隆方法，返回一个 TRTManager 的独占指针。
     * @return std::unique_ptr<TRTManager> 克隆的 TRTManager 对象。
     */
    std::unique_ptr<TRTManager> clone() const;

    /**
     * @brief 设置张量地址。
     * @param tensorName 张量名称。
     * @param data 张量数据指针。
     * @return bool 设置是否成功。
     */
    bool setTensorAddress(char const* tensorName, void* data);

    /**
     * @brief 设置输入形状。
     * @param tensorName 张量名称。
     * @param dims 张量维度。
     * @return bool 设置是否成功。
     */
    bool setInputShape(char const* tensorName, nvinfer1::Dims const& dims);

    /**
     * @brief 在指定的 CUDA 流上执行推理。
     * @param stream CUDA 流。
     * @return bool 推理是否成功。
     */
    bool enqueueV3(cudaStream_t stream);

    /**
     * @brief 获取张量的形状。
     * @param tensorName 张量名称。
     * @return nvinfer1::Dims 张量的形状。
     */
    nvinfer1::Dims getTensorShape(char const* tensorName) const noexcept;

    /**
     * @brief 获取张量的数据类型。
     * @param tensorName 张量名称。
     * @return nvinfer1::DataType 张量的数据类型。
     */
    nvinfer1::DataType getTensorDataType(char const* tensorName) const noexcept;

    /**
     * @brief 获取张量的输入输出模式。
     * @param tensorName 张量名称。
     * @return nvinfer1::TensorIOMode 张量的输入输出模式。
     */
    nvinfer1::TensorIOMode getTensorIOMode(char const* tensorName) const noexcept;

    /**
     * @brief 获取指定配置文件的张量形状。
     * @param tensorName 张量名称。
     * @param profileIndex 配置文件索引。
     * @param select 配置文件选择器。
     * @return nvinfer1::Dims 张量的形状。
     */
    nvinfer1::Dims getProfileShape(char const* tensorName, int32_t profileIndex, nvinfer1::OptProfileSelector select) const noexcept;

    /**
     * @brief 获取输入输出张量的数量。
     * @return int32_t 输入输出张量的数量。
     */
    int32_t getNbIOTensors() const noexcept;

    /**
     * @brief 获取指定索引的输入输出张量名称。
     * @param index 张量索引。
     * @return char const* 张量名称。
     */
    char const* getIOTensorName(int32_t index) const noexcept;

private:
    std::unique_ptr<nvinfer1::IExecutionContext> context_;  // < TensorRT 执行上下文
    std::shared_ptr<nvinfer1::ICudaEngine>       engine_;   // < TensorRT CUDA 引擎
    std::unique_ptr<nvinfer1::IRuntime>          runtime_;  // < TensorRT 运行时
    std::unique_ptr<TRTLogger>                   logger_;   // < TensorRT 日志记录器
};

}  // namespace deploy
