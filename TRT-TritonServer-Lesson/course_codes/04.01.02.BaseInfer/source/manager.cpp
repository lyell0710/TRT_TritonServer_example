/**
 * @file manager.cpp
 * @author laugh12321 (laugh12321@vip.qq.com)
 * @brief TensorRT 运行时管理器实现
 * @version 0.1
 * @date 2025-02-23
 *
 * @copyright Copyright (c) 2025
 *
 */

#include <iostream>

#include "manager.hpp"

namespace deploy {

// 定义日志级别与前缀的映射表
const std::map<nvinfer1::ILogger::Severity, std::string> TRTLogger::severity_map_ = {
    {nvinfer1::ILogger::Severity::kINTERNAL_ERROR, "INTERNAL_ERROR: "},
    {         nvinfer1::ILogger::Severity::kERROR,          "ERROR: "},
    {       nvinfer1::ILogger::Severity::kWARNING,        "WARNING: "},
    {          nvinfer1::ILogger::Severity::kINFO,           "INFO: "},
    {       nvinfer1::ILogger::Severity::kVERBOSE,        "VERBOSE: "}
};

// 构造函数
TRTLogger::TRTLogger(nvinfer1::ILogger::Severity severity) : severity_(severity) {}

TRTLogger::TRTLogger(TRTLogger&& other) noexcept
    : severity_(other.severity_) {
    other.severity_ = nvinfer1::ILogger::Severity::kINFO;  // 重置原对象的日志级别
}

TRTLogger& TRTLogger::operator=(TRTLogger&& other) noexcept {
    if (this != &other) {
        severity_       = other.severity_;
        other.severity_ = nvinfer1::ILogger::Severity::kINFO;  // 重置原对象的日志级别
    }
    return *this;
}

// 实现 log 方法
void TRTLogger::log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept {
    // 如果当前日志级别高于设置的日志级别，则忽略该日志
    if (severity > severity_) return;

    // 根据日志级别选择输出流（INFO 及以上输出到标准输出，其他输出到标准错误）
    std::ostream& stream = severity >= nvinfer1::ILogger::Severity::kINFO ? std::cout : std::cerr;

    // 查找日志级别的映射表，获取对应的前缀
    auto it = severity_map_.find(severity);
    if (it != severity_map_.end()) {
        stream << it->second << msg << '\n';  // 输出日志
    }
}

// 默认构造函数
TRTManager::TRTManager() : context_(nullptr), engine_(nullptr), runtime_(nullptr), logger_(nullptr) {}

TRTManager::TRTManager(TRTManager&& other) noexcept
    : context_(std::move(other.context_)),
      engine_(std::move(other.engine_)),
      runtime_(std::move(other.runtime_)),
      logger_(std::move(other.logger_)) {
    other.context_ = nullptr;
    other.engine_  = nullptr;
    other.runtime_ = nullptr;
    other.logger_  = nullptr;
}

TRTManager& TRTManager::operator=(TRTManager&& other) noexcept {
    if (this != &other) {
        context_ = std::move(other.context_);
        engine_  = std::move(other.engine_);
        runtime_ = std::move(other.runtime_);
        logger_  = std::move(other.logger_);

        other.context_ = nullptr;
        other.engine_  = nullptr;
        other.runtime_ = nullptr;
        other.logger_  = nullptr;
    }
    return *this;
}

// 初始化方法
void TRTManager::initialize(void const* blob, std::size_t size) {
    logger_ = std::make_unique<TRTLogger>(nvinfer1::ILogger::Severity::kWARNING);

    initLibNvInferPlugins(logger_.get(), "");

    // 创建 TensorRT runtime
    runtime_ = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(*logger_));
    if (!runtime_) {
        throw std::runtime_error("Failed to create TensorRT runtime.");
    }

    // 反序列化引擎
    engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(runtime_->deserializeCudaEngine(blob, size));
    if (!engine_) {
        throw std::runtime_error("Failed to deserialize CUDA engine.");
    }

    // 创建执行上下文
    context_ = std::unique_ptr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());
    if (!context_) {
        throw std::runtime_error("Failed to create execution context.");
    }
}

// 克隆方法
std::unique_ptr<TRTManager> TRTManager::clone() const {
    if (!engine_ || !runtime_) {
        throw std::runtime_error("Invalid engine or runtime in TRTManager.");
    }

    // 创建新的 TRTManager 实例
    auto newManager = std::make_unique<TRTManager>();

    // 共享 engine
    newManager->engine_ = engine_;

    // 创建新的 context
    newManager->context_ = std::unique_ptr<nvinfer1::IExecutionContext>(newManager->engine_->createExecutionContext());
    if (!newManager->context_) {
        throw std::runtime_error("Failed to create new execution context during clone.");
    }

    return newManager;
}

// 析构函数
TRTManager::~TRTManager() {
    context_.reset();
    engine_.reset();
    runtime_.reset();
    logger_.reset();
}

// nvinfer1::IExecutionContext 相关方法
bool TRTManager::setTensorAddress(char const* tensorName, void* data) {
    return context_->setTensorAddress(tensorName, data);
}

bool TRTManager::setInputShape(char const* tensorName, nvinfer1::Dims const& dims) {
    return context_->setInputShape(tensorName, dims);
}

bool TRTManager::enqueueV3(cudaStream_t stream) {
    return context_->enqueueV3(stream);
}

// nvinfer1::ICudaEngine 相关方法
nvinfer1::Dims TRTManager::getTensorShape(char const* tensorName) const noexcept {
    return engine_->getTensorShape(tensorName);
}

nvinfer1::DataType TRTManager::getTensorDataType(char const* tensorName) const noexcept {
    return engine_->getTensorDataType(tensorName);
}

nvinfer1::TensorIOMode TRTManager::getTensorIOMode(char const* tensorName) const noexcept {
    return engine_->getTensorIOMode(tensorName);
}

nvinfer1::Dims TRTManager::getProfileShape(char const* tensorName, int32_t profileIndex, nvinfer1::OptProfileSelector select) const noexcept {
    return engine_->getProfileShape(tensorName, profileIndex, select);
}

int32_t TRTManager::getNbIOTensors() const noexcept {
    return engine_->getNbIOTensors();
}

char const* TRTManager::getIOTensorName(int32_t index) const noexcept {
    return engine_->getIOTensorName(index);
}

}  // namespace deploy
