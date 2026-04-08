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

#include "buffer.hpp"
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

// 默认构造函数
GraphManager::GraphManager() : graph_(nullptr), graph_exec_(nullptr), nodes_(nullptr) {}

// 析构函数
GraphManager::~GraphManager() { destroy(); }

// 开始捕获 CUDA 图
void GraphManager::beginCapture(cudaStream_t stream) {
    // 开始捕获 CUDA 图操作，捕获模式为 `cudaStreamCaptureModeThreadLocal`
    CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
}

// 结束捕获 CUDA 图
void GraphManager::endCapture(cudaStream_t stream) {
    // 结束捕获 CUDA 图操作，获取图句柄
    CHECK(cudaStreamEndCapture(stream, &graph_));
}

// 实例化 CUDA 图
void GraphManager::instantiate(cudaStream_t stream) {
    // 实例化 CUDA 图操作，获取执行图句柄
    CHECK(cudaGraphInstantiate(&graph_exec_, graph_, nullptr, nullptr, 0));

    // ! CUDA 12.x 开始 cudaGraphInstantiate 进行了修改
    // ! 由 __host__​cudaError_t cudaGraphInstantiate ( cudaGraphExec_t* pGraphExec, cudaGraph_t graph, cudaGraphNode_t* pErrorNode, char* pLogBuffer, size_t bufferSize )
    // ! 变更为 __host__​cudaError_t cudaGraphInstantiate ( cudaGraphExec_t* pGraphExec, cudaGraph_t graph, unsigned long long flags = 0 )
    // ! 但老版本的 cudaGraphInstantiate 仍然可用
    // ! CUDA 12.x 新用法
    // CHECK(cudaGraphInstantiate(&graph_exec_, graph_));

    // 上载执行图到指定流
    CHECK(cudaGraphUpload(graph_exec_, stream));  // ! CUDA 11.1.0 新增
}

// 执行 CUDA 图
void GraphManager::launch(cudaStream_t stream) {
    // 执行 CUDA 图操作
    CHECK(cudaGraphLaunch(graph_exec_, stream));
}

// 销毁 CUDA 图
void GraphManager::destroy() {
    if (graph_) {
        CHECK(cudaGraphDestroy(graph_));
        graph_ = nullptr;
    }
    if (graph_exec_) {
        CHECK(cudaGraphExecDestroy(graph_exec_));
        graph_exec_ = nullptr;
    }
}

// 获取图节点参数
void GraphManager::getNodesParams(size_t num) {
    // 如果节点数量为0，自动获取图中的节点数量
    if (num == 0) {
        CHECK(cudaGraphGetNodes(graph_, nullptr, &num));
    }

    if (num > 0) {
        // 为节点分配内存
        nodes_ = std::make_unique<cudaGraphNode_t[]>(num);
        CHECK(cudaGraphGetNodes(graph_, nodes_.get(), &num));
    } else {
        // 如果图中没有节点，抛出异常
        throw std::runtime_error("Failed to initialize nodes: graph has no nodes.");
    }
}

// 设置内核节点参数
void GraphManager::setKernelNodeParams(size_t index, void** params) {
    // 检查指定节点是否为内核节点
    if (getNodeType(index) != cudaGraphNodeTypeKernel) {
        throw std::runtime_error("Node at index " + std::to_string(index) + " is not a kernel node.");
    }

    // 获取当前内核节点的参数
    cudaKernelNodeParams kernelNodeParams;
    CHECK(cudaGraphKernelNodeGetParams(nodes_[index], &kernelNodeParams));

    // 根据需要修改内核参数（例如，更改内核配置）
    kernelNodeParams.kernelParams = params;

    // 更新内核节点的参数
    CHECK(cudaGraphExecKernelNodeSetParams(graph_exec_, nodes_[index], &kernelNodeParams));
}

// 设置内存拷贝节点参数
void GraphManager::setMemcpyNodeParams(size_t index, void* src, void* dst, size_t size) {
    // 检查指定节点是否为 memcpy 节点
    if (getNodeType(index) != cudaGraphNodeTypeMemcpy) {
        throw std::runtime_error("Node at index " + std::to_string(index) + " is not a memcpy node.");
    }

    // 获取当前 memcpy 节点的参数
    cudaMemcpy3DParms memcpyNodeParams;
    CHECK(cudaGraphMemcpyNodeGetParams(nodes_[index], &memcpyNodeParams));

    // 设置源和目标内存地址
    memcpyNodeParams.srcPtr = make_cudaPitchedPtr(src, size, size, 1);
    memcpyNodeParams.dstPtr = make_cudaPitchedPtr(dst, size, size, 1);
    memcpyNodeParams.extent = make_cudaExtent(size, 1, 1);

    // 更新 memcpy 节点的参数
    CHECK(cudaGraphExecMemcpyNodeSetParams(graph_exec_, nodes_[index], &memcpyNodeParams));
}

cudaGraphNodeType GraphManager::getNodeType(size_t index) {
    cudaGraphNodeType nodeType;
    CHECK(cudaGraphNodeGetType(nodes_[index], &nodeType));
    return nodeType;
}

}  // namespace deploy
