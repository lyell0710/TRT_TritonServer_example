/**
 * @file manager.hpp
 * @author laugh12321 (laugh12321@vip.qq.com)
 * @brief TensorRT 运行时管理器定义
 * @version 0.1
 * @date 2025-02-23
 *
 * @copyright Copyright (c) 2025
 *
 */

#pragma once

#include <NvInferPlugin.h>

#include <atomic>
#include <memory>

namespace deploy {

/**
 * @brief TensorRT 运行时管理器类，用于单例模式管理 TensorRT 的运行时环境和日志记录
 *
 * ! 使用引用计数（std::atomic<int> refCount）来管理 TensorRT 运行时的释放时机，确保在所有反序列化引擎和执行上下文销毁之前，
 * ! 不会提前销毁运行时实例。这是为了解决在多执行上下文（IExecutionContext）场景下可能出现的错误：
 *
 * ! ERROR: IRuntime::~IRuntime: Error Code 3: API Usage Error (Parameter check failed, condition: mEngineCounter.use_count() == 1.
 * ! Destroying a runtime before destroying deserialized engines created by the runtime leads to undefined behavior.)
 *
 * ! 该错误是由于运行时（IRuntime）被提前销毁，而其创建的引擎（ICudaEngine）或执行上下文（IExecutionContext）尚未销毁，从而导致未定义行为。
 * ! 参考博客分析：https://blog.csdn.net/qq_42190134/article/details/135365945
 */
class TRTManager {
public:
    /**
     * @brief 获取 TensorRT 运行时对象。
     *
     * @return nvinfer1::IRuntime* 返回 TensorRT 的运行时实例。
     */
    static nvinfer1::IRuntime* getRuntime();

    /**
     * @brief 引擎删除器，用于安全释放 TensorRT 引擎资源。
     *
     * @param engine 要释放的 TensorRT 引擎对象。
     */
    static void engineDeleter(nvinfer1::ICudaEngine* engine);

private:
    TRTManager();
    ~TRTManager() = default;

    TRTManager(const TRTManager&)            = delete;
    TRTManager& operator=(const TRTManager&) = delete;

    struct TRTLogger;                             // TensorRT 日志记录器
    std::unique_ptr<TRTLogger>          logger;   // 日志记录器实例
    std::unique_ptr<nvinfer1::IRuntime> runtime;  // TensorRT 运行时实例

    static TRTManager&      instance();           // 获取单例实例
    static std::atomic<int> refCount;             // 引用计数，用于管理运行时的释放时机
};

}  // namespace deploy
