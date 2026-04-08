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

#include <filesystem>
#include <iostream>
#include <map>

#include "manager.hpp"

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

const std::string PLUGIN_LIB_PATH = "/your/CustomPlugin/libs/libcustom_plugins.so";  // 定义全局变量  // * 新增：加载路径

namespace deploy {

// 定义自定义日志记录器 TRTLogger
struct TRTManager::TRTLogger : public nvinfer1::ILogger {
    TRTManager& parent;  // 指向 TRTManager 的引用，用于访问父类的成员

    TRTLogger(const TRTLogger&)            = default;
    TRTLogger(TRTLogger&&)                 = default;
    TRTLogger& operator=(const TRTLogger&) = delete;
    TRTLogger& operator=(TRTLogger&&)      = delete;
    // 构造函数，初始化父类引用
    TRTLogger(TRTManager& parent) : parent(parent) {}

    // 重写 TensorRT 的日志记录方法
    void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override {
        // 如果日志级别低于 WARNING，则忽略该日志
        if (severity > nvinfer1::ILogger::Severity::kWARNING) return;

        // 根据日志级别选择输出流（INFO 及以上输出到标准输出，其他输出到标准错误）
        std::ostream& stream = severity >= nvinfer1::ILogger::Severity::kINFO ? std::cout : std::cerr;

        // 查找日志级别的映射表，获取对应的前缀
        auto it = severity_map_.find(severity);
        if (it != severity_map_.end()) {
            stream << it->second << msg << '\n';  // 输出日志
        }
    }

private:
    // 日志级别与前缀的映射表
    std::map<nvinfer1::ILogger::Severity, std::string> severity_map_ = {
        {nvinfer1::ILogger::Severity::kINTERNAL_ERROR, "INTERNAL_ERROR: "},
        {         nvinfer1::ILogger::Severity::kERROR,          "ERROR: "},
        {       nvinfer1::ILogger::Severity::kWARNING,        "WARNING: "},
        {          nvinfer1::ILogger::Severity::kINFO,           "INFO: "},
        {       nvinfer1::ILogger::Severity::kVERBOSE,        "VERBOSE: "}
    };
};

// 初始化静态成员变量：引用计数器
std::atomic<int> TRTManager::refCount(0);

TRTManager& TRTManager::instance() {
    static TRTManager manager;  // 静态局部变量，线程安全
    return manager;
}

nvinfer1::IRuntime* TRTManager::getRuntime() {
    refCount.fetch_add(1, std::memory_order_acq_rel);  // 增加引用计数
    return instance().runtime.get();                   // 返回运行时对象
}

void TRTManager::engineDeleter(nvinfer1::ICudaEngine* engine) {
    if (engine) {
        delete engine;  // 调用 TensorRT 的销毁方法释放引擎
    }
    // 减少引用计数，当引用计数为 0 时释放运行时对象
    if (refCount.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        instance().runtime.reset();
    }
}

TRTManager::TRTManager() {
    // 创建自定义日志记录器
    logger = std::make_unique<TRTLogger>(*this);

    // 初始化 TensorRT 插件库
    initLibNvInferPlugins(logger.get(), "");

    // * 新增：加载动态库
    loadPluginLibrary(PLUGIN_LIB_PATH);

    // 创建 TensorRT 运行时对象
    runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(*logger));
    if (!runtime) throw std::runtime_error("Failed to call createInferRuntime().");
}

// * 新增：加载动态库
void TRTManager::loadPluginLibrary(const std::string& path) {
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error("Plugin library does not exist: " + path);
    }

#ifdef _WIN32
    mHandle = LoadLibraryA(path.c_str());
    if (!mHandle) {
        throw std::runtime_error("Failed to load plugin library: " + path);
    }
#else
    int32_t flags = RTLD_LAZY;
#if ENABLE_ASAN
    flags |= RTLD_NODELETE;
#endif
    mHandle = dlopen(path.c_str(), flags);
    if (!mHandle) {
        throw std::runtime_error("Failed to load plugin library: " + path + ". Error: " + std::string(dlerror()));
    }
#endif
}

}  // namespace deploy
