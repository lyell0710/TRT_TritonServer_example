#include "NvInfer.h"  // namespace nvinfer1
#include "NvOnnxParser.h" // namespace nvonnxparser

#include <iostream>
#include <fstream>

// 自定义Logger类
class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        // 忽略信息级别消息
        if (severity <= nvinfer1::ILogger::Severity::kWARNING)
            std::cout << msg << std::endl;
    }
} logger;

int main() {
    const std::string onnxFile = "mnist-12.onnx"; // 替换为你的ONNX文件路径
    const std::string engineFile = "mnist-12.engine"; // 替换为输出引擎文件的路径

    // 创建 TensorRT 构建器
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);
    if (!builder) {
        std::cerr << "Failed to create TensorRT builder." << std::endl;
        return -1;
    }

    // 创建网络定义，使用显式批量模式
    uint32_t flags = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(flags);
    if (!network) {
        std::cerr << "Failed to create network definition." << std::endl;
        delete builder;
        return -1;
    }

    // 创建 ONNX 解析器
    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, logger);
    if (!parser) {
        std::cerr << "Failed to create ONNX parser." << std::endl;
        delete network;
        delete builder;
        return -1;
    }

    // 解析ONNX文件
    if (!parser->parseFromFile(onnxFile.c_str(), static_cast<int32_t>(nvinfer1::ILogger::Severity::kWARNING))) {
        // 打印错误信息并退出
        std::cerr << "Failed to parse ONNX file: " << onnxFile << std::endl;
        for (int i = 0; i < parser->getNbErrors(); ++i) {
            std::cerr << "Error: " << parser->getError(i)->desc() << std::endl;
        }
        delete parser;
        delete network;
        delete builder;
        return -1;
    }

    // 创建构建器配置
    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
    if (!config) {
        std::cerr << "Failed to create builder config." << std::endl;
        delete parser;
        delete network;
        delete builder;
        return -1;
    }

    // 设置构建器配置选项
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1U << 20);  // 设置最大工作区大小
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kTACTIC_SHARED_MEMORY, 48 << 10); // 设置最大共享内存 

    // 构建 TensorRT 引擎
    nvinfer1::IHostMemory* serializedModel = builder->buildSerializedNetwork(*network, *config);
    if (!serializedModel) {
        std::cerr << "Failed to build serialized network." << std::endl;
        delete config;
        delete parser;
        delete network;
        delete builder;
        return -1;
    }

    // 将构建好的引擎序列化到文件
    std::ofstream outFile(engineFile, std::ios::binary);
    if (!outFile) {
        std::cerr << "Failed to open engine file for writing: " << engineFile << std::endl;
        delete serializedModel;
        delete config;
        delete parser;
        delete network;
        delete builder;
        return -1;
    }
    outFile.write(reinterpret_cast<const char*>(serializedModel->data()), serializedModel->size());
    outFile.close();

    std::cout << "Engine has been successfully built and saved to " << engineFile << std::endl;

    // 释放创建的对象
    delete serializedModel;
    delete config;
    delete parser;
    delete network;
    delete builder;

    return 0;
}