#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <cuda_runtime.h>
#include <NvInfer.h>
#include <cmath>
#include <stdexcept>
#include <numeric>

// 自定义Logger类
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << msg << std::endl;
        }
    }
} logger;

// 从文件中读取二进制内容
std::string ReadBinaryFromFile(const std::string& file) {
    std::ifstream fin(file, std::ios::in | std::ios::binary);
    if (!fin.is_open()) {
        throw std::runtime_error("Failed to open file: " + file);
    }
    fin.seekg(0, std::ios::end);
    std::string contents(fin.tellg(), '\0');
    fin.seekg(0, std::ios::beg);
    fin.read(&contents[0], contents.size());
    return contents;
}

// 读取PGM文件
std::vector<uint8_t> ReadPGMFile(const std::string& fileName, int32_t inH, int32_t inW) {
    std::ifstream infile(fileName, std::ifstream::binary);
    if (!infile.is_open()) {
        throw std::runtime_error("Failed to open file: " + fileName);
    }

    std::string magic, w, h, max;
    infile >> magic >> w >> h >> max;
    if (!infile) {
        throw std::runtime_error("Failed to read header from file: " + fileName);
    }

    infile.seekg(1, infile.cur);
    std::vector<uint8_t> buffer(inH * inW);
    infile.read(reinterpret_cast<char*>(buffer.data()), buffer.size());
    if (!infile) {
        throw std::runtime_error("Failed to read image data from file: " + fileName);
    }

    return buffer;
}

// CUDA错误检查
inline void CheckCudaError(cudaError_t err, const std::string& msg) {
    if (err != cudaSuccess) {
        throw std::runtime_error(msg + ": " + cudaGetErrorString(err));
    }
}

// 计算Softmax
void Softmax(std::vector<float>& output) {
    float maxVal = *std::max_element(output.begin(), output.end());
    float sum = 0.0f;
    for (float& val : output) {
        val = std::exp(val - maxVal); // 减去最大值防止数值溢出
        sum += val;
    }
    for (float& val : output) {
        val /= sum;
    }
}

int main(int argc, char* argv[]) {
    // 检查命令行参数
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <engine_file> <image_index>" << std::endl;
        std::cerr << "Example: " << argv[0] << " mnist-12.engine 0" << std::endl;
        return -1;
    }

    std::string engineFile = argv[1]; // 模型文件
    std::string imageIndex = argv[2]; // 图片索引

    try {
        // 读取引擎文件
        std::string engineBuffer = ReadBinaryFromFile(engineFile);

        // 创建TensorRT运行时
        std::unique_ptr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(logger)};
        if (!runtime) {
            throw std::runtime_error("Failed to create TensorRT runtime");
        }

        // 反序列化引擎
        std::unique_ptr<nvinfer1::ICudaEngine> engine{
            runtime->deserializeCudaEngine(engineBuffer.data(), engineBuffer.size())};
        if (!engine) {
            throw std::runtime_error("Failed to deserialize CUDA engine");
        }

        // 创建执行上下文
        std::unique_ptr<nvinfer1::IExecutionContext> context{engine->createExecutionContext()};
        if (!context) {
            throw std::runtime_error("Failed to create execution context");
        }

        // 获取输入输出张量信息
        size_t inputSize = 1, outputSize = 1;
        int inputH = 0, inputW = 0;
        for (int32_t i = 0, e = engine->getNbIOTensors(); i < e; i++) {
            const char* name = engine->getIOTensorName(i);
            auto shape = engine->getTensorShape(name);
            bool isInput = (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT);

            if (isInput) {
                inputH = shape.d[2];
                inputW = shape.d[3];
                inputSize = std::accumulate(shape.d, shape.d + shape.nbDims, 1, std::multiplies<size_t>());
            } else {
                outputSize = std::accumulate(shape.d, shape.d + shape.nbDims, 1, std::multiplies<size_t>());
            }
        }

        // 分配GPU内存
        float* d_input;
        float* d_output;
        CheckCudaError(cudaMalloc(&d_input, inputSize * sizeof(float)), "Failed to allocate input memory");
        CheckCudaError(cudaMalloc(&d_output, outputSize * sizeof(float)), "Failed to allocate output memory");

        // 读取输入图像并预处理
        std::string imageFile = imageIndex + ".pgm"; // 根据索引生成图片文件名
        std::vector<uint8_t> fileData = ReadPGMFile(imageFile, inputH, inputW);
        std::vector<float> h_input(inputSize);
        for (int i = 0; i < inputH * inputW; i++) {
            h_input[i] = 1.0f - static_cast<float>(fileData[i]) / 255.0f;
        }

        // 将输入数据拷贝到GPU
        CheckCudaError(cudaMemcpy(d_input, h_input.data(), inputSize * sizeof(float), cudaMemcpyHostToDevice),
                      "Failed to copy input to GPU");

        // 执行推理
        void* bindings[2] = {d_input, d_output};
        if (!context->executeV2(bindings)) {
            throw std::runtime_error("Failed to execute inference");
        }

        // 将输出数据拷贝回CPU
        std::vector<float> h_output(outputSize);
        CheckCudaError(cudaMemcpy(h_output.data(), d_output, outputSize * sizeof(float), cudaMemcpyDeviceToHost),
                      "Failed to copy output to CPU");

        // 计算Softmax并获取预测结果
        Softmax(h_output);
        int predictedClass = std::distance(h_output.begin(), std::max_element(h_output.begin(), h_output.end()));

        std::cout << "Predicted: " << predictedClass << ", Truth: " << imageIndex << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}