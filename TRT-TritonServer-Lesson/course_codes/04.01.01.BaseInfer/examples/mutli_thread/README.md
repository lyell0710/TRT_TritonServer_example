# 多线程推理中的模型克隆

TensorRT 允许多个线程共享一个 engine 进行推理。一个 engine 可以支持多个 context 同时使用，这意味着多个线程可以共享同一份模型权重和参数，而仅在内存或显存中保留一份副本。因此，尽管复制了多个对象，但模型的内存占用并不会线性增加。

我们提供了以下接口用于模型克隆（以 DetectModel 为例）：

- C++: `DetectModel::clone()`

## C++ 示例

```cpp
#include <memory>
#include <opencv2/opencv.hpp>

// 为了方便调用，模块除使用CUDA、TensorRT外，其余均使用标准库实现
#include "deploy/model.hpp"  // 包含模型推理相关的类定义
#include "deploy/option.hpp"  // 包含推理选项的配置类定义
#include "deploy/result.hpp"  // 包含推理结果的定义

int main() {
    // 配置推理选项
    deploy::InferOption option;
    option.enableSwapRB();  // 启用通道交换（从BGR到RGB）

    // 初始化模型
    auto model = std::make_unique<deploy::DetectModel>("yolo11n-with-plugin.engine", option);

    // 加载图片
    cv::Mat cvim = cv::imread("test_image.jpg");
    deploy::Image im(cvim.data, cvim.cols, cvim.rows);

    // 模型预测
    deploy::DetResult result = model->predict(im);

    // 可视化（代码省略）
    // ...  // 可视化部分代码未提供，可根据需要实现

    // 克隆模型并进行预测
    auto clone_model = model->clone();
    deploy::DetResult clone_result = clone_model->predict(im);

    return 0;  // 程序正常结束
}
```

### 模型克隆与不克隆的显存占用对比

**硬件配置**:
- **CPU**: AMD Ryzen 7 5700X 8-Core Processor  
- **GPU**: NVIDIA GeForce RTX 2080 Ti 22GB  
**模型**: YOLO11x

| 模型数 | `model.clone()` | 不克隆模型 |
|:---   |:-----           |:-----    |
|1      |408M             |408M      |
|2      |536M             |716M      |
|3      |662M             |1092M      |
|4      |790M             |1470M      |

> [!IMPORTANT]
>
> 使用 `model.clone()` 后，显存占用仍会有少量增加，主要原因如下：
>
> 1. **输入输出的显存预分配**：
>    - 每个 `context` 需要为模型的输入和输出分配独立的显存缓冲区。即使多个 `context` 共享同一个 `engine`，输入输出缓冲区仍需单独分配，这会占用额外的显存。
>    - 输入输出缓冲区的显存占用取决于模型输入输出的形状和数据类型。对于较大的输入输出，这部分显存占用可能会比较显著。
>
> 2. **Context 的显存开销**：
>    - 每个 `context` 是独立的执行上下文，用于管理模型推理的执行流。即使共享了 `engine`，每个 `context` 仍然需要分配一些额外的显存来存储执行状态、中间结果和临时缓冲区。
>    - `context` 的显存占用通常比 `engine` 小得多，但随着 `context` 数量的增加，显存占用也会逐渐增加。
>
> 3. **CUDA 运行时开销**：
>    - CUDA 运行时需要为每个 `context` 分配一些额外的显存来管理执行流、内核启动和内存操作。
>    - TensorRT 在推理过程中可能会使用一些优化技术（如内存重用、内核融合等），这些优化技术可能会引入额外的显存开销。
