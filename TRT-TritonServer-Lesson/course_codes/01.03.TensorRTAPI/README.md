# TensorRT C++ API 使用示例

本项目展示了如何使用 NVIDIA TensorRT 和 C++ API 构建和运行深度学习推理任务。项目包含两个主要的可执行文件：`build_demo` 和 `infer_demo`，分别用于构建 TensorRT 引擎和执行推理任务。

## 快速开始

### 编译项目

使用以下命令编译项目：
```bash
cmake -S . -B build -DTENSORRT_PATH="/path/to/your/TensorRT" # 在本docker 环境中， TensorRT 的安装路径在 /usr 目录下
cmake --build build -j8 --config Release
```

编译完成后，可执行文件 `build_demo` 和 `infer_demo` 将生成在 `workspace` 文件夹中。

### 运行示例

1. **构建 TensorRT 引擎**  
   使用 `build_demo` 构建 TensorRT 引擎：
   ```bash
   ./workspace/build_demo
   ```
   输出示例：
   ```
   Engine has been successfully built and saved to mnist-12.engine
   ```

2. **执行推理任务**  
   使用 `infer_demo` 运行推理任务：
   ```bash
   ./workspace/infer_demo mnist-12.engine 1
   ```
   输出示例：
   ```
   Predicted: 1, Truth: 1
   ```

## 项目结构

- `src/`：源代码目录，包含 `build.cpp` 和 `infer.cpp`。
- `workspace/`：编译输出目录，包含生成的可执行文件和 TensorRT 引擎文件。
- `CMakeLists.txt`：CMake 配置文件，用于项目编译和配置。

## 编译选项

- **TensorRT 路径**  
  通过 `-DTENSORRT_PATH` 指定 TensorRT 的安装路径。例如：
  ```bash
  cmake -S . -B build -DTENSORRT_PATH="/usr" # 在本docker 环境中， TensorRT 的安装路径在 /usr 目录下
  ```

- **CUDA 架构版本**  
  在 `CMakeLists.txt` 中，`CMAKE_CUDA_ARCHITECTURES` 设置了支持的 CUDA 架构版本。根据目标硬件调整该设置。

## 注意事项

- 确保 CUDA 和 TensorRT 的版本兼容。
- 如果在 Windows 上运行，使用 `.\build_demo.exe` 和 `.\infer_demo.exe` 替代上述命令。
- 如果需要调试，可以将 `--config Release` 替换为 `--config Debug`。