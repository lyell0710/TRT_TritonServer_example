# 旋转目标检测推理示例

## 模型构建

使用 `trtexec` 工具将 ONNX 文件转换为 TensorRT 引擎（fp16）：

```bash
/usr/src/tensorrt/bin/trtexec --onnx=models/yolo11n-obb_with_plugin.onnx --saveEngine=models/yolo11n-obb_with_plugin.engine --fp16 --staticPlugins=/path/to/your/libcustom_plugins.so --setPluginsToSerialize=/path/to/your/libcustom_plugins.so
```

## 模型推理

1. 确保已按照 [模块编译](../../README.md) 对项目进行编译。
2. 将 `obb.cpp` 编译为可执行文件：

    ```bash
    # 使用 cmake 编译
    cmake -S . -B build -DTENSORRT_PATH="/path/to/your/TensorRT" -DDEPLOY_PATH="/path/to/your/BaseInfer" # 在本docker 环境中， TensorRT 的安装路径在 /usr 目录下
    cmake --build build -j8 --config Release
    ```

    编译完成后，可执行文件将生成在项目根目录的 `bin` 文件夹中。

3. 使用以下命令运行推理：

    ```bash
    cd bin
    ./obb -e ../models/yolo11n-obb_with_plugin.engine -i ../images -o ../output -l ../labels.txt
    ```

通过以上方式，您可以顺利完成模型推理。
