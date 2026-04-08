# MatrixAddPlugin：基于 TensorRT 的自定义矩阵相加插件

MatrixAddPlugin 是一个基于 NVIDIA TensorRT 10.x API 中的 `IPluginV3` 接口实现的自定义插件示例。该插件的核心功能是完成两个矩阵的相加操作，旨在为开发者提供一个清晰且实用的自定义插件开发模板。

## 实现细节

### 插件类 `MatrixAddPlugin` 的实现

`MatrixAddPlugin` 插件类实现了 TensorRT 插件的三大核心功能：**核心功能**、**构建功能** 和 **运行时功能**，分别对应插件在构建和运行阶段的共性行为、构建器所需行为以及推理执行所需行为。

- **核心功能**：定义插件在其生命周期内构建和运行阶段共有的属性和行为。
- **构建功能**：定义插件必须展示给 TensorRT 构建器的属性和行为，用于优化和配置。
- **运行时功能**：定义插件在 TensorRT 构建阶段的自动调优或运行时推理阶段必须展现的属性和行为。

插件的核心逻辑在 `enqueue` 方法中实现，该方法调用内部的 `MatrixAddInference` 函数来完成矩阵相加操作。

### 插件创建类 `MatrixAddPluginCreator` 的实现

在使用插件之前，必须先将其注册到 TensorRT 的 `PluginRegistry` 中。注册时并非直接注册插件本身，而是注册一个从 `IPluginCreatorInterface` 的子类派生的工厂类实例。插件创建器类提供了插件的名称、版本和字段参数等元信息。

### 将插件创建器类注册至插件注册表

TensorRT 提供了两种灵活的注册方式供开发者选择：

- **静态注册**：通过 TensorRT 提供的 `REGISTER_TENSORRT_PLUGIN` 宏，可将插件创建者便捷地注册到默认命名空间（""）下的注册表中。这种方式简单直接，但无法自定义命名空间。
- **动态注册**：通过构建类似于 `initLibNvInferPlugins` 的入口点，并在插件注册表中调用 `registerCreator` 方法来实现。这种方式更为推荐，因为它允许插件在唯一的命名空间下进行注册，从而有效避免在构建过程中因跨不同插件库而可能出现的名称冲突问题。

## 编译指南

按照以下步骤即可完成编译：
```bash
cmake -S . -B build -DTENSORRT_PATH="/your/tensorrt/dir" # 在本docker 环境中， TensorRT 的安装路径在 /usr 目录下
cmake --build build -j8 --config Release
```

编译完成后，插件动态库 `MatrixAddPlugin` 将被生成在 `libs` 文件夹中。

## 参考

> [TensorRT 10.x - Extending TensorRT with Custom Layers](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/extending-custom-layers.html)