# 模块编译

使用 CMake，可以按照以下步骤操作：

```bash
cmake -S . -B build -DTENSORRT_PATH=/usr/local/tensorrt # 在本docker 环境中， TensorRT 的安装路径在 /usr 目录下
cmake --build build -j$(nproc) --config Release
```

编译完成后，根目录下将生成一个名为 `lib` 的文件夹。`lib` 文件夹下包含一个名为 `deploy` 的动态库文件。

在调用动态库时，只需要引用 `include` 中的 `model.hpp` 和 `result.hpp` 头文件即可。