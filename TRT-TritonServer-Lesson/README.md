# 实战课程 | TensorRT x Triton Inference Server 模型部署 相关资料

## 使用指南

> **注意事项**
> - 确保 Docker 已正确安装，并且系统支持 GPU。
> - 如果遇到任何问题，请检查 Docker 镜像构建和容器运行的日志输出。

### 构建和运行 Docker 容器

1. **加载 Docker 镜像**：
   ```bash
   docker pull laugh12321/cuda-trt-lesson:12.6_10.3.0
   ```

2. **进入项目目录**：在终端中，导航到 `TRT-TritonServer-Lesson` 的父目录。

3. **运行 Docker 容器**：运行以下命令启动 Docker 容器，并映射端口和目录。

   ```bash
   docker run --gpus=all --network=host --rm -it -v ./TRT-TritonServer-Lesson:/home/Lesson --name lesson cuda-trt-lesson:12.6_10.3.0
   ```

### 访问 Jupyter Notebook

在容器中运行以下命令启动 Jupyter Notebook：

```bash
nohup jupyter-notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --notebook-dir=/home/Lesson --NotebookApp.token='' > jupyter.log 2>&1 & /bin/bash
```

在浏览器中输入 `127.0.0.1:8888` 即可访问并使用 Jupyter Notebook。

### 编译相关代码

#### 示例：编译 `01.04.MatrixAddPlugin`

1. **进入 Docker 容器**：
   确保你已经按照前面的步骤启动了 Docker 容器，并且进入了容器内部。

2. **导航到代码目录**：
   执行以下命令，切换到示例代码所在的目录：
   ```bash
   cd /home/Lesson/course_codes/01.04.MatrixAddPlugin
   ```

3. **配置编译环境**：
   使用 `cmake` 配置项目。在本 Docker 环境中，TensorRT 的安装路径默认为 `/usr`。运行以下命令进行配置：
   ```bash
   cmake -S . -B build -DTENSORRT_PATH="/usr"
   ```
   - `-S .`：指定源代码目录为当前目录。
   - `-B build`：指定构建输出目录为 `build`。
   - `-DTENSORRT_PATH="/usr"`：指定 TensorRT 的安装路径。

4. **开始编译**：
   在配置完成后，运行以下命令开始编译项目：
   ```bash
   cmake --build build -j8 --config Release
   ```
   - `--build build`：指定构建目录为 `build`。
   - `-j8`：指定使用 8 个线程进行并行编译，加快构建速度。
   - `--config Release`：指定构建配置为 Release 模式，生成优化后的可执行文件。

#### 注意事项

- 如果你的 TensorRT 安装路径不是 `/usr`，请将 `-DTENSORRT_PATH="/usr"` 替换为实际的安装路径。
- 如果在编译过程中遇到错误，请检查错误信息，确保所有依赖项已正确安装，尤其是 CUDA 和 TensorRT 的版本是否匹配。
- 如果需要调试代码，可以将 `--config Release` 替换为 `--config Debug`，以生成调试版本的可执行文件。

### 结构说明

- `course_codes`：课程相关 C++、CUDA 源码。
- `course_data`：配合 Jupyter 文件的相关数据（图片、模型）。
- `course_functions`：配合 Jupyter 文件的相关 Python 脚本。
- `01` 对应视频课程的【TensorRT】章节。
- `02` 对应视频课程的【YOLO5-11模型特点讲解和Plugin开发】章节。
- `03` 对应视频课程的【实战-TensorRT部署YOLO Detect、OBB、Seg】章节。
- `04` 对应视频课程的【实战-深度优化TensorRT部署YOLO Detect、OBB、Seg】章节。
- `05` 对应视频课程的【实战-Triton Server 上线YOLO OBB】章节。

## 版权说明

本课程由 [laugh12321](https://space.bilibili.com/86034462) 与[不归牛顿管的熊猫](https://space.bilibili.com/393625476) 联合发布，版权所有。未经授权，禁止复制、分发或修改本项目的任何内容。如有疑问，请联系 laugh12321@vip.qq.com。