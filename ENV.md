# 环境复现指南

## 测试环境
- OS: Ubuntu 22.04
- GPU: RTX 4070 Laptop（8GB VRAM）
- CUDA Driver: 595.71.05
- Python: 3.10+（Jupyter notebook 环境）

## 依赖

### 1. TensorRT 安装
TensorRT 需与 CUDA 版本严格匹配，从 NVIDIA 官方下载：
```
https://developer.nvidia.com/tensorrt
```
推荐用 pip wheel 方式安装（省去 LD_LIBRARY_PATH 配置）：
```bash
pip install tensorrt
```

### 2. Python 依赖
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install jupyter notebook
pip install cuda-python          # cuda-python binding（cudart）
pip install netron               # 网络结构可视化
pip install trex                 # TensorRT Engine Explorer
pip install opencv-python opencv-contrib-python opencv-python-headless
pip install PyYAML tqdm pycocotools
```

### 3. Triton Server（05.TritonYOLOServer）
```bash
# Triton 推理服务器通过 Docker 启动：
docker pull nvcr.io/nvidia/tritonserver:<tag>-py3
# tag 对应你的 TensorRT 版本，例如 23.10

cd TRT-TritonServer-Lesson/course_codes/05.TritonYOLOServer
pip install -r requirements.txt
# requirements: tritonclient[all], opencv-python, pycocotools, tqdm, PyYAML
```

### 4. 启动 Jupyter
```bash
cd TRT-TritonServer-Lesson
jupyter notebook
```

## Notebook 学习顺序
```
01-01  TensorRT 简介
01-02  Engine 构建流程
01-03  TensorRT C++ API
01-04  TensorRT Plugin 开发
02-01~06  YOLO 系列（预处理/后处理/检测/分割/姿态估计）
04-01  Multiple Execution Contexts
05     Triton Inference Server + YOLO 部署
```

## 说明
- `course_data/` 包含测试图片/视频，未入 git（.gitignore 忽略），需自行准备
- `course_functions/` 是公共工具库，随 notebook 一起使用
- TensorRT Plugin（01-04）需要编译 C++ 代码，确保 `libnvinfer_plugin.so` 在路径中
