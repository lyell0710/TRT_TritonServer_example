# WarpAffine：基于 CUDA 的仿射变换实现

`warp_affine_kernel` 是一个基于 CUDA 实现的高效仿射变换（Warp Affine）操作。它能够快速对图像进行平移、旋转、缩放等几何变换，广泛应用于计算机视觉和图像处理领域。

## 实现细节

已知仿射变换矩阵 $M = \begin{pmatrix} a & b & e \\ c & d & f \end{pmatrix}$ 对于一个点 $(x, y)$ ，其经过变换后的坐标 $(x', y')$ 可以通过以下矩阵运算得到：

仿射变换是一种二维坐标到二维坐标的线性变换，可以通过一个 $2×3$ 的变换矩阵 $M$ 来表示。已知仿射变换矩阵：

$$
M = \begin{pmatrix} a & b & e \\ c & d & f \end{pmatrix}
$$

对于一个点 $(x, y)$ ，其经过变换后的坐标 $(x', y')$ 可以通过以下矩阵运算得到：

$$
\begin{pmatrix}
x' \\
y' \\
1
\end{pmatrix}
=
\begin{pmatrix}
a & b & e \\
c & d & f \\
0 & 0 & 1
\end{pmatrix}
\begin{pmatrix}
x \\
y \\
1
\end{pmatrix}
$$

在 CUDA 实现中，假设变换矩阵的两行分别存储为 `float3` 类型的变量：

```cpp
float3 m0 = make_float3(a, b, e);
float3 m1 = make_float3(c, d, f);
```

则目标图像中的每个像素 $(dx,dy)$ 对应的源图像坐标 $(srcx, srcy)$ 可以通过以下代码计算得到：

```cpp
// 计算源图像中的对应坐标
float src_x = m0.x * dx + m0.y * dy + m0.z;
float src_y = m1.x * dx + m1.y * dy + m1.z;
```

## 编译指南

按照以下步骤即可完成编译：
```bash
cmake -S . -B build
cmake --build build -j8 --config Release
```

编译完成后，可执行文件 `warp_affine` 将被生成在 `workspace` 文件夹中。