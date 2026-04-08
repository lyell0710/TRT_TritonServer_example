# BilinearWarpAffine：基于 CUDA 的双线性插值仿射变换实现

`warp_affine_bilinear_kernel` 是一个基于 CUDA 实现的高效仿射变换操作，采用双线性插值方法来保持图像的细节信息。与传统的最近邻插值相比，双线性插值能够显著提升图像质量，从而提高模型的检测性能，尤其适用于对精度要求较高的图像处理任务。

## 实现细节

假设我们想要得到未知函数 $f$ 在点 $P = (x, y)$ 的值，并且已知函数 $f$ 在四个点 $Q_{11} = (x_1, y_1)$、$Q_{12} = (x_1, y_2)$、$Q_{21} = (x_2, y_1)$ 和 $Q_{22} = (x_2, y_2)$ 的值。

首先，在 $x$ 方向进行线性插值，得到：

$$
\begin{aligned}
f(x, y_1) &\approx \frac{x_2 - x}{x_2 - x_1} f(Q_{11}) + \frac{x - x_1}{x_2 - x_1} f(Q_{21}), \\
f(x, y_2) &\approx \frac{x_2 - x}{x_2 - x_1} f(Q_{12}) + \frac{x - x_1}{x_2 - x_1} f(Q_{22}).
\end{aligned}
$$

接着，在 $y$ 方向进行线性插值，得到：

$$
\begin{aligned}
f(x, y) &\approx \frac{y_2 - y}{y_2 - y_1} f(x, y_1) + \frac{y - y_1}{y_2 - y_1} f(x, y_2) \\
&= \frac{y_2 - y}{y_2 - y_1} \left( \frac{x_2 - x}{x_2 - x_1} f(Q_{11}) + \frac{x - x_1}{x_2 - x_1} f(Q_{21}) \right) \\
&\quad + \frac{y - y_1}{y_2 - y_1} \left( \frac{x_2 - x}{x_2 - x_1} f(Q_{12}) + \frac{x - x_1}{x_2 - x_1} f(Q_{22}) \right) \\
&= \frac{1}{(x_2 - x_1)(y_2 - y_1)} \Big( f(Q_{11})(x_2 - x)(y_2 - y) + f(Q_{21})(x - x_1)(y_2 - y) \\
&\quad + f(Q_{12})(x_2 - x)(y - y_1) + f(Q_{22})(x - x_1)(y - y_1) \Big) \\
&= \frac{1}{(x_2 - x_1)(y_2 - y_1)} \begin{bmatrix} x_2 - x & x - x_1 \end{bmatrix} \begin{bmatrix} f(Q_{11}) & f(Q_{12}) \\ f(Q_{21}) & f(Q_{22}) \end{bmatrix} \begin{bmatrix} y_2 - y \\ y - y_1 \end{bmatrix}.
\end{aligned}
$$

## 编译指南

按照以下步骤即可完成编译：
```bash
cmake -S . -B build
cmake --build build -j8 --config Release
```

编译完成后，可执行文件 `bilinear_warp_affine` 将被生成在 `workspace` 文件夹中。