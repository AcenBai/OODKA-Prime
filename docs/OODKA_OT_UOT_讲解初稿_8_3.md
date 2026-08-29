## 1. 为什么在 OODKA 里引入 OT

OODKA 的训练过程里有两个角色：一个是提供医学分割知识的专家网络 nnUNet，另一个是最终负责推理的 BiomedParse 学生网络。两个网络的特征分辨率、通道语义和空间响应并不完全一致，因此不能简单地把同一坐标上的特征直接做 MSE。即便经过通道适配，相同位置也未必表达相同的解剖内容。

最优传输（Optimal Transport, OT）提供了一种更自然的对齐方式：先把两组空间特征看成两个离散分布，再根据特征相似度、空间距离和语义信息，在两组 token 之间寻找总代价最小的软匹配。

设学生侧 token 为 $\{x_i\}_{i=1}^{N}$，专家侧 token 为 $\{y_j\}_{j=1}^{K}$，对应质量分别为

$$
\mathbf a\in\Delta^N,\qquad \mathbf b\in\Delta^K,
$$

其中 $\Delta$ 表示概率单纯形。代价矩阵 $\mathbf C\in\mathbb R_+^{N\times K}$ 描述把学生位置 $i$ 与专家位置 $j$ 对齐需要付出的代价。平衡 OT 求解

$$
\min_{\mathbf T\ge 0}
\langle \mathbf T,\mathbf C\rangle
+\varepsilon\sum_{i,j}T_{ij}(\log T_{ij}-1),
$$

$$
\text{s.t.}\qquad
\mathbf T\mathbf 1=\mathbf a,\qquad
\mathbf T^\top\mathbf 1=\mathbf b.
$$

$\mathbf T$ 是传输矩阵，$T_{ij}$ 越大，表示学生 token $x_i$ 越应该从专家 token $y_j$ 中吸收知识。熵正则项让问题更平滑，也使它可以用 Sinkhorn 迭代高效求解。

### 1.1 Sinkhorn 如何求解 OT

记

$$
\mathbf K=\exp(-\mathbf C/\varepsilon),
$$

则传输矩阵可写成

$$
\mathbf T=\operatorname{diag}(\mathbf u)\mathbf K
\operatorname{diag}(\mathbf v).
$$

Sinkhorn 通过交替更新两个缩放向量，使传输矩阵满足两侧的边缘质量约束。直观地说，就是反复进行“行归一化”和“列归一化”，直到学生侧收到的质量接近 $\mathbf a$，专家侧送出的质量接近 $\mathbf b$。

### 1.2 UOT 与普通 OT 的区别

平衡 OT 有一个较强的假设：两边给出的质量都必须被完整匹配。但对于特异性特征，这个假设并不总是合理。专家网络中有些响应可能对当前学生没有帮助，甚至来自域偏移或专家误差。如果强制全部对齐，就可能把不可靠的信息也蒸馏给学生。

非平衡最优传输（Unbalanced OT, UOT）把硬边缘约束改成 KL 软约束：

$$
\min_{\mathbf T\ge 0}
\langle\mathbf T,\mathbf C\rangle
+\varepsilon\sum_{i,j}T_{ij}(\log T_{ij}-1)
+\rho_b\,\mathrm{KL}(\mathbf T\mathbf 1\Vert\mathbf a)
+\rho_e\,\mathrm{KL}(\mathbf T^\top\mathbf 1\Vert\mathbf b).
$$

这里 $\rho_b$ 和 $\rho_e$ 分别控制学生侧需求、专家侧供给偏离原质量的代价。对应的 Sinkhorn 更新会增加松弛指数

$$
\tau_b=\frac{\rho_b}{\rho_b+\varepsilon},\qquad
\tau_e=\frac{\rho_e}{\rho_e+\varepsilon}.
$$

当 $\rho$ 很大时，UOT 接近平衡 OT；当 $\rho$ 较小时，传输可以主动减少不合适的质量。在当前设计中，专家侧 $\rho_e=0.2$ 小于学生侧 $\rho_b=1.0$，因此专家供给更容易被拒绝。这正好对应我们的目标：学生的学习需求相对稳定，但并不是所有专家残差都值得接收。

## 2. 本次升级的核心：P 用 OT，S 用 UOT

经过解耦后，每一层特征被分成 P 和 S 两部分。这里可以把 P 理解为更稳定、可共享的结构信息，把 S 理解为与任务难点、模型差异和域特征有关的特异信息。

这两类信息不适合使用同一种传输约束：

- P 分支强调解剖结构的完整对齐，因此采用 balanced OT；
- S 分支强调有选择地吸收专家残差，因此采用 UOT；
- 两个分支分别构造质量和代价，而不是共用一套均匀分布；
- OT 只负责在训练期构造动态教师，最终推理仍然是纯学生网络。

### 2.1 P 分支的结构质量

P 分支的质量不是均匀分配，而是由标注结构和特征能量共同决定。对类别 $c$，先从 GT 得到三种空间响应：

$$
O_c=\operatorname{AvgPool}(\mathbb 1[Y=c]),
$$

$$
B_c=\operatorname{AvgPool}(\operatorname{Dilate}(Y_c)
-\operatorname{Erode}(Y_c)),
$$

$$
R_c=\operatorname{MaxPool}(\mathbb 1[Y=c]).
$$

其中 $O_c$ 表示区域占据率，$B_c$ 强调边界，$R_c$ 是小结构的保底项。仅用平均池化时，小器官可能在低分辨率 OT 网格上几乎消失，max-pooling rescue 就是max池化小结构

不同器官的体积差异很大，因此又定义逆平方根类别预算：

$$
\pi_c=
\frac{(|Y_c|+\delta)^{-1/2}}
{\sum_{c'}(|Y_{c'}|+\delta)^{-1/2}}.
$$

没有出现在当前切片中的类别预算记为 0。最终的结构强度为

$$
h_i=\sum_c\pi_c
\left(
w_o O_{c,i}+w_b B_{c,i}+w_r R_{c,i}
\right).
$$

当前代码取 $w_o=1.0,\ w_b=1.5,\ w_r=0.2$，即边界比普通区域得到更高权重。

在结构强度上再加入学生和专家各自的局部特征能量：

$$
q_i^{b}=h_i(1+\lambda_E e_i^{b}),\qquad
q_j^{e}=h_j(1+\lambda_E e_j^{e}),
$$

$$
\mathbf a^P=\operatorname{Norm}(\mathbf q^b),\qquad
\mathbf b^P=\operatorname{Norm}(\mathbf q^e).
$$

当前 $\lambda_E=0.1$。GT 决定主体结构，特征能量只做轻量调节；若当前切片完全没有有效前景，则退化为按特征能量分配质量，避免出现零质量问题。

### 2.2 S 分支的任务感知质量

S 分支没有直接沿用 P 的结构质量，更关心“学生哪里困难”以及“专家到底能改善多少”。

对每个位置，先计算 S 特征能量与特异性比例：

$$
e_i=\frac{\lVert s_i\rVert_2}
{\operatorname{mean}_k\lVert s_k\rVert_2+\delta},
\qquad
r_i=\frac{\lVert s_i\rVert_2}
{\lVert p_i\rVert_2+\lVert s_i\rVert_2+\delta}.
$$

然后由分割误差构造学生难度和专家相对优势。误差使用多类别 logits 与 GT 的 BCE，计算时全部 `detach`：

$$
d_i=\operatorname{Pool}(\ell_{\text{base},i}),
$$

$$
a_i=d_i-e_i^{\mathrm{err}},
\qquad
e_i^{\mathrm{err}}=\operatorname{Pool}(\ell_{\text{expert},i}).
$$

这里 $d_i$ 表示学生在该区域有多难，$a_i$ 是专家相对学生的误差优势：$a_i>0$ 表示专家更好，$a_i=0$ 表示二者相当，$a_i<0$ 表示专家更差。

再定义温度和连续 gain：

$$
\tau=\gamma s_a,\qquad \gamma=0.5,
$$

$$
g_i=2\,\sigma\!\left(\frac{a_i}{\tau}\right).
$$

这个映射的语义非常明确：$g_i=1$ 是中性，$g_i>1$ 表示专家更好，$g_i<1$ 表示专家更差；其理论范围为 $(0,2)$。代码仍统一执行 $[0,3]$ 截断，但 smooth 模式正常情况下不会超过 2。每张切片用自身 $\operatorname{mean}|a|$ 定标。

学生难度仍按样本内均值归一化并截断到 $[0,3]$：

$$
\hat d_i=\operatorname{clip}\!\left(
\frac{d_i}{\operatorname{mean}_k d_k+\delta},0,3
\right).
$$

S 分支两侧的候选质量定义为

$$
q_i^b=e_i^b r_i^b(\lambda_0+\hat d_i),
$$

$$
q_j^e=e_j^e r_j^e(\lambda_0+g_j),
$$

$$
\mathbf a^S=\operatorname{Norm}(\mathbf q^b),\qquad
\mathbf b^S=\operatorname{Norm}(\mathbf q^e),
$$

其中 $\lambda_0=0.1$。也就是说，学生侧质量集中在困难且 S 响应明显的位置；专家侧质量不再只保留 hard-positive 区域，而是连续区分“更好 / 相当 / 更差”，再与专家 S 能量和特异性共同决定相对供给。这里的 gain 决定的是专家质量在空间上的相对分布，而不是直接把某个 token 判定为接收或拒绝。真正的软拒绝仍由 UOT 的传输代价和松弛边缘共同完成。

### 2.3 代价矩阵

四个尺度 `res2`—`res5` 分别建立 OT 问题。对齐后的学生、专家特征先池化到不超过 $32\times32$ 的网格，再展平成 token。特征代价采用 cosine distance：

$$
C_{ij}^{\text{feat}}
=1-
\frac{\langle x_i,y_j\rangle}
{\lVert x_i\rVert_2\lVert y_j\rVert_2}.
$$

位置代价采用归一化坐标上的“免罚半径 + 平方惩罚”：

$$
C_{ij}^{\text{coord}}
=\left[\lVert p_i-p_j\rVert_2-r_0\right]_+^2,
\qquad p_i,p_j\in[-1,1]^2.
$$

当前 $r_0=0.25$。半径以内的局部移动不增加坐标代价，超过半径后才平方惩罚。这允许小范围的跨模型错位，又能压制不合理的远距离传输。

P 分支还加入 GT 语义代价：

$$
C_{ij}^{\text{sem}}
=1-\langle m_i^b,m_j^e\rangle,
$$

其中 $m_i$ 是池化后的类别分布。于是

$$
\mathbf C^P
=\lambda_f\mathbf C^{\text{feat}}
+\lambda_p\mathbf C^{\text{coord}}
+\lambda_s\mathbf C^{\text{sem}},
$$

$$
\mathbf C^S
=\lambda_f\mathbf C^{\text{feat}}
+\lambda_p\mathbf C^{\text{coord}}.
$$

当前参数为 $\lambda_f=1.0,\ \lambda_p=0.25,\ \lambda_s=0.25,\ r_0=0.25$。P 分支既要特征相似，也要尽量保持解剖位置和类别一致；S 分支则不施加额外语义硬引导，让 UOT 根据特异特征的兼容性决定接收多少专家信息。

### 2.4 传输矩阵与动态教师

对 P 分支求得

$$
\mathbf T^P
=\operatorname{Sinkhorn}
(\mathbf a^P,\mathbf b^P,\mathbf C^P),
$$

它满足近似的双边质量守恒。对 S 分支求得

$$
\mathbf T^S
=\operatorname{UOTSinkhorn}
(\mathbf a^S,\mathbf b^S,\mathbf C^S),
$$

它允许

$$
\mathbf T^S\mathbf 1\ne\mathbf a^S,\qquad
(\mathbf T^S)^\top\mathbf 1\ne\mathbf b^S.
$$

专家 token 通过重心投影变成与学生位置一一对应的动态教师：

$$
\tilde y_i=
\frac{\sum_jT_{ij}y_j}
{\sum_jT_{ij}+\delta}.
$$

这一步不是把专家特征直接复制过来，而是让每个学生位置根据传输计划，从多个专家位置聚合适合自己的监督信号。

### 2.5 OT 蒸馏损失

蒸馏使用加权 cosine distance：

$$
\mathcal L_{\mathrm{distill}}
=
\frac{\sum_iw_i
\left(1-\cos(x_i,\tilde y_i)\right)}
{\sum_iw_i+\delta}.
$$

P 分支使用结构质量 $w_i=a_i^P$：

$$
\mathcal L_{P\text{-OT}}
=
\frac{\sum_i a_i^P
\left(1-\cos(x_i^P,\tilde y_i^P)\right)}
{\sum_i a_i^P+\delta}.
$$

S 分支使用 UOT 后学生实际收到的质量

$$
r_i^S=\sum_jT_{ij}^S
$$

作为权重：

$$
\mathcal L_{S\text{-UOT}}
=
\frac{\sum_i r_i^S
\left(1-\cos(x_i^S,\tilde y_i^S)\right)}
{\sum_i r_i^S+\delta}.
$$

因此，被 UOT 拒绝的专家信息不会以同样强度进入蒸馏损失。四个尺度分别计算损失后取平均。

总训练目标为

$$
\mathcal L=
\lambda_{\mathrm{seg}}\mathcal L_{\mathrm{seg}}
+\lambda_{\mathrm{ae}}\mathcal L_{\mathrm{rec}}
+\lambda_{\mathrm{ort}}\mathcal L_{\mathrm{ort}}
+\lambda_{\mathrm{route}}\mathcal L_{\mathrm{route}}
+\lambda_P\mathcal L_{P\text{-OT}}
+\lambda_S\mathcal L_{S\text{-UOT}}.
$$

当前主要权重为

$$
\lambda_{\mathrm{seg}}=3.0,\quad
\lambda_{\mathrm{ae}}=0.2,\quad
\lambda_{\mathrm{ort}}=0.3,\quad
\lambda_{\mathrm{route}}=10^{-3},\quad
\lambda_P=\lambda_S=0.1.
$$

## 3. Prompt-conditioned 空间 Beta 门控

两种版本：propmt->产生标量门控，类特异凸组合产生prompt->conditioned  视觉memory and mask features

旧式的固定比例、单个 sigmoid 门控，以及更早的“每层一个标量 Beta gate”，都无法直接表达同一类别在不同空间位置对 P/S 的不同依赖。当前路由器改成 prompt-conditioned spatial Beta field：每个文本 prompt 预测一张最高分辨率的空间概率场，再把同一张场用 area downsampling 共享到 Predictor 的各个尺度。

路由器先把冻结的文本嵌入 $t_c$ 映射为一组径向基函数（RBF）系数。设最高分辨率位置为 $u$，空间基函数为 $\phi_k(u)$，则

$$
a_c(u)=b_\alpha+
\frac{1}{\sqrt K}\sum_{k=1}^{K}w_{c,k}^{\alpha}\phi_k(u),
$$

$$
b_c(u)=b_\beta+
\frac{1}{\sqrt K}\sum_{k=1}^{K}w_{c,k}^{\beta}\phi_k(u),
$$

并得到

$$
\alpha_c(u)=1+\operatorname{softplus}(a_c(u)),
\qquad
\beta_c(u)=1+\operatorname{softplus}(b_c(u)),
$$

$$
g_c(u)\sim\operatorname{Beta}(\alpha_c(u),\beta_c(u)).
$$

当前基函数中心使用 $8\times8$ 网格； $\sigma$ 为网格间距的 1.5 倍融合形式为

$$
F_c(u)=g_c(u)P(u)+[1-g_c(u)]S(u).
$$

这里 $g$ 是 P 分支权重，$1-g$ 是 S 分支权重。门控只依赖冻结的文本语义和坐标基函数，不读取图像特征或专家特征，因此不会在推理阶段形成专家依赖。

训练时使用 Beta 分布的可重参数化采样。同一个 2.5D block 内所有 Z 切片共享一次 prompt-specific 空间采样，不同 block 独立采样，避免相邻切片的路由发生无意义抖动。推理时不再采样，而使用确定性均值

$$
\bar g_c(u)=
\frac{\alpha_c(u)}{\alpha_c(u)+\beta_c(u)}.
$$

当前所有 prompt、所有位置初始化为同一个

$$
\operatorname{Beta}(7,3)
$$

先验，即 P 权重均值 0.7、浓度 10。两组 RBF coefficient head 零初始化，所以训练起点是空间均匀的 0.7；随着训练更新，才逐渐形成 prompt-specific 的空间差异。路由约束为整个 prompt-spatial field 上的平均 KL：

$$
\mathcal L_{\mathrm{route}}
=\operatorname{mean}_{c,u}
\mathrm{KL}\left[
\operatorname{Beta}(\alpha_c(u),\beta_c(u))
\Vert
\operatorname{Beta}(7,3)
\right].
$$

路由 KL 在前 5 个 epoch 线性 warm-up；P-OT 和 S-UOT 也各自按配置 warm-up。

## 5. 当前实验结果

### 5.1 已完成的 CT 30-epoch 模型

| 数据划分 | 病例数 |        Mean Dice |
| -------- | -----: | ---------------: |
| 验证集   |      4 | **0.8303** |
| 测试集   |     20 | **0.8876** |

测试集各类别 Dice：

| 类别 | 解剖结构          |   Dice |
| ---: | ----------------- | -----: |
|    1 | 左心室血池（LV）  | 0.8969 |
|    2 | 右心室血池（RV）  | 0.8309 |
|    3 | 左心房血池（LA）  | 0.9422 |
|    4 | 右心房血池（RA）  | 0.8715 |
|    5 | 左心室心肌（MYO） | 0.8691 |
|    6 | 升主动脉（AO）    | 0.9629 |
|    7 | 肺动脉（PA）      | 0.8400 |

同一 20 例测试集上的 nnUNet baseline Mean Dice 为 0.8063；该 OODKA 模型为 0.8876，高 0.0813。
