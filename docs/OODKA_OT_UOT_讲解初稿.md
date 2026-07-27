# OODKA 的 OT / UOT 升级设计

> 这是一版讲解初稿，重点放在思路、形式化定义和当前代码实现。实验部分先放已有结果，后续可以再补完整的实验设置、更多消融和可视化。

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

当前实现没有直接在普通数值域里更新 $\mathbf u,\mathbf v$，而是在 log 域里用 `logsumexp` 计算，主要是为了避免 $\exp(-C/\varepsilon)$ 在混合精度训练中发生上溢或下溢。

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

其中 $O_c$ 表示区域占据率，$B_c$ 强调边界，$R_c$ 是小结构的保底项。仅用平均池化时，小器官可能在低分辨率 OT 网格上几乎消失，max-pooling rescue 可以保留“这里出现过该类别”的信号。

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

S 分支没有直接沿用 P 的结构质量，因为我们更关心“学生哪里困难”以及“专家到底能改善多少”。

对每个位置，先计算 S 特征能量与特异性比例：

$$
e_i=\frac{\lVert s_i\rVert_2}
{\operatorname{mean}_k\lVert s_k\rVert_2+\delta},
\qquad
r_i=\frac{\lVert s_i\rVert_2}
{\lVert p_i\rVert_2+\lVert s_i\rVert_2+\delta}.
$$

然后由分割误差构造学生难度和专家收益。误差使用多类别 logits 与 GT 的 BCE，计算时全部 `detach`：

$$
d_i=\operatorname{Pool}(\ell_{\text{base},i}),
$$

$$
g_i=
\left[
\operatorname{Pool}(\ell_{\text{base},i})
-\operatorname{Pool}(\ell_{\text{expert},i})
\right]_+.
$$

$d_i$ 表示学生在该区域有多难，$g_i$ 表示专家相对学生带来了多少正收益。二者在样本内按均值归一化，并截断到 $[0,3]$。S 分支两侧的候选质量定义为

$$
q_i^b=e_i^b r_i^b(\lambda_0+\hat d_i),
$$

$$
q_j^e=e_j^e r_j^e(\lambda_0+\hat g_j),
$$

$$
\mathbf a^S=\operatorname{Norm}(\mathbf q^b),\qquad
\mathbf b^S=\operatorname{Norm}(\mathbf q^e),
$$

其中 $\lambda_0=0.1$。也就是说，学生侧质量集中在“困难且 S 响应明显”的位置，专家侧质量集中在“确实优于学生且 S 响应明显”的位置。即使输入的 $\mathbf a^S,\mathbf b^S$ 被归一化，UOT 的实际行、列边缘仍然可以偏离它们，因此仍能产生接受和拒绝。

### 2.3 代价矩阵

四个尺度 `res2`—`res5` 分别建立 OT 问题。对齐后的学生、专家特征先池化到不超过 $32\times32$ 的网格，再展平成 token。特征代价采用 cosine distance：

$$
C_{ij}^{\text{feat}}
=1-
\frac{\langle x_i,y_j\rangle}
{\lVert x_i\rVert_2\lVert y_j\rVert_2}.
$$

位置代价采用归一化坐标上的平方欧氏距离：

$$
C_{ij}^{\text{coord}}
=\lVert p_i-p_j\rVert_2^2,
\qquad p_i,p_j\in[-1,1]^2.
$$

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

当前参数为 $\lambda_f=1.0,\ \lambda_p=0.1,\ \lambda_s=0.25$。P 分支既要特征相似，也要尽量保持解剖位置和类别一致；S 分支则不施加额外语义硬引导，让 UOT 根据特异特征的兼容性决定接收多少专家信息。

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

## 3. 新的 Beta 分布门控

旧式的固定比例或单个 sigmoid 门控只能给出一个确定权重，也很难表达“这个 prompt 在这个尺度上应该更依赖 P 还是 S，以及模型对此有多确定”。新版门控改为 prompt-conditioned Beta 分布。

对第 $c$ 个文本 prompt 和第 $l$ 个尺度，路由器输出

$$
\alpha_{c,l}=1+\operatorname{softplus}(f_\alpha(t_c)),
$$

$$
\beta_{c,l}=1+\operatorname{softplus}(f_\beta(t_c)),
$$

并定义

$$
g_{c,l}\sim
\operatorname{Beta}(\alpha_{c,l},\beta_{c,l}).
$$

融合特征为

$$
F_{c,l}
=g_{c,l}P_l+(1-g_{c,l})S_l.
$$

这里 $g$ 是 P 分支权重，$1-g$ 是 S 分支权重。门控只依赖冻结的文本语义，不读取专家特征，因此不会在推理阶段形成专家依赖。

训练时使用可重参数化采样，同一个 2.5D block 内的所有 Z 切片共享一次 prompt 门控采样，不同 block 独立采样。这样既引入了适度随机性，又避免相邻切片的路由发生无意义抖动。推理时不再采样，而是直接使用分布均值：

$$
\bar g_{c,l}
=\frac{\alpha_{c,l}}{\alpha_{c,l}+\beta_{c,l}}.
$$

四个尺度的 P 权重先验均值设为

$$
(0.5,\ 0.6,\ 0.7,\ 0.8),
$$

对应 `res2` 到 `res5`。先验浓度为 10，所以初始化参数分别是

$$
\alpha_0=(5,6,7,8),\qquad
\beta_0=(5,4,3,2).
$$

浅层保留更多 S 细节，深层更偏向稳定的 P 语义。训练时再通过

$$
\mathcal L_{\mathrm{route}}
=
\frac{1}{4C}\sum_{c,l}
\mathrm{KL}
\left[
\operatorname{Beta}(\alpha_{c,l},\beta_{c,l})
\Vert
\operatorname{Beta}(\alpha^0_l,\beta^0_l)
\right]
$$

约束门控不要在早期无序漂移。路由 KL 和 OT 权重都采用 warm-up。

## 4. 工程上是怎么实现的

### 4.1 多尺度与形状处理

OT 在 `res2`、`res3`、`res4`、`res5` 四个尺度独立计算。输入特征原本是

```text
[B, C, Z, H, W]
```

实现中先去掉 block 尾部为补齐长度而重复的无效切片，再整理成

```text
[M, C, H, W]
```

其中 $M$ 是当前 batch 中有效切片数。每个尺度只做降采样，不会为了 OT 上采样；每个空间维最大取 32。池化后展平为 $N=H_{\text{OT}}W_{\text{OT}}$ 个 token，传输矩阵形状为

```text
[M, N_base, N_expert]
```

当前学生和专家经过 adapter 后通道数一致，P/S 四组特征也具有一致形状。

### 4.2 质量和代价是不是固定的

不是。P 质量依赖当前切片的 GT 结构和当前 P 特征能量；S 质量依赖当前 P/S 特征、学生误差和专家相对收益；代价矩阵也依赖当前 batch 的学生、专家特征。因此它们都是动态构造的。

更准确地说，只要当前 epoch 对应的 OT 权重大于 0，每次 `forward_one_batch` 都会在四个尺度上重新完成：

```text
构造质量 → 构造代价 → Sinkhorn/UOT Sinkhorn
→ 重心投影 → 蒸馏损失
```

所以传输矩阵不是每个 epoch 求一次，也不是训练前离线算好，而是每个启用 OT 的训练/验证 batch 都重新求解。当前每个尺度运行 30 次 Sinkhorn 迭代。

P-OT 从第 2 个 epoch 开启，S-UOT 从第 3 个 epoch 开启，随后分别经过 5 个 epoch 线性 warm-up 到目标权重 0.1。这样可以先让基本的分割、解耦和重建关系稳定下来，再逐步增加跨模型传输监督。

### 4.3 梯度与数值稳定性

质量构造、代价中的匹配依据、Sinkhorn 迭代、传输矩阵和重心教师都在 `no_grad` 下完成，并强制使用 float32。教师 token 和误差图也会 `detach`。反向传播只通过最终 cosine 蒸馏损失更新学生侧特征。

这样做有两个实际好处：

- 不需要对 30 步 Sinkhorn 迭代保存完整反向图，显存和计算开销更可控；
- 学生学习的是当前传输计划给出的目标，不会通过反向传播“篡改”质量或匹配关系来降低损失。

质量归一化带有零质量的均匀回退，代价和传输结果也会显式检查 NaN/Inf。Sinkhorn 在关闭 AMP 的 float32 log 域中运行，所以外层即使使用 float16 AMP，也不会直接降低 OT 求解的稳定性。

### 4.4 训练和推理的边界

nnUNet 专家、专家误差、OT/UOT 和传输矩阵都只存在于训练监督路径中。部署 checkpoint 只保留学生侧解耦模块和 Beta router。推理阶段：

1. BiomedParse 提取多尺度特征；
2. 学生解耦器得到 P/S；
3. Beta router 根据 prompt 给出确定性均值门控；
4. 融合 P/S 后完成分割。

因此线上推理不需要 nnUNet 输入、nnUNet 模型、专家预处理、传输矩阵或缓存的专家统计。这一点对 OODKA 的实际部署很重要：训练时借助专家，推理时仍保持单学生模型。

## 5. 当前实验结果

### 5.1 30 epoch 当前模型

当前主实验为 `Dataset009_CT_OOD`、fold 0、2.5D block 长度 6。最佳 checkpoint 出现在第 28 个 epoch。按全体积评估，结果如下：

| 数据划分 | 病例数 | Mean Dice |
|---|---:|---:|
| 验证集 | 4 | **0.9017** |
| 测试集 | 20 | **0.8898** |

测试集各类别 Dice：

| 类别 | 解剖结构 | Dice |
|---:|---|---:|
| 1 | 左心室血池（LV） | 0.8918 |
| 2 | 右心室血池（RV） | 0.8482 |
| 3 | 左心房血池（LA） | 0.9281 |
| 4 | 右心房血池（RA） | 0.8804 |
| 5 | 左心室心肌（MYO） | 0.8784 |
| 6 | 升主动脉（AO） | 0.9626 |
| 7 | 肺动脉（PA） | 0.8391 |

同一 20 例测试集上的 nnUNet baseline Mean Dice 为 0.8063；当前 OODKA 为 0.8898，高 0.0835。不过这个对比主要说明当前学生系统的整体结果，不能替代严格的模块消融。

### 5.2 初步 OT 消融

仓库中已有一组 5 epoch 的同配置开关实验，二者仅 `w_p_ot`、`w_s_ot` 不同：

| 设置 | 验证病例数 | 全体积 Mean Dice |
|---|---:|---:|
| 不使用 OT/UOT | 4 | 0.8872 |
| P-OT + S-UOT | 4 | **0.8989** |
| 变化 | — | **+0.0117** |

这是一个正向的初步信号，但验证集只有 4 例、训练也只有 5 epoch。后续最好补充完整轮次、多 fold，以及只开 P-OT、只开 S-UOT、均匀质量、去掉坐标代价、去掉语义代价等消融。

### 5.3 UOT 的拒绝行为

受控压力测试给 S 分支代价统一增加惩罚，四个尺度的专家质量接受率都会随惩罚增加而下降。例如 `res4`：

| S 代价附加值 | 接受率 |
|---:|---:|
| 0.00 | 0.8675 |
| 0.25 | 0.7963 |
| 0.50 | 0.7152 |
| 1.00 | 0.5392 |
| 2.00 | 0.2590 |

这至少验证了 UOT 模块不是形式上的“软匹配”：当专家特征的传输成本被提高时，它确实会减少接收质量。不过自然扰动实验还不完全单调，因此更合适的表述是“拒绝机制在受控代价实验中有效”，暂时不宜扩大成“已经证明能识别所有错误专家特征”。

### 5.4 Beta 门控的当前表现

在 30 epoch 最佳模型上，7 个已见 prompt 的平均 P 权重为：

| 尺度 | `res2` | `res3` | `res4` | `res5` |
|---|---:|---:|---:|---:|
| 平均 P gate | 0.3353 | 0.6009 | 0.8422 | 0.8440 |

可以看到，模型在浅层更多保留 S 分支，在中深层逐渐偏向 P 分支，整体符合“浅层细节、深层稳定语义”的设计预期。对改写后的同义 prompt，四层均值分别为 0.3373、0.6033、0.8377、0.8409，与原 prompt 接近，说明当前门控对简单措辞变化具有一定稳定性。

## 6. 目前可以怎么概括这版方法

这次升级的重点并不是单纯“加了一个 Sinkhorn loss”，而是把 P/S 的语义分工真正写进了传输问题：

- P 分支通过 GT 占据率、边界、小结构保底和类别平衡构造结构质量，用 balanced OT 做完整对齐；
- S 分支用学生难度、专家收益、特征能量和特异性构造任务感知质量，用 UOT 选择性接收专家残差；
- 特征、空间和语义共同决定匹配代价，传输矩阵再通过重心投影生成动态教师；
- Beta router 用一个分布而不是单点权重描述 P/S 偏好，并让路由随 prompt 和尺度变化；
- 专家和 OT 都只服务于训练，部署阶段仍是纯 BiomedParse 学生路径。

一句话概括：**P-OT 负责把应该共享的结构对齐好，S-UOT 负责只接收值得学习的专家残差，Beta 门控再决定不同 prompt、不同尺度下两类信息该如何组合。**

## 7. 后续建议补充的实验

为了让讲解或论文中的结论更完整，后面可以优先补下面几组：

1. 完整轮次的 `no OT / P-OT only / S-UOT only / P-OT + S-UOT`；
2. P 质量消融：均匀质量、去边界、去 rescue、去类别预算；
3. S 质量消融：去 difficulty、去 expert gain、去 specificity；
4. 代价消融：feature only、`+ coordinate`、`+ semantic`；
5. balanced S-OT 与 S-UOT 的直接对比；
6. 固定 gate、sigmoid gate 与 Beta gate 的对比；
7. 更多 fold、更多外部域数据，以及推理时间和显存开销。
