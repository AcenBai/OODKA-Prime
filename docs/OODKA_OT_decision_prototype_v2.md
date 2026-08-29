# OODKA-OT 决策机制 Prototype v2

## 1. 基本前提

OODKA-OT 先将 BiomedParse 的中间特征分解为：

$$
F_{\mathrm{raw}} \rightarrow (P,S)
$$

并通过重构约束保证：

$$
P + S \approx F_{\mathrm{raw}}
$$

其中：

- **P**：稠密、稳定、偏解剖结构的主成分表示；
- **S**：稀疏、敏感、偏边缘与病理特异性的补充表示。

因此，$P+S$ 可以视为接近 BiomedParse 原始特征的安全锚点。

---

## 2. Candidate Routing：决定“整体想怎么选 P / S”

首先由文本提供任务语义先验，同时消费当前图像内部的 P/S 状态：

$$
\hat g = f(t,z_{\mathrm{state}})
$$

其中：

- $t$：BiomedParse 的文本嵌入；
- $z_{\mathrm{state}}$：当前图像内部少量 P/S 统计量，如能量比、稀疏度、分布状态等；
- $\hat g$：当前 image-prompt pair 的候选 P/S routing decision。

解释：

- $\hat g < 0.5$：整体更偏向稳定的 P；
- $\hat g > 0.5$：整体更偏向特异性的 S；
- $\hat g = 0.5$：保持 $P+S$ 的原始平衡。

---

## 3. Local Reliability $q_i$：决定“当前位置敢不敢执行这个决策”

$q_i$ 不额外训练新的空间卷积网络，而尽量消费已有 P/S 内部信息。

第一版建议使用 **跨层 P/S preference consistency**。

每一层定义：

$$
m_i^l =
\log
\frac{
\lVert S_i^l\rVert+\epsilon
}{
\lVert P_i^l\rVert+\epsilon
}
$$

将不同层（如 res2 / res3）对齐到同一空间尺度后，比较局部 P/S 偏好是否一致：

$$
q_i =
\exp
\left(
-\frac{
\left|
\bar m_i^{\mathrm{res2}}
-
\bar m_i^{\mathrm{res3}}
\right|
}{
\tau_q
}
\right)
$$

其中 $\bar m$ 可以采用局部平均池化后的结果，以减少单像素噪声。

解释：

- $q_i \approx 1$：当前位置的 P/S 证据跨层稳定，允许执行 routing；
- $q_i \approx 0$：当前位置的 P/S 证据不稳定，可能是噪声或假阳性，应收缩回 $P+S$。

---

## 4. Global Applicability $C$：决定“OODKA-OT 整体是否适合介入”

$C$ 不直接判断“测试图像和训练集像不像”，而判断：

> 当前 image-prompt pair 是否仍满足 OODKA-OT 学到的 P/S 表示假设。

建议由两部分构成：

### 4.1 Semantic Applicability

$$
C_{\mathrm{sem}}
$$

衡量当前 prompt 是否落在 OODKA-OT 训练过的语义范围内。

例如：

- cardiac prompt：scar、edema、myocardium、LV blood pool、RV blood pool；
- open semantic prompt：liver tumor、lung nodule、prostate lesion 等。

可以利用 BiomedParse 原始文本嵌入与 cardiac prompt prototypes 的相似度或距离进行刻画。

### 4.2 Representation Validity

$$
C_{\mathrm{repr}}
$$

衡量当前图像上 P/S 分解假设是否仍然成立，例如：

- $P+S$ 是否仍然能够很好重构原始特征；
- P 是否仍保持相对稠密、结构化；
- S 是否仍保持相对稀疏、敏感。

最终可定义：

$$
C =
\sqrt{
C_{\mathrm{sem}}
C_{\mathrm{repr}}
}
$$

解释：

- 同域或轻度 OOD：$C \approx 1$；
- scanner / device shift，但 P/S 结构仍成立：$C$ 适度下降；
- 真正跨器官、跨语义 OOD：$C \rightarrow 0$。

---

## 5. 最终统一门控

不再额外引入 $F_{\mathrm{raw}}$ 的推理分支。

直接定义：

$$
g_i
=
0.5
+
C q_i
(\hat g-0.5)
$$

这里：

$$
Cq_i
$$

可以理解为当前 routing 的 **deviation permission**：

> 它决定当前特征允许偏离 $P+S$ 原始平衡点多少。

最终特征：

$$
F_i^{\mathrm{out}}
=
2(1-g_i)P_i
+
2g_iS_i
$$

---

## 6. 自动回退机制

### 情况 1：CT，小 OOD gap

如果：

$$
C \approx 1,
\qquad
q_i \approx 1
$$

则：

$$
g_i \approx \hat g
$$

直接执行训练好的 P/S routing decision。

---

### 情况 2：MRI / LGE，设备间 OOD gap 较大

如果 scanner / device shift 较明显，$C$ 会适度下降。

但只要 P/S 分解仍稳定，并且局部跨层一致性较高：

$$
q_i \approx 1
$$

则仍然可以执行较保守的 P/S routing，实现同一医学领域内部的跨设备 OOD 泛化。

---

### 情况 3：Heart-trained OODKA-OT $\rightarrow$ Cancer / Open Semantics

如果当前 prompt 或 representation 已明显超出 cardiac OODKA-OT 的适用范围：

$$
C \rightarrow 0
$$

则：

$$
g_i \rightarrow 0.5
$$

因此：

$$
F_i^{\mathrm{out}}
\rightarrow
P_i + S_i
\approx
F_{i,\mathrm{raw}}
$$

OODKA-OT 自动静默，尽可能恢复 BiomedParse 原始表示，从而减少知识遗忘并保留开放语义能力。

---

## 7. 三个核心量的职责

### $\hat g$

**What should I use?**

决定当前 image-prompt pair 下，整体应该更偏向 P 还是 S。

### $q_i$

**Can I trust this decision here?**

决定当前局部位置是否允许执行该 P/S routing。

### $C$

**Should OODKA-OT intervene at all?**

决定整个 OODKA-OT 对当前样本的介入强度。

---

## 8. 当前版本的核心思想

OODKA-OT 不再设计额外的 raw-feature bypass 分支，而是利用：

$$
P+S \approx F_{\mathrm{raw}}
$$

构造天然的安全锚点。

最终：

$$
\boxed{
g_i
=
0.5
+
Cq_i(\hat g-0.5)
}
$$

使得：

- 证据可靠时，执行 task-specific P/S routing；
- 局部不可靠时，通过 $q_i$ 收缩回 $P+S$；
- 整体不适用时，通过 $C$ 收缩回 $P+S$；
- 从而以最少额外存储实现 OOD adaptation 与知识保持之间的平衡。

可以概括为：

$$
\boxed{
\text{Semantic/Image Routing}
\rightarrow
\text{Local Consistency}
\rightarrow
\text{Global Applicability}
\rightarrow
\text{Identity-like Shrinkage}
}
$$
