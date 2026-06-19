---
title: "Embodied Image Quality Assessment for Robotic Intelligence"
arxiv_id: "2412.18774"
version: "v3"
date: 2025-08-18
tags:
  - paper
  - alphaxiv
  - embodied-ai
  - image-quality-assessment
  - robotic-vision
  - robot-generated-content
  - moravec-paradox
  - human-robot-perception
  - robotic-intelligence
source: "https://alphaxiv.org/abs/2412.18774"
authors:
  - Jianbo Zhang
  - Chunyi Li
  - Jie Hao
  - Jun Jia
  - Huiyu Duan
  - Guoquan Zheng
  - Liang Yuan
  - Guangtao Zhai
aliases:
  - Embodied Image Quality Assessment for Robotic Intelligence
created: "2026-06-11 22:40"
published_date: 2024-12-25
---

# Embodied Image Quality Assessment for Robotic Intelligence

> **arXiv**: [2412.18774](https://arxiv.org/abs/2412.18774) | **Version**: v3 | **Published**: 2025-08-18

## 摘要

Image Quality Assessment (IQA) of User-Generated Content (UGC) is a critical technique for human Quality of Experience (QoE). However, does the the image quality of Robot-Generated Content (RGC) demonstrate traits consistent with the Moravec paradox, potentially conflicting with human perceptual norms? Human subjective scoring is more based on the attractiveness of the image. Embodied agent are required to interact and perceive in the environment, and finally perform specific tasks. Visual images as inputs directly influence downstream tasks. In this paper, we explore the perception mechanism of embodied robots for image quality. We propose the first Embodied Preference Database (EPD), which contains 12,500 distorted image annotations. We establish assessment metrics based on the downstream tasks of robot. In addition, there is a gap between UGC and RGC. To address this, we propose a novel Multi-scale Attention Embodied Image Quality Assessment called MA-EIQA. For the proposed EPD dataset, this is the first no-reference IQA model designed for embodied robot. Finally, the performance of mainstream IQA algorithms on EPD dataset is verified. The experiments demonstrate that quality assessment of embodied images is different from that of humans. We sincerely hope that the EPD can contribute to the development of embodied AI by focusing on image quality assessment. The benchmark is available at this https URL.

---

### AI 摘要

上海交通大学的研究人员开发了首个以机器人为中心的图像质量评估（IQA）框架，用于具身AI。他们建立了具身偏好数据库（EPD），其中包含机器人生成的标签，并提出了轻量级多尺度注意力具身IQA（MA-EIQA）模型。研究通过实证确认机器人对图像质量的评估方式与人类不同，MA-EIQA模型通过将图像质量与机器人任务成功直接关联，在EPD数据集上取得了最先进的性能。

### 要点

- 实证证据证实了IQA中的莫拉维克悖论，表明人类主观评分与机器人基于性能的图像质量评分之间存在低相关性（PLCC < 0.22）。
- 机器人对图像失真表现出不同的敏感度，发现色彩失真对任务性能影响最大，而噪声影响最小，这与人类感知形成对比。
- 机器人的图像质量评估具有高度任务特异性，这意味着即使是不同的操作子任务，最佳图像特征也可能显著不同。

### 问题

- 传统的图像质量评估（IQA）主要为人眼视觉系统（HVS）设计，导致其指标与机器人视觉系统（RVS）的功能需求不符。
- 缺乏专门的框架或数据集，无法从机器人的角度评估图像质量，即以完成下游具身任务的成功率来定义质量。
- 各种图像失真对机器人执行任务能力的影响尚未得到系统性理解，阻碍了鲁棒具身AI的发展。

### 方法

- 创建了具身偏好数据库（EPD），这是首个通过强化学习代理在模拟器中执行操作任务生成标注的IQA数据集。
- 开发了多尺度注意力具身图像质量评估（MA-EIQA），一个专为机器人设计的轻量级、无参考IQA模型。
- MA-EIQA包含一个多尺度特征编码器，用于捕获语义和细节信息，以及一个具身注意力模块，用于关注任务关键型特征。

### 结果

- MA-EIQA在EPD数据集上的无参考IQA方法中取得了最先进的性能（例如，
“所有任务”数据的SRCC为0.5755，KRCC为0.4032，PLCC为0.5836）。
- 该模型轻量级（4883万参数），适用于资源受限的具身代理，并且显著优于传统的人类中心IQA方法。
- 消融研究证实了多尺度特征编码器（SRCC提升6.55%）和具身注意力模块（SRCC提升3.85%）的有效性，显示出协同效应。
- 使用UR5机器人进行的真实世界实验验证了研究结果，表明变暗和JPEG压缩等失真对任务错误的影响大于噪声或遮挡，这与模拟机器人的偏好一致。

---

## AI 综述 (中文)

> AI 生成的内容

### 机器人中心图像质量评估导论

图像质量评估 (IQA) 传统上是围绕人类感知设计的，旨在优化与人类视觉系统 (HVS) 相符的美学吸引力和语义内容。然而，随着具身人工智能系统变得越来越普遍，一个基本问题出现了：机器人感知图像质量的方式与人类相同吗？Zhang 等人的这项研究通过开发第一个专门为具身智能应用中的机器人视觉系统 (RVS) 设计的综合框架，引入了 IQA 的范式转变。

![[300 Resources/320 References/attachments/db0f5eb0cf71cab3beb0a70721a9f063_MD5.png]]

*图 1：所提出的具身图像质量评估框架概述，展示了机器人如何根据任务性能而不是人类审美偏好来评估图像质量。*

这项工作解决了当前 IQA 研究中的一个关键空白，认识到具身机器人——那些通过物理与环境互动来完成特定任务的机器人——与人类有着根本不同的视觉要求。虽然人类优先考虑高级语义理解和美学吸引力，但机器人需要与它们成功执行抓取、推动或导航等物理任务的能力直接相关的图像质量指标。

### 视觉感知中的莫拉维克悖论

这项研究以莫拉维克悖论为基础，该悖论指出，对人类来说容易的任务对机器来说往往很难，反之亦然。应用于图像质量评估，这意味着严重影响人类感知的视觉失真可能对机器人性能影响最小，而看似微小的技术伪影却可能严重损害机器人的任务执行。

这种根本差异源于人类和机器人视觉系统不同的处理方法。人类擅长高级语义解释，并且当总体意义保持清晰时，通常可以忽略技术缺陷。相比之下，机器人通常严重依赖低级几何特征、纹理一致性和结构完整性，以实现准确的环境理解和操纵规划。

![[300 Resources/320 References/attachments/2bc8c880f7de8c5b8bb00580f1123516_MD5.png]]

*图 2：具身偏好数据库 (EPD) 的构建，展示了在各种图像失真下机器人任务性能如何用于生成质量分数。*

### 具身偏好数据库

这项工作的一个基石贡献是创建了具身偏好数据库 (EPD)——第一个 IQA 数据集，其中质量注释完全由机器人而不是人类受试者生成。该数据库的构建涉及几个关键组成部分：

**模拟环境**：研究人员利用 SAPIEN 模拟器和 ManiSkill 平台来创建逼真的机器人交互场景。一个固定基座的机械臂作为具身代理，配备了一个捕获 128×128 像素 RGB 图像的单目相机。

**任务框架**：两个基本的操纵任务构成了评估的基础：

- **推动任务**：机器人必须将盒子推到目标位置
- **拾取任务**：机器人必须抓取并举起一个球

**失真类别**：该数据库包含 25 种常见图像失真类型，分为五大类别，每种类别应用五种不同强度级别：

- 模糊失真（高斯模糊、镜头模糊、运动模糊）
- 颜色失真（颜色扩散、颜色偏移、量化）
- 压缩伪影（JPEG、JPEG2000）
- 噪声（白噪声、脉冲噪声、乘性噪声）
- 亮度和空间失真

**质量标注方法**：关键创新在于标注过程。该数据库不依赖人类主观评分，而是使用机器人实际任务性能作为质量指标。三种强化学习算法（PPO、SAC和TDMPC2）控制机器人，任务执行期间累积的奖励作为图像质量得分。这种方法直接将视觉输入质量与功能性能联系起来。

![[300 Resources/320 References/attachments/0b864a539e30bebced3b9813604fbfe2_MD5.png]]

*图3：EPD数据库中不同失真类型的分布分析，显示了各种图像退化如何影响机器人任务性能。*

### 关键发现：人类与机器人感知

EPD的实证分析揭示了人类和机器人图像质量评估之间显著的差异：

**与人类评分相关性低**：人类平均意见得分（HMOS）与具身平均意见得分（EMOS）之间的相关性极低，皮尔逊相关系数在0.12到0.21之间。这为莫拉维克悖论在图像质量评估背景下提供了强有力的经验证据。

![[300 Resources/320 References/attachments/43d643d12b7641ff7e53093af5ad1e26_MD5.png]]

*图4：散点图显示人类主观评分与机器人任务性能评分之间的低相关性，突出了感知的根本差异。*

**失真敏感性模式**：研究表明，机器人对不同类型的图像失真表现出独特的敏感性模式：

- **颜色失真**对机器人任务造成最显著的干扰
- **运动模糊**对任务性能影响最小
- **噪声伪影**虽然在视觉上对人类造成干扰，但对机器人功能影响相对较小
- **JPEG2000压缩**和**颜色扩散**是具身任务中最成问题的类型

**任务特异性**：即使在机器人领域内，不同任务对图像质量的敏感性也各不相同，推动（push）任务和抓取（pick）任务之间的相关系数相对较低。这表明机器人的最佳图像质量评估可能需要依赖于具体任务。

![[300 Resources/320 References/attachments/81fb11f4ef105ba161c35b0ce825c3b5_MD5.png]]

*图5：相关矩阵显示了不同机器人任务之间质量得分的关系。*

### MA-EIQA 模型架构

为了解决以机器人为中心的IQA的独特要求，研究人员开发了多尺度注意力具身图像质量评估（MA-EIQA），这是一种专为具身智能应用设计的轻量级无参考模型。

![[300 Resources/320 References/attachments/bbc75066fece1310ddd03a6fc5fa889d_MD5.png]]

*图6：所提出的MA-EIQA模型架构，具有多尺度特征编码和具身注意力机制。*

**多尺度特征编码器**：受路径聚合网络（PANet）启发，该组件捕获宏观语义信息和微观纹理细节，这些对于机器人感知至关重要。编码器通过以下方式处理ResNet50骨干网络不同阶段的特征：

- **自上而下路径**：上采样和1×1卷积，用于语义信息流
- **自下而上路径**：下采样和3×3卷积，用于空间细节增强

每个级别的特征融合的数学公式可以表示为：

$$
P_i = \text{Conv}_{1 \times 1}(C_i) + \text{Upsample}(P_{i+1})
$$

$$
N_i = \text{Conv}_{3 \times 3}(P_i + \text{Downsample}(N_{i-1}))
$$

其中$C_i$表示骨干特征，$P_i$表示自上而下特征，$N_i$表示最终聚合特征。

**具身注意力模块**：基于卷积块注意力模块（CBAM），该组件使模型能够专注于与任务最相关的视觉特征。注意力机制按顺序操作：

1.  **通道注意力**：通过平均池化和最大池化聚合空间信息，然后应用MLP和Sigmoid激活。
2.  **空间注意力**：通过池化操作聚合通道信息，然后进行卷积和Sigmoid激活。

注意力加权特征的计算方式如下：

$$
F_A = \text{SpatialAttention}(\text{ChannelAttention}(F_E) \times F_E) \times F_E
$$

**回归层**：最后一个组件由两个全连接层组成，它们将注意力加权特征映射到一个质量分数，并使用均方误差损失进行训练。

### 性能评估与结果

MA-EIQA模型在机器人中心任务评估中，表现出优于现有IQA方法的卓越性能：

**基准比较**：该模型持续优于16种主流IQA方法，包括全参考和无参考方法，并取得了最先进的结果：

- SRCC：0.5755
- KRCC：0.4032
- PLCC：0.5836

**计算效率**：MA-EIQA拥有4883万个参数，保持了轻量级结构，这对于部署在资源受限的机器人系统上至关重要，其使用的参数明显少于基于Transformer的替代方案。

**消融研究结果**：组件分析显示：

- 多尺度特征编码器使SRCC提高了6.55%
- 具身注意力模块使SRCC提高了3.85%
- 组合架构带来了12.03%的整体提升

### 真实世界验证

![[300 Resources/320 References/attachments/c8e530bbbb67a9f4f175763286ac74b9_MD5.png]]

*图7：使用UR5机器人进行的真实世界实验，展示了不同的图像失真如何在实际场景中影响任务性能。*

研究人员使用UR5机械臂通过真实世界实验验证了他们的发现。结果证实了模拟结果，表明：

- 变暗和JPEG压缩等失真显著影响机器人性能
- 噪声和遮挡对人类感知影响严重，但对机器人任务执行影响极小
- 机器人的任务性能与MA-EIQA预测的质量分数相关

### 启示和未来方向

这项研究为具身人工智能领域确立了几项重要启示：

**范式转变**：这项工作从根本上挑战了“以人类为中心的IQA指标适用于机器人应用”的假设，确立了具身智能中任务特定质量评估的必要性。

**实际应用**：该框架使机器人能够：

- 根据任务相关性自主评估和优先处理视觉输入
- 通过关注功能上重要的图像特征来优化计算资源
- 在存在视觉失真的情况下提高任务性能

**研究基础**：EPD数据库和MA-EIQA模型为未来机器人中心计算机视觉研究提供了基础资源，可能影响以下领域：

- 针对不同机器人任务的自适应视觉处理
- 改进具身AI系统的训练数据管理
- 开发任务感知图像增强技术

这项研究为未来的调查开辟了众多途径，包括开发适应特定机器人形态和环境条件的动态质量评估，以及探索不同传感器模态如何影响具身系统中的质量感知。

这项工作代表着向开发真正以机器人为中心的计算机视觉系统迈出了重要一步，这些系统优化的是功能性能而非人类审美偏好，最终有助于更强大、更自主的具身人工智能。

---

## 相关引用

1. **图像质量评价：从误差可见性到结构相似性**
   - 本文介绍了结构相似性指数（SSIM），这是一种基于人类视觉系统的基础性图像质量评估（IQA）指标。其高度相关性在于，它代表了IQA的经典方法，而主论文认为这种方法对于具身机器人而言是不足的。本文使用SSIM作为关键的比较基线，以论证这一差距。

2. **Sapien: 模拟的基于部件的交互式环境**
   - [AlphaXiv](https://alphaxiv.org/abs/2003.08515)
   - 这篇引用至关重要，因为它描述了SAPIEN仿真环境。本文的主要贡献——具身偏好数据库（EPD）——完全是在这个模拟器中通过让机械臂执行任务来构建的，这使得这项工作成为本文实验方法学的基石。

3. **路径聚合网络用于实例分割**
   - [AlphaXiv](https://alphaxiv.org/abs/1803.01534)
   - 本文被直接引用为多尺度特征编码器的灵感来源，而多尺度特征编码器是所提出的MA-EIQA网络的核心组件。因此，它对于理解主论文中开发的图像质量评估模型的架构设计和新颖性具有高度相关性。

4. **Maniqa：用于无参考图像质量评估的多维注意力网络**
   - [AlphaXiv](https://alphaxiv.org/abs/2204.08958)
   - 本文提出了一种最先进的无参考IQA模型，该模型在实验中作为关键性能基准。本文直接将其提出的MA-EIQA模型与MANIQA进行比较，以证明其具有竞争力的性能，并特别强调了其作为基于Transformer架构的轻量级替代方案的有效性。

5. **近端策略优化算法**
   - [AlphaXiv](https://alphaxiv.org/abs/1707.06347)
   - 本文介绍了近端策略优化（PPO）算法，这是一种经典的强化学习方法，在主论文中用于训练机器人代理。通过PPO训练的代理生成的奖励值被用于创建EPD数据库的质量分数，因此这项工作是数据收集过程的基础。

---

*Fetched from [AlphaXiv](https://alphaxiv.org/abs/2412.18774) on 2026-06-11 22:40*
