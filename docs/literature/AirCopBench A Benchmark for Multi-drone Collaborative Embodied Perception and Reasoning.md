---
title: "AirCopBench: A Benchmark for Multi-drone Collaborative Embodied Perception and Reasoning"
arxiv_id: "2511.11025"
version: "v2"
date: 2025-11-23
tags:
  - paper
  - alphaxiv
source: "https://alphaxiv.org/abs/2511.11025"
authors:
  - Jirong Zha
  - Yuxuan Fan
  - Tianyu Zhang
  - Geng Chen
  - Yingfeng Chen
  - Chen Gao
  - Xinlei Chen
aliases:
  - AirCopBench: A Benchmark for Multi-drone Collaborative Embodied Perception and Reasoning
created: "2026-06-18 17:52"
published_date: 2025-11-14
---

# AirCopBench: A Benchmark for Multi-drone Collaborative Embodied Perception and Reasoning

> **arXiv**: [2511.11025](https://arxiv.org/abs/2511.11025) | **Version**: v2 | **Published**: 2025-11-23

## 摘要

Multimodal Large Language Models (MLLMs) have shown promise in single-agent vision tasks, yet benchmarks for evaluating multi-agent collaborative perception remain scarce. This gap is critical, as multi-drone systems provide enhanced coverage, robustness, and collaboration compared to single-sensor setups. Existing multi-image benchmarks mainly target basic perception tasks using high-quality single-agent images, thus failing to evaluate MLLMs in more complex, egocentric collaborative scenarios, especially under real-world degraded perception this http URL address these challenges, we introduce AirCopBench, the first comprehensive benchmark designed to evaluate MLLMs in embodied aerial collaborative perception under challenging perceptual conditions. AirCopBench includes 14.6k+ questions derived from both simulator and real-world data, spanning four key task dimensions: Scene Understanding, Object Understanding, Perception Assessment, and Collaborative Decision, across 14 task types. We construct the benchmark using data from challenging degraded-perception scenarios with annotated collaborative events, generating large-scale questions through model-, rule-, and human-based methods under rigorous quality control. Evaluations on 40 MLLMs show significant performance gaps in collaborative perception tasks, with the best model trailing humans by 24.38% on average and exhibiting inconsistent results across tasks. Fine-tuning experiments further confirm the feasibility of sim-to-real transfer in aerial collaborative perception and reasoning.

---
### AI 摘要
AirCopBench被引入作为第一个综合基准，用于评估多模态大语言模型（MLLMs）在挑战性的真实世界条件下进行多无人机协作具身感知的能力。该基准包含一个由2.9k多视图图像和14.6k VQA对组成的数据集，涵盖14个任务，揭示了当前MLLMs与人类能力之间存在显著的性能差距，同时表明在该数据集上进行微调可以大幅提高模型的性能和无人机应用的仿真到现实迁移能力。

### 要点
- 当前的MLLMs在多无人机协作具身感知方面表现出显著的性能差距（比人类基线落后24.38%），表明有很大的改进空间。
- 执行“因果评估”（推断感知退化的原因）的能力与整体性能高度相关，表明它是具身感知的一项基本认知能力。
- 在AirCopBench上进行监督微调显著提升了MLLM的性能（例如，Qwen-2.5-VL-7B提升了26.97%），并且仿真到现实的迁移是可行的，平均真实世界准确率提高了19.64%。

### 问题
- 现有的MLLM基准主要关注单智能体视觉或多图像理解，忽略了复杂的多智能体协作感知场景。
- 当前的协作感知基准缺乏多样化、真实的感知退化（例如，噪声、运动模糊、数据丢失），这些是无人机在真实世界环境中经常面临的问题。
- 需要评估MLLMs的具身推理能力，特别是它们从自我中心视角评估感知质量和做出情境感知协作决策的能力。

### 方法
- 开发了AirCopBench，一个综合基准，为多无人机系统定义了涵盖场景理解、物体理解、感知评估和协作决策的14种任务类型。
- 创建了一个高质量的、包含来自联合仿真（Carla, AirSim）和真实世界无人机图像的多视图图像数据集，并辅以多样的感知退化和新颖的事件级协作事件标注。
- 采用了混合问题生成方法（基于模型、基于规则、基于人工）和严格的质量控制措施（标准检查、盲过滤、人工细化），以确保基准的可靠性和相关性。

### 结果
- 对40个MLLMs的评估显示，表现最佳的模型Ovis2-16B仅达到59.17%的准确率，远低于人类基线78.25%，证实了该基准的挑战性。
- MLLMs在基本感知任务上表现较好，但在具身感知和协作决策的更高层次推理方面（如“可用性评估”和“何时协作”）表现挣扎。
- 错误分析识别出感知幻觉、空间推理错误和多图像理解错误是主要的失败类型，强调了MLLM开发的关键领域。

---

## AI 综述 (中文)

> AI 生成的内容

### 引言

多模态大型语言模型（MLLMs）在处理文本和图像方面展现出卓越的能力，在单智能体视觉任务中取得了显著成功。然而，它们在涉及多智能体协作感知等更复杂场景中的表现仍未得到充分探索。这一差距对于无人机（UAV）等自主系统尤为关键，因为在这些系统中，多架无人机必须协同工作，在充满挑战的条件下理解和导航动态环境。

![多无人机协作感知场景](https://paper-assets.alphaxiv.org/figures/2511.11025v2/img-0.jpeg)

这项研究引入了 AirCopBench，这是第一个专门用于评估多无人机协作具身感知中 MLLMs 的综合基准。该基准通过纳入真实的感知退化，并要求模型从第一人称视角推理何时、何物、何人以及为何与其他无人机协作，解决了当前评估框架中的基本局限性。

### 对协作感知基准的需求

当前的 MLLM 基准主要侧重于单智能体视觉任务或使用高质量、理想化图像的多图像理解。这种方法未能捕捉真实世界多智能体场景的复杂性，在这些场景中，无人机必须在包括传感器噪声、遮挡、运动模糊、数据丢失和环境干扰在内的恶劣条件下运行。

现有协作感知基准存在两个关键局限。首先，它们采用过于简化的感知设置，只考虑有限的退化类型，如遮挡，未能考虑到无人机在实践中面临的各种挑战。其次，它们缺乏具身推理能力，转而依赖非以自我为中心、程序化的决策方案，无法从第一人称视角实现情境感知、类似人类的决策。

多无人机系统比单传感器设置具有显著优势，包括增强覆盖范围、提高对传感器故障的鲁棒性以及通过信息交换解决复杂视觉任务的更大灵活性。然而，要实现这些益处，需要 MLLMs 能够理解自身的感知状态，评估其观测质量，并在充满挑战的条件下做出智能的协作决策。

### 基准设计与任务框架

AirCopBench 引入了 14 种不同的任务类型，分为四个关键维度，全面评估 MLLM 在协作感知中的能力：

![任务框架概览](https://paper-assets.alphaxiv.org/figures/2511.11025v2/img-1.jpeg)

**场景理解**任务侧重于解释整体场景，包括场景描述、不同无人机视角下的场景比较，以及观察姿态以分析观察者与场景的关系，实现具身感知。

**物体理解**任务侧重于物体层面的分析，包括物体识别、计数、用于空间位置评估的定位，以及跨不同视图查找对应物体的匹配。

**感知评估**任务通过客观图像清晰度的质量评估、主观任务有效性的可用性评估以及识别感知退化原因的因果评估，评估感知信息的质量和可用性。

**协作决策**任务评估 MLLMs 推理无人机间协作的能力，确定何时需要协作、共享哪些信息、与谁协作以及为何协作是必要的。

### 数据收集与生成流程

该基准采用系统的四步方法：数据收集、数据标注、问题生成和质量控制，确保全面覆盖现实的多无人机场景。

![数据生成流程](https://paper-assets.alphaxiv.org/figures/2511.11025v2/img-2.jpeg)

**数据收集**结合了多个来源，以确保多样性和真实性。模拟器数据来源于使用Carla和AirSim进行的协同仿真，并利用Coperception-UAV和EmbodiedCity等现有数据集。真实世界数据整合自MDMT数据集，该数据集包含真实的无人机图像以及运动模糊和遮挡等真实挑战。派生数据通过包括噪声注入和部分遮蔽在内的后处理技术，扩展了退化多样性。

**数据标注**提供了丰富、合理的标注，这对于空中协同感知至关重要。事件级标注捕捉了无人机间的交互策略，包括图像质量、感知可用性的详细评分，以及退化原因的推理。对象级标注提供了传统的标注，如对象列表、边界框和属性。

**问题生成**采用了一种混合方法，结合了使用GPT-4o等强大MLLM的模型生成，以及任务分解、角色扮演设置、思维链提示和少样本学习等技术。基于规则的生成处理结构化任务，而基于人工的生成则解决复杂的多图像推理需求。

**质量控制**实施了严格的措施，包括根据内容完整性和答案有效性等标准进行常规检查，盲过滤掉无需视觉输入即可回答的问题，以及耗时超过800小时的广泛人工优化。

### 数据集构成和特点

最终的基准测试包括2,945张多视角图像和14,638个视觉问答（VQA）对，涵盖各种场景和退化类型。

![数据集统计数据](https://paper-assets.alphaxiv.org/figures/2511.11025v2/img-3.jpeg)

该数据集包含各种无人机群组配置（3-6架无人机），数据来源包括真实世界图像和模拟环境。感知退化的分布反映了真实的挑战，其中小目标占63.3%，其次是复杂背景（30.9%）和各种环境干扰类型，包括阴影、运动模糊和传感器故障。

任务分布确保了对所有四个维度的全面覆盖，其中协同决策任务占数据集的23.3%，感知评估占34.6%，对象理解占29.4%，场景理解占12.7%。这种分布反映了基准测试对更高层次推理和协同决策能力的重视。

### 实验结果与性能分析

对40个主流MLLM的广泛评估揭示了显著的性能差距以及关于当前多无人机协同感知能力的重要见解。

**总体性能差距**：所有评估的MLLM与人类能力相比都表现出显著的性能缺陷。表现最佳的模型Ovis2-16B仅达到59.17%的准确率，比人类表现（78.25%）低24.38%。这一巨大差异突出表明，当前的MLLM在复杂多智能体场景中远未达到人类水平的理解能力。

**任务特定性能差异**：MLLM在不同任务类别中表现出不一致的性能。它们在场景描述等基本感知任务上通常表现更好，但在需要更深层次领域知识和目标导向推理的任务（如可用性评估和协同决策）上则表现出显著的困难。

**相关性分析**：任务相关性分析揭示了不同能力之间的重要关系：

![任务相关性矩阵](https://paper-assets.alphaxiv.org/figures/2511.11025v2/img-4.jpeg)

因果评估显示与几乎所有其他任务高度相关，这表明理解感知退化的原因对于具身感知至关重要。目标匹配与目标识别和接地表现出强相关性，这强调了鲁棒的单图像感知对于多视角理解的重要性。

**常见错误类型**：分析确定了三种主要的推理失败类别：

![常见错误示例](https://paper-assets.alphaxiv.org/figures/2511.11025v2/img-5.jpeg)

感知幻觉错误涉及由于不正确的视觉输入处理而导致的误识别或未能识别物体。空间推理错误反映了在准确解释物体关系和位置方面的困难。多图像理解错误显示了在比较和整合多个图像信息方面的挑战。

### 训练有效性验证

**监督微调结果**：微调实验证明了数据集作为训练资源的有效性。Qwen-2.5-VL-7B 在微调后准确率提高了 26.97%（从 47.33% 到 74.30%），而 LLaVA-NeXT-13B 提高了 19.30%（从 38.31% 到 57.61%）。这些显著的改进验证了基准测试对于提升 MLLM 能力的实用性。

**仿真到现实迁移**：实验证实了在仿真数据上进行训练对于现实世界应用的可行性。AirCop-7B 仅在仿真数据上进行微调，在真实世界无人机图像测试中，整体准确率提高了 19.64%。显著的改进包括目标接地（+50.00%）和可用性评估（+53.20%），展示了显著的泛化能力。

### 意义与影响

AirCopBench 通过弥补当前单一智能体能力与多智能体协作系统复杂要求之间的差距，代表了 MLLM 评估的一个关键进展。该基准测试关注在逼真的退化条件下的具身推理，推动了我们对智能自主系统期望的界限。

仿真到现实迁移能力的验证对实际部署具有深远影响，因为在真实机器人系统上训练复杂的 AI 模型通常成本过高且在物流上面临挑战。证明知识从仿真到现实的有效迁移为更经济的研究和更快的开发周期开辟了道路。

基准测试揭示的巨大性能差距表明了未来研究的肥沃土壤，特别是在开发能够进行更高层次推理、因果理解和多智能体环境中战略决策的 MLLM 方面。全面的任务框架和高质量数据集为推进具身 AI 和多无人机系统研究提供了标准化基础。

### 未来方向

AirCopBench 为推进智能多无人机系统奠定了基础，其应用涵盖搜索与救援、监控、环境监测、物流和国防。该基准测试对协作决策和具身推理的强调，预示着未来在更复杂的、能够在复杂动态环境中可靠运行的多智能体 AI 系统方面的发展。

这项研究为未来的工作开辟了多条途径，包括开发能够处理各种退化的更鲁棒的感知模型、提高多图像理解能力以及创建更复杂的协作推理框架。随着该领域的发展，基准测试的可扩展设计允许通过新的场景、退化类型和任务类别进行持续扩展。

---

## 相关引用

1. **Where2comm: 通过空间置信地图实现通信高效的协同感知**
   - 本文是主要工作的关键来源，因为它提供了 Coperception-UAV 数据集。AirCopBench 的作者直接使用这些模拟数据作为生成其基准问题的依据，使其成为他们研究的基础组成部分。

2. **解决目标遮挡的鲁棒多无人机多目标跟踪：基准**
   - 这项工作很重要，因为它引入了MDMT数据集，主论文从中获取了其真实世界数据。通过整合来自这一基准的数据，AirCopBench将其评估建立在真实场景中，从而提高了其发现的可信度和适用性。

3. **Urbench：一个在多视角城市场景中评估大型多模态模型的综合性基准**
   - 这份引用具有高度相关性，因为它提出了一个用于多视图理解的基准，直接影响了 AirCopBench 的设计。主论文在定义其部分核心任务（例如场景比较和对象匹配）时明确引用了 UrBench，将其确立为一项重要的同期研究。

4. **具身城市：一个用于具身智能体在真实城市环境中的基准平台**
   - [AlphaXiv](https://alphaxiv.org/abs/2410.09604)
   - 本文介绍了 EmbodiedCity 平台，它是 AirCopBench 模拟器数据的另一个关键来源。该平台对具身智能体的关注点与主论文评估具身感知和协作推理的目标直接契合，使其成为该基准测试这一特定方面的重要参考。


---

*Fetched from [AlphaXiv](https://alphaxiv.org/abs/2511.11025) on 2026-06-18 17:52*
