# UAV VLA 数据集与 Benchmark 调研(面向三分数具身 IQA)

> 调研日期: 2026-07-12 | 约束: 无真机,仿真闭环优先 | 置信度: 中高
> 检索方式: WebSearch + curl 直读 arXiv/GitHub/HuggingFace(WebFetch 被企业网络策略拦截)

## 1. 背景与动机

本项目对齐 Embodied-IQA(arXiv [2505.16815](https://arxiv.org/abs/2505.16815))的
**perception-cognition-decision-execution** 四阶段框架,把"图像质量分数"重定义为
**图像对机器人任务的可用性**,由三个独立维度合成:

| 分数 | 含义 | 生成方式 | 量化工具 |
|---|---|---|---|
| ① **Cognition** | 失真导致"理解错误"的程度 | 对比 VLM 在参考图/失真图的文本输出差异 | BLEU / ROUGE / CIDEr |
| ② **Decision** | 失真导致"动作规划出错"的程度 | 对比 VLA 在参考图/失真图的动作向量偏差 | 位置欧氏距离 + 旋转余弦相似度(UAV 无 gripper 项) |
| ③ **Execution** | 失真在物理世界的最终后果 | 真机执行 VLA 决策(本项目无真机 → 仿真闭环代理) | 成功率 + 碰撞率 |

**问题**:现用数据集 AirCopBench(AAAI 2026)本质是多机协同 VQA/感知-理解 benchmark,
只覆盖 ① Cognition。② Decision 与 ③ Execution 两层缺标注来源。

**关键约束(本项目)**:无真机。③ Execution 只能用**仿真闭环成功率/碰撞**做地面真值代理,
论文中须明确表述为 "simulated execution success",不等同 Embodied-IQA 的 UR5 真机地面真值。

## 2. Niche 确认:UAV VLA 视觉退化鲁棒性是空白

补搜"VLA 视觉退化鲁棒性"命中一批**桌面操作域**先行工作,但无人做 UAV:

- **NEBULA**:VLA sensor-level 退化压测套件,退化类型与本项目 36 种畸变高度重叠
  (Poisson-Gaussian 噪声、rolling-shutter、光闪烁、分辨率退化、丢帧、color cast)。
- **RobustVLA**:π₀ 在 Gaussian noise 下 51%、dead-pixel 下仅 21% 成功率。
- **CRT / BYOVLA**:即插即用去退化模块。

检索明确指出:*"No existing work systematically evaluates UAV-specific VLAs under
combined visual corruptions"*,且这些工作全用 LIBERO/Meta-World(机械臂),非空中导航。

> **定位成立**:把 NEBULA/RobustVLA 的"畸变→动作鲁棒性"范式迁到 UAV VLA,
> 并做成 IQA 质量指标,是当前空白。

## 3. 候选数据集 × 三分数覆盖

要一个数据集同时喂三分数,它必须自带**可控仿真器**(能重渲染注入畸变的 `O_t` + 闭环跑执行)。

| 数据集 | ① Cognition | ② Decision | ③ Execution(仿真) | 畸变可注入 | arXiv |
|---|:---:|:---:|:---:|:---:|---|
| **TravelUAV** (TRAVEL) ⭐ | ✅ VLN 指令→VLM | ✅ 6-DoF 连续位姿 | ✅ AirSim 闭环 + SR/OSR/SPL/NE + 碰撞内建 | ✅ RGB+Depth 5 视角 | [2410.07087](https://arxiv.org/abs/2410.07087) |
| **UAV-Flow(-Sim)** | ✅ atomic 指令→VLM | ✅ 6D=`[xyz,cos(rpy)]`(与论文算法零改造对齐) | ✅ 闭环仿真套件(语义满足→成功) | ✅ UnrealCV egocentric | [2505.15725](https://arxiv.org/abs/2505.15725) |
| **BEDI** | ✅ semantic perception | ✅ action generation | ✅ motion control(UE+AirSim) | ⚠️ 偏 eval 框架,非可重放轨迹集 | [2505.18229](https://arxiv.org/abs/2505.18229) |
| **CognitiveDrone** | — | ✅ 实时 4D 动作 | ✅ 仿真 success rate(59.6%/77.2%) | ✅ first-person | [2503.01378](https://arxiv.org/abs/2503.01378) |
| **OpenFly** | ✅ 指令 | 4-DoF(粒度弱) | ⚠️ 无碰撞判定 | ✅ 3DGS 实景重渲染,100k 轨迹 | [2502.18041](https://arxiv.org/abs/2502.18041) |
| **HUGE-Bench** | — | 高层 VLA | ✅ Collision Rate / CSPL(安全指标最全) | ✅ 3DGS-Mesh 数字孪生 | [2603.19822](https://arxiv.org/abs/2603.19822) |

其它相关:AerialVLN(2023, 4-DoF, 首个)、CityNav(ICCV 2025, 真实 32k)、
CosFly-Track(2026, 6-DoF 跟踪)、AIR-VLA(2026, Isaac Sim 空中操作,需机械臂)、
ESARBench(2026, 搜救)。

## 4. 结论:主力选 TravelUAV(TRAVEL)

**唯一一个三分数所需要素原生齐备、执行指标开箱即用、且仿真可跑的单一数据集:**

- **一个 AirSim 场景,三分数一次产出**(无真机场景最省事)
- ① Cognition:用 VLN 指令让 VLM 描述"目标在哪/是什么" → BLEU/ROUGE/CIDEr
- ② Decision:6-DoF 动作向量 → 位置距离 + 旋转余弦
- ③ Execution:AirSim 闭环 → **SR + 碰撞率(内建,无需自己设计)**
- 三分数共享同一批 `O_t`,注入同一畸变,天然可比,可复现论文
  "decision/execution 比 cognition 更预测成功" 的结论

**取舍**:TravelUAV 的 Decision 是长程导航位姿;若要论文那种"细粒度短程动作偏差",
用 **UAV-Flow-Sim** 作补充(6D 格式与论文算法更严格对齐)。
**安全维度**追问时用 **HUGE-Bench**(Collision Rate/CSPL)。

### 数据规模(据检索,接入前需核对)
- 12,149 条轨迹 / 22 场景 / 96 asset 类型 / 76 目标类别
- 6-DoF 连续位姿 `(x,y,z,θ,φ,ψ)`,RGB+Depth(前/左/右/后/下 5 视角)+ velocity + IMU
- 碰撞检测(点云),碰撞=任务失败
- Splits: Train 9,152 / Test Seen 1,410 / Test Unseen Map 958 / Test Unseen Object 629
- 指标: SR(20m 内)/ OSR / SPL / NE

## 5. 下载与安装

> 官方命名: 项目=**TRAVEL**,数据集=**TravelUAV**。仓库 `prince687028/TravelUAV`(branch main)。

| 内容 | 链接 |
|---|---|
| 代码 | https://github.com/prince687028/TravelUAV |
| 项目页 | https://prince687028.github.io/Travel/ |
| 数据集(轨迹+指令+标注) | https://huggingface.co/datasets/wangxiangyu0814/TravelUAV |
| **仿真环境**(AirSim 地图可执行档,闭环必需) | https://huggingface.co/datasets/wangxiangyu0814/TravelUAV_env |
| 额外模型 GroundingDINO | https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth |

```bash
# 1. 代码
git clone https://github.com/prince687028/TravelUAV.git && cd TravelUAV

# 2. 环境
conda create -n llamauav python=3.10 -y && conda activate llamauav
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirement.txt
# 按 Model/LLaMA-UAV/README.md 装 LLaMA-UAV 依赖
# 应用 AirSim Python API 修复: https://github.com/microsoft/AirSim/issues/3333#issuecomment-827894198

# 3. 数据集(需 huggingface_hub / git-lfs)
huggingface-cli download wangxiangyu0814/TravelUAV --repo-type dataset --local-dir data/TravelUAV

# 4. 仿真环境(闭环执行必需,体积大,先确认磁盘)
huggingface-cli download wangxiangyu0814/TravelUAV_env --repo-type dataset --local-dir envs/

# 5. GroundingDINO → 放到 src/model_wrapper/utils/GroundingDINO/

# 6. 启动 AirSim 环境服务器(先在 airsim_plugin/AirVLNSimulatorServerTool.py 设 env_exec_path_dict)
cd airsim_plugin
python AirVLNSimulatorServerTool.py --port 30000 --root_path /path/to/your/envs
```

## 6. 接入本项目的方案

### 畸变注入点
AirSim 通过 `simGetImages` 返回 RGB/Depth。本项目 `src/uav_iqa/distortions/` 应插在
**"AirSim 取图 → 喂给 VLA"** 之间,对 `O_t` 做退化,三分数共用同一畸变管线:

```
TravelUAV 场景 → O_t (clean, RGB+Depth)
        │  注入 36 种畸变 (distortions/)
        ▼
      O_t' → VLA (LLaMA-UAV baseline / OpenVLA)
        │
        ├─► â_t 6-DoF ──vs── expert a_t  → ② Decision (位置距离 + 旋转余弦)
        └─► AirSim 闭环执行 → SR / 碰撞   → ③ Execution (仿真代理)

        与 ① Cognition (AirCopBench VLM 文本) 合成 → 具身 IQA 质量分
```

### 三分数计算落点
- ① Cognition:复用现有 `src/uav_iqa/vlm/` + `text_metrics.py`(BLEU/ROUGE/CIDEr)
- ② Decision:`src/uav_iqa/vla_scorer.py` 里实现 UAV 版 —— 位置欧氏距离 + 旋转余弦,
  去掉 gripper 项;TravelUAV 6-DoF 位姿格式直接对齐
- ③ Execution:调 TravelUAV/AirSim 闭环,取其内建 SR / 碰撞判定

## 7. 局限与待核实

- WebFetch 被企业网络策略拦截,细节均来自搜索摘要 + curl 直读,未逐篇通读全文。
- TravelUAV 数据规模/splits 来自检索摘要,**接入前须 clone repo 核对 AirSim 版本、
  数据字段、畸变注入接口**。
- `TravelUAV_env` 体积大(UE+AirSim 打包地图,通常数十 GB/图),先确认磁盘。
- 无真机是硬约束:Execution 为**仿真代理**,limitations 须写明。
- 部分 2026 arXiv(HUGE-Bench 2603.x、CosFly-Track 2605.x、NEBULA/RobustVLA 2512.x)
  较新,规模/指标以摘要为准,正式引用前核对原文。

## 8. 参考来源

- Embodied-IQA — [2505.16815](https://arxiv.org/abs/2505.16815)
- TRAVEL / TravelUAV — [2410.07087](https://arxiv.org/abs/2410.07087) · [repo](https://github.com/prince687028/TravelUAV) · [dataset](https://huggingface.co/datasets/wangxiangyu0814/TravelUAV) · [env](https://huggingface.co/datasets/wangxiangyu0814/TravelUAV_env)
- UAV-Flow Colosseo — [2505.15725](https://arxiv.org/abs/2505.15725) · [project](https://prince687028.github.io/UAV-Flow/)
- BEDI — [2505.18229](https://arxiv.org/abs/2505.18229)
- CognitiveDrone — [2503.01378](https://arxiv.org/abs/2503.01378)
- OpenFly — [2502.18041](https://arxiv.org/abs/2502.18041) · [agent](https://huggingface.co/IPEC-COMMUNITY/openfly-agent-7b)
- HUGE-Bench — [2603.19822](https://arxiv.org/abs/2603.19822)
- CityNav (ICCV 2025) — [CVF](https://www.openaccess.thecvf.com/content/ICCV2025/html/Lee_CityNav_A_Large-Scale_Dataset_for_Real-World_Aerial_Navigation_ICCV_2025_paper.html)
- CosFly-Track — [2605.17776](https://arxiv.org/abs/2605.17776)
- RaceVLA — [arxivlens](https://arxivlens.com/paperview/details/racevla-vla-based-racing-drone-navigation-with-human-like-behaviour-4262-f089db5a)
- RobustVLA / NEBULA — [2512.15258](https://arxiv.org/pdf/2512.15258v1.pdf)
