# PhiX 与开源仿真软件对标（v2.22.0，仅评估不改动）

> 日期：2026-07-03 · 对标范围：相场/微观组织开源软件 + 两个架构参照系
> 结论先行：PhiX 处在"单组自研 GPU 相场框架"生态位的健康位置——
> 调度层性能工程与定量 KKS 件在同类中少见；系统性差距集中在
> **多 GPU、模型符号层完备度、验证案例库与社区生态**三块。

---

## 1. 对标对象一览

| 软件 | 机构/来源 | 离散 | GPU | 并行规模 | 时间积分 | 特色 |
|------|-----------|------|-----|----------|----------|------|
| **MicroSim** | IISc/印度国家超算任务 | FD/FV/**FFT** | **CUDA + OpenCL** | **CUDA-aware MPI 多 GPU** | 显式为主 + FFT 谱 | Grand-potential/**KKS**/CH 全家桶、真实合金热力学耦合、**GUI + 后处理工具链** |
| **SymPhas 2.0** (2025.11) | McGill 系 | FD（编译期任意阶模板） | **CUDA**（符号表达式→优化 kernel） | MPI + GPU；32768²/1024³ 基准 | 显式为主 | **编译期符号代数**：自由能泛函→自动泛函求导→方程；对 v1 多线程 CPU 提速 ~1000× |
| **PRISMS-PF** (v2.4.1, 2025.02) | 密歇根大学 | **matrix-free FEM**（deal.II） | 未成气候 | MPI 多级并行，>10⁹ DOF | 显式为主 | **块结构 AMR**、成熟应用矩阵、PFHub 社区基准 |
| **MOOSE** | 爱达荷国家实验室 | FEM（libMesh/PETSc） | 有限（PETSc 后端间接） | MPI 大规模 | **全隐式 Newton/JFNK** | 多物理耦合生态之王、文档/教程/社区无出其右 |
| **OpenPhase / PACE3D** | 波鸿鲁尔/卡尔斯鲁厄 | FD 多相场 | **无**（CPU only） | OpenMP/MPI | 显式 | 大 N 相多相场（active phase tracking）、工业冶金案例 |
| **FiPy** | NIST | Python FV | 无直接 | PETSc/Trilinos 后端 | 隐式可用 | 教学/原型友好，规模小 |
| **JAX-PF** (2025) | 学术新秀 | JAX 数组 | GPU/TPU（XLA） | 单机为主 | 显式 | **可微分**相场（梯度反传做参数标定） |
| *架构参照* AMReX | LBNL/DOE | 块结构 AMR 框架 | CUDA/HIP/SYCL | 极大规模 | — | GPU-AMR 的工程标杆 |
| *架构参照* mumax3 | Ghent | FD 微磁学 | CUDA | 单 GPU | 显式 RK | **FP32 换吞吐**的成功先例（与我们 PHIX_PRECISION 同思路） |

## 2. PhiX 当前坐标（v2.22.0 快照）

FD/FV，单 GPU，`Real` 可切精度；显式（Euler/RK4/自适应 dt）+ **一阶
IMEX 半隐式**（matrix-free CG + CUDA graph，隐式扩散 200×dt 守恒到 0、
CH 分裂 50×dt 漂移 5e-12）；调度层经四轮优化（prev opt-in、去每步同步、
BC 批处理、CG 去同步 + 图重放：半隐式步累计 ~3×，显式步 ~1.8×）；
定量相场件：**KKS 闭式分配 + Karma 反截留流**（μ 跳变消除 96%）；
29 项 ctest 常驻含收敛阶回归；~1.5 万行，单人可整体把控。

---

## 3. 逐项差距分析

### 3.1 对 MicroSim（最直接的对标物：FD + GPU + KKS）

**我们缺**：
- **多 GPU**（他们 CUDA-aware MPI 跨节点）——我们 16 GB 单卡封顶；
- **FFT 谱求解器**（周期问题的谱半隐式，dt 收益比 CG 路线更狠且每步 O(N log N) 一锤定音）；
- **表格/CALPHAD 热力学的深度耦合工作流**（我们有 FreeEnergyTable 但 KKS 尚只支持抛物自由能；他们 GP/KKS/CH 全模型族 + 真实合金案例）；
- **GUI 与后处理工具链**（用户友好第二，但差距要记账）。

**我们强**：调度层工程（graphs/批处理/去同步——他们论文没有此层优化的证据）；
半隐式 CG 通用 BC 路线（FFT 只吃周期）；测试/收敛回归文化；反截留流模块化。

### 3.2 对 SymPhas 2.0（哲学最近的亲戚）

他们把"编译期表达式→融合 GPU kernel"走到了极致：**写自由能泛函，
编译期符号泛函求导自动生成演化方程**，任意阶差分模板也是编译期推导。
PhiX 的 FusedTerm 是同一思想的手动挡：用户写 RHS 而非泛函，融合树
手工搭。**差距 = 符号层完备度**（自动求导、任意阶模板、张量表达式）。
另外他们有 MPI+GPU 和 32768² 级基准。
我们强在：**隐式/半隐式能力**（他们显式为主）、KKS/AT 领域件、
自适应 dt 与运行时自治。

### 3.3 对 PRISMS-PF / MOOSE（FEM 双雄）

不同赛道（FEM vs FD），但三样东西与离散方法无关：
- **AMR**（PRISMS-PF 的块结构自适应网格）——相场界面局域时收益巨大；
- **全隐式非线性求解**（MOOSE 的 JFNK）——我们刻意不做，维持判断；
- **社区资产**：PFHub 社区基准参与、文档/教程/发表应用矩阵、贡献者
  生态。这是 PhiX 作为单人项目最本质的差距，也是最难靠代码补的。

值得注意：这两家的 **GPU 故事反而弱于我们**——文献里 MOOSE/PRISMS-PF
的 GPU 相场数据仍然稀缺，OpenPhase/PACE3D 明确 CPU-only。在
"GPU 原生 + 相场专用"的交集里，真正的同台竞技者只有 MicroSim 和
SymPhas。

### 3.4 对 OpenPhase（多相场模型完备度）

大 N 相（几十上百晶粒/取向）的 **active phase tracking**（每格只存
非零相）我们没有——GFA_4ph 的 4 相是硬编码的。晶粒长大/多晶凝固
方向若要走深，这是必补件。

### 3.5 新趋势记录（不构成当前差距）

JAX-PF 的**可微分相场**（用反传做 M_φ/界面能标定）与我们的 Fe-B
标定工作流在目标上同构——值得关注但路线成本高。2026 年已出现多家
凝固相场代码的公开横向基准（arXiv:2602.10316），未来 PhiX 若对外，
参与此类基准与 PFHub 是建立可信度的标准动作。

---

## 4. 差距汇总与轻重缓急（评估，不动手）

| 差距 | 谁有 | 对 GFA 工作流的影响 | 建议优先级 |
|------|------|--------------------|-----------|
| 多 GPU（CUDA-aware MPI） | MicroSim, SymPhas | 3D 大域 / 参数扫描受限于 16 GB | 中（撞显存墙时升为高） |
| FFT 谱半隐式 | MicroSim | 周期算例 dt 再上台阶 | 中 |
| KKS 表格自由能（Newton/KKSTableView） | MicroSim（GP 形式） | **高**——Fe-B FETAB 直接受益 | **高** |
| 符号层（泛函→方程自动导出） | SymPhas | 建模效率/防错 | 低-中（用户友好第二） |
| AMR / active phase tracking | PRISMS-PF / OpenPhase | 多晶方向的门票 | 低（当前模型界面遍布全域） |
| 验证案例库 + PFHub 基准 | PRISMS-PF, MOOSE | 对外可信度 | 中（develop/implicit_examples 是起点） |
| GUI/后处理/文档生态 | MicroSim, MOOSE | 单人使用影响小 | 低 |

## 5. 定位结论

PhiX 不必也不应对标 MOOSE 式生态——它的合理定位是 **MicroSim/SymPhas
一档的"研究组级 GPU 相场引擎"**，并且在三点上已经形成差异化：
调度层性能工程（graphs/批处理/去同步的量化闭环）、通用 BC 的半隐式
CG 路线（不吃周期性限制）、KKS+反截留流的定量凝固件。短板的主线
清晰：**KKS 表格自由能 → FFT 谱选项 → 多 GPU**，按 GFA 工作流的
实际需求逐个触发即可。

---

Sources:
- [PRISMS-PF 官网](https://prisms-center.github.io/phaseField/) / [GitHub](https://github.com/prisms-center/phaseField) / [npj Comput. Mater. 论文](https://www.nature.com/articles/s41524-020-0298-5)
- [MICROSIM arXiv:2404.01035](https://arxiv.org/abs/2404.01035) / [官网](https://microsim.co.in/) / [GitHub](https://github.com/syam-s/MicroSim)
- [SymPhas 2.0 arXiv:2511.10508](https://arxiv.org/abs/2511.10508) / [GitHub](https://github.com/SoftSimu/SymPhas)
- [PhaseFieldPet（PETSc 异构）](https://www.researchgate.net/publication/390626824_PhaseFieldPet_An_Open-Source_Phase-Field_Modeling_Software_for_Heterogeneous_Architectures)
- [OpenPhase 并行多相场](https://www.researchgate.net/publication/313460827_Parallel_multiphase_field_simulations_with_OpenPhase)
- [JAX-PF 可微分相场](https://www.researchgate.net/publication/399707932_Efficient_GPU-computing_simulation_platform_JAX-PF_for_differentiable_phase_field_model)
- [凝固相场代码横向基准 arXiv:2602.10316](https://arxiv.org/pdf/2602.10316)
