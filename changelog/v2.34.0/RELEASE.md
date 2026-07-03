# PhiX v2.34.0 发布说明

## 新增：DomainMask 非矩形计算域模块（PFHub 基础设施 6/6）

面向 PFHub BM1c/BM2c（T 形域）等结构化网格上的非矩形计算域，
新增 cell 掩码模块。至此 PFHub 六算例所需基础设施全部就绪
（能量积分 → PFHub 输出 → Poisson → LBM 边界 → FFT 弹性 → 掩码域）。

### `field/DomainMask.h` + `src/field/DomainMask.cu`

- **构造**：由 `inside(x,y,z)` 谓词在 cell 中心求值一次性建掩码
  （活跃 = 1，非活跃与所有 ghost = 0），构造即上载 GPU 常驻。
- **两条互补的内部无通量边界机制**：
  - `maskFaces(F)`：将所有**不介于两个活跃 cell 之间**的面通量置零，
    嵌入守恒面通量链（`faceGrad → facePW → maskFaces → divFace`）。
    活跃区总量守恒由望远镜求和**严格成立**——守恒金标准；
  - `applyClosure(f)`：镜像闭合——紧邻活跃区的非活跃 cell 取其活跃
    面邻居的平均值，使 cell 型 CD2 拉普拉斯在掩码边界看到零法向
    梯度。活跃 cell 只读不写，就地更新无竞态。适用 ghost-1 五/七点
    格式；更宽格式或严格守恒需求走面通量路线（头文件已文档化）。
- **掩码归约**：`sum(f)` 只累计活跃 cell；任意被积函数可将
  `cellMask()` 作为额外场传给 `reduce::fieldSumPW` 组合实现。
- 2D/3D 通用（closure 按 `mesh.dim` 循环方向）；谓词选不中任何
  cell 时抛 `std::invalid_argument`。

## 测试

新增 `test/moduleTest/field/test_mask.cu`（`module_mask`），五项校验
（T 形域：横梁 64×16 + 竖杆 16×48 = 1792 活跃 cell）：

1. 活跃 cell 计数与解析值一致，ghost 均非活跃；
2. `maskFaces` 精确性：逐面校验——恰好且仅有非"活跃-活跃"面被置零
   （1728 保留 / 2432 置零，全部逐面断言）；
3. **T 形域守恒扩散**：显式扩散 400 步（面通量链 + `maskFaces`），
   活跃区总量漂移 **4.6e-13**（机器精度），非活跃区泄漏**严格为 0**，
   场向活跃区均值弛豫；
4. 闭合平衡性：常数场经 `applyClosure` 后掩码边界处 CD2 拉普拉斯
   **精确为 0**（零法向梯度）；
5. 掩码求和对非活跃/ghost 毒化值（1e150）完全免疫。

测试规模：DOUBLE 38/38 通过（新增 1 项），FLOAT 3/3 通过。
