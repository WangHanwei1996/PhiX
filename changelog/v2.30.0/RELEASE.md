# v2.30.0 — PFHub 输出层 + 界面追踪（PFHub 基础设施 ②）

## 摘要

- **`IO::PFHubWriter`**：PFHub 提交口径的 CSV 时序（表头 + 逐行
  flush——中断的 run 保留全部已写行）+ `writeMeta` 生成 meta.yaml
  骨架（benchmark id、代码/仓库/GPU 元信息）；列数不匹配抛异常。
- **`interfacePosition`**（`diagnostics/Interface.h`）：沿指定轴在固定
  横向索引处找 level 交叉的物理坐标，亚格线性插值；只用 strided-gather
  kernel 拷贝**一条线**（缓存的小设备缓冲），不下载整场。
  `scanFromHigh` 从高索引端扫——枝晶尖端即最外层交叉（BM3 的
  tip position/velocity 由相邻两次输出差分即得）。

## 实测

- tanh 前沿（刻意取非格点中心 x0=0.61234）：定位偏差 **1.4e-5**
  （约 dx/550，远超亚格精度要求）；
- 圆界面沿轴半径恢复同精度；无交叉正确抛异常；
- CSV/meta 逐字段回读验证。

## 测试

`module_diagnostics`（已注册 ctest，新 diagnostics 测试子目录）。
全量 ctest **35/35**，FLOAT 3/3。

## 兼容性

纯新增（`src/IO/PFHubWriter.cpp`、`src/diagnostics/Interface.cu`
已注册 phix 库）。
