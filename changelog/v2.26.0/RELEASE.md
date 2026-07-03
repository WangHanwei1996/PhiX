# v2.26.0 — 开发者/使用者工具集 + bashrc 常用命令（OpenFOAM 风格）

## 摘要

参考 OpenFOAM 的工具生态（foamListTimes / foamCleanCase / foamMonitor /
foamLog / 教程 case 复制），为求解器开发者与使用者补齐一套轻量工具。
工具是自包含脚本（Python 标准库 / bash，matplotlib 可选），放在
`applications/tools/`，由 `etc/bashrc` 的 PATH 扫描自动进入命令行；
常用命令以 bash 函数形式加入 `etc/bashrc`。

## 工具（applications/tools/）

| 工具 | OpenFOAM 类比 | 功能 |
|------|----------------|------|
| **phixInfo** | foamListTimes | 读 `.field` 二进制头 + min/max/mean/L2/NaN 统计；`-l` 按场名分组列出 output/ 的可用步 |
| **phixPlot** | 轻量 paraFoam | `.field` 快速出图（2D 热图 / 1D 线图 / 3D 中间切片 → PNG）；`--diff a b` 打印两场 max/L2/相对差（回归排查利器） |
| **phixMonitor** | foamMonitor/foamLog | 解析 OutputWriter 的 `[progress]` 行：steps/s、`-f` 实时跟踪、`-n 总步数` 给 ETA |
| **phixClean** | foamCleanCase | 清理 case 产物（output*/、.vts/.dat/.field/.png/.log），默认先列出再确认，`-f` 直删；绝不碰源码/settings/tables |
| **phixNewCase** | 教程 case 复制 | 脚手架新 case：settings.jsonc 模板（mesh/time/constants/output 全节示例）+ 初始场生成脚本存根 + output/。与 makePhi/newSolver（造求解器）互补——这个造**算例** |

## bashrc 常用命令（etc/bashrc 追加）

```bash
phixBuild [目标...]   # build/ 不存在则自动配置（带 nvcc 路径与架构——
                      #   规避环境中空 CUDAARCHS 毒化 CMake 的坑）再并行编译
phixTest [-R 过滤]    # ctest --output-on-failure
phixBench             # bench_stencil + bench_semiimplicit 两个基线
phixVersion [-a]      # 最新（或全部）changelog 版本与标题
phixCase <名字>       # 按名跳到某求解器/算例目录
```

既有的 `whw` / `app` / `develop` / `solver -list` / `makePhi` 保持不变。

## 验证（对 C++ 真实产物冒烟）

- phixInfo/phixPlot 读取由 `ScalarField::write` 生成的真实 `.field`
  文件（格式解析与库端逐字节一致）；`--diff` 精确报出注入的 1e-4 扰动；
- phixMonitor 对构造日志给出 50.0 steps/s 与正确 ETA；
- phixNewCase 脚手架 + phixClean 只清产物不动 settings；
- bashrc 全部函数可加载，工具经 PATH 扫描可直呼；
- 全量 ctest 32/32 无回归。

## 兼容性

纯新增；`etc/bashrc` 追加式修改，原有行为不变。
