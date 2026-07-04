# PhiX v2.40.1 发布说明（补丁）

## 修复：OutputWriter 未知 format 静默零输出

### 问题

`OutputWriter` 只识别 `"BINARY" / "DAT" / "VTK" / "ALL"`；遇到未知
token（如 `"VTS"`）三个写开关全为 false，但 `writeFields` 仍打印
`written: output/...`——**日志显示已写、实际零文件落盘**。
PFHub 六算例配置全部写了 `"VTS"`（输出文件扩展名恰是 .vts，极易
误写），导致所有场快照静默丢失（结果 CSV 由求解器自写、不受影响）。

### 修复

- `"VTS"` 接受为 `"VTK"` 的别名（两者等价，均写 .vts）；
- 未知 format **构造时即抛 `std::invalid_argument`**，不再静默吞掉
  ——该缺陷若一开始就抛异常，冒烟测试第一步就会暴露。

### 算例重跑

BM1a/b、BM2a、BM3、BM4a、BM6a 全部以修复后二进制重跑，场快照
（.vts）补齐入 output/；结果 CSV 与已入库版本逐位一致
（确定性复算校验通过）。BM5（不经 OutputWriter）不受影响。
