# rms_norm_compat

## 定位

`rms_norm_compat` 是 **参考精确（reference-exact）的 RMSNorm epilogue**，用于对齐
Gemma-4 推理链（`vllm.ir.ops.rms_norm` / `RMSNorm.forward_static`）在 FP16 上的
逐位数值行为。与 `rms_norm` 的关键区别：**它不重算行统计量**——每行的 `rstd`
由调用方（参考链本身）作为输入传入，因此该算子只负责 epilogue：缩放 + 权重乘 +
（可选）残差加。

## 语义（与设备 kernel 的注释逐条对应）

- 纯 epilogue（3 输入 / 1 输出）：
  `s = fp16(fp32(x) * rstd)`；`y = fp16(fp32(s) * fp32(w))`
  ——即 **两次舍入**：`x * rstd` 先按 fp16 RNE 舍入，再与 fp16 权重做 fp16 乘。
- 融合残差（4 输入 / 2 输出）：
  `xr = fp32(x) + fp32(res)`；`residual_out = fp16(xr)`；
  `s = fp16(xr * rstd)`；`y = fp16(fp32(s) * fp32(w))`
  ——`residual_out` 的舍入就是参考链里残差分支的那次舍入。

正因为这两次显式舍入与"先算完再舍入一次"不同，本算子才能与参考链逐位一致；
这也是它与 `rms_norm`（单次舍入、内部自行求 `rstd`）不可互换的原因。

## 接口

| 层 | 入口 |
| --- | --- |
| Python | `tecoops.rms_norm_compat(input, weight, rstd, residual, output, residual_out)` |
| C ABI | `tecoopsRmsNormCompat(handle, input, weight, rstd, residual, output, residual_out, num_tokens, hidden_size)` |
| UAL | `teco/ual/ops/rms_norm/find_rms_norm_compat.cpp` → `teco_slave_rms_norm_fp16_compat[_add]` |
| Zoo | `test/zoo/teco/rms_norm_compat/`（case_0 纯 epilogue，case_1 融合残差） |
| Python 测试 | `python_api_test/test_rms_norm_compat.py` |

## 形状 / dtype 约束

- `input`、`residual`、`output`、`residual_out`：FP16，`[N, D]`，连续。
- `weight`：FP16，`[D]`。
- `rstd`：**FP32**，`[N]`（不是 FP16；kernel 按 `const float *` 读取）。
- `D` 必须为 32 的倍数：epilogue 以 16 宽 SIMD 前进，且 tile 宽度为 32
  （`head_dim % 32 != 0` 时尾列不会被写）。
- `residual` 与 `residual_out` 必须同时给出或同时为 `nullptr`，否则 C API 返回
  `TECOOPS_STATUS_BAD_PARAM`；`input/weight/rstd/output` 为 `nullptr` 同样拒绝。
- 算子绑定调用方**当前 SDAA stream**（保证它排在产出 `rstd` 的统计 kernel 之后），
  热路径无分支、无 `getenv`。

## 与 `rms_norm` 的差异（不可混用）

| 项 | `rms_norm` | `rms_norm_compat` |
| --- | --- | --- |
| 行统计量 | 算子内计算 `sum(x^2)` | **调用方传入 `rstd`（fp32）** |
| `eps` | 有 | 无 |
| 舍入 | `x * rstd * w` 后一次舍入 | `x * rstd` 先舍入到 fp16，再与 `w` 做 fp16 乘 |
| 残差 | 可选（同 `rstd` 语义） | 可选（`residual_out` 为参考链的舍入结果） |

## 验证状态与边界

- 官方固定形态 zoo：`./build/demo --gid=0 --perf_repeat=50 --warm_repeat=3 --gtest_repeat=1 --case_path=zoo/teco/rms_norm_compat/test_case/<c>.prototxt`；
  参考实现按上文两次舍入写，与设备输出逐例对照（DIFF1 门限 1e-2，实测见 PR #45 评论）。
- Python 侧：`python_api_test/test_rms_norm_compat.py` 覆盖绑定/导出契约、非默认
  stream、`rstd` 被原样消费（喂入被缩放过的 `rstd` 以排除"内部重算"）、以及拒绝路径。
- **未覆盖**：plugin 层（`teco/plugin/pluginRmsNormCompat*` 不存在，推理侧零覆盖）、
  FakeTensor/meta 与 `torch.compile`（无证据，不应视为已支持）、多流并发。
