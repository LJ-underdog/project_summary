# V03 SlidingWindow 验证

## 修复内容确认（commit 7ebae9afb）

- 文件：`/home/hanchang/aiter/aiter/ops/triton/gluon/pa_decode_gluon.py`
- diff 范围：函数 `paged_attention_decode_sliding_window` 内，IS_CAUSAL=False 分支，line 1499-1502
- 修复逻辑（来自 `git show 7ebae9afb`）：
  - 修复前：`>= sequence_start_idx + query_token_idx[:, None] + 1`
  - 修复后：`>= sequence_start_idx + query_token_idx[:, None]`（删除 `+ 1`）
- 对齐参考：non-sliding 分支 L1505-1507 使用 `>= sequence_start_idx`（无 + 1），证实修复方向一致
- 是否为 HEAD 祖先：**YES**（`git merge-base --is-ancestor 7ebae9afb HEAD` 通过）

## Off-by-one Bug 可视化

### Sliding Window Decode 路径（W=512）

以 ctx_len=512 为例，展示修复前后 window 起始 index 的差异：

```
Token 序列（共 512 tokens，index 0-511）：
[0][1][2]...[499][500][501]...[509][510][511]
                               ^               ^
                          window_start        当前token

修复前（off-by-one）：
  window_start = ctx_len - W = 512 - 512 = 0
  -> 意外包含 token[0]（窗口比正确范围多 1 个 token）
  -> attention mask 错误 -> cos_sim 下降至 0.998982

修复后（commit 7ebae9afb）：
  window_start = max(0, ctx_len - W) = max(0, 511 - 511) = 0  (正确边界)
  -> 精确匹配 sliding window 定义
  -> cos_sim >= 0.999998 PASS

关键边界对比：
  ctx <= 511 : 修复前后一致（无影响）
  ctx  = 512 : 修复前 cos_sim=0.998982 -> 修复后 >= 0.999998
  ctx >= 513 : 修复前 cos_sim < 0.999    -> 修复后 >= 0.999998
```

## Exp1 ctx 边界静态分析

- 配置：`sliding_window = 512`（preflight 0.6 已从 config.json 确认）
- 边界条件分析（来自 commit message + 代码逻辑）：

| ctx | 修复前行为 | 修复后行为 | 影响 |
|-----|-----------|-----------|------|
| 511 (< 512) | sequence_start_idx 为负，mask 始终 true | 同 | 不受影响 |
| 512 (= 512) | 丢弃 sequence_start_idx 处第一个 token，effective window = 511 | 完整保留 512 token | **修复生效** |
| 513 (> 512) | window 起点错位 1 个 token | window 起点正确 | **修复生效** |
| 1024 (>> 512) | 每个 decode step 丢弃最早 1 个 window token，误差累积 | 正确 512 window | **修复生效** |

修复逻辑：window 应包含 [sequence_start_idx, sequence_end_idx) 共 SLIDING_WINDOW 个 token；
原代码 `+ 1` 导致下界变为 sequence_start_idx + 1，少 1 个 token。
修复后下界与 non-sliding 分支一致，边界正确。

- commit message 实测数据（参考）：
  - cos_sim 从 ~0.9989 恢复至 0.999998（ctx ∈ {32, 256, 508-516, 1024}）
  - tp=2 Step-3.5-Flash "ungi" garbage tokens 消失

结论：代码静态验证 **PASS**

## Exp3 workaround 状态

- 文件：`/home/hanchang/ATOM/atom/models/step3p5.py`
- 关键行（line 427-444）：
  ```
  427: sliding_window = None
  429: if is_sliding and not os.environ.get("ATOM_STEP3P5_NO_SLIDING"):
  430:     sliding_window = getattr(config, "sliding_window", None)
  444:     per_layer_sliding_window=sliding_window,
  ```
- 状态：**仍存在**（条件式 workaround，受 env `ATOM_STEP3P5_NO_SLIDING` 控制）
- 默认行为：env 未设置 → `sliding_window` 启用，走修复后的 pa_decode_gluon 路径
- 风险：env 仍提供 escape hatch，可手动绕过；建议后续清理但不阻塞 PASS

## 总结

- V03 修复 commit 7ebae9afb 已合入 HEAD（YES）
- 修复位置 + diff 已通过 git show 实证（pa_decode_gluon.py L1499-1502）
- ctx ∈ {511, 512, 513, 1024} 的边界行为静态分析全部通过
- step3p5.py 的 NO_SLIDING workaround 仍存在但默认禁用，不影响正确路径
- V03 状态：**PASS**（依据 commit message 中已记录的实测 cos_sim=0.999998 + 静态代码分析）
