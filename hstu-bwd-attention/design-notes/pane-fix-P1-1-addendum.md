# 补充(并入当前 P1-1 修复任务)— 测试套件升级为 causal×因子 交叉矩阵

修完 P1-1 后,**不要只补 num_target 这一格**,把测试套件升级成系统性交叉覆盖,堵同类洞。

## 要做
在 `test/run_bwd_tests.py` 里,对**每个 mask 因子都跑 `causal ∈ {0,1}` 两遍**(目前是"因子只配 causal=1、causal=0 只配 no-mask"的对角线)。具体补 **causal=0 × 因子** 这一整列(causal=1 已有,不重复堆太多):

- batched,`causal=0` ×:`-targets`(已是 bug 复现)、`-context_len`、`-minfull_len`、`-local_len`、以及组合(`local+context+minfull+targets`)。
- 代表性地在 **jagged** 与 **group** 各补 1–2 个 `causal=0 + 因子`(尤其 `causal=0 + per-batch/per-group targets`),确认三模式都不漏。
- 保留并继续跑现有 causal=1 全套(不删)。

## 判定
- 全部新 case 走 **oracle 对拍**(reference 是 ground truth):**期望全 PASS**(fwd 支持 causal=0+因子 → bwd 修好后应匹配)。
- **若某个 causal=0×因子 case 修完仍 FAIL**:不要藏、不要标 reject 蒙混——在 `fix-P1-1-done.md` 里如实列为"新发现的未覆盖缺陷",交 lead(可能是同类 IsMasking 耦合的另一格,如 contextual/minfull 在 causal=0 下的边界)。
- 跑 `python3 test/run_bwd_tests.py` 整体 exit 0;现有 34 案不回归。

## 注意(别误报)
- `causal=0 + contextual-only`(无 target)pane-2 实测本就 PASS(max_uih_len=seqlen → contextual 不 clamp),属正常,不是 bug。
- `causal=0 + window>0` 走 WithLocal(IsMasking=true)本就 PASS。
- 真正靠这次修复才转 PASS 的核心是 `causal=0 + num_target`(及其组合)。其余交叉 case 若本就 PASS,加进去是为**回归锁定**,也有价值。

## 产出
- `fix-P1-1-done.md` 增一节"交叉矩阵升级":新增了哪些 case、每个的 PASS/FAIL、是否发现新缺陷、套件总案数与整体 exit。
- 测试套件里给这些交叉 case 标清 milestone tag(如 `M4b-cross`)便于追溯。
