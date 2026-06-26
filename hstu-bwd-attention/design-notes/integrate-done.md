# integrate DONE — DESIGN.md

整合三件套(part-algo / part-mask-modes / part-engineering)为单一可 review 方案 `/tmp/hstu-bwd-design/DESIGN.md`(21.5K 字符 / 310 行)。

## 结构(§0–§8)
§0 摘要+目标芯片(gfx942/950)+6 差异 · §1 3-kernel/KV外Q内/grid/数据流 · §2 七阶段双路+骨架伪代码 · §3 mask/三模式/索引/scale · §4 文件表/problem/policy/dispatch/dQ两路/canonical 字段表/scale 接线表/CMake · §5 对拍+容差+矩阵 · §6 里程碑 M0–M8(M1 闸门)· §7 复用vs新写总清单 · §8 风险与未决。

## 裁决的跨组点(全部给单一结论,§8.1 已决 7 条)
1. **mask Y 缺口 → 方案 A**(新增 GetTileRangeAlongY/IsEdgeTile,纯加不改 fwd),与 STAGE2/5 置零统一成一套谓词接口(真值源 IsTokenPairInsideMask)。
2. **is_tile_in_first_split 已查清并关闭 pane-2 §5.5**:源码证实 fwd kernel 按 Q-tile 重算该 flag(fwd_kernel.hpp:691-716),元素级谓词不依赖它;bwd 元素置零不受影响,GetTileRangeAlongY 首版保守超集。
3. **留 g(非整张 S)**,M1 实测 VGPR。
4. **masked-out 置零**:SiLU 必清 p,g(dsilu(0)=0.5),禁 -inf;STAGE2 用 set_tile_if。
5. **scale 接线表**固化:alpha 两处+dV 不吃+三模式单标量;scale_p 折 p/g+softmax 不用+group per-group;softmax exp(S−LSE) 自然对数域不重复乘 scale。
6. **group 取数**:alpha 全局标量、scale_p/mask 超参 per-group 指针、num_target per-batch。
7. **命名统一**:canonical params/kargs 字段表(§4.6),消解 lsed/d、kIsDeterministic 出入。

## 冲突标注(§8.4,4 条,均非事实级,已消解)
C1 置零位置 STAGE2 vs STAGE5、C2 「set S」vs「清输出」、C3 字段命名 → 全部裁决统一;架构/scale/复用边界三份一致,无事实冲突。

## 真未决(§8.3,需用户拍板)4 条
U1 bias/dbias 是否需要(默认不做) · U2 deterministic 是否首版默认(默认 atomic) · U3 MVP 覆盖面(dtype/hdim/模式) · U4 GQA/MQA(默认不做)。
建议(需 M1/实测,非用户阻塞)5 条:R1 policy 复用可行性、R2 g 的 VGPR、R3 GetTileRangeAlongY 边界、R4 dq_acc 显存、R5 编译规模。
