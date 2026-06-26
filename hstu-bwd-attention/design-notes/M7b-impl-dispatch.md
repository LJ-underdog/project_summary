# M7b 实现单 —— draft 已批准(coder pane 0.1)

你的 `draft-M7b.md` 经 lead 闸门审查 **批准**。设计正确(尤其抓到 ①shape 必须随 MaxK 选否则 silent-wrong、②harness kN0_bwd 对 hd256 失配、③非典范 hdim guard、④selector<64> 同型即同码保零回归)。按 draft §7 分阶段实现,叠加下面 lead 附加要求。全程对拍铁律 `-attn_scale=1.0`,不动 reference/promoted pipeline/kernel 逻辑。

## ⛔ 硬检查点(stage 1 后必停)
**stage 1 = shape selector + dispatch 重构(取 `HstuBwdShape<MaxK>::Type`)+ 解 3 处 throw 换典范值 guard + harness kN0_bwd 随 hdim,但【还不加新 hdim 到 generate_instances】。**
完成 stage 1 后**停下报 lead**,证明"重构本身零回归":
1. selector<64> 产出类型与现硬编码**逐字等价**(贴出 selector<64> 与原硬编码 FmhaBlockTile/WarpTile/BlockWarps 对照)。
2. 重生成的 hd64 instance 与基线 `bf82a1d2` 对应文件**内容不变**(`git diff bf82a1d2 -- instances/*backward*hdim... ` 或逐文件 diff,除非文件名/路径)。
3. 现 **106 套件全绿 + determ byte-identical repro 全绿 + 误差与基线同量级**。
**lead 亲验通过后才放行 stage 2(加新 hdim)。** 这一步保护 106/106 基线不被重构破坏,别跳。

## stage 2+(检查点放行后)
按 draft §7:hd128 symmetric(batched SiLU→全 no_group→group,每步对拍)→ hd96+hd256 → fp16×全 hdim → 收尾。

## lead 附加要求
1. **🟥 hd256 寄存器/occupancy 检查(必做)**:hd256(bm0=16/bn0=64/大 bk)在 gfx950 上按 M1 口径查 VGPR/AGPR/Scratch/occupancy(rocprof 或编译期 resource),确认无 spill 灾难;若 spill 严重影响正确性/可行性,**如实报 lead**(可能 hd256 需单独处置或降级)。证据落 `profile/M7b-hd256-resource.md`。
2. **🟥 每 hdim 必覆盖 P1-1 cross(causal=0+num_target)**——别重蹈 M6b 覆盖洞。
3. **容差不预松**:沿用 bf16(2e-2/5e-2)、fp16(5e-3/1e-2);hd256 若某案 FAIL,先判数值真错 vs 容差,如实报,**禁为凑 PASS 调容差**(M7a 纪律)。
4. **编译时间**:+48 bwd TU,若 group entry TU 成最慢瓶颈,**授权**你把 group 拆 per-hdim instance 文件(draft §3.4 选项),在 done.md 记明。
5. **诚实纪律**:带任何 FAIL 不标 promoted;日志数字与结论一致(M6b 首轮过度声称教训)。

## 产出
- stage 1 检查点:pane 里一句话报"stage1 零回归证毕,三项证据见 X",**停**。
- 全程:`runs/build-M7b*.log`、`runs/run-M7b-sweep.log`(全 hdim 全表)、`profile/M7b-hd256-resource.md`、新套件 `runs/test-*.log`、`docs/M7b-done.md`、`candidates.jsonl` 加行(status 据实,全绿=in-progress 待闭合)。
- **不 commit**(lead 闭合后统一 commit 立里程碑)。

reviewer(pane 0.2)会在 stage 1 检查点后或有完整码后并行对抗 review,你专注实现 + 自验对拍即可。
