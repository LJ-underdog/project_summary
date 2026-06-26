# M7b 对抗式 review —— reviewer(pane 0.2,已 /clear)

你是 HSTU bwd 项目独立 reviewer(较真、对抗式,**默认怀疑、不信自述**)。coder 刚完成 **M7b:symmetric hdim∈{64,96,128,256}(hdim_qk==hdim_v)**,停在闭合检查点。你做独立对抗 review。**只读不改库代码**,把 GREEN/RED + 证据列给 lead。

## 0. 背景(只读)
- 基线 HEAD=`bf82a1d2`(M7a)。M7b 未 commit。素材:
  - `/tmp/hstu-bwd-design/M7b-done.md`(coder 自述)、`M7b-stage1-checkpoint.md`、`M7b-draft.md`(已批准设计)。
  - 改面:新 `hstu_attention_bwd_shape.hpp`;8 改源 + 50 instance(`git -C /root/workspace/ck_hstu status` + `git diff bf82a1d2 -- example/ck_tile/18_hstu_attention/`)。
- lead 已亲核:171/171 套件、hd64 byte-identical、guard throw、hd256 一例 PASS。**你要独立复核、找洞,别复述 lead 结论。**
- 铁律:对拍 CPU reference,`-attn_scale=1.0`;bf16 2e-2/5e-2、fp16 5e-3/1e-2。

## 1. 对抗审查清单(逐条 GREEN/RED + 文件:行号)
1. **★ shape 真随 hdim 变(M7b 核心,silent-wrong 高发)**:draft 抓到"pre-M7b 把 hd64 tile 写死、MaxK 没用来选 shape"。验:`HstuBwdShape<MaxK>` 四个特化的 tile 值与 FMHA bwd codegen 蓝本(`example/ck_tile/01_fmha/codegen/ops/fmha_bwd.py` gfx9 fp16/bf16 非 trload)逐一对得上?96 的 warps2、128/256 的 bm0=16、256 的 bn0=64 是否正确?**反证:若 4 个 hdim 仍共用 hd64 tile,误差/行为应异常**——抽 hd128/hd256 实测误差量级是否"hdim 越大累加越多 err 略增"的合理曲线(共用错 tile 会 silent-wrong 但可能仍"PASS",故要看 tile 类型本身)。
2. **selector<64> 同型 = hd64 零回归**:`HstuBwdShape<64>::Type` 与 pre-M7b 硬编码逐字等价?hd64 maxk_64 instance 与 `bf82a1d2` byte-identical?(`git diff bf82a1d2 -- instances/*maxk_64*`)。
3. **guard 挡 silent-wrong**:非典范 hdim(80/100)+ asymmetric(64/128)是否真 throw 而非静默选大 tile 算错?亲测 what()。**反证**:临时把 guard 注释掉跑 hdim=100,看是否 silent-wrong(产出非 throw 的错值)→ 证 guard 真在防洞;验完恢复。
4. **harness kN0_bwd 修复(hd256 determ)**:hd256 bn0=64,harness `kN0_bwd=(hdim==256)?64:128`。验 hd256 determ 的 num_splits/workspace 不越界 + **hd256 determ 两次 byte-identical**(亲跑 cmp -s）。若 kN0 仍写死 128,hd256 determ 会越界/错。
4. **对拍公平 + 容差不放水**:reference 同走该 hdim;容差沿用未松。**revert 实验**:抽 hd256 一案,把容差收紧 5×看是否仍过(过=误差真小);别只看 PASS。
5. **库零回归 byte-level**:promoted pipeline/kernel/dispatch 逻辑/reference 是否 byte-identical 于 `bf82a1d2`?(`git diff bf82a1d2 -- <pipeline/kernel/reference 文件>` 应空,除 dispatch 头取 selector 的改动)。确认没动 promoted 逻辑。
6. **套件 171/171 独立复跑**:你自己的 `build_review`(干净 configure+build)+ `run_bwd_tests.py --bin <build_review>/...` → 171/171 exit 0?新增 2 guard reject + 60 pass + 4 repro 覆盖是否真(尤其**每 hdim 有 P1-1 cross**,别重蹈覆盖洞)。
7. **hd256 资源声称**:`profile/M7b-hd256-resource.md` 的 Scratch=0/VGPR 数据可信?(可复跑 rocprofv3 抽验,或至少逻辑自洽)。occupancy=1 归 M8 的判定对不对。

## 2. 产出
写 `/tmp/hstu-bwd-design/M7b-review-findings.md`,逐条 GREEN/RED + 证据行号;RED 给可复现配置。结论:M7b 可否 promote。**发现 RED 必给复现。** 完成 pane 里一句话报。
