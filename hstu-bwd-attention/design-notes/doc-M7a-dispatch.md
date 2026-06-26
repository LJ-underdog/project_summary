# 派单:写 M7a 讲义(fp16 dtype 加宽)

**先读硬规格:`/tmp/hstu-bwd-design/doc-series-spec.md`(全文)。本卡给 M7a 专属输入。**

- **里程碑**:M7a = fp16 dtype 加宽(纯 dtype 轴,fp16 复用 bf16 同模板码路,仍 hd64、hdim_qk==hdim_v)。
- **commit(行号锚定)**:`8b1fab06`。`cd /root/workspace/ck_hstu && git show --stat 8b1fab06`。
- **旧 HTML(参考叙事,行号/数值必重核)**:`/root/workspace/hstu-b1052-report/hstu-bwd-M7a-fp16-20260611.html`。
- **事实来源**:`/tmp/hstu-bwd-design/M7a-done.md`(+ review 若在)+ `candidates.jsonl` 里 `"id":"M7a-fp16"` 那条(106/106 exit0、fp16 sweep 66/66、9 库文件 byte-identical)。
- **输出**:`/root/workspace/hstu-b1052-report/hstu-bwd-M7a-fp16-20260625.html`
- **M7a 讲解重点**:
  1. **复用不重写**:dispatch/kernel/pipeline 本就模板化于 `InOutDataType`,fp16 仅在边界加 dtype。
  2. 改面:harness 运行时 `-prec` 选 fp16_t/bf16_t(no_group+group fwd/bwd wiring);`get_bwd_elimit<fp16_t>` rtol5e-3/atol1e-2(**比 bf16 的 2e-2/5e-2 更紧**,因 fp16 尾数 10bit);新 no_group/group fp16 bwd entry(bf16 逐字镜像);generate_instances dtype 轴 → 8 batched fp16 instance + ref.hpp;api.hpp 2 fp16 extern;CMake glob fp16 fwd/bwd。
  3. **零回归实证**:库/kernel/pipeline/dispatch/reference byte-identical 于 M6b(`git diff` 仅 4 文件 + fp16 新文件)。
  4. fp16 不溢出:max|ref|~10.9 << 65504,内部 accum 是 float,所有 softmax PASS。
- **诚实点(适合 note-block)**:容差比 bf16 紧是关键证据——reviewer 做了 tolerance-revert 双实验(放松到 bf16 仍 PASS=无隐藏误差;收紧 5-20x 仍 PASS=误差确实小)+ fp16-ULP 量级反证确跑 fp16 非静默 bf16 fallback。
- 写完按规格 §6 回报。
