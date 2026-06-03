# V06 — FP8 tp=4 Verification Plan

**Scope**: ATOM commit `ccb64621` — three independent bugs fixed simultaneously to enable Step-3.5-Flash-FP8 with tp=4. Source files:
- `/home/hanchang/ATOM/atom/model_ops/moe.py` (`_load_w13` L2287, `_load_w2` L2330)
- `/home/hanchang/project_fp8_tp4/06_fp8_tp4.md` (root-cause writeup)

**Bugs in scope**:
| # | Bug | Fix |
|---|-----|-----|
| 1 | `create_weights` ValueError check rejects padded inter | Padding-aware check |
| 2 | `_process_block_quant` does not pad inter dim | Zero-pad w13/w2 to `inter_pad=384` |
| 3 (root) | `_load_w13` / `_load_w2` scale TP shard uses `floor(N/tp)` → last partial scale block left at `torch.ones() = 1.0` → ~5000× dequant blowup → gibberish | Use ceil + `narrow` boundary protection |

---

## A. Code Review

### A.1 Fix 3: ceil arithmetic correctness

Code at `moe.py:2305-2310` (and mirror at 2347-2352):
```python
load_shard_size = (loaded_weight.shape[shard_dim] + self.tp_size - 1) // self.tp_size
start = load_shard_size * tp_rank
size = min(load_shard_size, loaded_weight.shape[shard_dim] - start)
loaded_weight = loaded_weight.narrow(shard_dim, start, size)
```

**Concrete walk-through (10 scale blocks, tp=4)**:
| rank | start | naive size | clipped size | covers blocks |
|------|-------|-----------|--------------|---------------|
| 0 | 0 | 3 | 3 | [0,1,2] |
| 1 | 3 | 3 | 3 | [3,4,5] |
| 2 | 6 | 3 | 3 | [6,7,8] |
| 3 | 9 | 3 | min(3, 10-9)=1 | [9] |

Total covered = 3+3+3+1 = 10 ✓ (no missing block, no double-coverage).

**Verification questions**:
- Is `(N + tp - 1) // tp` (ceil) the correct algebra for ensuring full coverage when `N % tp != 0`? → Standard formula, mirrors numerous sites in vLLM/Megatron.
- Is the `min(...)` clamp on `size` essential? → Yes. Without it, rank 3 would request `loaded_weight.narrow(dim, 9, 3)` on a length-10 axis → out-of-range. The `expert_data.narrow(dim, 0, load_shard_size)` at L2324 still uses unclamped `load_shard_size` to keep the destination view aligned to the padded buffer; only the source slice is clamped.
- Is `expert_data.copy_(loaded_weight)` shape-compatible on rank 3? On the last rank `loaded_weight.shape[dim] = 1` while `expert_data.shape[dim] = 3` (after narrow to `load_shard_size`). **This is a potential shape-mismatch bug to verify** — the broadcast would fail unless the destination is also narrowed to `size`. Need to inspect whether PyTorch silently broadcasts here or whether the narrow path is hit only when `load_shard_size == expert_shard_size` is false. → **Open question for Experiment 1b below.**

### A.2 narrow boundary protection

L2309 / L2351: `size = min(load_shard_size, loaded_weight.shape[shard_dim] - start)`.

- For ranks 0..tp-2: `size == load_shard_size` (no clipping).
- For rank tp-1 with `N % tp != 0`: `size < load_shard_size` (partial tail block).
- For all-other tp where `N % tp == 0`: `size == load_shard_size` for every rank (degenerate ceil = floor).

Edge case to confirm: when `load_shard_size * tp_rank >= N` (rank holds zero data, e.g. `N=3, tp=4, rank=3`: `start=3, size=min(1, 0)=0`). PyTorch `narrow(dim, 3, 0)` returns an empty tensor; subsequent `copy_` on a non-empty `expert_data` slice would broadcast or raise. **Should be exercised by experiment 1c (oversharded extreme case).**

### A.3 Regression safety across TP sizes

| inter_dim / block | TP=1 | TP=2 | TP=4 | TP=8 |
|-------------------|------|------|------|------|
| 1280 / per_1x128 → N=10 | ceil=10, floor=10 ✓ | ceil=5, floor=5 ✓ | ceil=3, floor=2 ⚠ (this fix) | ceil=2, floor=1 ⚠ |

Implications:
- tp=2 path: `ceil == floor`, behavior unchanged → safe regression target.
- tp=8 path: 4 blocks / 8 ranks → many ranks hold zero blocks. Confirms the empty-narrow concern in A.2. Even if tp=8 is currently blocked by GPU5, the loader must not crash at construction time. **Add static unit-test-style check or experiment.**
- Other inter_dims (e.g. 8192 / 128 = 64 blocks): ceil = floor for tp ∈ {1,2,4,8} → fully safe.

Sweep recommendation: enumerate `(inter_dim, tp_size)` pairs from the model config family (Step-3.5-Flash variants, plus any other FP8 model that uses `_load_w13`) and assert `ceil(N/tp) >= floor(N/tp)` cover-completeness.

### A.4 Fix 2 vs V04 Fix 1 — timing of padding

| | BF16 (V04 Fix 1) | FP8 (V06 Fix 2) |
|---|------------------|-----------------|
| Where padded | `process_weights_after_loading` (post-load) | `_process_block_quant` (during weight transform) |
| Why difference | BF16 weights live as `nn.Parameter` and can be reallocated after loading completes | FP8 block-quantized weights have a paired scale tensor that must match the padded shape; padding has to happen inside the quant transform so scales are produced for the padded buffer |
| Risk | Padding before the loader can write means the loader needs the padded-aware narrow logic above | Padding after loading would be too late — scales would be sized to the unpadded inter, mismatching the kernel's expected shape |

The two timings are not interchangeable; each is correct for its dtype path. Cross-reference test: dump `w13.shape`, `sc13.shape`, `inter_pad` after `process_weights_after_loading` for both BF16 and FP8 and confirm both match `inter_pad=384` on tp=4.

### A.5 Scale debug-print method (non-invasive)

Constraints (from MEMORY note + ATOM CLAUDE.md):
- Must not edit `@support_torch_compile`-decorated model files.
- Multiprocessing uses `spawn`, so monkey-patches in the parent process do not propagate to workers.

Recommended approaches in order of invasiveness:
1. **Env-gated print inside `_process_block_quant`** (temporary patch to `moe.py`, removed before commit). Uses `os.environ.get("ATOM_DEBUG_FP8_SCALE")` so it's a no-op in production.
2. **Workers-side hook via `ATOM_PRE_FORWARD_HOOK` / model-loader callback** (if such an env hook already exists — search `atom/utils/envs.py`).
3. **Post-hoc inspection of the saved checkpoint cache** at `/root/.cache/atom/` if scales are dumped there.
4. **Last resort**: add a one-line `torch.save(sc13[:1, :6, :1].clone(), "/tmp/sc13_dump.pt")` inside the loader, gated by env. Read back from parent process.

Method 1 is preferred for V06 — it directly observes the `1.0` artifact at the source.

---

## B. Experiment Design

### Exp 1 — Scale loading verification (most diagnostic)

**Goal**: prove that pre-fix produces `1.0` residue in the third scale block, and post-fix does not.

**Setup**:
1. Patch `_process_block_quant` (or `_load_w13` post-copy) to dump:
   ```python
   if os.environ.get("ATOM_DEBUG_FP8_SCALE") and expert_id == 0:
       torch.save(sc13[0, 0:6, 0].cpu(), f"/tmp/sc13_rank{tp_rank}.pt")
   ```
2. Run the inference command in Exp 2 with `ATOM_DEBUG_FP8_SCALE=1`.
3. Load `/tmp/sc13_rank{0..3}.pt` and inspect.

**Pass criteria**:
- All scale entries in [1e-5, 1e-1] range (typical FP8 block-scale magnitude).
- No entry equal to `1.0` (the `torch.ones()` initial value).
- Pre-fix run reproduces the documented `[0.000208, 0.000203, 1.0, ...]` pattern at rank 0 or 3.

**Variant 1b** — cover-completeness assertion (offline, no GPU): import the loader, monkey-patch `narrow` to record `(start, size)` per rank, assert `Σ size == loaded_weight.shape[shard_dim]` and ranges are non-overlapping.

**Variant 1c** — extreme oversharding: synthesize a `loaded_weight` with `shape[shard_dim] = 4` and `tp_size = 8`. Verify ranks 4..7 receive empty slices without crash.

### Exp 2 — FP8 tp=4 end-to-end (core)

```bash
cd /tmp && rm -rf /root/.cache/atom/* && CUDA_VISIBLE_DEVICES=0,1,2,3 \
  AITER_LOG_LEVEL=WARNING \
  /opt/venv/bin/python -m atom.examples.simple_inference \
  --model stepfun-ai/Step-3.5-Flash-FP8 --trust-remote-code \
  --tensor-parallel-size 4 --level 0 --temperature 0 --max-tokens 128 \
  --max-num-batched-tokens 4096 --max-num-seqs 256 \
  --gpu-memory-utilization 0.7
```

Avoid GPU 5 (per MEMORY: ~700ms/tensor hardware fault).

**Pass criteria**:
- Output is coherent (no gibberish, no token-repetition collapse).
- `TTFT < 200 ms` (target: ~93 ms per V06 summary).
- `TPOT < 20 ms`  (target: ~12.75 ms per V06 summary).
- No `ValueError` from `create_weights` (Fix 1 still effective).
- No shape mismatch in `_process_block_quant` (Fix 2 still effective).

### Exp 3 — Performance comparison FP8 tp=4 vs BF16 tp=4

Run Exp 2 twice with identical prompts and `max_tokens`:
- (a) `stepfun-ai/Step-3.5-Flash-FP8` tp=4 → record TPOT_fp8.
- (b) `stepfun-ai/Step-3.5-Flash` (BF16) tp=4 → record TPOT_bf16.

**Pass criteria**:
- `1 - TPOT_fp8 / TPOT_bf16 ∈ [0.15, 0.25]` (target: ~0.19 = 19% speedup).
- TTFT difference (fp8 ≈ 93 ms vs bf16 ≈ 88 ms): **expected** — fp8 has additional dequant overhead in compute-bound prefill where memory pressure is lower; the win shows up in decode (memory-bound). Document, do not gate on it.

### Exp 4 — FP8 tp=2 regression (Fix 3 should be a no-op)

Same command as Exp 2 with `--tensor-parallel-size 2` and `CUDA_VISIBLE_DEVICES=0,1`.

**Pass criteria**:
- TTFT ≈ 85 ms, TPOT ≈ 13.5 ms (matches MEMORY's recorded baseline).
- No new errors vs the pre-V06 tp=2 run (regression-free).
- Optional: re-run Exp 1 dump on tp=2 — every shard size should equal `loaded_weight.shape[dim] // 2` exactly (ceil = floor).

### Exp 5 — Gibberish reproduction (necessity of Fix 3)

Revert just the ceil→floor portion of Fix 3 (keep Fixes 1 and 2 in place):
```python
load_shard_size = loaded_weight.shape[shard_dim] // self.tp_size  # floor
```
Drop the `min(...)` clamp at the same time (since with floor it is unnecessary and was added together with ceil).

Re-run Exp 2.

**Pass criteria** (negative-control: fix is *necessary* iff the failure reappears):
- Output is gibberish (random tokens, repetition, or BOS spam).
- Exp 1 dump shows `1.0` re-appearing at the third scale-block index for at least one rank.
- Restore Fix 3 and confirm Exp 2 passes again.

If the failure does *not* reproduce, Fix 3 may be over-claimed and the root cause needs re-investigation.

---

## C. Key Open Questions

### C.1 Long-sequence (≥10k tokens) interaction
Per MEMORY, BF16 tp=4 has an open bug: "tp=4 long sequences (≥10k) outputs all BOS" (tracked as task 07). The V06 summary records this as an open bug for FP8 tp=4 as well.

**Decision for V06 verification**: cap the prompt at < 4 k tokens for Exps 2/3/4/5. Do NOT use V06 to validate long-sequence behavior — that is V07's scope. If a regression test must cover long sequences, run it but mark expected-failure and link to V07.

### C.2 Spawn-subprocess monkey-patch limitation
The parent-process monkey-patch will not be inherited by `spawn`-launched workers. The Exp 1 dump must therefore be written either (a) directly into the source file under env-gate, or (b) into a model-load callback that the workers themselves import. Option (a) is preferred for one-shot verification; remove the patch before merging.

Alternative: use `ATOM_*` env passed to workers + a file-based dump (`torch.save` to `/tmp/sc13_rank{rank}.pt`) so the parent can collate after the workers exit.

### C.3 TTFT 93 ms (FP8) vs 88 ms (BF16) on tp=4
Small (~5 ms, ~6%) regression in prefill. Likely sources, ranked by probability:
1. Per-block dequant overhead in the FP8 GEMM path (compute-bound during prefill, where memory-bandwidth savings of FP8 do not apply).
2. Extra padding to `inter_pad=384` introduces a few wasted flops in the padded tail.
3. Scale-tensor extra reads on the prologue.

Recommended: do not block V06 on this. Capture as a follow-up perf note and revisit when prefill becomes the critical path. Document in V06 sign-off.

---

## D. Sign-off Checklist

- [ ] A.1 ceil arithmetic walk-through validated against actual `loaded_weight.shape[shard_dim]` for the model.
- [ ] A.2 empty-shard edge case observed not to crash.
- [ ] A.3 TP-sweep table extended with all `(inter_dim, tp_size)` combos used by Step-3.5-Flash-FP8.
- [ ] A.4 padding shapes confirmed equal across BF16 and FP8 paths post-load.
- [ ] A.5 debug-print method chosen and documented.
- [ ] Exp 1 pre-fix shows `1.0` residue; post-fix clean.
- [ ] Exp 2 passes accuracy + perf gates.
- [ ] Exp 3 within [15%, 25%] decode speedup.
- [ ] Exp 4 tp=2 regression-free.
- [ ] Exp 5 negative control reproduces gibberish.
- [ ] C.1/C.2/C.3 documented as known limitations / follow-ups.
