# [BUG] tp=8 weight load crash: `_load_w2` / `_load_w13` compute negative `narrow()` size on last ranks (Step-3.5-Flash-FP8 / per_1x128 scale dim)

> **Status**: DRAFT — not yet filed upstream.
> **Author**: fp8-tp4-repro project, issue_wave (2026-04-29).
> **Scope**: bug report + RCA + advisory fix. **No source modification proposed in this draft.**

---

## Summary

When launching the OpenAI-compatible server with `-tp 8` on `stepfun-ai/Step-3.5-Flash-FP8`, ATOM **deterministically crashes during weight load** (before any inference / cudagraph / batching). The root cause is in `atom/model_ops/moe.py` — both `_load_w2` (line 2335-2364) and `_load_w13` (line 2292-2333) use a `ceil`-based shard split that does not guard against `start >= loaded_weight.shape[shard_dim]` on the trailing ranks, resulting in a negative `size` argument to `torch.Tensor.narrow()`.

The crash is **not** related to CUDAGraph capture sizes, batch size, sampling parameters, or `max_tokens`. It is purely a function of `(D, tp_size)` where `D = loaded_weight.shape[shard_dim]` for a given expert/scale tensor.

The same bug pattern exists in `_load_w13` and is reachable on the same model under `-tp 8`.

---

## Environment

| Item | Value |
|------|-------|
| ATOM commit | `acff926` |
| AITER commit | `0f8164017` |
| CK (Composable Kernel) commit | `defd7ad29` |
| Model | `stepfun-ai/Step-3.5-Flash-FP8` |
| Hardware | AMD Instinct MI308X UBB (8 GPU/node, gfx942, e4m3fnuz) |
| Launch | `python -m atom.entrypoints.openai_server --model stepfun-ai/Step-3.5-Flash-FP8 --kv_cache_dtype fp8 -tp 8` |
| Sampling (irrelevant — never reached) | `temperature=0`, `top_p=1`, `max_tokens=512`, `cudagraph_capture_sizes=[1,2,4]` |

`-tp 2` and `-tp 4` work end-to-end on this same commit set.

---

## Reproduction

```bash
cd /path/to/ATOM
# Same script that PASSes at tp=2 and tp=4
python -m atom.entrypoints.openai_server \
  --model stepfun-ai/Step-3.5-Flash-FP8 \
  --kv_cache_dtype fp8 \
  -tp 8
```

Expected: server reaches `Loaded model and starting engine ...`
Actual: `ModelRunner6/8` and `ModelRunner7/8` raise `RuntimeError` during `Loading safetensors shards: 1/44`. EngineCoreMgr then receives `SHUTDOWN` from all ranks and the process exits.

A fully scripted reproducer with 4 prompts is at `correctness_eval/correctness_bench.py` of the fp8-tp4-repro project; the prompt set / sampling parameters do not matter for this bug — the crash happens before any prompt is consumed.

---

## Traceback (verbatim, abridged for clarity)

### Symptom A — rank 6/7: negative `size` to `narrow()`

```
File "atom/models/step3p5.py", line 897, in load_fused_expert_weights
    weight_loader(param, loaded_weight[expert_id], name, shard_id, expert_id)
File "atom/model_ops/moe.py", line 2610, in weight_loader
    self._load_model_weight_or_group_weight_scale(...)
File "atom/model_ops/moe.py", line 2256, in _load_model_weight_or_group_weight_scale
    self._load_w2(...)
File "atom/model_ops/moe.py", line 2357, in _load_w2
    loaded_weight = loaded_weight.narrow(shard_dim, start, size)
RuntimeError: narrow(): length must be non-negative.
```

### Symptom B — rank 5: `narrow()` succeeds with `size=0` but downstream `copy_` shape mismatches

```
RuntimeError: The size of tensor a (2) must match the size of tensor b (0)
              at non-singleton dimension 1
```

(Triggered at the `expert_data.copy_(loaded_weight)` call inside the same `_load_w2` path.)

Both symptoms are the **same root cause** seen at different ranks: rank 5 lands exactly at `start == D` (so `size = 0`, `narrow` does not raise but the downstream `copy_` fails); ranks 6 and 7 land at `start > D` (so `size < 0` and `narrow` raises).

Full log: `correctness_eval/logs/tp8_full.log` of the fp8-tp4-repro project.

---

## Root cause

The relevant code (verbatim, line numbers from ATOM `acff926`):

```python
# atom/model_ops/moe.py:2335-2364   _load_w2  (down_proj, RowParallel on input_dim)
def _load_w2(self, expert_data, shard_dim, loaded_weight, tp_rank, load_full=False):
    shard_size = expert_data.shape[shard_dim]
    if not load_full:
        load_shard_size = (
            loaded_weight.shape[shard_dim] + self.tp_size - 1
        ) // self.tp_size                                      # ceil split
        start = load_shard_size * tp_rank
        size  = min(load_shard_size, loaded_weight.shape[shard_dim] - start)
        loaded_weight = loaded_weight.narrow(shard_dim, start, size)   # ← line 2357
        if load_shard_size != shard_size:
            expert_data = expert_data.narrow(shard_dim, 0, load_shard_size)
    ...
    expert_data.copy_(loaded_weight)                            # ← line 2364
```

```python
# atom/model_ops/moe.py:2310-2314   _load_w13  (gate_up_proj, MergedColumnParallel on output_dim)
load_shard_size = (
    loaded_weight.shape[shard_dim] + self.tp_size - 1
) // self.tp_size                                              # same ceil split
start = load_shard_size * tp_rank
size  = min(load_shard_size, loaded_weight.shape[shard_dim] - start)
loaded_weight = loaded_weight.narrow(shard_dim, start, size)   # ← line 2315 (same bug)
```

Let `D = loaded_weight.shape[shard_dim]` and `C = ceil(D / tp_size) = load_shard_size`.

For trailing ranks the assignment `start = C * tp_rank` can satisfy `start >= D`, in which case:
- if `start == D` → `size = 0` → `narrow` returns an empty slice but the subsequent `copy_` fails because `expert_data` along that dim is non-zero (shape mismatch);
- if `start > D` → `size = D - start < 0` → `narrow` raises `RuntimeError: narrow(): length must be non-negative.`

### Trigger condition

`start_{last} = C * (tp_size - 1) > D` whenever the ceil-rounding "absorbs" more than one rank of capacity, i.e. whenever

> `D < C * (tp_size - 1)`     (condition for `narrow` raise on `tp_rank = tp_size - 1`)

Equivalently: when `D / tp_size` is not close enough to an integer that the ceil bump still leaves work for the last rank.

For `_load_w13` the existing code comment (line 2306-2309) gives the example `inter=1280, tp=4 → 10 scale blocks / 4 = 2.5 → ceil=3`, which works fine at `tp=4`. Re-applied to `tp=8` the same shape gives:

> `D = 10`, `C = ceil(10/8) = 2`
> `rank 5: start=10 = D → size = 0`     ← Symptom B
> `rank 6: start=12 > D → size = -2`    ← Symptom A
> `rank 7: start=14 > D → size = -4`    ← Symptom A

This matches the observed symptom split (rank 5 `copy_` mismatch + rank 6/7 `narrow` raise) exactly, so `D = 10` (the per_1x128 scale block count for `inter_size = 1280`) is the most likely trigger on Step-3.5-Flash-FP8.

The bug is **deterministic** (no race / no GPU non-determinism) and is a pure function of `(D, tp_size)`.

---

## Affected configurations

The bug surfaces in two related symptoms whose boundary is whether `start == D` or `start > D` on the last rank:

- Symptom A (`narrow` raises): `start_{last} = C * (tp_size - 1) > D`     ⟺ `D < C * (tp_size - 1)`
- Symptom B (`copy_` shape mismatch): `start_{last} = D` exactly, so `size = 0` but `expert_data` along that dim is non-zero

Both reduce to the same root cause: the ceil-based split assigns a `start` that exits the loaded tensor for at least one trailing rank.

For `tp_size = 2` the trailing rank is rank 1 with `start = ceil(D/2) ≤ D` for every `D ≥ 1`, so `tp = 2` is provably safe. For `tp_size ≥ 4` the trigger is satisfied by many small `D` values — exactly the regime where fp8 per-block scale dimensions live (e.g. `inter_size / 128`, which produces single-digit `D` for typical MoE expert sizes).

**Concrete observed case**: Step-3.5-Flash-FP8 at `tp_size = 8` with `D = 10` (the per_1x128 scale block count for `inter_size = 1280`, matching the `_load_w13` comment example at line 2306-2309) — yields rank 5 in Symptom B and ranks 6/7 in Symptom A.

We do not enumerate the full `(D, tp_size)` trigger set in this draft; the closed-form conditions above are what should drive the fix and the unit-test parameterisation.

---

## Proposed fix (advisory only — not implemented in this draft)

Two natural options. Neither has been implemented and either should be discussed with ATOM maintainers before landing.

### Option A — early return when the rank holds no slice

```python
# Pseudocode, NOT a patch
start = load_shard_size * tp_rank
if start >= loaded_weight.shape[shard_dim]:
    # This rank does not own any slice of this expert's weight.
    return  # skip narrow + copy_ entirely
size = min(load_shard_size, loaded_weight.shape[shard_dim] - start)
```

Pros: minimal change; matches the physical interpretation ("the trailing rank has nothing to load"); preserves the existing `ceil` logic that the comment at line 2306-2309 explicitly justifies for partial-scale-block correctness.
Cons: needs careful audit that downstream consumers tolerate a rank that "skipped" loading the slice (e.g. expert_data initial value, all-reduce semantics, mxf4 dtype paths).

### Option B — even split with remainder absorbed by rank 0

```python
# Pseudocode, NOT a patch
base, rem = divmod(loaded_weight.shape[shard_dim], self.tp_size)
load_shard_size = base + (1 if tp_rank < rem else 0)
start = base * tp_rank + min(tp_rank, rem)
```

Pros: every rank always gets `start + size <= D`, no negative-size case is reachable.
Cons: changes the split distribution; the comment at line 2306-2309 says `ceil` is required so that the last partial scale block is included — Option B places that partial block on rank 0, not on the trailing rank, which may or may not be what the kernel expects (especially the AITER fused MoE kernel's per-block dequant path). This also reverses the per-rank residual size ordering (under `ceil` the trailing rank holds the partial block; under Option B rank 0 does), which any downstream consumer that relies on a specific ordering for per-block dequant indexing would need to be audited for. This needs upstream review.

### Sweep target

The same fix needs to be applied to **both** `_load_w2` (line 2355-2357) **and** `_load_w13` (line 2313-2315) — they use identical formulas. Per ATOM project rule "fix-then-sweep", a single PR should patch both call sites and add a unit test parameterised over `(D, tp_size)` that exercises `D ∈ {tp_size-1, tp_size, tp_size+1, tp_size+rem}` style boundaries.

---

## Why this matters / scope

- **100 % blocking** for any user trying to serve a model at `tp_size ≥ 4` whose per-block scale dimension hits the trigger condition. Step-3.5-Flash-FP8 + `tp=8` is the concrete observed case; Step-3.5-Flash-FP8 + `tp=4` happens to be safe only because `D=10 ≥ ceil(10/4)*3 = 9`.
- This bug is in the **weight-load path only**. Inference / CUDAGraph / batching / sampling code is **not implicated**. There is no need to investigate the runtime engine.
- Cross-references the upstream symptom of the fp8-tp4-repro project's main RCA (Step-3.5 MoE w2/w13 sharding misalignment at tp boundaries — three separate root causes already landed in `aiter/fused_moe.py:881-886`, `atom/model_ops/moe.py:1709-1746` padding, and `atom/model_ops/utils.py:79` weight_scale handling). This `_load_w2` / `_load_w13` boundary case is the fourth and most upstream of the family.

---

## References

| Source | Path |
|--------|------|
| Code, `_load_w2` | `atom/model_ops/moe.py:2335-2364` (ATOM `acff926`) |
| Code, `_load_w13` (same bug pattern) | `atom/model_ops/moe.py:2292-2333` |
| Full crash log (rank-by-rank) | `correctness_eval/logs/tp8_full.log` (fp8-tp4-repro) |
| Correctness wave RCA | `correctness_eval/CORRECTNESS_REPORT.md` §4 (fp8-tp4-repro) |
| Per-rank reproduction notes | `correctness_eval/progress/corr-t1.md` §3 |
| Project handoff packet | `handoff_wave/HANDOFF_PACKET.md` §4.1 F-OPEN-1 |
| Three-repo commit pin | ATOM `acff926` / AITER `0f8164017` / CK `defd7ad29` |

---

## Caveats / what this draft does NOT claim

1. The exact value `D = 10` is **inferred** from matching the observed rank-5/6/7 symptom split against the ceil-split arithmetic. It is consistent with `inter_size = 1280` and per_1x128 scale blocks, but the precise tensor name and dtype are not directly extracted from the dump in this draft. Furthermore, `inter_size = 1280` for Step-3.5-Flash-FP8 is itself inferred by matching the `_load_w13` comment example at line 2306-2309 against the observed crash pattern; we have not extracted it from the model config in this draft. Upstream maintainers can confirm both by adding `print(name, loaded_weight.shape)` immediately before `moe.py:2357` on a `-tp 8` run.
2. We have **not** verified whether `_load_w13` actually crashes on Step-3.5-Flash-FP8 at `tp=8` — the `_load_w2` crash kills the loader before `_load_w13` is reached on the affected expert. The static argument that `_load_w13` shares the bug is based on identical code structure, not an independent run.
3. We have **not** implemented or tested either Option A or Option B. Both are sketches for upstream discussion.
4. There is a **separately reported** observation in our internal benchmarking that `tp=8` PASSed under a different configuration (`cudagraph_capture_sizes=[1]` and a single short prompt). We currently **cannot** reconcile that PASS with the deterministic crash described here; one explanation is that the benchmark run used a different launch path, model variant, or expert-loading branch. We will resolve this independently in our project tracker; it is not a blocker for triaging the upstream bug, since the crash described above is fully deterministic against the reproduction command in this issue.

---

**End of draft.** Ready for issue_wave reviewer pass.
