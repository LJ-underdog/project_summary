# [Bug] HSTU group forward: `group_max_seqlens_q/kv` under-estimated → wrong O & LSE (NaN) when a group's longest batch is not its first batch

**Repo:** ROCm/composable_kernel
**Area:** `example/ck_tile/18_hstu_attention` (HSTU attention, group mode forward)
**Branch observed:** `hstu_attention_fwd` (tip `cde6f606` at time of report)

---

## Summary

In the HSTU group-mode example, the host-side `group_max_seqlens_q` (and `group_max_seqlens_kv`) are computed by indexing the **per-batch** `num_targets[]` array with the **group** index. When a group contains batches of differing length and the group's longest *packed* batch (uih + target) is **not** the batch at the group index, `max_seqlen_q` is under-estimated. Because `max_seqlen_q` drives the kernel launch config (`GridSize`, m-tile selection, split-kv decision), the GPU forward then produces **globally wrong `O` and `LSE`, including NaN/inf** — not just at the uncovered tail.

This triggers under the **default** parameter derivation (i.e. when `-g_max_seqlens` is *not* supplied), which is the normal usage path.

## Impact

- GPU forward `O`: ~87.5% of elements wrong (incl. NaN/inf) in the repro below.
- Forward `LSE` (training mode): ~84.6% wrong.
- Affects group mode with >1 batch per group and intra-group length heterogeneity. Both softmax and SiLU activation paths are affected via the launch config; the softmax LSE/O corruption is the most visible.

## Root cause

`example/ck_tile/18_hstu_attention/example_hstu_attention_fwd.cpp` (around L850–854):

```cpp
group_max_seqlens_q[i_grp] =
    group_max_uih_seqlens_q[i_grp] + group_contextual_seqlens[i_grp] + num_targets[i_grp];
group_max_seqlens_kv[i_grp] =
    group_max_uih_seqlens_kv[i_grp] + group_contextual_seqlens[i_grp] + num_targets[i_grp];
```

- `group_max_uih_seqlens_q[i_grp]` is the per-group max of the **uih** lengths — correct.
- But `num_targets[i_grp]` indexes the **per-batch** `num_targets[]` array (supplemented to `num_batch` length, see L725/L776) using the **group** index `i_grp`. It therefore picks the target count of *the i_grp-th batch*, not the target of the group's longest packed batch.

When the longest `(uih + target)` batch in a group differs from batch `i_grp`, `group_max_seqlens_q` < that batch's true packed `seqlen`. Since `hstu_attention_group_forward_dispatch.hpp` uses `param.max_seqlen_q` for `GridSize` (L166), `get_hstu_attention_fwd_mtile` (L186), and `shall_use_splitkv` (L205), an under-estimate corrupts the entire launch, not only the uncovered rows.

## Minimal reproduction

```
build/bin/tile_example_hstu_attention \
  -prec=bf16 -hdim_qk=64 -hdim_v=64 -nhead=4 -b=4 -g=2 \
  -softmax=1 -training=1 -causal=1 -v=1 \
  -g_context_lens=0,0 -g_local_lens=0,0 -g_minfull_lens=0,0 -g_attn_scales=1.0,1.0 \
  -seqlens=100 -targets=0,0,0,200
```

Here group 1 = {batch2, batch3}. `num_targets[1] = 0` (group-indexed) → `group_max_seqlens_q = 100`, but batch3's true packed length is `uih 100 + target 200 = 300`. Under-estimate = 200.

Result: `O` max_err = inf (NaN present), 87.5% wrong; `LSE` max_err = 5.06, 84.6% wrong.

## Evidence that this is a genuine GPU error (not a reference artifact)

`reference_hstu_attention_fwd.hpp` (group, softmax path) computes each batch's `O`/`LSE` from the **true** per-batch offset length (`seqlen_q = seq_q_offsets[i_batch+1] - seq_q_offsets[i_batch]`, L433) and does **not** depend on `max_seqlen_q` for the softmax output (it only uses it for the SiLU `scale_p = 1/max_seqlen_q` and mask tensor dims). Verified empirically: the reference `output_host.dat`/`lse_host.dat` are **byte-identical** between the trigger run and a coverage-forced control run → the reference is independent of `max_seqlen_q`, so the corrpair FAIL reflects a real GPU error, not a "both-wrong-equal" artifact.

| Experiment | O | LSE |
|---|---|---|
| trigger (no override) | max err inf, 87.5% wrong (NaN) | max err 5.06, 84.6% wrong |
| control: `-g_max_seqlens=300,300` (force coverage) | bf16 ULP (PASS) | 0 errs |
| fix the host formula + rebuild (no override) | byte-identical to control (PASS) | 0 errs |

The only changed variable between trigger and control/fix is `max_seqlen_q` (100 → 300), which isolates the cause.

## Suggested fix

Compute the per-group max over the group's batches, consistent with the packed-offset definition `batch_seqlen = seq_lengths_q[b] + num_targets[b] (+ contextual)`:

```cpp
int gmax_q = 0, gmax_kv = 0;
for (int b : batches_of_group(i_grp)) {
    int tgt = num_targets.empty() ? 0 : num_targets[b];
    gmax_q  = std::max(gmax_q,  seq_lengths_q[b]  + tgt);
    gmax_kv = std::max(gmax_kv, seq_lengths_kv[b] + tgt);
}
group_max_seqlens_q[i_grp]  = gmax_q  + group_contextual_seqlens[i_grp];
group_max_seqlens_kv[i_grp] = gmax_kv + group_contextual_seqlens[i_grp];
// if -g_max_seqlens override is supported, max() it in rather than replace.
```

Optionally add an assertion that `max_seqlen_q >= ` each batch's packed seqlen, so a future mis-setup fails loudly instead of silently producing wrong results.

## Notes

- The same underestimate pattern exists in the HSTU **backward** group example we are developing; we fixed it on our side with the formula above. We are reporting the forward occurrence here since it lives in the upstream-maintained `example_hstu_attention_fwd.cpp`.
- Environment: gfx950 / MI350X (CDNA4), ROCm `/opt/rocm`, `-DBUILD_DEV=OFF`. The logic bug is host-side and architecture-independent.
