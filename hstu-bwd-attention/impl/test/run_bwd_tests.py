#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
HSTU attention backward — automated regression test runner.

Drives the bwd harness binary (tile_example_hstu_attention_bwd) over a fixed
test matrix and asserts, per case, that the binary either:
  - PASS  : exits 0 AND prints exactly 3 "[PASS]" and no "[FAIL]"  (a fully-
            validated SiLU/no-mask/bf16/hd64/atomic run), or
  - REJECT: exits non-zero AND does NOT report all-3-PASS  (an un-implemented
            path that is correctly refused — dispatch throw, harness guard, or
            CLI-parse rejection), or
  - SKIP  : N/A on current code (reported, does not affect overall exit).

Overall exit is non-zero iff any non-skipped case did not meet its expectation
(CI-friendly). Pure python3 + subprocess, no third-party deps.

Designed as the regression gate for milestones M2-M8: when a milestone lands,
flip its REJECT case to PASS (see test/README.md).
"""

import argparse
import datetime
import os
import re
import subprocess
import sys

DEFAULT_BIN = "/root/workspace/ck_hstu/build/bin/tile_example_hstu_attention_bwd"
DEFAULT_BUILD_DIR = "/root/workspace/ck_hstu"
BUILD_TARGET = "tile_example_hstu_attention_bwd"
LOG_DIR = "/root/workspace/hstu-bwd-impl/runs"

PASS_RE = re.compile(r"\[PASS\]")
FAIL_RE = re.compile(r"\[FAIL\]")
ERR_LINE_RE = re.compile(r"^\s*d[QKV]:")  # "  dQ: max_abs_err=..."

# Common args shared by every "should PASS" baseline.
_COMMON = ["-prec=bf16", "-hdim_qk=64", "-hdim_v=64", "-softmax=0", "-causal=0", "-v=1"]
# Common args for group (M4) cases (causal set per-case; -g enables group).
_GB = ["-prec=bf16", "-hdim_qk=64", "-hdim_v=64", "-softmax=0", "-v=1"]
# Common prefix for the causal=0 x factor cross matrix (M4b).
_C0 = ["-prec=bf16", "-hdim_qk=64", "-hdim_v=64", "-softmax=0", "-v=1", "-attn_scale=1.0",
       "-causal=0", "-b=2", "-nhead=2", "-seqlens=128"]
# Common prefix for the M5 softmax cases (causal / -b / -seqlens / factors set per-case).
_SM = ["-prec=bf16", "-hdim_qk=64", "-hdim_v=64", "-softmax=1", "-v=1", "-attn_scale=1.0"]
# Common prefix for M5b group softmax cases (-g / per-group params set per-case).
_GSM = ["-prec=bf16", "-hdim_qk=64", "-hdim_v=64", "-softmax=1", "-v=1", "-attn_scale=1.0"]
# Common prefix for M6 deterministic correctness cases (softmax/causal/factors per-case).
_DET = ["-prec=bf16", "-hdim_qk=64", "-hdim_v=64", "-v=1", "-attn_scale=1.0", "-deterministic=1"]
# Common prefix for M6b group deterministic cases (-g / per-group params per-case).
_GDET = ["-prec=bf16", "-hdim_qk=64", "-hdim_v=64", "-v=1", "-attn_scale=1.0", "-deterministic=1"]
# M7a fp16 prefix (pure dtype widening; fp16 reuses the bf16 code path, hd64).
# Tolerance is the templated fp16 elimit rtol5e-3/atol1e-2 (tighter than bf16) — set in
# the harness, not here. These are real pass-assertions (no longer reject).
_FP16 = ["-prec=fp16", "-hdim_qk=64", "-hdim_v=64", "-v=1", "-attn_scale=1.0"]


def _c(name, args, expect, milestone="", note=""):
    return {"name": name, "args": args, "expect": expect, "milestone": milestone, "note": note}


# ---- Test matrix -----------------------------------------------------------
# expect ∈ {"pass", "reject", "skip"}.  Verified against real binary 2026-06-05.
MATRIX = [
    # ---- PASS baseline (M1-verified). Meaningful gradient magnitude via attn_scale=1.0,
    #      plus one default-scale case. Includes two non-tile-divisible seqlens. ----
    _c("pass-basic-attnscale1",
       _COMMON + ["-b=2", "-nhead=2", "-seqlens=128", "-attn_scale=1.0"], "pass", "M1"),
    _c("pass-b4-nhead8-seq256",
       _COMMON + ["-b=4", "-nhead=8", "-seqlens=256", "-attn_scale=1.0"], "pass", "M1"),
    _c("pass-b1-nhead1-seq512",
       _COMMON + ["-b=1", "-nhead=1", "-seqlens=512", "-attn_scale=1.0"], "pass", "M1"),
    _c("pass-seq200-non-kN0-128",
       _COMMON + ["-b=2", "-nhead=2", "-seqlens=200", "-attn_scale=1.0"], "pass", "M1",
       "seqlen not divisible by kN0=128"),
    _c("pass-seq130-non-kM0-32",
       _COMMON + ["-b=2", "-nhead=2", "-seqlens=130", "-attn_scale=1.0"], "pass", "M1",
       "seqlen not divisible by kM0=32 -> exercises OOB zero-fill via buffer_load"),
    _c("pass-default-attn_scale",
       _COMMON + ["-b=2", "-nhead=2", "-seqlens=128"], "pass", "M1",
       "default scale_p=1/max_seqlen_q (small gradients, still must PASS)"),

    # ---- M2 HSTU mask (causal + 5 factors), SiLU/bf16/hd64/batched. Landed -> pass. ----
    _c("pass-mask-causal",
       ["-prec=bf16", "-hdim_qk=64", "-hdim_v=64", "-softmax=0", "-causal=1", "-v=1",
        "-attn_scale=1.0", "-b=2", "-nhead=2", "-seqlens=128"], "pass", "M2", "causal only"),
    _c("pass-mask-window",
       ["-prec=bf16", "-hdim_qk=64", "-hdim_v=64", "-softmax=0", "-causal=1", "-v=1",
        "-attn_scale=1.0", "-b=2", "-nhead=2", "-seqlens=128", "-local_len=16"], "pass", "M2",
       "+ window"),
    _c("pass-mask-contextual",
       ["-prec=bf16", "-hdim_qk=64", "-hdim_v=64", "-softmax=0", "-causal=1", "-v=1",
        "-attn_scale=1.0", "-b=2", "-nhead=2", "-seqlens=128", "-context_len=8"], "pass", "M2",
       "+ contextual"),
    _c("pass-mask-minfull",
       ["-prec=bf16", "-hdim_qk=64", "-hdim_v=64", "-softmax=0", "-causal=1", "-v=1",
        "-attn_scale=1.0", "-b=2", "-nhead=2", "-seqlens=128", "-minfull_len=16"], "pass", "M2",
       "+ min_full"),
    _c("pass-mask-numtarget",
       ["-prec=bf16", "-hdim_qk=64", "-hdim_v=64", "-softmax=0", "-causal=1", "-v=1",
        "-attn_scale=1.0", "-b=2", "-nhead=2", "-seqlens=128", "-targets=16"], "pass", "M2",
       "+ num_target (num_targets supplemented to num_batch)"),
    _c("pass-mask-numtarget-perbatch",
       ["-prec=bf16", "-hdim_qk=64", "-hdim_v=64", "-softmax=0", "-causal=1", "-v=1",
        "-attn_scale=1.0", "-b=2", "-nhead=2", "-seqlens=128", "-targets=8,24"], "pass", "M2",
       "per-batch num_target"),
    _c("pass-mask-combo",
       ["-prec=bf16", "-hdim_qk=64", "-hdim_v=64", "-softmax=0", "-causal=1", "-v=1",
        "-attn_scale=1.0", "-b=2", "-nhead=2", "-seqlens=128",
        "-local_len=16", "-context_len=8", "-minfull_len=16", "-targets=16"], "pass", "M2",
       "all 5 factors combined"),
    _c("pass-mask-combo-seq200",
       ["-prec=bf16", "-hdim_qk=64", "-hdim_v=64", "-softmax=0", "-causal=1", "-v=1",
        "-attn_scale=1.0", "-b=2", "-nhead=2", "-seqlens=200",
        "-local_len=24", "-context_len=12", "-minfull_len=20", "-targets=12"], "pass", "M2",
       "combo + non-tile-divisible seqlen"),

    # ---- M3 jagged (variable-length packed [1,ΣL,h,d] + cu_seqlens). Landed -> pass. ----
    #      Per-batch seqlens differ (comma list); includes non-tile-divisible, single
    #      batch, large length spread, and tiny seqlens, plus jagged × mask combos. ----
    _c("pass-jagged-nomask-varying",
       ["-prec=bf16", "-hdim_qk=64", "-hdim_v=64", "-softmax=0", "-v=1", "-attn_scale=1.0",
        "-jagged=1", "-causal=0", "-b=3", "-nhead=2", "-seqlens=128,200,96"], "pass", "M3",
       "per-batch varying seqlen incl non-divisible 200/96"),
    _c("pass-jagged-causal-varying",
       ["-prec=bf16", "-hdim_qk=64", "-hdim_v=64", "-softmax=0", "-v=1", "-attn_scale=1.0",
        "-jagged=1", "-causal=1", "-b=3", "-nhead=2", "-seqlens=128,200,96"], "pass", "M3",
       "jagged + causal"),
    _c("pass-jagged-causal-window",
       ["-prec=bf16", "-hdim_qk=64", "-hdim_v=64", "-softmax=0", "-v=1", "-attn_scale=1.0",
        "-jagged=1", "-causal=1", "-b=4", "-nhead=4", "-seqlens=128,256,200,96",
        "-local_len=16"], "pass", "M3", "jagged + window"),
    _c("pass-jagged-causal-numtarget-perbatch",
       ["-prec=bf16", "-hdim_qk=64", "-hdim_v=64", "-softmax=0", "-v=1", "-attn_scale=1.0",
        "-jagged=1", "-causal=1", "-b=3", "-nhead=2", "-seqlens=128,200,96",
        "-targets=8,24,16"], "pass", "M3", "jagged + per-batch num_target"),
    _c("pass-jagged-5factor-combo",
       ["-prec=bf16", "-hdim_qk=64", "-hdim_v=64", "-softmax=0", "-v=1", "-attn_scale=1.0",
        "-jagged=1", "-causal=1", "-b=4", "-nhead=4", "-seqlens=256,200,128,96",
        "-local_len=24", "-context_len=12", "-minfull_len=20", "-targets=12,8,16,4"],
       "pass", "M3", "jagged + all 5 mask factors, per-batch varying"),
    _c("pass-jagged-single-batch",
       ["-prec=bf16", "-hdim_qk=64", "-hdim_v=64", "-softmax=0", "-v=1", "-attn_scale=1.0",
        "-jagged=1", "-causal=1", "-b=1", "-nhead=2", "-seqlens=300"], "pass", "M3",
       "single jagged batch"),
    _c("pass-jagged-large-spread",
       ["-prec=bf16", "-hdim_qk=64", "-hdim_v=64", "-softmax=0", "-v=1", "-attn_scale=1.0",
        "-jagged=1", "-causal=0", "-b=3", "-nhead=2", "-seqlens=512,32,256"], "pass", "M3",
       "large per-batch length spread (grid sized to max, short batches early-exit)"),
    _c("pass-jagged-tiny-seqlens",
       ["-prec=bf16", "-hdim_qk=64", "-hdim_v=64", "-softmax=0", "-v=1", "-attn_scale=1.0",
        "-jagged=1", "-causal=1", "-b=3", "-nhead=2", "-seqlens=1,128,7"], "pass", "M3",
       "tiny seqlens (1,7) mixed with full tile"),

    # ---- M4 group HSTU (packed + per-group hyper-params; alpha global). Landed -> pass. ----
    #      group = jagged superset; per-group window/contextual/min_full/max_seqlen/attn_scale
    #      indexed by i_group. Includes mixed per-group window (exercises the runtime
    #      with-local/without-local pipeline branch) and fully heterogeneous params. ----
    _c("pass-group-g2-nomask",
       _GB + ["-causal=0", "-b=4", "-nhead=2", "-g=2", "-seqlens=128,200,96,160"], "pass", "M4",
       "g=2 no-mask, per-batch varying packed seqlen"),
    _c("pass-group-g2-causal",
       _GB + ["-causal=1", "-b=4", "-nhead=2", "-g=2", "-seqlens=128,200,96,160"], "pass", "M4",
       "g=2 causal"),
    _c("pass-group-g2-pergroup-window",
       _GB + ["-causal=1", "-b=4", "-nhead=4", "-g=2", "-seqlens=128,200,96,160",
              "-g_local_lens=16,0"], "pass", "M4",
       "per-group window 16,0 -> mixes with-local & without-local groups in one launch"),
    _c("pass-group-g2-pergroup-attnscale",
       _GB + ["-causal=1", "-b=4", "-nhead=2", "-g=2", "-seqlens=128,200,96,160",
              "-g_attn_scales=1.0,0.5"], "pass", "M4",
       "per-group attn_scale 1.0,0.5"),
    _c("pass-group-g2-attnscale-fallback",
       _GB + ["-causal=1", "-b=4", "-nhead=2", "-g=2", "-seqlens=128,200,96,160",
              "-g_attn_scales=0,1.0"], "pass", "M4",
       "group attn_scale=0 -> scale_p=1/group_max_seqlen_q fallback"),
    _c("pass-group-g2-heterogeneous",
       _GB + ["-causal=1", "-b=4", "-nhead=4", "-g=2", "-seqlens=128,200,96,160",
              "-g_local_lens=16,0", "-g_context_lens=8,0", "-g_minfull_lens=16,0",
              "-g_attn_scales=1.0,0.5", "-targets=8,24,0,16"], "pass", "M4",
       "ALL per-group params differ + per-batch num_target (proves real i_group indexing)"),
    _c("pass-group-g3",
       _GB + ["-causal=1", "-b=6", "-nhead=2", "-g=3", "-seqlens=128,200,96,160,64,256",
              "-g_local_lens=16,0,32", "-g_attn_scales=1.0,0.5,0"], "pass", "M4",
       "g=3, per-group window + attn_scale (incl fallback)"),
    _c("pass-group-g4-singleton",
       _GB + ["-causal=1", "-b=4", "-nhead=2", "-g=4", "-seqlens=128,200,96,160",
              "-g_local_lens=0,16,0,32", "-g_context_lens=0,8,4,0",
              "-g_attn_scales=1.0,0.5,0,2.0"], "pass", "M4",
       "g=4 (one batch per group), fully heterogeneous"),

    # ---- M4b P1-1 fix + causal=0 x factor CROSS MATRIX (systematic, not just the
    #      one repro). Pre-fix, causal=0 compiled out STAGE2 masking (NoLocal
    #      IsMasking=kUseCausal); fix gates it on runtime IsEdgeTile. The existing
    #      sweeps only cover (causal=1 x factor) and (causal=0 x no-mask) -- the
    #      diagonal. These add the (causal=0 x factor) column for all three modes so
    #      the same IsMasking-coupling hole can't reopen on any factor. All verified
    #      against the CPU oracle -> expect PASS. ----
    #   batched, causal=0 x each factor:
    _c("pass-c0-batched-target",       _C0 + ["-targets=8"], "pass", "M4b-cross",
       "causal=0 + num_target (CORE repro: was FAIL dQ~1.16 pre-fix)"),
    _c("pass-c0-batched-context",      _C0 + ["-context_len=8"], "pass", "M4b-cross",
       "causal=0 contextual-only (max_uih_len=seqlen -> no clamp; lock as PASS)"),
    _c("pass-c0-batched-minfull",      _C0 + ["-minfull_len=16"], "pass", "M4b-cross",
       "causal=0 minfull-only (window=0 -> minfull ignored by without_local; lock)"),
    _c("pass-c0-batched-window",       _C0 + ["-local_len=16"], "pass", "M4b-cross",
       "causal=0 + window (WithLocal IsMasking=true path; lock)"),
    _c("pass-c0-batched-context-target",  _C0 + ["-context_len=8", "-targets=8"], "pass",
       "M4b-cross", "causal=0 + contextual + num_target"),
    _c("pass-c0-batched-minfull-target",  _C0 + ["-minfull_len=16", "-targets=8"], "pass",
       "M4b-cross", "causal=0 + minfull + num_target"),
    _c("pass-c0-batched-combo",        _C0 + ["-local_len=16", "-context_len=8",
       "-minfull_len=16", "-targets=8"], "pass", "M4b-cross",
       "causal=0 + all factors combined"),
    #   jagged, causal=0 x factor (per-batch):
    _c("pass-c0-jagged-target",
       ["-prec=bf16", "-hdim_qk=64", "-hdim_v=64", "-softmax=0", "-v=1", "-attn_scale=1.0",
        "-causal=0", "-b=3", "-nhead=2", "-jagged=1", "-seqlens=128,200,96", "-targets=8,24,16"],
       "pass", "M4b-cross", "jagged causal=0 + per-batch num_target"),
    _c("pass-c0-jagged-context",
       ["-prec=bf16", "-hdim_qk=64", "-hdim_v=64", "-softmax=0", "-v=1", "-attn_scale=1.0",
        "-causal=0", "-b=3", "-nhead=2", "-jagged=1", "-seqlens=128,200,96", "-context_len=8"],
       "pass", "M4b-cross", "jagged causal=0 + contextual"),
    #   group, causal=0 x factor (per-batch / per-group):
    _c("pass-c0-group-target",
       _GB + ["-causal=0", "-b=4", "-nhead=2", "-g=2", "-seqlens=128,200,96,160",
              "-attn_scale=1.0", "-targets=8,24,0,16"], "pass", "M4b-cross",
       "group causal=0 + per-batch num_target (was FAIL dQ~2.18 pre-fix)"),
    _c("pass-c0-group-window",
       _GB + ["-causal=0", "-b=4", "-nhead=4", "-g=2", "-seqlens=128,200,96,160",
              "-attn_scale=1.0", "-g_local_lens=16,0"], "pass", "M4b-cross",
       "group causal=0 + per-group window 16,0 (mixes WithLocal/NoLocal)"),
    _c("pass-c0-group-context-target",
       _GB + ["-causal=0", "-b=4", "-nhead=2", "-g=2", "-seqlens=128,200,96,160",
              "-attn_scale=1.0", "-g_context_lens=8,0", "-targets=8,24,0,16"], "pass",
       "M4b-cross", "group causal=0 + per-group contextual + per-batch num_target"),

    # ---- M5 softmax (no_group = batched + jagged). fwd stores LSE (natural log) ->
    #      bwd PRE computes D=rowsum(O*dO) -> softmax pipeline: P=exp(aS-LSE), dS=P(dP-D).
    #      Verified vs CPU oracle (GPU-produced LSE fed to both). softmax x causal{0,1}
    #      x factor x {batched,jagged}. group softmax is M5b (not here). ----
    _c("pass-sm-b-c0-nomask",   _SM + ["-causal=0", "-b=2", "-nhead=2", "-seqlens=128"],
       "pass", "M5", "softmax batched no-mask"),
    _c("pass-sm-b-c1-causal",   _SM + ["-causal=1", "-b=2", "-nhead=2", "-seqlens=128"],
       "pass", "M5", "softmax batched causal"),
    _c("pass-sm-b-c1-window",   _SM + ["-causal=1", "-b=2", "-nhead=2", "-seqlens=128",
       "-local_len=16"], "pass", "M5", "softmax batched causal+window"),
    _c("pass-sm-b-c0-window",   _SM + ["-causal=0", "-b=2", "-nhead=2", "-seqlens=128",
       "-local_len=16"], "pass", "M5", "softmax batched causal=0+window"),
    _c("pass-sm-b-c1-context",  _SM + ["-causal=1", "-b=2", "-nhead=2", "-seqlens=128",
       "-context_len=8"], "pass", "M5", "softmax batched causal+contextual"),
    _c("pass-sm-b-c1-numtarget", _SM + ["-causal=1", "-b=2", "-nhead=2", "-seqlens=128",
       "-targets=8"], "pass", "M5", "softmax batched causal+num_target"),
    _c("pass-sm-b-c0-numtarget", _SM + ["-causal=0", "-b=2", "-nhead=2", "-seqlens=128",
       "-targets=8"], "pass", "M5", "softmax batched causal=0+num_target"),
    _c("pass-sm-b-c1-combo",    _SM + ["-causal=1", "-b=2", "-nhead=2", "-seqlens=128",
       "-local_len=16", "-context_len=8", "-minfull_len=16", "-targets=8"], "pass", "M5",
       "softmax batched all factors"),
    _c("pass-sm-b-c1-seq200",   _SM + ["-causal=1", "-b=3", "-nhead=4", "-seqlens=200"],
       "pass", "M5", "softmax batched non-tile-divisible seqlen"),
    _c("pass-sm-j-c0-nomask",   _SM + ["-causal=0", "-b=3", "-nhead=2", "-jagged=1",
       "-seqlens=128,200,96"], "pass", "M5", "softmax jagged no-mask, per-batch varying"),
    _c("pass-sm-j-c1-causal",   _SM + ["-causal=1", "-b=3", "-nhead=2", "-jagged=1",
       "-seqlens=128,200,96"], "pass", "M5", "softmax jagged causal"),
    _c("pass-sm-j-c1-window",   _SM + ["-causal=1", "-b=3", "-nhead=2", "-jagged=1",
       "-seqlens=128,200,96", "-local_len=16"], "pass", "M5", "softmax jagged causal+window"),
    _c("pass-sm-j-c1-numtarget", _SM + ["-causal=1", "-b=3", "-nhead=2", "-jagged=1",
       "-seqlens=128,200,96", "-targets=8,24,16"], "pass", "M5",
       "softmax jagged causal+per-batch num_target"),
    _c("pass-sm-j-c0-numtarget", _SM + ["-causal=0", "-b=3", "-nhead=2", "-jagged=1",
       "-seqlens=128,200,96", "-targets=8,24,16"], "pass", "M5",
       "softmax jagged causal=0+per-batch num_target"),
    _c("pass-sm-j-c1-combo",    _SM + ["-causal=1", "-b=4", "-nhead=4", "-jagged=1",
       "-seqlens=256,200,128,96", "-local_len=24", "-context_len=12", "-minfull_len=20",
       "-targets=12,8,16,4"], "pass", "M5", "softmax jagged all factors, per-batch varying"),

    # ---- M5b group softmax (M4 group + M5 softmax合流). group=packed, per-group hyper-
    #      params via i_group + double pipeline + LSE/D. softmax x causal{0,1} x g{2,3,4}
    #      x per-group heterogeneous. group fwd produces LSE; reference uses GPU LSE. ----
    _c("pass-gsm-g2-c1-causal", _GSM + ["-causal=1", "-b=4", "-nhead=2", "-g=2",
       "-seqlens=128,200,96,160"], "pass", "M5b", "group softmax g=2 causal"),
    _c("pass-gsm-g2-c0-nomask", _GSM + ["-causal=0", "-b=4", "-nhead=2", "-g=2",
       "-seqlens=128,200,96,160"], "pass", "M5b", "group softmax g=2 causal=0 no-mask"),
    _c("pass-gsm-g2-c1-pergroup-window", _GSM + ["-causal=1", "-b=4", "-nhead=4", "-g=2",
       "-seqlens=128,200,96,160", "-g_local_lens=16,0"], "pass", "M5b",
       "group softmax per-group window 16,0 (mixes with/without-local pipeline)"),
    _c("pass-gsm-g2-c1-attnscale-fallback", _GSM + ["-causal=1", "-b=4", "-nhead=2", "-g=2",
       "-seqlens=128,200,96,160", "-g_attn_scales=0,1.0"], "pass", "M5b",
       "group softmax g_attn_scale=0 (softmax ignores scale_p; lock that it's harmless)"),
    _c("pass-gsm-g2-c1-heterogeneous", _GSM + ["-causal=1", "-b=4", "-nhead=4", "-g=2",
       "-seqlens=128,200,96,160", "-g_local_lens=16,0", "-g_context_lens=8,0",
       "-g_minfull_lens=16,0", "-g_attn_scales=1.0,0.5", "-targets=8,24,0,16"], "pass", "M5b",
       "group softmax ALL per-group params differ + per-batch num_target (real i_group)"),
    _c("pass-gsm-g2-c0-numtarget", _GSM + ["-causal=0", "-b=4", "-nhead=2", "-g=2",
       "-seqlens=128,200,96,160", "-targets=8,24,0,16"], "pass", "M5b",
       "group softmax causal=0 + per-batch num_target (P1-1 class)"),
    _c("pass-gsm-g3-c1", _GSM + ["-causal=1", "-b=6", "-nhead=2", "-g=3",
       "-seqlens=128,200,96,160,64,256", "-g_local_lens=16,0,32",
       "-g_attn_scales=1.0,0.5,0"], "pass", "M5b", "group softmax g=3 per-group window"),
    _c("pass-gsm-g4-c1-singleton", _GSM + ["-causal=1", "-b=4", "-nhead=2", "-g=4",
       "-seqlens=128,200,96,160", "-g_local_lens=0,16,0,32", "-g_context_lens=0,8,4,0",
       "-g_attn_scales=1.0,0.5,0,2.0"], "pass", "M5b",
       "group softmax g=4 (one batch/group) fully heterogeneous"),

    # ---- REJECT: un-implemented paths must NOT silently produce an all-PASS run. ----
    # Mechanism today is recorded in `note`; when the milestone lands, upgrade to "pass".
    # (M7a landed fp16 -> its reject case became the pass-fp16-* block.
    #  M7b landed symmetric hdim {64,96,128,256} -> hdim128/256 are now pass-hdim-* below.
    #  M7c landed asymmetric + non-canonical hdim via head-dim padding -> the two former reject/
    #  skip cases are now pass-ASYM/NONCANON below, asserted WITH -poison_pad=1 so the suite
    #  itself positively proves OOB head-dim load-zero / store-skip (a 4th [PASS] store-skip
    #  marker is required; the runner counts 4 markers for poison cases). The remaining
    #  STRUCTURAL reject (hdim>256, HDIM_SWITCH else-throw) is locked here.)
    _c("reject-hdim-gt256",
       ["-prec=bf16", "-hdim_qk=512", "-hdim_v=512", "-softmax=0", "-causal=0", "-v=1",
        "-b=2", "-nhead=2", "-seqlens=128", "-attn_scale=1.0"], "reject", "M7c",
       "structural reject: hdim>256 -> HDIM_SWITCH else-throw (kept)"),

    # ---- M6 deterministic (no_group batched+jagged x SiLU+softmax). dQ via per-KV-block
    #      split slots (no atomic) + POST reduce -> bit-reproducible. Correctness vs oracle
    #      here; bit-reproducibility (same case twice -> byte-identical dQ) asserted separately
    #      by run_repro_checks() below. ----
    _c("pass-det-silu-b-c0",  _DET + ["-softmax=0", "-causal=0", "-b=2", "-nhead=2",
       "-seqlens=128"], "pass", "M6", "determ SiLU batched no-mask"),
    _c("pass-det-silu-b-c1-combo", _DET + ["-softmax=0", "-causal=1", "-b=2", "-nhead=2",
       "-seqlens=128", "-local_len=16", "-context_len=8", "-minfull_len=16", "-targets=8"],
       "pass", "M6", "determ SiLU batched all factors"),
    _c("pass-det-silu-b-seq512", _DET + ["-softmax=0", "-causal=1", "-b=2", "-nhead=4",
       "-seqlens=512"], "pass", "M6", "determ SiLU multi-KV-block (4 splits)"),
    _c("pass-det-silu-j-c1",  _DET + ["-softmax=0", "-causal=1", "-b=3", "-nhead=2",
       "-jagged=1", "-seqlens=128,200,96"], "pass", "M6", "determ SiLU jagged causal"),
    _c("pass-det-sm-b-c0",    _DET + ["-softmax=1", "-causal=0", "-b=2", "-nhead=2",
       "-seqlens=128"], "pass", "M6", "determ softmax batched no-mask"),
    _c("pass-det-sm-b-c1-window", _DET + ["-softmax=1", "-causal=1", "-b=2", "-nhead=2",
       "-seqlens=128", "-local_len=16"], "pass", "M6", "determ softmax batched causal+window"),
    _c("pass-det-sm-j-c1-numtgt", _DET + ["-softmax=1", "-causal=1", "-b=3", "-nhead=2",
       "-jagged=1", "-seqlens=128,200,96", "-targets=8,24,16"], "pass", "M6",
       "determ softmax jagged causal+per-batch num_target"),

    # ---- M6b group deterministic (M6 determ x M4/M5b group). Fixes O1 (group+determ was
    #      silently atomic via hardcoded false). Cases kept within the M5b-validated group
    #      envelope (correctness gate); bit-reproducibility asserted by run_repro_checks(). ----
    _c("pass-gdet-silu-g2-c1",  _GDET + ["-softmax=0", "-causal=1", "-b=4", "-nhead=2",
       "-g=2", "-seqlens=128,200,96,160"], "pass", "M6b", "group determ SiLU g2 causal"),
    _c("pass-gdet-silu-g2-hetero", _GDET + ["-softmax=0", "-causal=1", "-b=4", "-nhead=4",
       "-g=2", "-seqlens=128,200,96,160", "-g_local_lens=16,0", "-g_context_lens=8,0",
       "-g_minfull_lens=16,0", "-g_attn_scales=1.0,0.5", "-targets=8,24,0,16"], "pass", "M6b",
       "group determ SiLU fully heterogeneous"),
    _c("pass-gdet-silu-g2-seq512", _GDET + ["-softmax=0", "-causal=1", "-b=2", "-nhead=4",
       "-g=2", "-seqlens=512,300"], "pass", "M6b", "group determ SiLU multi-split"),
    _c("pass-gdet-silu-g4", _GDET + ["-softmax=0", "-causal=1", "-b=4", "-nhead=2", "-g=4",
       "-seqlens=128,200,96,160", "-g_local_lens=0,16,0,32", "-g_attn_scales=1.0,0.5,0,2.0"],
       "pass", "M6b", "group determ SiLU g4 singleton heterogeneous"),
    _c("pass-gdet-sm-g2-c1",    _GDET + ["-softmax=1", "-causal=1", "-b=4", "-nhead=2",
       "-g=2", "-seqlens=128,200,96,160"], "pass", "M6b", "group determ softmax g2 causal"),
    _c("pass-gdet-sm-g2-window", _GDET + ["-softmax=1", "-causal=1", "-b=4", "-nhead=4",
       "-g=2", "-seqlens=128,200,96,160", "-g_local_lens=16,0"], "pass", "M6b",
       "group determ softmax per-group window (mixed with/without-local)"),
    _c("pass-gdet-sm-g2-seq512", _GDET + ["-softmax=1", "-causal=1", "-b=2", "-nhead=4",
       "-g=2", "-seqlens=512,300"], "pass", "M6b", "group determ softmax multi-split"),
    _c("pass-gdet-sm-g3", _GDET + ["-softmax=1", "-causal=1", "-b=6", "-nhead=2", "-g=3",
       "-seqlens=128,200,96,160,64,256", "-g_local_lens=16,0,32", "-g_attn_scales=1.0,0.5,0"],
       "pass", "M6b", "group determ softmax g3 per-group window"),

    # ---- M6b/M5b regression lock for the group_max_seqlens_q under-cover bug (harness):
    #      nbpg>1 + differing seqlen within a group + the group's longer batch carries the
    #      larger num_target + window>0. Pre-fix: PRE D under-covered the long batch's tail
    #      tokens -> wrong dQ. The bug is softmax-PRE-D specific, so only the two softmax
    #      cases below actually LOCK it (verified: revert the harness formula -> they FAIL).
    #      The SiLU case is general coverage only (SiLU has no PRE-D -> cannot detect it). ----
    _c("pass-gtrig-sm-atomic",
       ["-prec=bf16", "-hdim_qk=64", "-hdim_v=64", "-v=1", "-attn_scale=1.0", "-softmax=1",
        "-causal=1", "-b=4", "-nhead=4", "-g=2", "-seqlens=128,200,96,160", "-g_local_lens=16,16",
        "-targets=8,24,8,16"], "pass", "M5b",
       "group softmax atomic — group_max_seqlens_q under-cover trigger (was dQ 0.0626 FAIL)"),
    _c("pass-gtrig-sm-determ",
       ["-prec=bf16", "-hdim_qk=64", "-hdim_v=64", "-v=1", "-attn_scale=1.0", "-deterministic=1",
        "-softmax=1", "-causal=1", "-b=4", "-nhead=4", "-g=2", "-seqlens=128,200,96,160",
        "-g_local_lens=16,16", "-targets=8,24,8,16"], "pass", "M6b",
       "group softmax determ — same trigger (shares PRE D; was FAIL)"),
    _c("pass-gtrig-silu-atomic",
       ["-prec=bf16", "-hdim_qk=64", "-hdim_v=64", "-v=1", "-attn_scale=1.0", "-softmax=0",
        "-causal=1", "-b=4", "-nhead=4", "-g=2", "-seqlens=128,200,96,160", "-g_local_lens=16,16",
        "-targets=8,24,8,16"], "pass", "M5b",
       "group SiLU same config — general coverage only (SiLU has no PRE-D so this canNOT "
       "detect the under-cover bug; the sm-* cases above are what lock it)"),

    # ---- M7a fp16: dtype widening (fp16 reuses the bf16 template code path, hd64).
    #      Representative cross of {no_group batched/jagged, group} x {SiLU, softmax} x
    #      causal{0,1} x a few mask factors + determ (no_group + group). fp16 elimit is
    #      TIGHTER than bf16 (rtol5e-3/atol1e-2). All verified vs the CPU oracle -> pass.
    #      Byte-identical fp16 determ repro is asserted by REPRO_CASES below. ----
    #   no_group batched, SiLU:
    _c("pass-fp16-silu-b-c0",     _FP16 + ["-softmax=0", "-causal=0", "-b=2", "-nhead=2",
       "-seqlens=128"], "pass", "M7a", "fp16 SiLU batched no-mask (was reject-fp16)"),
    _c("pass-fp16-silu-b-c1-combo", _FP16 + ["-softmax=0", "-causal=1", "-b=2", "-nhead=2",
       "-seqlens=128", "-local_len=16", "-context_len=8", "-minfull_len=16", "-targets=8"],
       "pass", "M7a", "fp16 SiLU batched all 5 factors"),
    _c("pass-fp16-silu-b-c0-target", _FP16 + ["-softmax=0", "-causal=0", "-b=2", "-nhead=2",
       "-seqlens=128", "-targets=8"], "pass", "M7a", "fp16 SiLU causal=0+num_target (P1-1 cross)"),
    _c("pass-fp16-silu-b-seq200", _FP16 + ["-softmax=0", "-causal=1", "-b=2", "-nhead=2",
       "-seqlens=200"], "pass", "M7a", "fp16 non-tile-divisible seqlen"),
    #   no_group jagged, SiLU:
    _c("pass-fp16-silu-j-c1-combo", _FP16 + ["-softmax=0", "-causal=1", "-b=4", "-nhead=4",
       "-jagged=1", "-seqlens=256,200,128,96", "-local_len=24", "-context_len=12",
       "-minfull_len=20", "-targets=12,8,16,4"], "pass", "M7a", "fp16 SiLU jagged all factors"),
    #   no_group batched/jagged, softmax:
    _c("pass-fp16-sm-b-c1-causal", _FP16 + ["-softmax=1", "-causal=1", "-b=2", "-nhead=2",
       "-seqlens=128"], "pass", "M7a", "fp16 softmax batched causal"),
    _c("pass-fp16-sm-b-c0-target", _FP16 + ["-softmax=1", "-causal=0", "-b=2", "-nhead=2",
       "-seqlens=128", "-targets=8"], "pass", "M7a", "fp16 softmax causal=0+num_target (P1-1)"),
    _c("pass-fp16-sm-j-c1-combo", _FP16 + ["-softmax=1", "-causal=1", "-b=4", "-nhead=4",
       "-jagged=1", "-seqlens=256,200,128,96", "-local_len=24", "-context_len=12",
       "-minfull_len=20", "-targets=12,8,16,4"], "pass", "M7a", "fp16 softmax jagged all factors"),
    #   group, SiLU + softmax, causal{0,1} + heterogeneous:
    _c("pass-fp16-silu-g2-hetero", _FP16 + ["-softmax=0", "-causal=1", "-b=4", "-nhead=4",
       "-g=2", "-seqlens=128,200,96,160", "-g_local_lens=16,0", "-g_context_lens=8,0",
       "-g_minfull_lens=16,0", "-g_attn_scales=1.0,0.5", "-targets=8,24,0,16"], "pass", "M7a",
       "fp16 group SiLU fully heterogeneous"),
    _c("pass-fp16-sm-g2-c1", _FP16 + ["-softmax=1", "-causal=1", "-b=4", "-nhead=2", "-g=2",
       "-seqlens=128,200,96,160"], "pass", "M7a", "fp16 group softmax g2 causal"),
    _c("pass-fp16-sm-g2-c0-target", _FP16 + ["-softmax=1", "-causal=0", "-b=4", "-nhead=2",
       "-g=2", "-seqlens=128,200,96,160", "-targets=8,24,0,16"], "pass", "M7a",
       "fp16 group softmax causal=0+num_target (P1-1 class)"),
    #   deterministic (no_group + group), SiLU + softmax:
    _c("pass-fp16-det-silu-b-seq512", _FP16 + ["-deterministic=1", "-softmax=0", "-causal=1",
       "-b=2", "-nhead=4", "-seqlens=512"], "pass", "M7a", "fp16 determ SiLU multi-split"),
    _c("pass-fp16-det-sm-j-numtgt", _FP16 + ["-deterministic=1", "-softmax=1", "-causal=1",
       "-b=3", "-nhead=2", "-jagged=1", "-seqlens=128,200,96", "-targets=8,24,16"], "pass",
       "M7a", "fp16 determ softmax jagged + per-batch num_target"),
    _c("pass-fp16-gdet-sm-g2", _FP16 + ["-deterministic=1", "-softmax=1", "-causal=1", "-b=4",
       "-nhead=2", "-g=2", "-seqlens=128,200,96,160"], "pass", "M7a",
       "fp16 group determ softmax g2 causal"),

    # ---- M7b symmetric hdim {96,128,256} (hdim_qk==hdim_v). Tile shape selected by MaxK via
    #      HstuBwdShape<MaxK>; hd64 unchanged (byte-identical). For each hdim: SiLU+softmax x
    #      bf16+fp16 x causal{0,1} x representative mask x {batched,jagged,group} x atomic/determ,
    #      P1-1 cross (causal=0+num_target) PER HDIM. Tolerance = templated elimit (NOT loosened).
    #      hd256 uses tile bn0=64 (determ kN0=64); verified no scratch spill (profile/M7b-hd256-resource.md).
    #      Helper: build a per-(hdim,dtype) arg prefix. ----
] + [
    _c(f"pass-h{hd}-{dt}-{name}",
       [f"-prec={dt}", f"-hdim_qk={hd}", f"-hdim_v={hd}", "-v=1", "-attn_scale=1.0"] + extra,
       "pass", "M7b", note)
    for hd in (96, 128, 256)
    for dt in ("bf16", "fp16")
    for name, extra, note in [
        ("silu-b-c1",       ["-softmax=0", "-causal=1", "-b=2", "-nhead=2", "-seqlens=128"],
         "SiLU batched causal"),
        ("silu-b-c1-combo", ["-softmax=0", "-causal=1", "-b=2", "-nhead=2", "-seqlens=128",
                             "-local_len=16", "-context_len=8", "-minfull_len=16", "-targets=8"],
         "SiLU batched all 5 factors"),
        ("silu-b-c0-target",["-softmax=0", "-causal=0", "-b=2", "-nhead=2", "-seqlens=128",
                             "-targets=8"], "SiLU causal=0+num_target (P1-1 cross, per hdim)"),
        ("silu-j-combo",    ["-softmax=0", "-causal=1", "-b=4", "-nhead=4", "-jagged=1",
                             "-seqlens=256,200,128,96", "-local_len=24", "-context_len=12",
                             "-minfull_len=20", "-targets=12,8,16,4"], "SiLU jagged all factors"),
        ("sm-b-c1",         ["-softmax=1", "-causal=1", "-b=2", "-nhead=2", "-seqlens=128"],
         "softmax batched causal"),
        ("sm-b-c0-target",  ["-softmax=1", "-causal=0", "-b=2", "-nhead=2", "-seqlens=128",
                             "-targets=8"], "softmax causal=0+num_target (P1-1 cross, per hdim)"),
        ("silu-g2-hetero",  ["-softmax=0", "-causal=1", "-b=4", "-nhead=4", "-g=2",
                             "-seqlens=128,200,96,160", "-g_local_lens=16,0", "-g_context_lens=8,0",
                             "-g_minfull_lens=16,0", "-g_attn_scales=1.0,0.5", "-targets=8,24,0,16"],
         "group SiLU fully heterogeneous"),
        ("sm-g2-c0-target", ["-softmax=1", "-causal=0", "-b=4", "-nhead=2", "-g=2",
                             "-seqlens=128,200,96,160", "-targets=8,24,0,16"],
         "group softmax causal=0+num_target (P1-1 cross, per hdim)"),
        ("det-silu-b-512",  ["-deterministic=1", "-softmax=0", "-causal=1", "-b=2", "-nhead=4",
                             "-seqlens=512"], "determ SiLU multi-split (hd256: kN0=64)"),
        ("gdet-sm-g2",      ["-deterministic=1", "-softmax=1", "-causal=1", "-b=4", "-nhead=2",
                             "-g=2", "-seqlens=128,200,96,160"], "group determ softmax"),
    ]
] + [
    # ---- M7c asymmetric / non-canonical hdim via head-dim padding. ASSERTED WITH -poison_pad=1
    #      so the suite itself positively proves OOB head-dim load-zero / store-skip (NaN-filled
    #      input pad tails + pre-poisoned output tails -> any leak = NaN = FAIL; a 4th [PASS]
    #      store-skip marker is required). no_group + group x {bf16,fp16} x representative pairs
    #      (both directions 64/128 & 128/64, non-canonical 100/100 & 80/128, */256 determ lock),
    #      each pair incl a P1-1 cross (causal=0 + num_target). Tolerance NOT loosened. ----
    _c(f"pass-m7c-{tag}-{hq}x{hv}-{dt}",
       [f"-prec={dt}", f"-hdim_qk={hq}", f"-hdim_v={hv}", "-v=1", "-attn_scale=1.0",
        "-poison_pad=1"] + extra, "pass", "M7c", note)
    for hq, hv in [(64, 128), (128, 64), (100, 100), (80, 128), (128, 256)]
    for dt in ("bf16", "fp16")
    for tag, extra, note in [
        ("ng-silu-c1",   ["-softmax=0", "-causal=1", "-b=2", "-nhead=2", "-seqlens=128"],
         "no_group SiLU causal (poison)"),
        ("ng-sm-c0tgt",  ["-softmax=1", "-causal=0", "-b=2", "-nhead=2", "-seqlens=128",
                          "-targets=8"], "no_group softmax causal=0+num_target (P1-1, poison)"),
        ("ng-det-sm-c1", ["-deterministic=1", "-softmax=1", "-causal=1", "-b=2", "-nhead=4",
                          "-seqlens=512"], "no_group determ softmax (poison; */256 -> kN0=64)"),
        ("g2-sm-c1",     ["-softmax=1", "-causal=1", "-b=4", "-nhead=2", "-g=2",
                          "-seqlens=128,200,96,160"], "group softmax causal (poison)"),
        ("g2-silu-c0tgt",["-softmax=0", "-causal=0", "-b=4", "-nhead=2", "-g=2",
                          "-seqlens=128,200,96,160", "-targets=8,24,0,16"],
         "group SiLU causal=0+num_target (P1-1, poison)"),
    ]
] + [
    # ---- Mcross: cross-attention (seqlen_q != seqlen_kv) via -seqlens_kv (draft §6).
    #      cross is a RUNTIME switch (kIsCrossAttention BOOL_SWITCH in dispatch + if constexpr
    #      mask builder in kernel); self path (-seqlens_kv absent) is byte-identical (co_symbols
    #      486/486). These flip the formerly-REJECT cross cases (CLI didn't know -seqlens_kv) to
    #      PASS. BOTH directions (q<kv & q>kv) x {no_group jagged, group, batched-uniform} x
    #      SiLU/softmax x causal{0,1} x P1-1 (Q-side target / contextual<=min(q,kv) / local /
    #      minfull) x atomic/determ + non-divisible + determ kv>q multi-KV-block (R4) + fp16.
    #      target_in_kv==false (KV has contextual, no targets). -attn_scale=1.0, elimit NOT
    #      loosened. Full 32-case 对拍 sweep: test/sweep_cross.py (runs/run-cross-sweep.log). ----
    _c(f"xattn-{name}", ["-prec=" + dt, "-hdim_qk=64", "-hdim_v=64", "-v=1", "-attn_scale=1.0"] + extra,
       "pass", "Mcross", note)
    for name, dt, extra, note in [
        # no_group jagged, both directions, SiLU/softmax x causal{0,1}
        ("j-qlt-silu-c1", "bf16", ["-jagged=1", "-b=2", "-nhead=2", "-seqlens=128", "-seqlens_kv=256", "-softmax=0", "-causal=1"], "q<kv jagged SiLU causal"),
        ("j-qgt-silu-c1", "bf16", ["-jagged=1", "-b=2", "-nhead=2", "-seqlens=256", "-seqlens_kv=128", "-softmax=0", "-causal=1"], "q>kv jagged SiLU causal"),
        ("j-qlt-silu-c0", "bf16", ["-jagged=1", "-b=2", "-nhead=2", "-seqlens=128", "-seqlens_kv=256", "-softmax=0", "-causal=0"], "q<kv jagged SiLU no-causal"),
        ("j-qgt-silu-c0", "bf16", ["-jagged=1", "-b=2", "-nhead=2", "-seqlens=256", "-seqlens_kv=128", "-softmax=0", "-causal=0"], "q>kv jagged SiLU no-causal"),
        ("j-qlt-sm-c1",   "bf16", ["-jagged=1", "-b=2", "-nhead=2", "-seqlens=128", "-seqlens_kv=256", "-softmax=1", "-causal=1"], "q<kv jagged softmax causal"),
        ("j-qgt-sm-c1",   "bf16", ["-jagged=1", "-b=2", "-nhead=2", "-seqlens=256", "-seqlens_kv=128", "-softmax=1", "-causal=1"], "q>kv jagged softmax causal"),
        ("j-qlt-sm-c0",   "bf16", ["-jagged=1", "-b=2", "-nhead=2", "-seqlens=128", "-seqlens_kv=256", "-softmax=1", "-causal=0"], "q<kv jagged softmax no-causal"),
        ("j-qgt-sm-c0",   "bf16", ["-jagged=1", "-b=2", "-nhead=2", "-seqlens=256", "-seqlens_kv=128", "-softmax=1", "-causal=0"], "q>kv jagged softmax no-causal"),
        # P1-1 factors, both directions (Q-side targets; contextual<=min(q,kv)=128)
        ("j-qlt-c0-target",  "bf16", ["-jagged=1", "-b=2", "-nhead=2", "-seqlens=128", "-seqlens_kv=256", "-softmax=0", "-causal=0", "-targets=8"], "q<kv causal=0+target (P1-1)"),
        ("j-qgt-c0-target",  "bf16", ["-jagged=1", "-b=2", "-nhead=2", "-seqlens=256", "-seqlens_kv=128", "-softmax=0", "-causal=0", "-targets=8"], "q>kv causal=0+target (P1-1)"),
        ("j-qlt-c1-context", "bf16", ["-jagged=1", "-b=2", "-nhead=2", "-seqlens=128", "-seqlens_kv=256", "-softmax=0", "-causal=1", "-context_len=8"], "q<kv contextual"),
        ("j-qgt-c1-context", "bf16", ["-jagged=1", "-b=2", "-nhead=2", "-seqlens=256", "-seqlens_kv=128", "-softmax=0", "-causal=1", "-context_len=8"], "q>kv contextual"),
        ("j-qlt-c1-local",   "bf16", ["-jagged=1", "-b=2", "-nhead=2", "-seqlens=128", "-seqlens_kv=256", "-softmax=0", "-causal=1", "-local_len=16"], "q<kv local window"),
        ("j-qgt-c1-local",   "bf16", ["-jagged=1", "-b=2", "-nhead=2", "-seqlens=256", "-seqlens_kv=128", "-softmax=0", "-causal=1", "-local_len=16"], "q>kv local window"),
        ("j-qlt-c1-minfull", "bf16", ["-jagged=1", "-b=2", "-nhead=2", "-seqlens=128", "-seqlens_kv=256", "-softmax=0", "-causal=1", "-minfull_len=16"], "q<kv min_full"),
        ("j-qlt-c1-combo",   "bf16", ["-jagged=1", "-b=3", "-nhead=2", "-seqlens=128,160,96", "-seqlens_kv=256,200,300", "-softmax=1", "-causal=1", "-local_len=16", "-context_len=8", "-minfull_len=16", "-targets=8,16,0"], "jagged combo all factors"),
        # non-divisible (q non-kM0=32-div, kv non-kN0=128-div)
        ("j-nondiv-qlt", "bf16", ["-jagged=1", "-b=2", "-nhead=2", "-seqlens=130", "-seqlens_kv=200", "-softmax=0", "-causal=1"], "q<kv non-divisible 130/200"),
        ("j-nondiv-qgt", "bf16", ["-jagged=1", "-b=2", "-nhead=2", "-seqlens=200", "-seqlens_kv=130", "-softmax=1", "-causal=1"], "q>kv non-divisible 200/130"),
        # determ (kN0=128 for hd64): >=1 case kv>q crossing multiple KV blocks (R4)
        ("j-determ-qlt-multiblk-sm",   "bf16", ["-jagged=1", "-b=2", "-nhead=4", "-seqlens=128", "-seqlens_kv=512", "-softmax=1", "-causal=1", "-deterministic=1"], "determ kv=512>q=128 multi-block (R4)"),
        ("j-determ-qlt-multiblk-silu", "bf16", ["-jagged=1", "-b=2", "-nhead=4", "-seqlens=128", "-seqlens_kv=512", "-softmax=0", "-causal=1", "-deterministic=1", "-targets=8"], "determ kv>q multi-block SiLU+target"),
        ("j-determ-qgt",               "bf16", ["-jagged=1", "-b=2", "-nhead=4", "-seqlens=512", "-seqlens_kv=128", "-softmax=0", "-causal=1", "-deterministic=1"], "determ q>kv"),
        # group (per-group kv lengths), both directions
        ("g2-qlt-sm-c1",     "bf16", ["-g=2", "-b=4", "-nhead=2", "-seqlens=128", "-seqlens_kv=256", "-softmax=1", "-causal=1"], "group q<kv softmax"),
        ("g2-qgt-silu-c1",   "bf16", ["-g=2", "-b=4", "-nhead=2", "-seqlens=256", "-seqlens_kv=128", "-softmax=0", "-causal=1"], "group q>kv SiLU"),
        ("g2-qlt-c0-target", "bf16", ["-g=2", "-b=4", "-nhead=2", "-seqlens=128", "-seqlens_kv=256", "-softmax=0", "-causal=0", "-targets=8"], "group q<kv causal=0+target (P1-1)"),
        ("g2-het-qlt-sm-c1", "bf16", ["-g=2", "-b=4", "-nhead=2", "-seqlens=128,160,96,200", "-seqlens_kv=256,300,200,256", "-softmax=1", "-causal=1"], "group heterogeneous q<kv"),
        ("g2-determ-qlt-multiblk", "bf16", ["-g=2", "-b=4", "-nhead=4", "-seqlens=128", "-seqlens_kv=512", "-softmax=1", "-causal=1", "-deterministic=1"], "group determ kv>q multi-block"),
        # batched uniform, both directions (raw scalar seqlen_kv path)
        ("b-qlt-silu-c1", "bf16", ["-jagged=0", "-b=2", "-nhead=2", "-seqlens=128", "-seqlens_kv=256", "-softmax=0", "-causal=1"], "batched q<kv SiLU"),
        ("b-qgt-sm-c1",   "bf16", ["-jagged=0", "-b=2", "-nhead=2", "-seqlens=256", "-seqlens_kv=128", "-softmax=1", "-causal=1"], "batched q>kv softmax"),
        ("b-qlt-determ-multiblk", "bf16", ["-jagged=0", "-b=2", "-nhead=4", "-seqlens=128", "-seqlens_kv=512", "-softmax=1", "-causal=1", "-deterministic=1"], "batched determ kv>q multi-block"),
        # fp16 (tighter elimit), both directions
        ("j-qlt-silu-c1-fp16", "fp16", ["-jagged=1", "-b=2", "-nhead=2", "-seqlens=128", "-seqlens_kv=256", "-softmax=0", "-causal=1"], "fp16 q<kv jagged SiLU"),
        ("g2-qgt-sm-c1-fp16",  "fp16", ["-g=2", "-b=4", "-nhead=2", "-seqlens=256", "-seqlens_kv=128", "-softmax=1", "-causal=1"], "fp16 group q>kv softmax"),
    ]
]

# Deterministic bit-reproducibility checks (M6 core): run the same case twice with
# -dump_grad and assert dq_dev.dat is byte-identical. Multi-KV-block (seq512 -> 4 splits)
# so the split machinery is actually exercised. (name, extra-args)
REPRO_CASES = [
    ("repro-det-silu-seq512",
     ["-softmax=0", "-causal=1", "-b=2", "-nhead=4", "-seqlens=512", "-local_len=64",
      "-targets=16"]),
    ("repro-det-softmax-seq512",
     ["-softmax=1", "-causal=1", "-b=2", "-nhead=4", "-seqlens=512"]),
    ("repro-det-jagged-softmax",
     ["-softmax=1", "-causal=1", "-b=3", "-nhead=2", "-jagged=1", "-seqlens=512,300,400",
      "-targets=16,8,32"]),
    # M6b group deterministic bit-reproducibility (multi-split, per-group heterogeneous)
    ("repro-gdet-silu-g2",
     ["-softmax=0", "-causal=1", "-b=2", "-nhead=4", "-g=2", "-seqlens=512,300",
      "-g_local_lens=16,0"]),
    ("repro-gdet-softmax-g2",
     ["-softmax=1", "-causal=1", "-b=2", "-nhead=4", "-g=2", "-seqlens=512,300"]),
    ("repro-gdet-silu-g3",
     ["-softmax=0", "-causal=1", "-b=6", "-nhead=2", "-g=3",
      "-seqlens=300,512,256,400,128,480", "-g_local_lens=16,0,32"]),
    # M7a fp16 deterministic bit-reproducibility (-prec=fp16 overrides the bf16 in the
    # common prefix; last value wins). no_group multi-split + group multi-split.
    ("repro-fp16-det-softmax-seq512",
     ["-softmax=1", "-causal=1", "-b=2", "-nhead=4", "-seqlens=512", "-prec=fp16"]),
    ("repro-fp16-gdet-silu-g2",
     ["-softmax=0", "-causal=1", "-b=2", "-nhead=4", "-g=2", "-seqlens=512,300",
      "-g_local_lens=16,0", "-prec=fp16"]),
    # M7b per-hdim deterministic bit-reproducibility (-hdim_qk/-hdim_v override the 64 in the
    # common prefix; last value wins). hd256 is the key one: tile bn0=64 -> kN0=64 split path.
    ("repro-h96-det-softmax-seq512",
     ["-softmax=1", "-causal=1", "-b=2", "-nhead=4", "-seqlens=512", "-hdim_qk=96", "-hdim_v=96"]),
    ("repro-h128-det-silu-seq512",
     ["-softmax=0", "-causal=1", "-b=2", "-nhead=4", "-seqlens=512", "-local_len=64",
      "-targets=16", "-hdim_qk=128", "-hdim_v=128"]),
    ("repro-h256-det-softmax-seq512",
     ["-softmax=1", "-causal=1", "-b=2", "-nhead=4", "-seqlens=512", "-hdim_qk=256", "-hdim_v=256"]),
    ("repro-h256-gdet-silu-g2",
     ["-softmax=0", "-causal=1", "-b=2", "-nhead=4", "-g=2", "-seqlens=512,300",
      "-g_local_lens=16,0", "-hdim_qk=256", "-hdim_v=256"]),
    # Mcross deterministic bit-reproducibility: kv>q multi-KV-block (kv=512 -> 4 splits at
    # kN0=128) so the cross grid/num_splits-over-max_seqlen_kv split machinery is exercised (R4).
    ("repro-xattn-det-qlt-multiblk",
     ["-softmax=1", "-causal=1", "-jagged=1", "-b=2", "-nhead=4", "-seqlens=128", "-seqlens_kv=512"]),
    ("repro-xattn-gdet-qlt-multiblk",
     ["-softmax=1", "-causal=1", "-g=2", "-b=4", "-nhead=4", "-seqlens=128", "-seqlens_kv=512"]),
]
_DET_REPRO_COMMON = ["-prec=bf16", "-hdim_qk=64", "-hdim_v=64", "-attn_scale=1.0", "-v=0",
                     "-deterministic=1", "-dump_grad=1"]


def count_markers(text):
    return len(PASS_RE.findall(text)), len(FAIL_RE.findall(text))


def all_pass(text):
    n_pass, n_fail = count_markers(text)
    return n_pass == 3 and n_fail == 0


def key_info(text, expect):
    """Short human-readable detail for the summary line."""
    lines = [l.rstrip() for l in text.splitlines() if l.strip()]
    if expect == "pass":
        errs = [l.strip() for l in lines if ERR_LINE_RE.match(l)]
        if errs:
            return " | ".join(errs)
        return lines[-1] if lines else "(no output)"
    # reject / skip: the most informative line is usually the last (throw/guard/parse msg)
    for key in ("what():", "Failed to parse", "only supports", "not implemented"):
        for l in reversed(lines):
            if key in l:
                return l.strip()
    return lines[-1] if lines else "(no output)"


def run_case(binary, case, timeout):
    cmd = [binary] + case["args"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        out = (proc.stdout or "") + (proc.stderr or "")
        rc = proc.returncode
        timed_out = False
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or "") + (e.stderr or "") if isinstance(e.stdout, str) else ""
        out += f"\n<<< TIMEOUT after {timeout}s >>>"
        rc = None
        timed_out = True

    n_pass, n_fail = count_markers(out)
    # M7c poison cases print a 4th marker ([PASS]/[FAIL] store-skip dK/dV) on top of the 3
    # gradient markers; assert the full set so a leaked OOB store/load is caught here too.
    poison = "-poison_pad=1" in case["args"]
    need = 4 if poison else 3
    passed3 = (n_pass == need and n_fail == 0)
    expect = case["expect"]

    if timed_out:
        ok, verdict = False, "FAIL"
        reason = f"timeout after {timeout}s"
    elif expect == "pass":
        ok = (rc == 0 and passed3)
        verdict = "PASS" if ok else "FAIL"
        reason = "" if ok else f"expected exit0 + {need}xPASS, got exit={rc} PASS={n_pass} FAIL={n_fail}"
    elif expect == "reject":
        ok = (rc != 0 and not passed3)
        verdict = "PASS" if ok else "FAIL"
        if ok:
            reason = ""
        elif rc == 0 and passed3:
            reason = ("REGRESSION/false-positive: path now produces all-PASS -> "
                      "this milestone may be implemented; upgrade this case to expect='pass'")
        else:
            reason = f"expected reject(exit!=0 & not all-PASS), got exit={rc} PASS={n_pass} FAIL={n_fail}"
    elif expect == "skip":
        ok = True
        verdict = "SKIP"
        reason = f"N/A: exit={rc} PASS={n_pass} (not asserted)"
    else:
        ok, verdict, reason = False, "FAIL", f"unknown expect='{expect}'"

    return {
        "name": case["name"], "expect": expect, "milestone": case["milestone"],
        "note": case["note"], "cmd": " ".join(cmd), "exit": rc, "n_pass": n_pass,
        "n_fail": n_fail, "verdict": verdict, "ok": ok, "reason": reason,
        "info": key_info(out, expect), "output": out,
    }


def run_repro_checks(binary, timeout):
    """M6 core: each deterministic case run twice must produce byte-identical dQ.
    Runs in a temp cwd (harness writes dq_dev.dat there), returns list of result dicts."""
    import tempfile, filecmp, shutil
    results = []
    for name, extra in REPRO_CASES:
        cmd = [binary] + _DET_REPRO_COMMON + extra
        ok, reason = True, ""
        with tempfile.TemporaryDirectory() as wd:
            try:
                p1 = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=wd)
                shutil.copy(os.path.join(wd, "dq_dev.dat"), os.path.join(wd, "a.dat"))
                p2 = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=wd)
                shutil.copy(os.path.join(wd, "dq_dev.dat"), os.path.join(wd, "b.dat"))
                if p1.returncode != 0 or p2.returncode != 0:
                    ok, reason = False, f"nonzero exit ({p1.returncode}/{p2.returncode})"
                elif not filecmp.cmp(os.path.join(wd, "a.dat"), os.path.join(wd, "b.dat"),
                                     shallow=False):
                    ok, reason = False, "dQ differs between two runs (NOT bit-reproducible)"
            except Exception as e:  # noqa
                ok, reason = False, f"exception: {e}"
        results.append({"name": name, "verdict": "PASS" if ok else "FAIL",
                        "reason": reason, "cmd": " ".join(cmd)})
    return results


def do_build(build_dir):
    print(f"[build] cmake --build {build_dir}/build --target {BUILD_TARGET}")
    r = subprocess.run(
        ["cmake", "--build", os.path.join(build_dir, "build"), "--target", BUILD_TARGET,
         "-j", str(os.cpu_count() or 4)],
        capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stdout + r.stderr)
        sys.stderr.write("\n[build] FAILED\n")
        return False
    print("[build] ok")
    return True


def main():
    ap = argparse.ArgumentParser(description="HSTU bwd regression test runner")
    ap.add_argument("--bin", default=DEFAULT_BIN, help="path to bwd harness binary")
    ap.add_argument("--build", action="store_true",
                    help="cmake --build the bwd target before testing")
    ap.add_argument("--build-dir", default=DEFAULT_BUILD_DIR, help="ck_hstu repo root")
    ap.add_argument("--filter", default="", help="only run cases whose name contains this substring")
    ap.add_argument("--timeout", type=int, default=120, help="per-case timeout in seconds")
    ap.add_argument("--log-dir", default=LOG_DIR, help="dir for timestamped result log")
    args = ap.parse_args()

    if args.build and not do_build(args.build_dir):
        return 3

    if not (os.path.isfile(args.bin) and os.access(args.bin, os.X_OK)):
        sys.stderr.write(
            f"ERROR: binary not found or not executable: {args.bin}\n"
            f"  build it with:  cmake --build {args.build_dir}/build "
            f"--target {BUILD_TARGET} -j\n"
            f"  or pass --build / --bin <path>\n")
        return 4

    cases = [c for c in MATRIX if args.filter in c["name"]]
    if not cases:
        sys.stderr.write(f"ERROR: no cases match --filter '{args.filter}'\n")
        return 4

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(args.log_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, f"test-{stamp}.log")

    n_pass_cases = sum(1 for c in cases if c["expect"] == "pass")
    n_rej_cases = sum(1 for c in cases if c["expect"] == "reject")
    n_skip_cases = sum(1 for c in cases if c["expect"] == "skip")

    header = (f"HSTU bwd regression — {stamp}\n"
              f"binary : {args.bin}\n"
              f"matrix : {len(cases)} cases  (pass={n_pass_cases} reject={n_rej_cases} "
              f"skip={n_skip_cases})  timeout={args.timeout}s\n"
              + "=" * 96)
    print(header)

    results = []
    for c in cases:
        r = run_case(args.bin, c, args.timeout)
        results.append(r)
        tag = {"PASS": "PASS", "FAIL": "FAIL", "SKIP": "N/A "}[r["verdict"]]
        line = (f"[{tag}] {r['name']:<26} expect={r['expect']:<6} {r['milestone']:<3} "
                f"exit={str(r['exit']):<5} P/F={r['n_pass']}/{r['n_fail']}")
        print(line)
        print(f"        {r['info']}")
        if r["reason"]:
            print(f"        -> {r['reason']}")

    # M6 bit-reproducibility checks (only when not filtering, or filter matches "repro"/"det")
    repro_results = []
    if (not args.filter) or any(k in args.filter for k in ("repro", "det", "M6")):
        print("-" * 96)
        print("BIT-REPRODUCIBILITY (M6): deterministic dQ must be byte-identical across two runs")
        repro_results = run_repro_checks(args.bin, args.timeout)
        for r in repro_results:
            tag = "PASS" if r["verdict"] == "PASS" else "FAIL"
            print(f"[{tag}] {r['name']:<26} {('byte-identical' if r['verdict']=='PASS' else r['reason'])}")

    passed = sum(1 for r in results if r["verdict"] == "PASS") \
        + sum(1 for r in repro_results if r["verdict"] == "PASS")
    failed = sum(1 for r in results if r["verdict"] == "FAIL") \
        + sum(1 for r in repro_results if r["verdict"] == "FAIL")
    skipped = sum(1 for r in results if r["verdict"] == "SKIP")
    total = len(results) + len(repro_results)

    summary = ("=" * 96 + "\n"
               f"TOTAL {total}   PASSED {passed}   FAILED {failed}   SKIPPED {skipped}\n"
               f"RESULT: {'OK (all expectations met)' if failed == 0 else 'FAILURES PRESENT'}")
    print(summary)

    # full log (every case's full captured output)
    with open(log_path, "w") as f:
        f.write(header + "\n")
        for r in results:
            f.write("\n" + "-" * 96 + "\n")
            f.write(f"CASE   {r['name']}  (expect={r['expect']}, {r['milestone']}) -> {r['verdict']}\n")
            f.write(f"CMD    {r['cmd']}\n")
            f.write(f"EXIT   {r['exit']}   PASS={r['n_pass']} FAIL={r['n_fail']}\n")
            if r["note"]:
                f.write(f"NOTE   {r['note']}\n")
            if r["reason"]:
                f.write(f"REASON {r['reason']}\n")
            f.write("OUTPUT:\n")
            f.write(r["output"].rstrip() + "\n")
        f.write("\n" + summary + "\n")
    print(f"\nlog: {log_path}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
