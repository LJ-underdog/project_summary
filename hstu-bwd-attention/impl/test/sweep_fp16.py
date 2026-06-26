#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""M7a fp16 correctness sweep. Mirrors the M5/M5b/M6/M6b bf16-validated configs
with -prec=fp16 -attn_scale=1.0, runs the bwd harness per config, and records
PASS/FAIL + dQ/dK/dV max_abs_err vs max|ref|. No tolerance is touched here; the
harness uses the templated fp16 elimit (rtol5e-3/atol1e-2). Honest: a FAIL is
reported as a FAIL, never re-tuned away."""
import re, subprocess, sys, datetime, os

BIN = "/root/workspace/ck_hstu/build/bin/tile_example_hstu_attention_bwd"
LOG = "/root/workspace/hstu-bwd-impl/runs/run-M7a-sweep.log"
COMMON = ["-prec=fp16", "-hdim_qk=64", "-hdim_v=64", "-v=1", "-attn_scale=1.0"]
ERR_RE = re.compile(r"^\s*(d[QKV]): max_abs_err=(\S+) mean_abs_err=(\S+) \(max\|ref\|=(\S+)\)")
PASS_RE = re.compile(r"\[PASS\]"); FAIL_RE = re.compile(r"\[FAIL\]")

def c(name, extra, note=""):
    return {"name": name, "args": COMMON + extra, "note": note}

CASES = [
  # ---- SiLU batched: causal {0,1} x 5 factors individual + combo ----
  c("silu-b-c0-nomask",   ["-softmax=0","-causal=0","-b=2","-nhead=2","-seqlens=128"]),
  c("silu-b-c1-causal",   ["-softmax=0","-causal=1","-b=2","-nhead=2","-seqlens=128"]),
  c("silu-b-c1-window",   ["-softmax=0","-causal=1","-b=2","-nhead=2","-seqlens=128","-local_len=16"]),
  c("silu-b-c1-context",  ["-softmax=0","-causal=1","-b=2","-nhead=2","-seqlens=128","-context_len=8"]),
  c("silu-b-c1-minfull",  ["-softmax=0","-causal=1","-b=2","-nhead=2","-seqlens=128","-minfull_len=16"]),
  c("silu-b-c1-numtarget",["-softmax=0","-causal=1","-b=2","-nhead=2","-seqlens=128","-targets=16"]),
  c("silu-b-c1-combo",    ["-softmax=0","-causal=1","-b=2","-nhead=2","-seqlens=128","-local_len=16","-context_len=8","-minfull_len=16","-targets=16"]),
  c("silu-b-c0-target",   ["-softmax=0","-causal=0","-b=2","-nhead=2","-seqlens=128","-targets=8"], "P1-1 cross: causal=0+num_target"),
  c("silu-b-c0-window",   ["-softmax=0","-causal=0","-b=2","-nhead=2","-seqlens=128","-local_len=16"]),
  c("silu-b-c0-combo",    ["-softmax=0","-causal=0","-b=2","-nhead=2","-seqlens=128","-local_len=16","-context_len=8","-minfull_len=16","-targets=8"]),
  # non-divisible / single / tiny
  c("silu-b-seq200",      ["-softmax=0","-causal=1","-b=2","-nhead=2","-seqlens=200"], "non kN0=128 divisible"),
  c("silu-b-seq130",      ["-softmax=0","-causal=1","-b=2","-nhead=2","-seqlens=130"], "non kM0=32 divisible"),
  c("silu-b1-seq512",     ["-softmax=0","-causal=1","-b=1","-nhead=1","-seqlens=512"], "single batch"),
  c("silu-b-default-scale",["-softmax=0","-causal=0","-b=2","-nhead=2","-seqlens=128","-attn_scale=0"], "default scale (override attn_scale)"),

  # ---- softmax batched: causal {0,1} x factors ----
  c("sm-b-c0-nomask",   ["-softmax=1","-causal=0","-b=2","-nhead=2","-seqlens=128"]),
  c("sm-b-c1-causal",   ["-softmax=1","-causal=1","-b=2","-nhead=2","-seqlens=128"]),
  c("sm-b-c1-window",   ["-softmax=1","-causal=1","-b=2","-nhead=2","-seqlens=128","-local_len=16"]),
  c("sm-b-c0-window",   ["-softmax=1","-causal=0","-b=2","-nhead=2","-seqlens=128","-local_len=16"]),
  c("sm-b-c1-context",  ["-softmax=1","-causal=1","-b=2","-nhead=2","-seqlens=128","-context_len=8"]),
  c("sm-b-c1-minfull",  ["-softmax=1","-causal=1","-b=2","-nhead=2","-seqlens=128","-minfull_len=16"]),
  c("sm-b-c1-numtarget",["-softmax=1","-causal=1","-b=2","-nhead=2","-seqlens=128","-targets=8"]),
  c("sm-b-c0-numtarget",["-softmax=1","-causal=0","-b=2","-nhead=2","-seqlens=128","-targets=8"], "P1-1 cross softmax"),
  c("sm-b-c1-combo",    ["-softmax=1","-causal=1","-b=2","-nhead=2","-seqlens=128","-local_len=16","-context_len=8","-minfull_len=16","-targets=8"]),
  c("sm-b-c1-seq200",   ["-softmax=1","-causal=1","-b=3","-nhead=4","-seqlens=200"], "non-divisible"),

  # ---- SiLU jagged ----
  c("silu-j-c0-varying", ["-softmax=0","-causal=0","-b=3","-nhead=2","-jagged=1","-seqlens=128,200,96"]),
  c("silu-j-c1-causal",  ["-softmax=0","-causal=1","-b=3","-nhead=2","-jagged=1","-seqlens=128,200,96"]),
  c("silu-j-c1-window",  ["-softmax=0","-causal=1","-b=4","-nhead=4","-jagged=1","-seqlens=128,256,200,96","-local_len=16"]),
  c("silu-j-c1-numtgt",  ["-softmax=0","-causal=1","-b=3","-nhead=2","-jagged=1","-seqlens=128,200,96","-targets=8,24,16"]),
  c("silu-j-c1-combo",   ["-softmax=0","-causal=1","-b=4","-nhead=4","-jagged=1","-seqlens=256,200,128,96","-local_len=24","-context_len=12","-minfull_len=20","-targets=12,8,16,4"]),
  c("silu-j-single",     ["-softmax=0","-causal=1","-b=1","-nhead=2","-jagged=1","-seqlens=300"], "single jagged"),
  c("silu-j-spread",     ["-softmax=0","-causal=0","-b=3","-nhead=2","-jagged=1","-seqlens=512,32,256"], "large spread"),
  c("silu-j-tiny",       ["-softmax=0","-causal=1","-b=3","-nhead=2","-jagged=1","-seqlens=1,128,7"], "tiny seqlens 1,7"),

  # ---- softmax jagged ----
  c("sm-j-c0-varying", ["-softmax=1","-causal=0","-b=3","-nhead=2","-jagged=1","-seqlens=128,200,96"]),
  c("sm-j-c1-causal",  ["-softmax=1","-causal=1","-b=3","-nhead=2","-jagged=1","-seqlens=128,200,96"]),
  c("sm-j-c1-window",  ["-softmax=1","-causal=1","-b=3","-nhead=2","-jagged=1","-seqlens=128,200,96","-local_len=16"]),
  c("sm-j-c1-numtgt",  ["-softmax=1","-causal=1","-b=3","-nhead=2","-jagged=1","-seqlens=128,200,96","-targets=8,24,16"]),
  c("sm-j-c0-numtgt",  ["-softmax=1","-causal=0","-b=3","-nhead=2","-jagged=1","-seqlens=128,200,96","-targets=8,24,16"], "P1-1 jagged softmax"),
  c("sm-j-c1-combo",   ["-softmax=1","-causal=1","-b=4","-nhead=4","-jagged=1","-seqlens=256,200,128,96","-local_len=24","-context_len=12","-minfull_len=20","-targets=12,8,16,4"]),

  # ---- SiLU group ----
  c("silu-g2-c0",      ["-softmax=0","-causal=0","-b=4","-nhead=2","-g=2","-seqlens=128,200,96,160"]),
  c("silu-g2-c1",      ["-softmax=0","-causal=1","-b=4","-nhead=2","-g=2","-seqlens=128,200,96,160"]),
  c("silu-g2-window",  ["-softmax=0","-causal=1","-b=4","-nhead=4","-g=2","-seqlens=128,200,96,160","-g_local_lens=16,0"], "mixed with/without-local"),
  c("silu-g2-hetero",  ["-softmax=0","-causal=1","-b=4","-nhead=4","-g=2","-seqlens=128,200,96,160","-g_local_lens=16,0","-g_context_lens=8,0","-g_minfull_lens=16,0","-g_attn_scales=1.0,0.5","-targets=8,24,0,16"]),
  c("silu-g2-c0-target",["-softmax=0","-causal=0","-b=4","-nhead=2","-g=2","-seqlens=128,200,96,160","-targets=8,24,0,16"], "group P1-1 cross"),
  c("silu-g3",         ["-softmax=0","-causal=1","-b=6","-nhead=2","-g=3","-seqlens=128,200,96,160,64,256","-g_local_lens=16,0,32","-g_attn_scales=1.0,0.5,0"]),
  c("silu-g4-singleton",["-softmax=0","-causal=1","-b=4","-nhead=2","-g=4","-seqlens=128,200,96,160","-g_local_lens=0,16,0,32","-g_context_lens=0,8,4,0","-g_attn_scales=1.0,0.5,0,2.0"]),

  # ---- softmax group ----
  c("sm-g2-c1",        ["-softmax=1","-causal=1","-b=4","-nhead=2","-g=2","-seqlens=128,200,96,160"]),
  c("sm-g2-c0",        ["-softmax=1","-causal=0","-b=4","-nhead=2","-g=2","-seqlens=128,200,96,160"]),
  c("sm-g2-window",    ["-softmax=1","-causal=1","-b=4","-nhead=4","-g=2","-seqlens=128,200,96,160","-g_local_lens=16,0"]),
  c("sm-g2-hetero",    ["-softmax=1","-causal=1","-b=4","-nhead=4","-g=2","-seqlens=128,200,96,160","-g_local_lens=16,0","-g_context_lens=8,0","-g_minfull_lens=16,0","-g_attn_scales=1.0,0.5","-targets=8,24,0,16"]),
  c("sm-g2-c0-target", ["-softmax=1","-causal=0","-b=4","-nhead=2","-g=2","-seqlens=128,200,96,160","-targets=8,24,0,16"], "group softmax P1-1"),
  c("sm-g3",           ["-softmax=1","-causal=1","-b=6","-nhead=2","-g=3","-seqlens=128,200,96,160,64,256","-g_local_lens=16,0,32","-g_attn_scales=1.0,0.5,0"]),
  c("sm-g4-singleton", ["-softmax=1","-causal=1","-b=4","-nhead=2","-g=4","-seqlens=128,200,96,160","-g_local_lens=0,16,0,32","-g_context_lens=0,8,4,0","-g_attn_scales=1.0,0.5,0,2.0"]),

  # ---- M6b group_max_seqlens_q under-cover TRIGGER config (the old harness hole) ----
  c("gtrig-sm",        ["-softmax=1","-causal=1","-b=4","-nhead=4","-g=2","-seqlens=128,200,96,160","-g_local_lens=16,16","-targets=8,24,8,16"], "M6b trigger: long batch big target+window"),
  c("gtrig-silu",      ["-softmax=0","-causal=1","-b=4","-nhead=4","-g=2","-seqlens=128,200,96,160","-g_local_lens=16,16","-targets=8,24,8,16"], "M6b trigger SiLU"),

  # ---- deterministic no_group ----
  c("det-silu-b-c0",   ["-deterministic=1","-softmax=0","-causal=0","-b=2","-nhead=2","-seqlens=128"]),
  c("det-silu-b-combo",["-deterministic=1","-softmax=0","-causal=1","-b=2","-nhead=2","-seqlens=128","-local_len=16","-context_len=8","-minfull_len=16","-targets=8"]),
  c("det-silu-b-seq512",["-deterministic=1","-softmax=0","-causal=1","-b=2","-nhead=4","-seqlens=512"], "multi-split (4)"),
  c("det-silu-j-c1",   ["-deterministic=1","-softmax=0","-causal=1","-b=3","-nhead=2","-jagged=1","-seqlens=128,200,96"]),
  c("det-sm-b-c0",     ["-deterministic=1","-softmax=1","-causal=0","-b=2","-nhead=2","-seqlens=128"]),
  c("det-sm-b-window", ["-deterministic=1","-softmax=1","-causal=1","-b=2","-nhead=2","-seqlens=128","-local_len=16"]),
  c("det-sm-j-numtgt", ["-deterministic=1","-softmax=1","-causal=1","-b=3","-nhead=2","-jagged=1","-seqlens=128,200,96","-targets=8,24,16"]),

  # ---- deterministic group ----
  c("gdet-silu-g2",    ["-deterministic=1","-softmax=0","-causal=1","-b=4","-nhead=2","-g=2","-seqlens=128,200,96,160"]),
  c("gdet-silu-g2-512",["-deterministic=1","-softmax=0","-causal=1","-b=2","-nhead=4","-g=2","-seqlens=512,300"], "multi-split"),
  c("gdet-sm-g2",      ["-deterministic=1","-softmax=1","-causal=1","-b=4","-nhead=2","-g=2","-seqlens=128,200,96,160"]),
  c("gdet-sm-g2-window",["-deterministic=1","-softmax=1","-causal=1","-b=4","-nhead=4","-g=2","-seqlens=128,200,96,160","-g_local_lens=16,0"]),
  c("gdet-sm-g3",      ["-deterministic=1","-softmax=1","-causal=1","-b=6","-nhead=2","-g=3","-seqlens=128,200,96,160,64,256","-g_local_lens=16,0,32","-g_attn_scales=1.0,0.5,0"]),
]

def run(case):
    p = subprocess.run([BIN]+case["args"], capture_output=True, text=True, timeout=180)
    out = (p.stdout or "")+(p.stderr or "")
    errs = {}
    for line in out.splitlines():
        m = ERR_RE.match(line)
        if m: errs[m.group(1)] = (m.group(2), m.group(4))  # tensor -> (max_abs_err, max|ref|)
    npass = len(PASS_RE.findall(out)); nfail = len(FAIL_RE.findall(out))
    ok = (p.returncode == 0 and npass == 3 and nfail == 0)
    return ok, p.returncode, npass, nfail, errs, out

def main():
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    lines = [f"M7a fp16 sweep — {stamp}", f"binary: {BIN}", f"cases: {len(CASES)}  (all -prec=fp16 -attn_scale=1.0; fp16 elimit rtol5e-3/atol1e-2)", "="*100]
    print("\n".join(lines))
    npass_c = nfail_c = 0
    full = []
    for cse in CASES:
        try:
            ok, rc, npass, nfail, errs, out = run(cse)
        except subprocess.TimeoutExpired:
            ok, rc, npass, nfail, errs, out = False, None, 0, 0, {}, "<<TIMEOUT>>"
        tag = "PASS" if ok else "FAIL"
        if ok: npass_c += 1
        else: nfail_c += 1
        es = "  ".join(f"{t}:err={errs[t][0]} |ref|={errs[t][1]}" for t in ("dQ","dK","dV") if t in errs)
        line = f"[{tag}] {cse['name']:<22} exit={str(rc):<4} P/F={npass}/{nfail}  {es}"
        print(line)
        if cse["note"]: print(f"        note: {cse['note']}")
        full.append((cse, tag, rc, npass, nfail, es, out))
    summary = "="*100 + f"\nTOTAL {len(CASES)}  PASS {npass_c}  FAIL {nfail_c}\nRESULT: {'ALL PASS' if nfail_c==0 else 'FAILURES PRESENT'}"
    print(summary)
    with open(LOG, "w") as f:
        f.write("\n".join(lines)+"\n")
        for cse, tag, rc, npass, nfail, es, out in full:
            f.write("\n"+"-"*100+f"\nCASE {cse['name']} -> {tag}\nCMD  {' '.join([BIN]+cse['args'])}\nEXIT {rc} P/F={npass}/{nfail}\n{es}\n")
            if cse["note"]: f.write(f"NOTE {cse['note']}\n")
            f.write("OUTPUT:\n"+out.rstrip()+"\n")
        f.write("\n"+summary+"\n")
    print(f"\nlog: {LOG}")
    return 0 if nfail_c==0 else 1

if __name__ == "__main__":
    sys.exit(main())
