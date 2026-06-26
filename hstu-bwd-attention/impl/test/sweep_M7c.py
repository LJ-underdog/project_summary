#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""M7c Stage-2 batched poison-pad sweep (no_group). Runs the asymmetric / non-canonical
(hdim_qk, hdim_v) pair matrix with -poison_pad=1 so OOB head-dim load-zero / store-skip is
POSITIVELY proven (NaN-filled input pad tails + pre-poisoned output tails -> any leak = NaN =
hard FAIL). Tolerance is the harness templated elimit (bf16 2e-2/5e-2, fp16 5e-3/1e-2), NOT
loosened. Every pair includes a P1-1 cross (causal=0 + num_target). group is Stage 3."""
import re, subprocess, sys, datetime

BIN = "/root/workspace/ck_hstu/build/bin/tile_example_hstu_attention_bwd"
LOG = "/root/workspace/hstu-bwd-impl/runs/run-M7c-stage2-sweep.log"
DTYPES = ["bf16", "fp16"]
FAIL_RE = re.compile(r"\[FAIL\]")
NP_RE = re.compile(r"numeric_pass=(true|false)")
SK_RE = re.compile(r"\[(PASS|FAIL)\] store-skip")
ERR_RE = re.compile(r"^\s*(d[QKV]): max_abs_err=(\S+) .* \(max\|ref\|=(\S+)\)")

# (hdim_qk, hdim_v) pairs (draft §6): asymmetric-canonical, non-canonical-symmetric,
# asymmetric+non-canonical, both directions 64/128 & 128/64.
PAIRS = [
    (64, 128), (128, 64), (96, 256), (128, 256),     # asymmetric-canonical
    (80, 80), (48, 48), (192, 192), (100, 100),      # non-canonical-symmetric
    (80, 128), (100, 64), (48, 96), (192, 256),      # asymmetric + non-canonical
]

def sub(hq, hv, dt):
    base = [f"-prec={dt}", f"-hdim_qk={hq}", f"-hdim_v={hv}", "-v=1", "-attn_scale=1.0",
            "-poison_pad=1", "-b=2", "-nhead=2"]
    def c(name, extra): return {"name": f"{hq}x{hv}-{dt}-{name}", "args": base + extra}
    return [
        c("silu-c1",         ["-softmax=0", "-causal=1", "-seqlens=128"]),
        c("sm-c1",           ["-softmax=1", "-causal=1", "-seqlens=128"]),
        c("silu-c0-target",  ["-softmax=0", "-causal=0", "-seqlens=128", "-targets=8"]),  # P1-1
        c("sm-c0-target",    ["-softmax=1", "-causal=0", "-seqlens=128", "-targets=8"]),  # P1-1
        c("silu-c1-combo",   ["-softmax=0", "-causal=1", "-seqlens=200",
                              "-local_len=16", "-context_len=8", "-minfull_len=16", "-targets=8"]),
        c("sm-j-c1",         ["-softmax=1", "-causal=1", "-jagged=1", "-b=3", "-nhead=2",
                              "-seqlens=128,200,96", "-targets=8,24,16"]),
        c("det-sm-c1",       ["-softmax=1", "-causal=1", "-nhead=4", "-seqlens=512",
                              "-deterministic=1"]),  # determ lock (hd256 pairs -> kN0=64)
    ]

def run(case):
    p = subprocess.run([BIN] + case["args"], capture_output=True, text=True, timeout=240)
    out = (p.stdout or "") + (p.stderr or "")
    np_m = NP_RE.search(out); sk_m = SK_RE.search(out)
    errs = {m.group(1): (m.group(2), m.group(3)) for m in
            (ERR_RE.match(l) for l in out.splitlines()) if m}
    nan = ("nan" in out.lower() or "inf" in out.lower())
    ok = (p.returncode == 0 and np_m and np_m.group(1) == "true" and not FAIL_RE.search(out))
    sk = sk_m.group(1) if sk_m else "n/a"
    return ok, p.returncode, sk, errs, out

def main():
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    cases = [c for (hq, hv) in PAIRS for dt in DTYPES for c in sub(hq, hv, dt)]
    head = [f"M7c Stage-2 batched poison-pad sweep — {stamp}", f"binary: {BIN}",
            f"cases: {len(cases)}  pairs={PAIRS}  (-poison_pad=1; elimit NOT loosened)", "="*108]
    print("\n".join(head))
    npass = nfail = 0; full = []
    for cse in cases:
        try:
            ok, rc, sk, errs, out = run(cse)
        except subprocess.TimeoutExpired:
            ok, rc, sk, errs, out = False, None, "timeout", {}, "<<TIMEOUT>>"
        npass += ok; nfail += (not ok)
        es = "  ".join(f"{t}:e={errs[t][0]}|ref|={errs[t][1]}" for t in ("dQ","dK","dV") if t in errs)
        print(f"[{'PASS' if ok else 'FAIL'}] {cse['name']:<26} exit={str(rc):<4} store-skip={sk:<5} {es}")
        full.append((cse, ok, rc, sk, es, out))
    summ = "="*108 + f"\nTOTAL {len(cases)}  PASS {npass}  FAIL {nfail}\nRESULT: {'ALL PASS' if nfail==0 else 'FAILURES PRESENT'}"
    print(summ)
    with open(LOG, "w") as f:
        f.write("\n".join(head)+"\n")
        for cse, ok, rc, sk, es, out in full:
            f.write("\n"+"-"*108+f"\nCASE {cse['name']} -> {'PASS' if ok else 'FAIL'}\n"
                    f"CMD  {' '.join([BIN]+cse['args'])}\nEXIT {rc} store-skip={sk}\n{es}\n"
                    "OUTPUT:\n"+out.rstrip()+"\n")
        f.write("\n"+summ+"\n")
    print(f"\nlog: {LOG}")
    return 0 if nfail==0 else 1

if __name__ == "__main__":
    sys.exit(main())
