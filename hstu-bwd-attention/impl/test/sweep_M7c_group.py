#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""M7c Stage-3 GROUP poison-pad sweep. g{2,3,4} x asymmetric/non-canonical (hdim_qk,hdim_v)
pair x {bf16,fp16} x {SiLU,softmax} x P1-1 cross + group determ, all -poison_pad=1 -> positively
prove group OOB head-dim load-zero / store-skip. Tolerance NOT loosened."""
import re, subprocess, sys, datetime

BIN = "/root/workspace/ck_hstu/build/bin/tile_example_hstu_attention_bwd"
LOG = "/root/workspace/hstu-bwd-impl/runs/run-M7c-stage3-group-sweep.log"
DTYPES = ["bf16", "fp16"]
FAIL_RE = re.compile(r"\[FAIL\]")
NP_RE = re.compile(r"numeric_pass=(true|false)")
SK_RE = re.compile(r"\[(PASS|FAIL)\] store-skip")
ERR_RE = re.compile(r"^\s*(d[QKV]): max_abs_err=(\S+) .* \(max\|ref\|=(\S+)\)")

PAIRS = [(64, 128), (128, 64), (80, 80), (100, 64), (48, 96), (192, 256)]

def sub(hq, hv, dt):
    base = [f"-prec={dt}", f"-hdim_qk={hq}", f"-hdim_v={hv}", "-v=1", "-attn_scale=1.0",
            "-poison_pad=1"]
    def c(name, extra): return {"name": f"{hq}x{hv}-{dt}-{name}", "args": base + extra}
    return [
        c("g2-silu-c1",       ["-softmax=0", "-causal=1", "-b=4", "-nhead=2", "-g=2",
                               "-seqlens=128,200,96,160"]),
        c("g2-sm-c1",         ["-softmax=1", "-causal=1", "-b=4", "-nhead=2", "-g=2",
                               "-seqlens=128,200,96,160"]),
        c("g2-silu-c0-target",["-softmax=0", "-causal=0", "-b=4", "-nhead=2", "-g=2",
                               "-seqlens=128,200,96,160", "-targets=8,24,0,16"]),    # P1-1
        c("g2-sm-c0-target",  ["-softmax=1", "-causal=0", "-b=4", "-nhead=2", "-g=2",
                               "-seqlens=128,200,96,160", "-targets=8,24,0,16"]),    # P1-1
        c("g2-silu-hetero",   ["-softmax=0", "-causal=1", "-b=4", "-nhead=4", "-g=2",
                               "-seqlens=128,200,96,160", "-g_local_lens=16,0",
                               "-g_context_lens=8,0", "-g_minfull_lens=16,0",
                               "-g_attn_scales=1.0,0.5", "-targets=8,24,0,16"]),
        c("g3-sm-c1",         ["-softmax=1", "-causal=1", "-b=6", "-nhead=2", "-g=3",
                               "-seqlens=128,200,96,160,64,256", "-g_local_lens=16,0,32"]),
        c("g4-silu-c1",       ["-softmax=0", "-causal=1", "-b=4", "-nhead=2", "-g=4",
                               "-seqlens=128,200,96,160", "-g_local_lens=0,16,0,32"]),
        c("g2-det-sm-c1",     ["-softmax=1", "-causal=1", "-b=2", "-nhead=4", "-g=2",
                               "-seqlens=512,300", "-deterministic=1"]),            # determ lock
    ]

def run(case):
    p = subprocess.run([BIN] + case["args"], capture_output=True, text=True, timeout=300)
    out = (p.stdout or "") + (p.stderr or "")
    np_m = NP_RE.search(out); sk_m = SK_RE.search(out)
    errs = {m.group(1): (m.group(2), m.group(3)) for m in
            (ERR_RE.match(l) for l in out.splitlines()) if m}
    ok = (p.returncode == 0 and np_m and np_m.group(1) == "true" and not FAIL_RE.search(out))
    return ok, p.returncode, (sk_m.group(1) if sk_m else "n/a"), errs, out

def main():
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    cases = [c for (hq, hv) in PAIRS for dt in DTYPES for c in sub(hq, hv, dt)]
    head = [f"M7c Stage-3 GROUP poison-pad sweep — {stamp}", f"binary: {BIN}",
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
        print(f"[{'PASS' if ok else 'FAIL'}] {cse['name']:<28} exit={str(rc):<4} store-skip={sk:<5} {es}")
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
