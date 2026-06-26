#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""M7b multi-hdim correctness sweep. For each symmetric hdim in {64,96,128,256} x
{bf16,fp16} runs a representative cross of {SiLU,softmax} x causal{0,1} x mask x
{batched,jagged,group} x {atomic,determ}, all -attn_scale=1.0. Records PASS/FAIL +
dQ/dK/dV max_abs_err vs max|ref|. Tolerance is the harness templated elimit
(bf16 2e-2/5e-2, fp16 5e-3/1e-2) — NOT touched here; a FAIL is reported honestly.

P1-1 cross (causal=0 + num_target) is included for EVERY hdim (batched + group)."""
import re, subprocess, sys, datetime

BIN = "/root/workspace/ck_hstu/build/bin/tile_example_hstu_attention_bwd"
LOG = "/root/workspace/hstu-bwd-impl/runs/run-M7b-sweep.log"
HDIMS = [64, 96, 128, 256]
DTYPES = ["bf16", "fp16"]
ERR_RE = re.compile(r"^\s*(d[QKV]): max_abs_err=(\S+) mean_abs_err=(\S+) \(max\|ref\|=(\S+)\)")
PASS_RE = re.compile(r"\[PASS\]"); FAIL_RE = re.compile(r"\[FAIL\]")


def gen(hdim, dtype):
    """Representative config list for one (hdim, dtype)."""
    base = [f"-prec={dtype}", f"-hdim_qk={hdim}", f"-hdim_v={hdim}", "-v=1", "-attn_scale=1.0"]
    def c(name, extra, note=""):
        return {"name": f"h{hdim}-{dtype}-{name}", "args": base + extra, "note": note}
    return [
        # SiLU batched: causal{0,1}, combo, P1-1 cross
        c("silu-b-c1",        ["-softmax=0","-causal=1","-b=2","-nhead=2","-seqlens=128"]),
        c("silu-b-c0",        ["-softmax=0","-causal=0","-b=2","-nhead=2","-seqlens=128"]),
        c("silu-b-c1-combo",  ["-softmax=0","-causal=1","-b=2","-nhead=2","-seqlens=128","-local_len=16","-context_len=8","-minfull_len=16","-targets=8"]),
        c("silu-b-c0-target", ["-softmax=0","-causal=0","-b=2","-nhead=2","-seqlens=128","-targets=8"], "P1-1 cross"),
        c("silu-b-seq200",    ["-softmax=0","-causal=1","-b=2","-nhead=2","-seqlens=200"], "non-divisible"),
        # softmax batched: causal, P1-1 cross
        c("sm-b-c1",          ["-softmax=1","-causal=1","-b=2","-nhead=2","-seqlens=128"]),
        c("sm-b-c0-target",   ["-softmax=1","-causal=0","-b=2","-nhead=2","-seqlens=128","-targets=8"], "P1-1 cross softmax"),
        c("sm-b-c1-combo",    ["-softmax=1","-causal=1","-b=2","-nhead=2","-seqlens=128","-local_len=16","-context_len=8","-minfull_len=16","-targets=8"]),
        # jagged
        c("silu-j-combo",     ["-softmax=0","-causal=1","-b=4","-nhead=4","-jagged=1","-seqlens=256,200,128,96","-local_len=24","-context_len=12","-minfull_len=20","-targets=12,8,16,4"]),
        c("sm-j-numtgt",      ["-softmax=1","-causal=1","-b=3","-nhead=2","-jagged=1","-seqlens=128,200,96","-targets=8,24,16"]),
        # group
        c("silu-g2-hetero",   ["-softmax=0","-causal=1","-b=4","-nhead=4","-g=2","-seqlens=128,200,96,160","-g_local_lens=16,0","-g_context_lens=8,0","-g_minfull_lens=16,0","-g_attn_scales=1.0,0.5","-targets=8,24,0,16"]),
        c("sm-g2-c1",         ["-softmax=1","-causal=1","-b=4","-nhead=2","-g=2","-seqlens=128,200,96,160"]),
        c("sm-g2-c0-target",  ["-softmax=1","-causal=0","-b=4","-nhead=2","-g=2","-seqlens=128,200,96,160","-targets=8,24,0,16"], "group P1-1 cross"),
        # deterministic (multi-split; hd256 exercises kN0=64 split path)
        c("det-silu-b-512",   ["-deterministic=1","-softmax=0","-causal=1","-b=2","-nhead=4","-seqlens=512"], "determ multi-split"),
        c("det-sm-j-numtgt",  ["-deterministic=1","-softmax=1","-causal=1","-b=3","-nhead=2","-jagged=1","-seqlens=128,200,96","-targets=8,24,16"]),
        c("gdet-sm-g2",       ["-deterministic=1","-softmax=1","-causal=1","-b=4","-nhead=2","-g=2","-seqlens=128,200,96,160"], "group determ"),
    ]


def run(case):
    p = subprocess.run([BIN]+case["args"], capture_output=True, text=True, timeout=240)
    out = (p.stdout or "")+(p.stderr or "")
    errs = {}
    for line in out.splitlines():
        m = ERR_RE.match(line)
        if m: errs[m.group(1)] = (m.group(2), m.group(4))
    npass = len(PASS_RE.findall(out)); nfail = len(FAIL_RE.findall(out))
    ok = (p.returncode == 0 and npass == 3 and nfail == 0)
    return ok, p.returncode, npass, nfail, errs, out


def main():
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    cases = [cs for hd in HDIMS for dt in DTYPES for cs in gen(hd, dt)]
    head = [f"M7b multi-hdim sweep — {stamp}", f"binary: {BIN}",
            f"cases: {len(cases)}  hdims={HDIMS} dtypes={DTYPES}  (-attn_scale=1.0; "
            f"elimit bf16 2e-2/5e-2, fp16 5e-3/1e-2)", "="*108]
    print("\n".join(head))
    npass_c = nfail_c = 0; full = []; per_hd = {}
    for cse in cases:
        try:
            ok, rc, npass, nfail, errs, out = run(cse)
        except subprocess.TimeoutExpired:
            ok, rc, npass, nfail, errs, out = False, None, 0, 0, {}, "<<TIMEOUT>>"
        tag = "PASS" if ok else "FAIL"
        npass_c += ok; nfail_c += (not ok)
        hd = cse["name"].split("-")[0]
        per_hd.setdefault(hd, [0,0]); per_hd[hd][0 if ok else 1] += 1
        es = "  ".join(f"{t}:err={errs[t][0]} |ref|={errs[t][1]}" for t in ("dQ","dK","dV") if t in errs)
        print(f"[{tag}] {cse['name']:<26} exit={str(rc):<4} P/F={npass}/{nfail}  {es}")
        if cse["note"]: print(f"        note: {cse['note']}")
        full.append((cse, tag, rc, npass, nfail, es, out))
    summ = "="*108 + "\nper-hdim/dtype: " + "  ".join(f"{k}:{v[0]}P/{v[1]}F" for k,v in sorted(per_hd.items())) \
        + f"\nTOTAL {len(cases)}  PASS {npass_c}  FAIL {nfail_c}\nRESULT: {'ALL PASS' if nfail_c==0 else 'FAILURES PRESENT'}"
    print(summ)
    with open(LOG, "w") as f:
        f.write("\n".join(head)+"\n")
        for cse, tag, rc, npass, nfail, es, out in full:
            f.write("\n"+"-"*108+f"\nCASE {cse['name']} -> {tag}\nCMD  {' '.join([BIN]+cse['args'])}\nEXIT {rc} P/F={npass}/{nfail}\n{es}\n")
            if cse["note"]: f.write(f"NOTE {cse['note']}\n")
            f.write("OUTPUT:\n"+out.rstrip()+"\n")
        f.write("\n"+summ+"\n")
    print(f"\nlog: {LOG}")
    return 0 if nfail_c==0 else 1


if __name__ == "__main__":
    sys.exit(main())
