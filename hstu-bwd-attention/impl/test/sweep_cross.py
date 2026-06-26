#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""cross-attention (seqlen_q != seqlen_kv) full 对拍 sweep (draft §6).

Both directions (q<kv & q>kv) x {no_group jagged, group, batched-uniform} x SiLU/softmax x
causal{0,1} x P1-1 factors (num_target Q-side / contextual<=min(q,kv) / local / minfull) x
atomic/determ, + non-divisible + a determ kv>q multi-KV-block case (R4) + a couple fp16.
-attn_scale=1.0 (gradient magnitude meaningful). Tolerance = harness templated elimit
(bf16 2e-2/5e-2, fp16 5e-3/1e-2), NOT loosened. cross enabled via -seqlens_kv != -seqlens."""
import re, subprocess, sys, datetime

BIN = "/root/workspace/ck_hstu/build/bin/tile_example_hstu_attention_bwd"
LOG = "/root/workspace/hstu-bwd-impl/runs/run-cross-sweep.log"
FAIL_RE = re.compile(r"\[FAIL\]")
NP_RE = re.compile(r"numeric_pass=(true|false)")
ERR_RE = re.compile(r"^\s*(d[QKV]): max_abs_err=(\S+) .* \(max\|ref\|=(\S+)\)")
A = "-attn_scale=1.0"


def c(name, extra):
    return {"name": name, "args": ["-v=1", A] + extra}


# hd64 bf16 unless noted. q<kv uses seqlens=128 seqlens_kv=256; q>kv swaps.
CASES = [
    # ---- no_group jagged, both directions, SiLU/softmax x causal{0,1} ----
    c("j-qlt-silu-c1", ["-jagged=1", "-b=2", "-nhead=2", "-seqlens=128", "-seqlens_kv=256", "-softmax=0", "-causal=1"]),
    c("j-qgt-silu-c1", ["-jagged=1", "-b=2", "-nhead=2", "-seqlens=256", "-seqlens_kv=128", "-softmax=0", "-causal=1"]),
    c("j-qlt-silu-c0", ["-jagged=1", "-b=2", "-nhead=2", "-seqlens=128", "-seqlens_kv=256", "-softmax=0", "-causal=0"]),
    c("j-qgt-silu-c0", ["-jagged=1", "-b=2", "-nhead=2", "-seqlens=256", "-seqlens_kv=128", "-softmax=0", "-causal=0"]),
    c("j-qlt-sm-c1",   ["-jagged=1", "-b=2", "-nhead=2", "-seqlens=128", "-seqlens_kv=256", "-softmax=1", "-causal=1"]),
    c("j-qgt-sm-c1",   ["-jagged=1", "-b=2", "-nhead=2", "-seqlens=256", "-seqlens_kv=128", "-softmax=1", "-causal=1"]),
    c("j-qlt-sm-c0",   ["-jagged=1", "-b=2", "-nhead=2", "-seqlens=128", "-seqlens_kv=256", "-softmax=1", "-causal=0"]),
    c("j-qgt-sm-c0",   ["-jagged=1", "-b=2", "-nhead=2", "-seqlens=256", "-seqlens_kv=128", "-softmax=1", "-causal=0"]),
    # ---- P1-1 factors, both directions (Q-side targets; contextual<=min(q,kv)=128) ----
    c("j-qlt-c0-target",  ["-jagged=1", "-b=2", "-nhead=2", "-seqlens=128", "-seqlens_kv=256", "-softmax=0", "-causal=0", "-targets=8"]),
    c("j-qgt-c0-target",  ["-jagged=1", "-b=2", "-nhead=2", "-seqlens=256", "-seqlens_kv=128", "-softmax=0", "-causal=0", "-targets=8"]),
    c("j-qlt-c1-target",  ["-jagged=1", "-b=2", "-nhead=2", "-seqlens=128", "-seqlens_kv=256", "-softmax=1", "-causal=1", "-targets=8"]),
    c("j-qlt-c1-context", ["-jagged=1", "-b=2", "-nhead=2", "-seqlens=128", "-seqlens_kv=256", "-softmax=0", "-causal=1", "-context_len=8"]),
    c("j-qgt-c1-context", ["-jagged=1", "-b=2", "-nhead=2", "-seqlens=256", "-seqlens_kv=128", "-softmax=0", "-causal=1", "-context_len=8"]),
    c("j-qlt-c1-local",   ["-jagged=1", "-b=2", "-nhead=2", "-seqlens=128", "-seqlens_kv=256", "-softmax=0", "-causal=1", "-local_len=16"]),
    c("j-qgt-c1-local",   ["-jagged=1", "-b=2", "-nhead=2", "-seqlens=256", "-seqlens_kv=128", "-softmax=0", "-causal=1", "-local_len=16"]),
    c("j-qlt-c1-minfull", ["-jagged=1", "-b=2", "-nhead=2", "-seqlens=128", "-seqlens_kv=256", "-softmax=0", "-causal=1", "-minfull_len=16"]),
    c("j-qlt-c1-combo",   ["-jagged=1", "-b=3", "-nhead=2", "-seqlens=128,160,96", "-seqlens_kv=256,200,300", "-softmax=1", "-causal=1",
                           "-local_len=16", "-context_len=8", "-minfull_len=16", "-targets=8,16,0"]),
    # ---- non-divisible (q non-kM0=32-div, kv non-kN0=128-div) ----
    c("j-nondiv-qlt", ["-jagged=1", "-b=2", "-nhead=2", "-seqlens=130", "-seqlens_kv=200", "-softmax=0", "-causal=1"]),
    c("j-nondiv-qgt", ["-jagged=1", "-b=2", "-nhead=2", "-seqlens=200", "-seqlens_kv=130", "-softmax=1", "-causal=1"]),
    # ---- determ (kN0=128 for hd64): >=1 case kv>q crossing multiple KV blocks (R4) ----
    c("j-determ-qlt-multiblk-sm", ["-jagged=1", "-b=2", "-nhead=4", "-seqlens=128", "-seqlens_kv=512", "-softmax=1", "-causal=1", "-deterministic=1"]),
    c("j-determ-qlt-multiblk-silu", ["-jagged=1", "-b=2", "-nhead=4", "-seqlens=128", "-seqlens_kv=512", "-softmax=0", "-causal=1", "-deterministic=1", "-targets=8"]),
    c("j-determ-qgt", ["-jagged=1", "-b=2", "-nhead=4", "-seqlens=512", "-seqlens_kv=128", "-softmax=0", "-causal=1", "-deterministic=1"]),
    # ---- group (per-group kv lengths via supplement), both directions ----
    c("g2-qlt-sm-c1",     ["-g=2", "-b=4", "-nhead=2", "-seqlens=128", "-seqlens_kv=256", "-softmax=1", "-causal=1"]),
    c("g2-qgt-silu-c1",   ["-g=2", "-b=4", "-nhead=2", "-seqlens=256", "-seqlens_kv=128", "-softmax=0", "-causal=1"]),
    c("g2-qlt-c0-target", ["-g=2", "-b=4", "-nhead=2", "-seqlens=128", "-seqlens_kv=256", "-softmax=0", "-causal=0", "-targets=8"]),
    c("g2-het-qlt-sm-c1", ["-g=2", "-b=4", "-nhead=2", "-seqlens=128,160,96,200", "-seqlens_kv=256,300,200,256", "-softmax=1", "-causal=1"]),
    c("g2-determ-qlt-multiblk", ["-g=2", "-b=4", "-nhead=4", "-seqlens=128", "-seqlens_kv=512", "-softmax=1", "-causal=1", "-deterministic=1"]),
    # ---- batched uniform, both directions (nraw scalar seqlen_kv path) ----
    c("b-qlt-silu-c1", ["-jagged=0", "-b=2", "-nhead=2", "-seqlens=128", "-seqlens_kv=256", "-softmax=0", "-causal=1"]),
    c("b-qgt-sm-c1",   ["-jagged=0", "-b=2", "-nhead=2", "-seqlens=256", "-seqlens_kv=128", "-softmax=1", "-causal=1"]),
    c("b-qlt-determ-multiblk", ["-jagged=0", "-b=2", "-nhead=4", "-seqlens=128", "-seqlens_kv=512", "-softmax=1", "-causal=1", "-deterministic=1"]),
    # ---- fp16 (tighter elimit), both directions ----
    c("j-qlt-silu-c1-fp16", ["-prec=fp16", "-jagged=1", "-b=2", "-nhead=2", "-seqlens=128", "-seqlens_kv=256", "-softmax=0", "-causal=1"]),
    c("g2-qgt-sm-c1-fp16",  ["-prec=fp16", "-g=2", "-b=4", "-nhead=2", "-seqlens=256", "-seqlens_kv=128", "-softmax=1", "-causal=1"]),
]


def run(case):
    p = subprocess.run([BIN] + case["args"], capture_output=True, text=True, timeout=240)
    out = (p.stdout or "") + (p.stderr or "")
    np_m = NP_RE.search(out)
    errs = {m.group(1): (m.group(2), m.group(3)) for m in
            (ERR_RE.match(l) for l in out.splitlines()) if m}
    nan = ("nan" in out.lower())
    ok = (p.returncode == 0 and np_m and np_m.group(1) == "true" and not FAIL_RE.search(out) and not nan)
    return ok, p.returncode, errs, out


def main():
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    head = [f"cross-attention 对拍 sweep — {stamp}", f"binary: {BIN}",
            f"cases: {len(CASES)}  (-attn_scale=1.0; elimit NOT loosened; cross via -seqlens_kv)",
            "=" * 108]
    print("\n".join(head))
    npass = nfail = 0
    full = []
    for cse in CASES:
        try:
            ok, rc, errs, out = run(cse)
        except subprocess.TimeoutExpired:
            ok, rc, errs, out = False, None, {}, "<<TIMEOUT>>"
        npass += ok
        nfail += (not ok)
        es = "  ".join(f"{t}:e={errs[t][0]}|ref|={errs[t][1]}" for t in ("dQ", "dK", "dV") if t in errs)
        print(f"[{'PASS' if ok else 'FAIL'}] {cse['name']:<28} exit={str(rc):<4} {es}")
        full.append((cse, ok, rc, es, out))
    summ = "=" * 108 + f"\nTOTAL {len(CASES)}  PASS {npass}  FAIL {nfail}\nRESULT: {'ALL PASS' if nfail == 0 else 'FAILURES PRESENT'}"
    print(summ)
    with open(LOG, "w") as f:
        f.write("\n".join(head) + "\n")
        for cse, ok, rc, es, out in full:
            f.write("\n" + "-" * 108 + f"\nCASE {cse['name']} -> {'PASS' if ok else 'FAIL'}\n"
                    f"CMD  {' '.join([BIN] + cse['args'])}\nEXIT {rc}\n{es}\n"
                    "OUTPUT:\n" + out.rstrip() + "\n")
        f.write("\n" + summ + "\n")
    print(f"\nlog: {LOG}")
    return 0 if nfail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
