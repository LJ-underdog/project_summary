#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""M8 MI baseline driver — run the bwd harness with -perf over the canonical /
hd256 / window configs (SiLU + softmax each) and append per-kernel + envelope rows
to benchmark.csv (10-col schema: candidate,arch,mode,activation,dtype,hdim,kernel,
metric,value,date).

The harness prints lines:  PERF kernel=<k> metric=<m> value=<v>
                           PERF total_gemm_flops=<n>
We parse those (the -perf path runs AFTER validation, so a config must also PASS).

Usage:
  run_perf_baseline.py [--bin PATH] [--csv PATH] [--candidate NAME] [--dry-run]
"""
import argparse, csv, datetime, os, re, subprocess, sys

DEFAULT_BIN = "/root/workspace/ck_hstu/build/bin/tile_example_hstu_attention_bwd"
DEFAULT_CSV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "benchmark.csv")
ARCH = "gfx950"
PERF_RE = re.compile(r"^PERF kernel=(\S+) metric=(\S+) value=(\S+)\s*$")

# Canonical small config knobs (draft-M8-perf §1): b=2 nhead=8 seqlens=2048 causal=1.
# Each entry: (candidate, mode, activation, hdim, extra_args)
def cfg(suffix, act, hdim, extra):
    softmax = "1" if act == "softmax" else "0"
    base = ["-prec=bf16", "-b=2", "-nhead=8", "-seqlens=2048", "-causal=1",
            f"-softmax={softmax}", f"-hdim_qk={hdim}", f"-hdim_v={hdim}", "-v=1", "-perf=1"]
    return (suffix, "batched", act, hdim, base + extra)

# (suffix, mode, activation, hdim, args). The candidate column = f"{--candidate}-{suffix}".
CONFIGS = [
    cfg("canonical", "softmax", 64,  []),
    cfg("canonical", "silu",    64,  []),
    cfg("hd256",     "softmax", 256, []),
    cfg("hd256",     "silu",    256, []),
    cfg("window256", "softmax", 64,  ["-local_len=256"]),
    cfg("window256", "silu",    64,  ["-local_len=256"]),
    # B3: narrow/mid windows show the largest local-tightening win (full-scan baseline is
    # window-size-independent, so compare any of these to the MI-baseline-window256 rows).
    cfg("window64",  "softmax", 64,  ["-local_len=64"]),
    cfg("window64",  "silu",    64,  ["-local_len=64"]),
    cfg("window16",  "softmax", 64,  ["-local_len=16"]),
    cfg("window16",  "silu",    64,  ["-local_len=16"]),
]

# kernels that also get a TFLOPS row (GEMM-only); others only time_ms.
TFLOPS_KERNELS = {"envelope", "MAIN"}


def run_one(binary, args, timeout=600):
    proc = subprocess.run([binary] + args, capture_output=True, text=True, timeout=timeout)
    out = proc.stdout
    passed = proc.returncode == 0 and "numeric_pass=true" in out
    perf = {}  # (kernel,metric) -> value
    for line in out.splitlines():
        m = PERF_RE.match(line)
        if m:
            perf[(m.group(1), m.group(2))] = m.group(3)
    return passed, perf, proc.returncode, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default=DEFAULT_BIN)
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--date", default=datetime.date.today().isoformat())
    ap.add_argument("--candidate", default="MI-baseline",
                    help="candidate column prefix; row candidate = <candidate>-<suffix>")
    ap.add_argument("--dry-run", action="store_true", help="print rows, do not append")
    args = ap.parse_args()

    new_rows = []
    all_ok = True
    for suffix, mode, act, hdim, cargs in CONFIGS:
        name = f"{args.candidate}-{suffix}"
        print(f"[perf] {name} {act} hd{hdim}: {' '.join(cargs)}")
        passed, perf, rc, out = run_one(args.bin, cargs)
        if not passed:
            all_ok = False
            print(f"   !! NOT PASS (exit={rc}); skipping rows. tail:")
            print("\n".join("      " + l for l in out.splitlines()[-8:]))
            continue
        if not perf:
            all_ok = False
            print("   !! no PERF lines parsed; skipping")
            continue
        # emit time_ms for every kernel, TFLOPS for envelope/MAIN
        for (kern, metric), val in sorted(perf.items()):
            new_rows.append([name, ARCH, mode, act, "bf16", str(hdim), kern, metric, val, args.date])
        # console summary
        mm = perf.get(("MAIN", "time_ms"), "?")
        mt = perf.get(("MAIN", "TFLOPS"), "?")
        ev = perf.get(("envelope", "time_ms"), "?")
        print(f"   MAIN={mm}ms ({mt} TFLOPS)  envelope={ev}ms  "
              f"PRE={perf.get(('PRE','time_ms'),'?')} memset={perf.get(('memset','time_ms'),'?')} "
              f"POST={perf.get(('POST','time_ms'),'?')}")

    print(f"\n[perf] {len(new_rows)} rows from {len(CONFIGS)} configs; all_pass={all_ok}")
    if args.dry_run:
        for r in new_rows:
            print("  ", ",".join(r))
        return 0 if all_ok else 1

    # append (assumes header already present / 10-col schema)
    with open(args.csv, "a", newline="") as f:
        csv.writer(f).writerows(new_rows)
    print(f"[perf] appended to {args.csv}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
