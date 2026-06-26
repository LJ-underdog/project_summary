#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Device-code byte-identity helper (M7c zero-regression gate).

For each object file, extract its gfx950 device code object (llvm-objdump
--offloading), disassemble (llvm-objdump -d), and build {symbol -> sha256 of the
instruction-ENCODING stream} (the hex after '//', address column stripped, so
function reordering does not perturb a function's hash).

Usage:
  co_symbols.py dump   <out.json> <obj> [obj...]    # build a baseline map
  co_symbols.py verify <baseline.json> <obj> [obj...]  # every baseline symbol must
                                                       # reappear byte-identical
                                                       # (new symbols are allowed)
"""
import hashlib, json, os, re, subprocess, sys, tempfile

OBJDUMP = "/opt/rocm/llvm/bin/llvm-objdump"
SYM_RE = re.compile(r"^[0-9a-f]+ <(.+)>:\s*$")
# disasm line: "\ts_load_dword s5, ... // 000000003500: C0020140 00000038"
ENC_RE = re.compile(r"//\s*[0-9A-Fa-f]+:\s*([0-9A-Fa-f ]+)\s*$")


def co_for(obj, workdir):
    """Extract the gfx950 device code object for `obj`, return its path or None."""
    local = os.path.join(workdir, os.path.basename(obj))
    if local != obj:
        subprocess.run(["cp", obj, local], check=True)
    subprocess.run([OBJDUMP, "--offloading", local],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=workdir)
    cos = [f for f in os.listdir(workdir)
           if f.startswith(os.path.basename(local) + ".") and "gfx950" in f and "host" not in f]
    return os.path.join(workdir, cos[0]) if cos else None


def sym_map(obj, workdir):
    """{symbol -> sha256(encoding stream)} for all device functions in obj."""
    co = co_for(obj, workdir)
    if not co:
        return {}
    out = subprocess.run([OBJDUMP, "-d", co], capture_output=True, text=True).stdout
    m, cur, enc = {}, None, []
    def flush():
        if cur is not None:
            m[cur] = hashlib.sha256("\n".join(enc).encode()).hexdigest()
    for line in out.splitlines():
        s = SYM_RE.match(line)
        if s:
            flush(); cur = s.group(1); enc = []; continue
        e = ENC_RE.search(line)
        if e and cur is not None:
            enc.append("".join(e.group(1).split()))
    flush()
    return m


def build(objs):
    res = {}
    with tempfile.TemporaryDirectory() as wd:
        for o in objs:
            sub = tempfile.mkdtemp(dir=wd)
            res[os.path.basename(o)] = sym_map(o, sub)
    return res


def main():
    mode = sys.argv[1]
    if mode == "dump":
        out, objs = sys.argv[2], sys.argv[3:]
        res = build(objs)
        json.dump(res, open(out, "w"), indent=0)
        n = sum(len(v) for v in res.values())
        print(f"dumped {len(res)} objects, {n} device symbols -> {out}")
    elif mode == "verify":
        base = json.load(open(sys.argv[2]))
        objs = sys.argv[3:]
        cur = build(objs)
        total = ident = miss = diff = 0
        bad = []
        for ofile, syms in base.items():
            csyms = cur.get(ofile, {})
            for sym, h in syms.items():
                total += 1
                if sym not in csyms:
                    miss += 1; bad.append(("MISSING", ofile, sym[:80]))
                elif csyms[sym] != h:
                    diff += 1; bad.append(("DIFF", ofile, sym[:80]))
                else:
                    ident += 1
        print(f"baseline symbols: {total}  byte-identical: {ident}  MISSING: {miss}  DIFF: {diff}")
        new = sum(len(cur.get(o, {})) for o in base) - total + miss
        print(f"(new pad-true symbols in refactored objects: ~{new}, allowed)")
        for tag, of, s in bad[:20]:
            print(f"  [{tag}] {of}: {s}")
        sys.exit(0 if (miss == 0 and diff == 0) else 1)
    else:
        print(__doc__); sys.exit(2)


if __name__ == "__main__":
    main()
