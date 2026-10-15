"""Bundle attentris/data (gitignored, ~11 GB) into a few zips for Google Drive or similar, and restore them
on another machine. Three tiers by what you need:
    core   models (model_*.pt), the packed v5.1 datasets (v51/), small eval/result dirs  -> train, fine-tune, play
    labels the labelled jsonl the packed sets came from (raw*, dagger2..5all, state dumps) -> re-tokenize / re-pack
    old    v3 / v5 datasets, structure_study, audit, scaling-curve checkpoints           -> history only
.npz / .npy / .pt are already compressed and are stored as-is; text is deflated. Every zip carries its own
manifest (path + size) and CRCs; `verify` re-reads the zip, `unpack` extracts and checks sizes.

    python attentris/data_bundle.py list
    python attentris/data_bundle.py pack --tier core --out bundle      # bundle/attentris_data_core.zip
    python attentris/data_bundle.py pack --tier all --out bundle       # one zip per tier
    python attentris/data_bundle.py verify bundle/attentris_data_core.zip
    python attentris/data_bundle.py unpack bundle/attentris_data_*.zip # into attentris/data (skips identical files)
"""
import argparse
import fnmatch
import glob
import json
import os
import sys
import time
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
TIERS = ("core", "labels", "old")
LABELS = ["raw", "raw_1m", "raw_dagger1", "raw_dagger2", "raw_dagger2_tr", "raw_dagger2_val",
          "dagger2", "dagger3", "dagger4", "dagger5", "dagger5b", "dagger5all", "*.jsonl"]
OLD = ["v3", "v5", "raw_v3", "raw_dagger1_v3", "structure_study", "audit", "curve*"]
STORED = (".npz", ".npy", ".pt", ".zip", ".png")
MANIFEST = "MANIFEST.json"


def tier_of(top):
    """Tier of a top-level entry of attentris/data (file or directory name)."""
    if any(fnmatch.fnmatch(top, p) for p in OLD):
        return "old"
    if any(fnmatch.fnmatch(top, p) for p in LABELS):
        return "labels"
    return "core"


def files(tier):
    out = []
    for top in sorted(os.listdir(DATA)):
        if tier_of(top) != tier:
            continue
        p = os.path.join(DATA, top)
        if os.path.isfile(p):
            out.append(top)
        else:
            for root, _, names in os.walk(p):
                for n in sorted(names):
                    out.append(os.path.relpath(os.path.join(root, n), DATA).replace(os.sep, "/"))
    return [f for f in out if not f.endswith((".part", ".tmp"))]


def gb(n):
    return "%.2f GB" % (n / 1e9)


def cmd_list(_):
    for t in TIERS:
        fs = files(t)
        size = sum(os.path.getsize(os.path.join(DATA, f)) for f in fs)
        tops = sorted({f.split("/")[0] for f in fs})
        print("%-6s %5d files %9s  %s" % (t, len(fs), gb(size), " ".join(tops[:12]) + (" ..." if len(tops) > 12 else "")))


def cmd_pack(args):
    os.makedirs(args.out, exist_ok=True)
    for t in TIERS if args.tier == "all" else (args.tier,):
        fs = files(t)
        total = sum(os.path.getsize(os.path.join(DATA, f)) for f in fs)
        path = os.path.join(args.out, "attentris_data_%s.zip" % t)
        tmp = path + ".part"
        print("%s: %d files, %s -> %s" % (t, len(fs), gb(total), path), flush=True)
        done, t0 = 0, time.time()
        last = t0
        with zipfile.ZipFile(tmp, "w", allowZip64=True) as z:
            man = {}
            for f in fs:
                src = os.path.join(DATA, f)
                comp = zipfile.ZIP_STORED if f.endswith(STORED) else zipfile.ZIP_DEFLATED
                z.write(src, f, compress_type=comp, compresslevel=6 if comp == zipfile.ZIP_DEFLATED else None)
                man[f] = os.path.getsize(src)
                done += man[f]
                if time.time() - last > 5:
                    last = time.time()
                    print("  %5.1f%%  %s / %s  %.0f MB/s" % (100 * done / max(total, 1), gb(done), gb(total),
                                                            done / 1e6 / max(last - t0, 1e-9)), flush=True)
            z.writestr(MANIFEST, json.dumps({"tier": t, "files": man}, indent=0))
        os.replace(tmp, path)
        print("  done: %s (%.0f s)" % (gb(os.path.getsize(path)), time.time() - t0), flush=True)


def cmd_verify(args):
    bad = 0
    for path in args.zips:
        with zipfile.ZipFile(path) as z:
            man = json.loads(z.read(MANIFEST))["files"]
            first = z.testzip()   # reads every member, checks its CRC
            sizes = {i.filename: i.file_size for i in z.infolist() if i.filename != MANIFEST}
            ok = first is None and sizes == man
            bad += not ok
            print("%s: %d files, %s, %s" % (path, len(man), gb(sum(man.values())),
                                            "OK" if ok else "BAD (first bad member: %s)" % first))
    sys.exit(1 if bad else 0)


def cmd_unpack(args):
    zips = [p for a in args.zips for p in sorted(glob.glob(a))]
    if not zips:
        sys.exit("no zips match %s" % args.zips)
    dest = args.dest or DATA
    os.makedirs(dest, exist_ok=True)
    for path in zips:
        with zipfile.ZipFile(path) as z:
            man = json.loads(z.read(MANIFEST))["files"]
            n = skip = 0
            for f, size in man.items():
                dst = os.path.join(dest, f)
                if os.path.isfile(dst) and os.path.getsize(dst) == size and not args.force:
                    skip += 1
                    continue
                z.extract(f, dest)
                if os.path.getsize(dst) != size:
                    sys.exit("size mismatch after extracting %s" % f)
                n += 1
            print("%s: %d extracted, %d already present -> %s" % (path, n, skip, dest), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    p = sub.add_parser("pack")
    p.add_argument("--tier", choices=TIERS + ("all",), default="core")
    p.add_argument("--out", required=True, help="output dir (keep it out of the repo, or use bundle/, which is gitignored)")
    p = sub.add_parser("verify")
    p.add_argument("zips", nargs="+")
    p = sub.add_parser("unpack")
    p.add_argument("zips", nargs="+", help="zip paths or globs")
    p.add_argument("--force", action="store_true", help="overwrite files that are already present with the same size")
    p.add_argument("--dest", help="extract here instead of attentris/data")
    args = ap.parse_args()
    sys.path.insert(0, HERE)
    import pcores
    pcores.pin(verbose=False)
    {"list": cmd_list, "pack": cmd_pack, "verify": cmd_verify, "unpack": cmd_unpack}[args.cmd](args)


if __name__ == "__main__":
    main()
