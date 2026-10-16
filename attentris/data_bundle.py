"""Bundle attentris/data (gitignored, ~11 GB) into zips for Google Drive or similar, restore them on another
machine, and ship new data from either machine as small delta zips. Three tiers by what you need:
    core   models (model_*.pt), the packed v5.1 datasets (v51/), small eval/result dirs  -> train, fine-tune, play
    labels the labelled jsonl the packed sets came from (raw*, dagger2..5all, state dumps) -> re-tokenize / re-pack
    old    v3 / v5 datasets, structure_study, audit, scaling-curve checkpoints           -> history only
The bundle folder (BUNDLE below) is a local copy of the Drive folder: the full tier zips plus any
attentris_data_delta_<time>_<host>.zip. .npz / .npy / .pt are stored as-is (already compressed), text is deflated.
Every zip carries a manifest (files, sizes, creation time, host) and per-file CRCs.

    python attentris/data_bundle.py list                                  # tiers and sizes of attentris/data
    python attentris/data_bundle.py pack --tier all --out BUNDLE          # full zips (first upload)
    python attentris/data_bundle.py status BUNDLE                         # local vs bundle: new / changed / missing
    python attentris/data_bundle.py pack --delta --out BUNDLE             # zip only new / changed files -> upload it
    python attentris/data_bundle.py unpack "BUNDLE/attentris_data_*.zip"  # restore / apply deltas (oldest first)
    python attentris/data_bundle.py verify BUNDLE/attentris_data_core.zip

unpack skips files whose content is already the same (size + CRC), applies zips in creation order so newer
versions win, restores each file's time, and never overwrites a local file that is newer than the zipped copy
(it reports it; --force overrides). Keep BUNDLE off a nearly full disk.
"""
import argparse
import datetime
import fnmatch
import glob
import json
import os
import socket
import sys
import time
import zipfile
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
TIERS = ("core", "labels", "old")
LABELS = ["raw", "raw_1m", "raw_dagger1", "raw_dagger2", "raw_dagger2_tr", "raw_dagger2_val",
          "dagger2", "dagger3", "dagger4", "dagger5", "dagger5b", "dagger5all", "plan", "*.jsonl"]
OLD = ["v3", "v5", "raw_v3", "raw_dagger1_v3", "structure_study", "audit", "curve*"]
# derived data rebuilt on each machine, never bundled: v51plan = attentris-autoreg's legal moves
# along every plan (~10 GB), rebuilt from data/plan by pack_plan.py in a few minutes
SKIP = ["v51plan"]
STORED = (".npz", ".npy", ".pt", ".zip", ".png")
MANIFEST = "MANIFEST.json"


def tier_of(top):
    """Tier of a top-level entry of attentris/data (file or directory name)."""
    if any(fnmatch.fnmatch(top, p) for p in OLD):
        return "old"
    if any(fnmatch.fnmatch(top, p) for p in LABELS):
        return "labels"
    return "core"


def files(tier=None):
    out = []
    for top in sorted(os.listdir(DATA)):
        if any(fnmatch.fnmatch(top, p) for p in SKIP):
            continue
        if tier is not None and tier_of(top) != tier:
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


def crc_of(path):
    c = 0
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            c = zlib.crc32(block, c)
    return c & 0xFFFFFFFF


def zip_time(info):
    return time.mktime(info.date_time + (0, 0, -1))


def bundle_index(zips):
    """Latest entry per path over the given zips, applied in creation order: {path: (zip, ZipInfo, mtime)}."""
    order = []
    for path in zips:
        with zipfile.ZipFile(path) as z:
            man = json.loads(z.read(MANIFEST))
            mt = man.get("mtime", {})   # UTC file times (zip times are local time); older zips lack them
            order.append((man.get("created", ""), path, {i.filename: (i, mt.get(i.filename, zip_time(i)))
                                                         for i in z.infolist() if i.filename != MANIFEST}))
    order.sort(key=lambda t: (t[0], t[1]))      # full zips from before deltas existed have no "created": first
    idx = {}
    for _, path, infos in order:
        for f, (info, mt) in infos.items():
            idx[f] = (path, info, mt)
    return idx, [p for _, p, _ in order]


def same(local, info, mtime):
    """Local file has the zipped content: same size and (same time, or same CRC)."""
    st = os.stat(local)
    if st.st_size != info.file_size:
        return False
    if abs(st.st_mtime - mtime) <= 2:   # zip-time fallback has 2 s resolution
        return True
    return crc_of(local) == info.CRC


def diff(bundle_dir):
    zips = sorted(glob.glob(os.path.join(bundle_dir, "attentris_data_*.zip")))
    idx, _ = bundle_index(zips)
    new, changed, unchanged = [], [], []
    for f in files():
        if f not in idx:
            new.append(f)
        elif same(os.path.join(DATA, f), idx[f][1], idx[f][2]):
            unchanged.append(f)
        else:
            changed.append(f)
    local = set(files())
    missing = [f for f in idx if f not in local]
    return zips, new, changed, unchanged, missing


def write_zip(path, fs, label):
    total = sum(os.path.getsize(os.path.join(DATA, f)) for f in fs)
    tmp = path + ".part"
    print("%s: %d files, %s -> %s" % (label, len(fs), gb(total), path), flush=True)
    done, t0 = 0, time.time()
    last = t0
    with zipfile.ZipFile(tmp, "w", allowZip64=True) as z:
        man, mt = {}, {}
        for f in fs:
            src = os.path.join(DATA, f)
            comp = zipfile.ZIP_STORED if f.endswith(STORED) else zipfile.ZIP_DEFLATED
            z.write(src, f, compress_type=comp, compresslevel=6 if comp == zipfile.ZIP_DEFLATED else None)
            man[f] = os.path.getsize(src)
            mt[f] = os.path.getmtime(src)
            done += man[f]
            if time.time() - last > 5:
                last = time.time()
                print("  %5.1f%%  %s / %s  %.0f MB/s" % (100 * done / max(total, 1), gb(done), gb(total),
                                                        done / 1e6 / max(last - t0, 1e-9)), flush=True)
        z.writestr(MANIFEST, json.dumps({"tier": label, "created": datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
                                         "host": socket.gethostname(), "files": man, "mtime": mt}, indent=0))
    os.replace(tmp, path)
    print("  done: %s (%.0f s)" % (gb(os.path.getsize(path)), time.time() - t0), flush=True)


def cmd_list(_):
    for t in TIERS:
        fs = files(t)
        size = sum(os.path.getsize(os.path.join(DATA, f)) for f in fs)
        tops = sorted({f.split("/")[0] for f in fs})
        print("%-6s %5d files %9s  %s" % (t, len(fs), gb(size), " ".join(tops[:12]) + (" ..." if len(tops) > 12 else "")))


def cmd_status(args):
    zips, new, changed, unchanged, missing = diff(args.bundle)
    print("bundle %s: %d zips" % (args.bundle, len(zips)))
    for name, fs in (("new (not in any zip)", new), ("changed since the zipped copy", changed)):
        size = sum(os.path.getsize(os.path.join(DATA, f)) for f in fs)
        print("%-30s %5d files %9s" % (name, len(fs), gb(size)))
        for f in fs[:args.show]:
            print("    " + f)
        if len(fs) > args.show:
            print("    ... (%d more)" % (len(fs) - args.show))
    print("%-30s %5d files" % ("unchanged", len(unchanged)))
    print("%-30s %5d files%s" % ("in the bundle, not here", len(missing), "  (unpack to get them)" if missing else ""))
    if new or changed:
        print("-> pack --delta --out %s, then upload the new delta zip" % args.bundle)


def cmd_pack(args):
    os.makedirs(args.out, exist_ok=True)
    if args.delta:
        zips, new, changed, _, _ = diff(args.out)
        if not zips:
            sys.exit("no attentris_data_*.zip in %s to compare with: pack the full tiers first (or copy them there)" % args.out)
        fs = sorted(new + changed)
        if not fs:
            print("nothing new or changed vs the %d zips in %s" % (len(zips), args.out))
            return
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        host = "".join(ch for ch in socket.gethostname() if ch.isalnum() or ch in "-_")[:24] or "host"
        write_zip(os.path.join(args.out, "attentris_data_delta_%s_%s.zip" % (stamp, host)), fs,
                  "delta (%d new, %d changed)" % (len(new), len(changed)))
        return
    for t in TIERS if args.tier == "all" else (args.tier,):
        write_zip(os.path.join(args.out, "attentris_data_%s.zip" % t), files(t), t)


def cmd_verify(args):
    bad = 0
    for path in args.zips:
        with zipfile.ZipFile(path) as z:
            man = json.loads(z.read(MANIFEST))
            first = z.testzip()   # reads every member, checks its CRC
            sizes = {i.filename: i.file_size for i in z.infolist() if i.filename != MANIFEST}
            ok = first is None and sizes == man["files"]
            bad += not ok
            print("%s: %s, %d files, %s, created %s on %s: %s" % (path, man.get("tier"), len(sizes), gb(sum(sizes.values())),
                  man.get("created", "?"), man.get("host", "?"), "OK" if ok else "BAD (first bad member: %s)" % first))
    sys.exit(1 if bad else 0)


def cmd_unpack(args):
    zips = sorted({p for a in args.zips for p in glob.glob(a)})
    if not zips:
        sys.exit("no zips match %s" % args.zips)
    dest = args.dest or DATA
    os.makedirs(dest, exist_ok=True)
    idx, order = bundle_index(zips)
    todo = {}
    for f, (path, info, mt) in idx.items():
        todo.setdefault(path, []).append((info, mt))
    kept = []
    for path in order:
        n = skip = 0
        with zipfile.ZipFile(path) as z:
            for info, mt in todo.get(path, []):
                dst = os.path.join(dest, info.filename)
                if os.path.isfile(dst) and not args.force:
                    if same(dst, info, mt):
                        skip += 1
                        continue
                    if os.stat(dst).st_mtime > mt + 2:
                        kept.append(info.filename)   # local copy is newer: probably made here after the zip
                        continue
                z.extract(info, dest)
                if os.path.getsize(dst) != info.file_size:
                    sys.exit("size mismatch after extracting %s" % info.filename)
                os.utime(dst, (mt, mt))
                n += 1
        print("%s: %d extracted, %d already the same" % (os.path.basename(path), n, skip), flush=True)
    if kept:
        print("kept %d local files that are newer than the zipped copy (pack --delta to ship them; --force to overwrite):" % len(kept))
        for f in kept[:20]:
            print("    " + f)
    print("-> %s" % dest)


def main():
    global DATA
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=DATA, help="data dir (default attentris/data)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    p = sub.add_parser("status")
    p.add_argument("bundle", help="folder with the attentris_data_*.zip files (local copy of the Drive folder)")
    p.add_argument("--show", type=int, default=15, help="list this many new / changed files")
    p = sub.add_parser("pack")
    p.add_argument("--tier", choices=TIERS + ("all",), default="core")
    p.add_argument("--delta", action="store_true", help="only files that are new or changed vs the zips already in --out")
    p.add_argument("--out", required=True, help="bundle folder (keep it out of the repo, or use bundle/, which is gitignored)")
    p = sub.add_parser("verify")
    p.add_argument("zips", nargs="+")
    p = sub.add_parser("unpack")
    p.add_argument("zips", nargs="+", help="zip paths or globs")
    p.add_argument("--force", action="store_true", help="overwrite local files even if they are newer than the zipped copy")
    p.add_argument("--dest", help="extract here instead of the data dir")
    args = ap.parse_args()
    DATA = os.path.abspath(args.data)
    sys.path.insert(0, HERE)
    import pcores
    pcores.pin(verbose=False)
    {"list": cmd_list, "status": cmd_status, "pack": cmd_pack, "verify": cmd_verify, "unpack": cmd_unpack}[args.cmd](args)


if __name__ == "__main__":
    main()
