#!/usr/bin/env python3
"""List and download TETR.IO Zenith replays for a player.

The 51 replays already in data/replays/ arrived without a script; this is that step made
reproducible, so a new player can be added or a run re-pulled without guesswork.

    python tools/fetch_replays.py rilakkuma_                    # list, download nothing
    python tools/fetch_replays.py rilakkuma_ --min-pieces 250   # list the runs worth having
    python tools/fetch_replays.py rilakkuma_ --get ID [ID ...]  # download those
    python tools/fetch_replays.py rilakkuma_ --min-pieces 250 --get-all

Two services are involved, both public:
  ch.tetr.io        the record lists. Gives replayid, mods, altitude, piece count.
  inoue.szy.lol     the replay bodies. tetr.io's own /api/games/ endpoint returns 406
                    to anything but the game client, so this is the usable route.

Files land as data/replays/{user}_{mods}__{id}.ttr, matching what harvest.mjs parses.
Existing files are skipped, and downloads are spaced out -- these are somebody's free
servers and we are pulling for a private dataset.
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

API = "https://ch.tetr.io/api"
REPLAY = "https://inoue.szy.lol/api/replay/%s"
UA = "tetris-research/1.0 (personal ML dataset; low volume)"
OUT_DIR = os.path.join("data", "replays")
## Zenith Quick Play and Zenith Expert are separate leaderboards; a player's interesting
## runs can sit in either, and `top` holds records that have aged out of `recent`.
MODES = ["zenith", "zenithex"]
BOARDS = ["recent", "top"]
PAUSE_SEC = 2.0


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=45) as r:
        return r.read()


def get_json(url):
    return json.loads(get(url).decode("utf-8"))


def slug(mods):
    """Mod list -> a filename token: lower case, hyphen-joined, no underscores.

    Underscores are the field separator in the filename, so `allspin_reversed` becomes
    `allspinrev` rather than splitting into two fields. Empty list is `nomod`, which is
    what the existing files use."""
    if not mods:
        return "nomod"
    parts = [m.replace("_reversed", "rev").replace("_", "") for m in mods]
    return "-".join(parts)


def records(user_id):
    """Every distinct record across both modes and both leaderboards."""
    seen, out = set(), []
    for mode in MODES:
        for board in BOARDS:
            url = "%s/users/%s/records/%s/%s?limit=100" % (API, user_id, mode, board)
            try:
                d = get_json(url)
            except urllib.error.HTTPError as e:
                print("  ! %s/%s: HTTP %s" % (mode, board, e.code), file=sys.stderr)
                continue
            if not d.get("success"):
                continue
            for e in d.get("data", {}).get("entries", []):
                rid = e.get("replayid")
                if not rid or rid in seen:
                    continue
                seen.add(rid)
                st = e.get("results", {}).get("stats", {})
                ag = e.get("results", {}).get("aggregatestats", {})
                z = st.get("zenith", {})
                ex = (e.get("extras") or {}).get("zenith", {}) or {}
                out.append({
                    "id": rid, "mode": e.get("gamemode"), "ts": (e.get("ts") or "")[:10],
                    "mods": ex.get("mods") or [], "alt": z.get("altitude") or 0.0,
                    "floor": z.get("floor") or 0, "pieces": st.get("piecesplaced") or 0,
                    "lines": st.get("lines") or 0, "apm": ag.get("apm") or 0.0,
                    "pps": ag.get("pps") or 0.0, "topbtb": st.get("topbtb") or 0,
                })
    out.sort(key=lambda r: -r["pieces"])
    return out


def table(rows):
    print("%-14s %-9s %-24s %7s %3s %7s %6s %6s %6s %5s" % (
        "replayid", "mode", "mods", "alt", "fl", "pieces", "lines", "apm", "pps", "btb"))
    for r in rows:
        print("%-14s %-9s %-24s %7.0f %3d %7d %6d %6.1f %6.2f %5d" % (
            r["id"], r["mode"], ",".join(r["mods"]) or "nomod", r["alt"], r["floor"],
            r["pieces"], r["lines"], r["apm"], r["pps"], r["topbtb"]))


def download(user, rows, want):
    os.makedirs(OUT_DIR, exist_ok=True)
    by_id = {r["id"]: r for r in rows}
    got, skipped, failed = [], [], []
    for i, rid in enumerate(want):
        r = by_id.get(rid)
        if r is None:
            failed.append((rid, "not in this player's records"))
            continue
        name = "%s_%s__%s.ttr" % (user.rstrip("_"), slug(r["mods"]), rid)
        dest = os.path.join(OUT_DIR, name)
        if os.path.exists(dest):
            skipped.append(name)
            continue
        if i:
            time.sleep(PAUSE_SEC)
        try:
            body = get(REPLAY % rid)
        except Exception as e:                     # noqa: BLE001 -- report and carry on
            failed.append((rid, "%s: %s" % (type(e).__name__, e)))
            continue
        # Validate before writing: a rate-limit or error page is also a 200 with a body,
        # and a half-written .ttr in data/replays/ would break the next harvest.
        try:
            d = json.loads(body.decode("utf-8"))
            assert d.get("id") == rid, "body is for %s, not %s" % (d.get("id"), rid)
            assert d.get("replay"), "no replay payload"
        except Exception as e:                     # noqa: BLE001
            failed.append((rid, "not a replay: %s" % e))
            continue
        with open(dest, "wb") as f:
            f.write(body)
        got.append((name, len(body)))
        print("  got %-52s %7.1f KB" % (name, len(body) / 1024.0))
    return got, skipped, failed


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("user")
    ap.add_argument("--min-pieces", type=int, default=0,
                    help="hide (and, with --get-all, skip) runs shorter than this")
    ap.add_argument("--get", nargs="*", default=[], metavar="ID")
    ap.add_argument("--get-all", action="store_true",
                    help="download every listed run that passes --min-pieces")
    a = ap.parse_args(argv[1:])

    u = get_json("%s/users/%s" % (API, a.user.lower()))
    if not u.get("success"):
        print("no such user: %s" % a.user, file=sys.stderr)
        return 1
    data = u.get("data", {})
    info = data.get("user", data)
    uid = info.get("_id")
    print("%s  id=%s  country=%s  games=%s\n" % (
        info.get("username"), uid, info.get("country"), info.get("gamesplayed")))

    rows = records(uid)
    shown = [r for r in rows if r["pieces"] >= a.min_pieces]
    table(shown)
    print("\n%d records, %d at >= %d pieces" % (len(rows), len(shown), a.min_pieces))

    want = list(a.get)
    if a.get_all:
        want = [r["id"] for r in shown]
    if not want:
        print("\nnothing downloaded (pass --get ID... or --get-all)")
        return 0
    print("\ndownloading %d into %s/" % (len(want), OUT_DIR))
    got, skipped, failed = download(info.get("username"), rows, want)
    print("\n%d downloaded, %d already present, %d failed" % (len(got), len(skipped), len(failed)))
    for name in skipped:
        print("  skip %s" % name)
    for rid, why in failed:
        print("  FAIL %s -- %s" % (rid, why))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
