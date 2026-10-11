"""Split labelled states (bot label output) by game seed into training and held-out shards for
pack.py --only --jobs: every --every-th seed (whole games) goes to validation.

    python attentris/split_by_seed.py --src data/dagger3/labelled.jsonl --tr data/dagger3/raw_tr --val data/dagger3/raw_val
"""
import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--tr", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--shards", type=int, default=32)
    ap.add_argument("--name", default="danger")
    args = ap.parse_args()
    for d in (args.tr, args.val):
        os.makedirs(d, exist_ok=True)
    nv = max(1, args.shards // 4)
    files = {("tr", i): open(os.path.join(args.tr, "%s_s%d.jsonl" % (args.name, i)), "w") for i in range(args.shards)}
    files.update({("val", i): open(os.path.join(args.val, "%s_s%d.jsonl" % (args.name, i)), "w") for i in range(nv)})
    cnt = {"tr": 0, "val": 0}
    for l in open(args.src):
        seed = json.loads(l)["seed"]
        part = "val" if seed % args.every == 0 else "tr"
        files[(part, (seed // args.every) % (args.shards if part == "tr" else nv))].write(l)
        cnt[part] += 1
    for f in files.values():
        f.close()
    print("train %d, held out %d" % (cnt["tr"], cnt["val"]))


if __name__ == "__main__":
    main()
