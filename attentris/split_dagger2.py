"""DAgger round 2: split the labelled student states (data/raw_dagger2/selected_labelled.jsonl) by
game seed into a training part and a held-out "student positions" validation part (every 10th
seed: whole games, both students' games of that seed), in shards for pack.py --only --jobs.

    python attentris/split_dagger2.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "data", "raw_dagger2", "selected_labelled.jsonl")
SHARDS = 32

out = {"tr": os.path.join(HERE, "data", "raw_dagger2_tr"), "val": os.path.join(HERE, "data", "raw_dagger2_val")}
for d in out.values():
    os.makedirs(d, exist_ok=True)
files = {("tr", i): open(os.path.join(out["tr"], "normal_dagger2_s%d.jsonl" % i), "w") for i in range(SHARDS)}
files.update({("val", i): open(os.path.join(out["val"], "normal_dagger2_s%d.jsonl" % i), "w") for i in range(SHARDS // 4)})
cnt = {"tr": 0, "val": 0}
for l in open(SRC):
    seed = json.loads(l)["seed"]
    part = "val" if seed % 10 == 0 else "tr"
    files[(part, seed % (SHARDS if part == "tr" else SHARDS // 4))].write(l)
    cnt[part] += 1
for f in files.values():
    f.close()
print("train %d, val (student positions) %d" % (cnt["tr"], cnt["val"]))
