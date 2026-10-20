"""Add token SHAP to an attention page: for every move the page shows (the model's top 5 and the bot's move),
the Shapley values from token_shap.py --keep PAGE --top 5, as a "SHAP" view next to the attention.

    attentris/.venv/bin/python attentris/shap_page.py --page attentris/attn_v51_d128_dagger5b.html \
        --shap attentris/data/shap/attn_v51_d128_dagger5b.jsonl --out attentris/attn_shap_v51_d128_dagger5b.html
"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--page", required=True, help="an attn_viz.py page")
    ap.add_argument("--shap", required=True, help="token_shap.py --keep PAGE --top N output")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="AtTenTris Attention and SHAP")
    args = ap.parse_args()
    html = open(args.page, encoding="utf-8").read()
    data = json.JSONDecoder().raw_decode(html[html.index('{"summary"'):])[0]
    recs = [json.loads(l) for l in open(args.shap)]
    assert len(recs) == len(data["positions"]), "SHAP file has %d positions, the page %d" % (len(recs), len(data["positions"]))
    n_moves = 0
    for p, r in zip(data["positions"], recs):
        kinds = [t["kind"] for t in p["tokens"]]
        assert kinds == r["kinds"], "token order differs (page %s vs SHAP %s)" % (kinds[:4], r["kinds"][:4])
        by_move = {c["i"]: c for c in r.get("cands", [])}
        for cand in p["top"] + [p["chosen"]]:
            c = by_move.get(cand["i"])
            if c:
                cand["shap"], cand["shap_all"], cand["shap_none"] = c["phi"], c["v_all"], c["v_none"]
                n_moves += 1
    tpl = open(os.path.join(HERE, "attn_template.html"), encoding="utf-8").read()
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(tpl.replace("__TITLE__", args.title).replace("__DATA__", json.dumps(data, separators=(",", ":"))))
    print("%d positions, %d moves with SHAP -> %s" % (len(recs), n_moves, args.out))


if __name__ == "__main__":
    main()
