"""AtTenTris-ToMe: no rule tokenizer. The board enters as 200 cell tokens (filled / empty +
2D position) plus the 7 piece tokens; token merging (ToMe, Bolya et al., ICLR 2023) inside the
attention layers groups cells into blobs. Merges are constrained to 4-adjacent blobs, so every
token is always a 4-connected region of the board. Candidates and heads as in AtTenTris
(model.py), cross-attending to the final blobs.

Per merging layer (after attention, before the feed-forward), as in ToMe's bipartite matching:
  * the cell tokens alive are split into sets A and B (a checkerboard on the first merge, so
    every 4-neighbour pair crosses the split; then alternating in token order),
  * each A token picks its most similar ADJACENT B token (cosine of the attention keys),
  * the r best A tokens (r = frac x alive cells) are merged into their B partners: features
    averaged by size, masks unioned, boxes widened, sizes added,
  * dead tokens are dropped (the batch is compacted to the most tokens any position keeps).
Attention adds log(size) to the scores (proportional attention). The self-attention pair bias
is cheaper than AtTenTris's FLAT-2D MLP (which at 207 x 207 pairs x 512 positions dominated time
and memory): a learned table over the row / column offset between the two blobs' centres, plus
a learned value for blobs that touch (4-adjacent), per head. The candidates' cross-attention
keeps the FLAT-2D MLP (<= 96 x ~41 pairs).

sky=True (v2, "pre-tokenized sky"): the empty cells above each column's top are one token per
column from the start (its cells' mean embedding + a sky embedding + the column height), so
positions start with ~77 tokens (stack cells + sky columns) instead of 200 and the merge budget
goes to the stack. Sky columns are ordinary mergeable tokens: they may join each other or the
cells next to them.

pretok=True (v3): the rule part of tome_tok.py, precomputed per position (batch["pretok"], one group
id per cell: 0 = sky, 1.. = solid blocks, -1 = single cell): one sky token (the rows above the
tallest column, with the 10 column heights as extras, as AtTenTris's sky), one token per solid
filled block (>= 2 x 2), and single cells for everything else below the sky (the air above lower
columns, wells, holes, gaps, ragged surface cells). All of them merge as usual. flat_bias=True uses
AtTenTris's FLAT-2D pair-feature MLP for the self-attention instead of the centre-offset table
(affordable at ~45 tokens).

streams=K > 1 (v4, multi-partition): at the first merging layer the cell tokens are copied into K
streams (+ a learned stream embedding); each stream merges only within itself, judging similarity
with attention head s's keys, so the board ends up tiled K different ways and one cell can belong
to a blob in every stream (a TSD and the well under it can both be blobs). The streams never
branch further; all their tokens attend to each other (FLAT-2D's overlap feature then sees blobs of
different streams sharing cells), and the candidates cross-attend to the blobs of every stream.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import model as M
import tokenizer as T

PIECE_FIRST = T.KINDS.index("current")       # current, hold, preview1..5 are contiguous in KINDS
N_PIECE_TOK = 7
N_GROUPS = 17                                   # sky + up to 16 blocks (pretok.py caps at 24; data max 12)
# A/B split after the first merge (the model's `split`, train.py --tome-split):
#   "token" (models up to v4.1): alternate in token order. Token order is row-major, so side-by-side
#           neighbours always cross the split and stacked ones only ~half the time (tome_orient.py):
#           merging is biased sideways;
#   "hash"  : each blob's side is a hashed bit of (board, merge step, stream, its first cell): any two
#           adjacent blobs cross with probability 1/2 whatever their direction; deterministic and the same
#           for a position in any batch (hash_side);
#   "greedy": no A/B sets at all: a greedy matching over all adjacent same-stream pairs (greedy_match);
#   analysis only: "col" (alternate in column-major order of the blobs' lowest-left corner), "random".
# SPLIT_MODE (analysis) overrides every model's own split when set; MERGE_HOOK(dict) sees every merge's
# sets, similarities and choices.
MERGE_HOOK = None
SPLIT_MODE = None


_W = torch.randint(1, 1 << 20, (200,), generator=torch.Generator().manual_seed(20261013))   # board -> position key


def _mix(x):
    """A 31-bit integer hash (multiply / xor-shift), elementwise on an int64 tensor."""
    m = (1 << 31) - 1
    x = ((x & m) * 1103515245 + 12345) & m
    x = x ^ (x >> 16)
    x = (x * 1664525 + 1013904223) & m
    x = x ^ (x >> 13)
    x = (x * 22695477 + 1) & m
    return x ^ (x >> 16)


def hash_side(cells, anchor, step, stream):
    """The "hash" split: A/B side (0/1) of each token from (board, merge step, stream, the token's first
    cell). A fixed table of bits per cell is not enough: the cells in play cluster in a few rows, where a
    fixed table is off 50% by up to ~7 points in one direction (test_tome_split.py); keying on the board
    makes every position's split a fresh draw, still deterministic and the same in any batch."""
    key = (cells.long().reshape(cells.shape[0], -1) * _W.to(cells.device)).sum(-1) & ((1 << 24) - 1)     # (B,)
    x = _mix(key[:, None] * 64 + step * 4 + stream)
    return (_mix(x * 256 + anchor) >> 11) & 1


def greedy_match(S, valid, r, stream, anchor, n_streams):
    """The "greedy" merge (no A/B sets): over the allowed pairs (valid: adjacent, same stream, both
    board blobs), take pairs best-first by key similarity, skipping any pair with a blob already taken,
    and merge the r best of them per stream. Computed in parallel by locally dominant rounds -- a pair
    is taken when it is the best remaining pair of both its blobs; repeat on the free blobs -- which
    gives exactly the sequential greedy matching (ties broken by the pair's cells, so distinct weights:
    deterministic, batch-independent, and every round takes at least the best free pair). Each blob
    joins at most one partner per step; if the matching has fewer than r pairs, the step merges fewer.
    Returns sel (B, N) = the blob that merges away, dst = its partner, score = the pair's similarity."""
    B, N, _ = S.shape
    dev = S.device
    ar = torch.arange(N, device=dev)
    a = anchor.double()
    lo, hi = torch.minimum(a[:, :, None], a[:, None, :]), torch.maximum(a[:, :, None], a[:, None, :])
    W = (S.double() - 1e-9 * (lo * 200 + hi) / 40000.0).masked_fill(~valid, float("-inf"))
    partner = torch.full((B, N), -1, dtype=torch.long, device=dev)
    free = valid.any(-1)
    while True:
        v, j = W.masked_fill(~(free[:, :, None] & free[:, None, :]), float("-inf")).max(-1)
        ok = torch.isfinite(v)
        mutual = ok & ok.gather(1, j) & (j.gather(1, j) == ar)
        if not bool(mutual.any()):
            break
        partner = torch.where(mutual, j, partner)
        free = free & ~mutual
    w = W.gather(2, partner.clamp(min=0)[..., None]).squeeze(-1)
    src = (partner >= 0) & (partner < ar)                      # each pair once: the later token merges into the earlier
    sel = torch.zeros_like(src)
    for st in range(n_streams):
        ins = src & (stream == st)
        sc = w.masked_fill(~ins, float("-inf"))
        order = sc.argsort(-1, descending=True)
        rank = torch.empty_like(order).scatter_(1, order, ar.expand(B, N))
        sel = sel | (ins & (rank < r[:, st, None]))
    return sel, partner.clamp(min=0), w.float()


class Attn(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.h, self.dh = heads, d // heads
        self.q, self.k, self.v, self.o = (nn.Linear(d, d) for _ in range(4))

    def forward(self, xq, xk, bias, key_pad, key_logsize=None):
        B, Nq, d = xq.shape
        Nk = xk.shape[1]
        q = self.q(xq).view(B, Nq, self.h, self.dh).transpose(1, 2)
        k = self.k(xk).view(B, Nk, self.h, self.dh).transpose(1, 2)
        v = self.v(xk).view(B, Nk, self.h, self.dh).transpose(1, 2)
        s = q @ k.transpose(-1, -2) / math.sqrt(self.dh) + bias
        if key_logsize is not None:
            s = s + key_logsize[:, None, None, :]
        s = s.masked_fill(key_pad[:, None, None, :], float("-inf"))
        a = s.softmax(-1)
        out = (a @ v).transpose(1, 2).reshape(B, Nq, d)
        return self.o(out), a, k.transpose(1, 2).reshape(B, Nk, d)


class Block(nn.Module):
    def __init__(self, d, heads, ff=4):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.att = Attn(d, heads)
        self.ff = nn.Sequential(nn.Linear(d, ff * d), nn.GELU(), nn.Linear(ff * d, d))


class RelBias(nn.Module):
    """Per head: table[round(centre row offset), round(centre col offset)] + touching * w;
    pairs without a box (piece tokens) get a learned constant."""

    def __init__(self, heads):
        super().__init__()
        self.table = nn.Parameter(torch.zeros(39 * 19, heads))
        self.touch = nn.Parameter(torch.zeros(heads))
        self.nobox = nn.Parameter(torch.zeros(heads))

    def forward(self, mask, size, has_box, adj):
        r = torch.arange(200, device=mask.device) // 10
        c = torch.arange(200, device=mask.device) % 10
        sz = size.clamp(min=1.0)
        cr = (mask @ r.float()) / sz
        cc = (mask @ c.float()) / sz
        dr = (cr[:, :, None] - cr[:, None, :]).round().long().clamp(-19, 19) + 19
        dc = (cc[:, :, None] - cc[:, None, :]).round().long().clamp(-9, 9) + 9
        b = F.embedding(dr * 19 + dc, self.table) + adj[..., None].float() * self.touch   # (B, N, N, H)
        both = has_box[:, :, None] & has_box[:, None, :]
        b = torch.where(both[..., None], b, self.nobox)
        return b.permute(0, 3, 1, 2)


def adjacency(mask):
    """mask (B, N, 200) float -> (B, N, N) bool: some cell of i is 4-adjacent to some cell of j."""
    B, N, _ = mask.shape
    g = mask.view(B, N, 20, 10)
    dil = torch.zeros_like(g)
    dil[:, :, 1:, :] += g[:, :, :-1, :]
    dil[:, :, :-1, :] += g[:, :, 1:, :]
    dil[:, :, :, 1:] += g[:, :, :, :-1]
    dil[:, :, :, :-1] += g[:, :, :, 1:]
    adj = (dil.view(B, N, 200) > 0).float() @ mask.transpose(1, 2) > 0
    return adj & ~torch.eye(N, dtype=torch.bool, device=mask.device)


def override_of(s):
    """--merge-override "6:0.30,5:0.2" -> {6: 0.3, 5: 0.2} (empty -> {})."""
    return {int(k): float(v) for k, v in (p.split(":") for p in s.split(",") if p)} if s else {}


def budget_of(s):
    """--tome-budget BASE,AT,SLOPE,FIRST (a checkpoint's args) -> the `budget` dict (None if empty)."""
    if not s:
        return None
    base, at, slope, first = (float(v) for v in s.split(","))
    return {"base": base, "at": at, "slope": slope, "first": first}


class AtTenTrisToMe(nn.Module):
    def __init__(self, d=64, heads=4, layers=4, merge=(0.0, 0.45, 0.45, 0.45), n_teachers=2, sky=False,
                 pretok=False, flat_bias=False, streams=1, budget=None, split="token"):
        super().__init__()
        assert split in ("token", "hash", "greedy")
        self.split = split
        # budget (dict or None): a per-position merge plan instead of fixed fractions. The layers with
        # merge > 0 merge; at the first, each position's board-token count n0 sets a budget
        # T = base + max(0, (n0 - at) * slope) blobs per stream; the first step merges `first` of the
        # tokens, the rest reach T in equal-ratio steps at the last merging layer; no merging if
        # n0 <= T. E.g. {"base": 20, "at": 36, "slope": 0.25, "first": 0.10}.
        self.budget = budget
        # inference-time override (analysis): {layer (1-based): fraction} -- that layer merges this share
        # of each stream's board tokens instead of its scheduled / budgeted step. Not saved, not trained.
        self.frac_override = {}
        assert streams <= heads
        self.streams, self.heads = streams, heads
        if streams > 1:
            self.stream_emb = nn.Embedding(streams, d)
        assert len(merge) == layers
        assert not (sky and pretok)
        self.d, self.merge, self.sky, self.pretok, self.flat_bias = d, tuple(merge), sky, pretok, flat_bias
        if pretok:
            self.sky_emb = nn.Parameter(torch.zeros(d))
            self.block_emb = nn.Parameter(torch.zeros(d))
            self.sky_heights = nn.Linear(10, d)                   # the 10 column heights / 20, as AtTenTris's sky extras
        if sky:
            self.sky_emb = nn.Parameter(torch.zeros(d))
            self.sky_h = nn.Embedding(21, d)                      # the column's height 0..20
        self.random_merge = False      # analysis baseline: merge random adjacent pairs (same schedule)
        self.val = nn.Embedding(2, d)
        self.pos = M.Pos2D(d)
        self.slot = nn.Embedding(N_PIECE_TOK, d)                 # current / hold / preview1..5
        self.piece = nn.Embedding(M.N_PIECES, d)
        self.is_cell = nn.Parameter(torch.zeros(d))
        self.bias = M.Bias(heads) if flat_bias else RelBias(heads)
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        # candidates and heads: as AtTenTris
        self.c_piece = nn.Embedding(M.N_PIECES, d)
        self.c_hold, self.c_spin, self.c_lines = nn.Embedding(2, d), nn.Embedding(3, d), nn.Embedding(5, d)
        self.c_attack = nn.Linear(1, d)
        self.style = nn.Embedding(n_teachers, d)
        self.c_bias = M.Bias(heads)
        self.cross = M.Block(d, heads)
        self.c_norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, 1)
        self.q1 = nn.Linear(d, 1)
        self.qs = nn.Linear(d, 1)

    # ---------------------------------------------------------------- merging
    def budget_plan(self, n0, n_steps):
        """Planned board-token counts after each merging step, per position and stream (n0: (B, S))."""
        bp = self.budget
        T = torch.minimum(bp["base"] + ((n0 - bp["at"]) * bp["slope"]).clamp(min=0), n0)
        n1 = torch.maximum(n0 - (bp.get("first", 0.0) * n0).floor(), T)
        out = [n1]
        for j in range(1, n_steps):
            out.append(torch.maximum(n1 * (T / n1.clamp(min=1)) ** (j / max(1, n_steps - 1)), T))
        return torch.stack(out, -1).round()

    @staticmethod
    def _merge(x, size, mask, box, alive, is_cell, keys, frac, first, adj, par0, stream=None, n_streams=1, heads=4, r_plan=None,
               split="token", step=0, cells=None):
        B, N, d = x.shape
        dev = x.device
        split = SPLIT_MODE or split
        if stream is None:
            stream = torch.zeros(B, N, dtype=torch.long, device=dev)
        cand = alive & is_cell
        if n_streams > 1:
            # stream s judges similarity with head s's keys
            kh = F.normalize(keys.view(B, N, heads, d // heads), dim=-1)
            Sh = torch.einsum("bnhd,bmhd->bhnm", kh, kh)
            S = Sh.gather(1, (stream % heads)[:, None, :, None].expand(B, 1, N, N)).squeeze(1)
        else:
            kn = F.normalize(keys, dim=-1)
            S = kn @ kn.transpose(1, 2)
        S0 = S
        same = stream[:, :, None] == stream[:, None, :]
        r_all = torch.stack([(frac * (cand & (stream == st)).sum(1).float()).floor().long() if r_plan is None else r_plan[:, st]
                             for st in range(n_streams)], 1)                      # (B, streams): merges this step
        if split == "greedy":
            # no A/B sets: a greedy matching over all adjacent same-stream pairs (greedy_match)
            A = Bs = None
            eye = torch.eye(N, dtype=torch.bool, device=dev)[None]
            valid = cand[:, :, None] & cand[:, None, :] & adj & same & ~eye
            sel, dst, score = greedy_match(S, valid, r_all, stream, mask.argmax(-1), n_streams)
        else:
            if first:
                par = par0                                           # checkerboard for cells (sky columns: by column)
            elif split == "hash":
                par = hash_side(cells, mask.argmax(-1), step, stream)
            elif split == "token":
                par = torch.zeros_like(stream)
                for st in range(n_streams):                          # alternate within each stream
                    cs = cand & (stream == st)
                    par = torch.where(stream == st, (torch.cumsum(cs.long(), 1) - 1) % 2, par)
            else:
                if split == "col":
                    key = (box[..., 2] * 100 + box[..., 0]).float() + torch.arange(N, device=dev) / (10.0 * N)
                else:
                    key = torch.rand(B, N, device=dev)
                par = torch.zeros_like(stream)
                for st in range(n_streams):
                    cs = cand & (stream == st)
                    order = key.masked_fill(~cs, float("inf")).argsort(-1)
                    rank = torch.empty_like(order).scatter_(1, order, torch.arange(N, device=dev).expand(B, N))
                    par = torch.where(cs, rank % 2, par)
            A = cand & (par == 0)
            Bs = cand & (par == 1)
            valid = A[:, :, None] & Bs[:, None, :] & adj & same
            S = S.masked_fill(~valid, float("-inf"))
            score, dst = S.max(-1)                                   # (B, N): best adjacent B per A token
            ok = torch.isfinite(score)
            sel = torch.zeros_like(ok)
            for st in range(n_streams):                              # each stream merges its own share
                ins = stream == st
                sc = score.masked_fill(~(ok & ins), float("-inf"))
                order = sc.argsort(-1, descending=True)
                rank = torch.empty_like(order).scatter_(1, order, torch.arange(N, device=dev).expand(B, N))
                sel = sel | (ok & ins & (rank < r_all[:, st, None]))
        if MERGE_HOOK is not None:
            MERGE_HOOK(dict(A=A, Bs=Bs, S=S0, dst=dst, sel=sel, mask=mask, size=size, cand=cand, stream=stream, first=first, keys=keys,
                            adj=adj, r=r_all, split=split))
        self_w = size * sel
        xs = x * size[..., None]
        add = torch.zeros_like(xs).scatter_add(1, dst[..., None].expand(-1, -1, d), xs * sel[..., None])
        add_s = torch.zeros_like(size).scatter_add(1, dst, self_w)
        new_size = size + add_s
        x = torch.where(sel[..., None], x, (xs + add) / new_size.clamp(min=1e-6)[..., None])
        mask = mask + torch.zeros_like(mask).scatter_add(1, dst[..., None].expand(-1, -1, mask.shape[-1]), mask * sel[..., None])
        big = torch.full_like(box, 1000)
        lo = torch.where(sel[..., None], box, big)
        hi = torch.where(sel[..., None], box, -big)
        blo = box.scatter_reduce(1, dst[..., None].expand(-1, -1, 4), lo, reduce="amin")
        bhi = box.scatter_reduce(1, dst[..., None].expand(-1, -1, 4), hi, reduce="amax")
        box = torch.cat([blo[..., 0:1], bhi[..., 1:2], blo[..., 2:3], bhi[..., 3:4]], -1)
        size = torch.where(sel, size, new_size)
        alive = alive & ~sel
        # compact: alive tokens first (stable), drop the columns no position needs
        keep = int(alive.sum(1).max())
        o = (~alive).int().argsort(dim=1, stable=True)[:, :keep]
        g = lambda t: torch.gather(t, 1, o.view(B, keep, *([1] * (t.dim() - 2))).expand(B, keep, *t.shape[2:]))
        return g(x), g(size), g(mask), g(box), g(alive), g(is_cell), g(par0), g(stream)

    @staticmethod
    def _compact(alive, *ts):
        B = alive.shape[0]
        keep = int(alive.sum(1).max())
        o = (~alive).int().argsort(dim=1, stable=True)[:, :keep]
        g = lambda t: torch.gather(t, 1, o.view(B, keep, *([1] * (t.dim() - 2))).expand(B, keep, *t.shape[2:]))
        return [g(alive)] + [g(t) for t in ts]

    # ---------------------------------------------------------------- forward
    def forward(self, batch, return_attn=False, heads=False):
        cells = batch["cells"]
        B = cells.shape[0]
        dev = cells.device
        phi = self.val(cells) + self.pos.cells()                  # (B, 200, d)
        # the 7 piece tokens from the packed token list
        kind = batch["tok_kind"]
        is_p = (kind >= PIECE_FIRST) & (kind < PIECE_FIRST + N_PIECE_TOK)
        po = (~is_p).int().argsort(dim=1, stable=True)[:, :N_PIECE_TOK]
        pk = torch.gather(kind, 1, po) - PIECE_FIRST
        pp = torch.gather(batch["tok_piece"], 1, po).clamp(min=0)
        pvalid = torch.gather(is_p, 1, po)
        xp = self.slot(pk.clamp(0, N_PIECE_TOK - 1)) + self.piece(pp)
        x = torch.cat([phi + self.is_cell, xp], 1)                # (B, 207, d)
        N = x.shape[1]
        size = torch.ones(B, N, device=dev)
        eye = torch.eye(200, device=dev)
        mask = torch.cat([eye.expand(B, 200, 200), torch.zeros(B, N_PIECE_TOK, 200, device=dev)], 1)
        r = torch.arange(200, device=dev) // 10
        c = torch.arange(200, device=dev) % 10
        cbox = torch.stack([r, r, c, c], -1).expand(B, 200, 4)
        box = torch.cat([cbox, torch.full((B, N_PIECE_TOK, 4), -1, device=dev, dtype=torch.long)], 1)
        alive = torch.cat([torch.ones(B, 200, dtype=torch.bool, device=dev), pvalid], 1)
        is_cell = torch.cat([torch.ones(B, 200, dtype=torch.bool, device=dev),
                             torch.zeros(B, N_PIECE_TOK, dtype=torch.bool, device=dev)], 1)
        par0 = torch.cat([((r + c) % 2).expand(B, 200), torch.zeros(B, N_PIECE_TOK, dtype=torch.long, device=dev)], 1)
        if self.sky:
            g = cells.view(B, 20, 10).bool()
            rows = torch.arange(1, 21, device=dev).view(1, 20, 1)
            h = torch.where(g, rows, torch.zeros_like(rows)).amax(1)               # (B, 10) column heights
            sky_cell = (r.view(1, 20, 10) >= h[:, None, :]).view(B, 200)          # above each column's top
            smask = (sky_cell.view(B, 20, 10).unsqueeze(1) & (c.view(1, 1, 20, 10) == torch.arange(10, device=dev).view(1, 10, 1, 1))).view(B, 10, 200).float()
            ssize = smask.sum(-1)
            sx = (smask @ phi) / ssize.clamp(min=1.0)[..., None] + self.is_cell + self.sky_emb + self.sky_h(h)
            cols = torch.arange(10, device=dev)
            sbox = torch.stack([h, torch.full_like(h, 19), cols.expand(B, 10), cols.expand(B, 10)], -1)
            x = torch.cat([x[:, :200], sx, x[:, 200:]], 1)
            size = torch.cat([size[:, :200], ssize.clamp(min=1.0), size[:, 200:]], 1)
            mask = torch.cat([mask[:, :200], smask, mask[:, 200:]], 1)
            box = torch.cat([box[:, :200], sbox, box[:, 200:]], 1)
            alive = torch.cat([alive[:, :200] & ~sky_cell, ssize > 0, alive[:, 200:]], 1)
            is_cell = torch.cat([is_cell[:, :200], torch.ones(B, 10, dtype=torch.bool, device=dev), is_cell[:, 200:]], 1)
            par0 = torch.cat([par0[:, :200], (cols % 2).expand(B, 10), par0[:, 200:]], 1)
            alive, x, size, mask, box, is_cell, par0 = self._compact(alive, x, size, mask, box, is_cell, par0)
        if self.pretok:
            grp = batch["pretok"].long()                                            # (B, 200)
            K = N_GROUPS
            gmask = (grp[:, None, :] == torch.arange(K, device=dev).view(1, K, 1)).float()   # (B, K, 200)
            gsize = gmask.sum(-1)
            gx = (gmask @ phi) / gsize.clamp(min=1.0)[..., None] + self.is_cell
            g = cells.view(B, 20, 10).bool()
            h = torch.where(g, torch.arange(1, 21, device=dev).view(1, 20, 1), torch.zeros(1, 20, 1, dtype=torch.long, device=dev)).amax(1)
            gx[:, 0] = gx[:, 0] + self.sky_emb + self.sky_heights(h.float() / 20.0)
            gx[:, 1:] = gx[:, 1:] + self.block_emb
            big = torch.full_like(r, 99)
            rr, cc = r.expand(B, K, 200), c.expand(B, K, 200)
            on = gmask > 0
            gbox = torch.stack([torch.where(on, rr, big).amin(-1), torch.where(on, rr, -big).amax(-1),
                                torch.where(on, cc, big).amin(-1), torch.where(on, cc, -big).amax(-1)], -1)
            galive = gsize > 0
            gbox = torch.where(galive[..., None], gbox, torch.full_like(gbox, -1))
            first = on.float().argmax(-1)                                           # a cell of the group
            gpar = (first // 10 + first % 10) % 2
            x = torch.cat([x[:, :200], gx, x[:, 200:]], 1)
            size = torch.cat([size[:, :200], gsize.clamp(min=1.0), size[:, 200:]], 1)
            mask = torch.cat([mask[:, :200], gmask, mask[:, 200:]], 1)
            box = torch.cat([box[:, :200], gbox, box[:, 200:]], 1)
            alive = torch.cat([alive[:, :200] & (grp < 0), galive, alive[:, 200:]], 1)
            is_cell = torch.cat([is_cell[:, :200], torch.ones(B, K, dtype=torch.bool, device=dev), is_cell[:, 200:]], 1)
            par0 = torch.cat([par0[:, :200], gpar, par0[:, 200:]], 1)
            alive, x, size, mask, box, is_cell, par0 = self._compact(alive, x, size, mask, box, is_cell, par0)
        merged_yet = False
        stream = torch.zeros(B, x.shape[1], dtype=torch.long, device=dev)
        attn, trace = [], []
        plan, step, n_steps = None, 0, sum(1 for f in self.merge if f > 0)
        for li, (blk, frac) in enumerate(zip(self.blocks, self.merge)):
            adj = adjacency(mask)
            if self.flat_bias:
                bias = self.bias(*M.pair_features(box, box, mask, mask))
            else:
                bias = self.bias(mask, size, is_cell, adj)
            h = blk.n1(x)
            y, a, keys = blk.att(h, h, bias, ~alive, torch.log(size.clamp(min=1.0)))
            x = x + y
            attn.append(a)
            if frac > 0:
                if self.random_merge:
                    keys = torch.randn_like(keys)
                if self.streams > 1 and not merged_yet:
                    # split: copy the tokens into K streams (pieces only once), then compact
                    K = self.streams
                    x = torch.cat([x + self.stream_emb.weight[st] for st in range(K)], 1)
                    keys = torch.cat([keys] * K, 1)
                    size, mask, box, is_cell, par0 = [torch.cat([t_] * K, 1) for t_ in (size, mask, box, is_cell, par0)]
                    alive = torch.cat([alive] + [alive & is_cell[:, :alive.shape[1]]] * (K - 1), 1)
                    stream = torch.cat([torch.full_like(stream, st) for st in range(K)], 1)
                    alive, x, size, mask, box, is_cell, par0, stream, keys = self._compact(alive, x, size, mask, box, is_cell, par0, stream, keys)
                    adj = adjacency(mask)
                r_plan = None
                if self.budget is not None or (li + 1) in self.frac_override:
                    cand = alive & is_cell
                    cnt = torch.stack([(cand & (stream == s_)).sum(1) for s_ in range(self.streams)], 1).float()   # (B, S)
                    if (li + 1) in self.frac_override:
                        r_plan = (cnt * self.frac_override[li + 1]).floor().long()
                    else:
                        if plan is None:
                            plan = self.budget_plan(cnt, n_steps)                  # (B, S, n_steps) planned counts
                        r_plan = (cnt - plan[..., step]).clamp(min=0).floor().long()
                x, size, mask, box, alive, is_cell, par0, stream = self._merge(
                    x, size, mask, box, alive, is_cell, keys, frac, not merged_yet, adj, par0, stream, self.streams, self.heads, r_plan,
                    self.split, step, cells)
                step += 1
                merged_yet = True
            x = x + blk.ff(blk.n2(x))
            if return_attn:
                trace.append({"mask": mask.detach(), "alive": alive.detach(), "size": size.detach(), "stream": stream.detach()})
        hb, pad = x, ~alive

        # --- candidates (as AtTenTris) ---
        cc = batch["cand_cells"]
        cpad = cc[..., 0] == -1
        onb = cc >= 0
        idx = cc.clamp(min=0)
        cphi = torch.gather(phi.unsqueeze(1).expand(-1, cc.shape[1], -1, -1), 2, idx.unsqueeze(-1).expand(-1, -1, -1, self.d))
        cphi = (cphi * onb.unsqueeze(-1)).sum(2) / onb.sum(-1, keepdim=True).clamp(min=1)
        meta = batch["cand_meta"]
        rr, cl = idx // 10, idx % 10
        bigv = torch.full_like(rr, 99)
        cb = torch.stack([torch.where(onb, rr, bigv).min(-1).values, torch.where(onb, rr, -bigv).max(-1).values,
                          torch.where(onb, cl, bigv).min(-1).values, torch.where(onb, cl, -bigv).max(-1).values], -1)
        cb = torch.where((onb.any(-1) & ~cpad).unsqueeze(-1), cb, torch.full_like(cb, -1))
        y = (cphi + self.c_piece(meta[..., 0]) + self.c_hold(meta[..., 1]) + self.c_spin(meta[..., 2])
             + self.c_lines(meta[..., 3].clamp(max=4)) + self.c_attack(meta[..., 4:5].float() / 10.0)
             + self.pos.box(cb) + self.style(batch["teacher"]).unsqueeze(1))
        cmask = torch.zeros(B, cc.shape[1], 200, device=dev)
        cmask.scatter_(2, idx, onb.float())
        cfeats, cboth = M.pair_features(cb, box, cmask, mask)
        y, ca = self.cross(y, self.c_bias(cfeats, cboth), pad, ctx=hb, ctx_norm=self.c_norm)
        logit = self.head(y).squeeze(-1).masked_fill(cpad, float("-inf"))
        out = {"logit": logit, "q1": self.q1(y).squeeze(-1), "qs": self.qs(y).squeeze(-1)} if heads else logit
        if return_attn:
            return out, {"self": attn, "cross": ca, "merges": trace}
        return out
