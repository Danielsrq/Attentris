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

gate={"lambda": L} (MergeTris): no merge budget. At each merging layer a learned scorer (MergeGate) gives every
adjacent same-stream pair a merge score s(i, j) from the two blobs (projected features: sum, |difference|,
product), their geometry and their key cosine; FiLM from the board context (mean board blob, current, hold,
preview 1, preview 2) scales and shifts the scorer's hidden units per position. Greedy matching on s, and the
pairs with s > 0 merge. gate["registers"] = R > 0: the context is R learned register tokens instead of the mean
board blob (they join the self-attention from layer 1 like the piece tokens, never merge, one copy for all
streams, and are kept out of the candidates' cross-attention: their only job is the gate's context). Trained straight-through: forward a hard merge, backward the merge weight acts as
sigmoid(s). The loss adds L x ((board blobs left - T) / board blobs at the first merge)^2, T = the budget's target
(--tome-budget, as l4g: 20 + (n0 - 36) / 4 per stream): a soft budget on how many, the gate learns which. (A linear
cost L x blobs left, without T, is all-or-nothing in pilots: L = 0 stops merging, L >= 0.1 merges every matched pair.)

metric=True (DTEM, Lee & Hong, NeurIPS 2024, arXiv 2412.10569; reference code github.com/movinghoon/dtem): the
merge similarity comes from a learned merge metric per merging layer (a linear map of the layer's normed input,
detached, so the backbone never learns from it; stream s uses its slice), not from the attention keys. It starts
as a copy of the key rows each stream used (init_metric_from_keys), so at the start it merges exactly as ToMe.
At inference (and in hard training steps) the merging is the usual one with this similarity. In training with
soft_merge=True (train.py --freeze-base: the backbone frozen, only the metric trains, DTEM's modular recipe) the
merge is relaxed (_soft_merge): soft greedy matching (r rounds of a softmax over the allowed pairs, each round's
picks suppressed by log(1 - weight) at both their blobs) and soft merging (the source keeps its feature, its mass
and cell weights shrink by the share it gives; the partner gains them; attention adds log mass), so the task loss
gives the metric a true gradient. The r blobs per stream that gave the most (lowest share still present) take no
part in later merges (DTEM's size sort, by share instead of mass since our blobs start at different sizes), which
rules out merge cycles.
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
# analysis only (structure_study.py --force): MERGE_GUIDE = (B, 200) long, a group id per cell (-1 none) for
# the batch about to run. Two blobs whose cells all lie in the same group get GUIDE_BONUS added to their key
# similarity, so they merge first (forced recovery of a rule token); everything else is the model's own merging.
MERGE_GUIDE = None
GUIDE_BONUS = 4.0


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


class MergeGate(nn.Module):
    """MergeTris: s(i, j) = w . ReLU(gamma * (W u + b) + beta) + alpha * cos(keys) + c0, for an adjacent pair
    with u = [p_i + p_j, |p_i - p_j|, p_i * p_j, geometry] (symmetric in i, j) and p = a projection of the
    blob; (gamma, beta) = FiLM(board context). Starts as ToMe's cosine rule: FiLM gives gamma = 1, beta = 0
    and w = 0, so s = 4 cos - 2 (merge above cosine 0.5) until training moves it.
    Context: [mean board blob or the R registers ; current ; hold ; preview 1 ; preview 2]."""
    N_GEO = 6

    def __init__(self, d, hid=64, proj=32, registers=0):
        super().__init__()
        self.hid, self.n_reg = hid, registers
        self.N_CTX = 4 + max(1, registers)
        self.norm = nn.LayerNorm(d)
        self.proj = nn.Linear(d, proj)
        self.film = nn.Sequential(nn.Linear(self.N_CTX * d, hid), nn.ReLU(), nn.Linear(hid, 2 * hid))
        nn.init.zeros_(self.film[2].weight)
        nn.init.zeros_(self.film[2].bias)
        self.w_in = nn.Linear(3 * proj + self.N_GEO, hid)
        self.w_out = nn.Linear(hid, 1)
        nn.init.zeros_(self.w_out.weight)
        nn.init.zeros_(self.w_out.bias)
        self.alpha = nn.Parameter(torch.tensor(4.0))
        self.c0 = nn.Parameter(torch.tensor(-2.0))

    def context(self, x, alive, is_cell, slot):
        """slot: 0.. the piece tokens (current, hold, previews), 10.. the registers, -1 board tokens."""
        h = self.norm(x)
        if self.n_reg:
            ctx = [(h * (alive & (slot == 10 + k)).float()[..., None]).sum(1) for k in range(self.n_reg)]
        else:
            board = (alive & is_cell).float()[..., None]
            ctx = [(h * board).sum(1) / board.sum(1).clamp(min=1.0)]
        for k in range(4):                                        # current, hold, preview 1, preview 2 (0 if absent)
            ctx.append((h * (alive & (slot == k)).float()[..., None]).sum(1))
        return torch.cat(ctx, -1)

    def forward(self, x, ctx, bi, ii, jj, geo, cos):
        g = self.film(ctx)
        gamma, beta = 1.0 + g[:, :self.hid], g[:, self.hid:]
        p = self.proj(self.norm(x))
        pi, pj = p[bi, ii], p[bi, jj]
        a = gamma[bi] * self.w_in(torch.cat([pi + pj, (pi - pj).abs(), pi * pj, geo], -1)) + beta[bi]
        return self.w_out(F.relu(a)).squeeze(-1) + self.alpha * cos + self.c0


def gate_of(a):
    """A checkpoint's args (dict) -> the `gate` dict (None for budgeted ToMe)."""
    return {"lambda": a.get("gate_lambda", 1.0), "target": budget_of(a.get("tome_budget", "")),
            "registers": a.get("gate_registers", 0)} if a.get("tome_gate") else None


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


def blob_kind(mask, cells):
    """(B, N) blob kind for pure merges: 0 = all empty, 1 = all filled, 2 = mixed. Relative to the blob's
    mass, so fractional (soft-merged) masks work; for 0/1 masks the same as nf == 0 / nf == size."""
    B = mask.shape[0]
    nf = (mask * cells.float().reshape(B, 1, -1)).sum(-1)
    ms = mask.sum(-1)
    return torch.where(nf <= 1e-3 * ms, 0, torch.where(nf >= (1 - 1e-3) * ms, 1, 2))


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


def apply_variant(net, spec):
    """Inference-time merge options from a spec like "split=greedy,pure,6:x2,5:0.3" ('none' or '' = as
    trained): split=token|hash|greedy|col|random, pure (blobs merge only with blobs of the same kind: both
    all-empty or both all-filled), LAYER:xK (that layer merges K times the merges its budget asks for),
    LAYER:F (that layer merges a fraction F of its board blobs), tetro (tetromino merges, implies pure: two
    empty blobs of a and b cells merge only if a + b <= 4 or both are multiples of 4, and never with the sky
    pre-token, so every 4-cell empty blob is a tetromino shape), rounds=K (with tetro: after each merging
    layer's budgeted merges, up to K more matching rounds of only the a + b <= 4 empty pairs, every matched
    pair merging, outside the budget). Returns a readable name."""
    net.split = net.trained_split
    net.pure, net.r_scale, net.frac_override = net.trained_pure, {}, {}
    net.tetro, net.tetro_rounds, net.sky_gap = net.trained_tetro, 0, net.trained_sky_gap
    names = []
    for it in ([] if spec in ("", "none") else spec.split(",")):
        if it == "tetro":
            net.tetro = net.pure = True
            names.append("tetromino merges")
        elif it == "skygap":
            net.sky_gap = True
            names.append("sky one row higher")
        elif it.startswith("rounds="):
            net.tetro_rounds = int(it[7:])
            names.append("+%d completion rounds per layer" % net.tetro_rounds)
        elif it.startswith("split="):
            net.split = it[6:]
            names.append({"hash": "hash split", "greedy": "greedy matching", "token": "token split"}.get(net.split, net.split + " split"))
        elif it == "pure":
            net.pure = True
            names.append("pure (no filled + empty)")
        elif ":x" in it:
            L, k = it.split(":x")
            net.r_scale[int(L)] = float(k)
            names.append("layer %s merges x%s" % (L, k))
        else:
            L, f = it.split(":")
            net.frac_override[int(L)] = float(f)
            names.append("layer %s merges %d%%" % (L, round(100 * float(f))))
    return ", ".join(names) or "as trained"


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
                 pretok=False, flat_bias=False, streams=1, budget=None, split="token", pure=False, gate=None, metric=False):
        super().__init__()
        assert split in ("token", "hash", "greedy")
        self.split = self.trained_split = split
        # pure=True: blobs merge only with blobs of the same kind (both all-empty or both all-filled), so a
        # blob never mixes filled and empty cells (the pre-tokens are all pure)
        self.pure = self.trained_pure = pure
        self.tetro = self.trained_tetro = False   # tetromino merges (apply_variant "tetro"; _merge sky_cells)
        self.tetro_rounds = 0
        self.sky_gap = self.trained_sky_gap = False   # pre-tokens: the sky's lowest row as play cells (apply_variant "skygap")
        self.r_scale = {}              # inference-time: {layer (1-based): factor} on the merges the budget asks for
        # budget (dict or None): a per-position merge plan instead of fixed fractions. The layers with
        # merge > 0 merge; at the first, each position's board-token count n0 sets a budget
        # T = base + max(0, (n0 - at) * slope) blobs per stream; the first step merges `first` of the
        # tokens, the rest reach T in equal-ratio steps at the last merging layer; no merging if
        # n0 <= T. E.g. {"base": 20, "at": 36, "slope": 0.25, "first": 0.10}.
        self.budget = budget
        self.gate = gate
        if gate is not None:
            assert split == "greedy", "the merge gate scores pairs for the greedy matching"
            nr = gate.get("registers", 0)
            self.gates = nn.ModuleList(MergeGate(d, registers=nr) for f in merge if f > 0)
            if nr:
                self.registers = nn.Parameter(torch.randn(nr, d) * 0.02)
        self.aux_loss, self.gate_log = None, []
        # DTEM merge metric: per merging layer a linear map of the (detached) normed layer input; streams > 1:
        # one head-sized slice per stream (as the key head it replaces), else d numbers (as the full keys)
        self.soft_merge, self.soft_tau = False, (0.1, 0.1)       # DTEM tau1, tau2 (scores / tau1, softmax / tau2)
        self.use_metric = metric
        if metric:
            mdim = d if streams == 1 else streams * (d // heads)
            self.metric = nn.ModuleList(nn.Linear(d, mdim) for f in merge if f > 0)
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
    @torch.no_grad()
    def init_metric_from_keys(self):
        """The merge metric starts as the key rows each stream's similarity used (head s for stream s; all keys
        with one stream), so a fresh metric merges exactly as the ToMe model it is added to."""
        dh = self.d // self.heads
        layers = [li for li, f in enumerate(self.merge) if f > 0]
        for m, li in zip(self.metric, layers):
            k = self.blocks[li].att.k
            rows = torch.arange(self.d) if self.streams == 1 else torch.cat([torch.arange(st * dh, (st + 1) * dh) for st in range(self.streams)])
            m.weight.copy_(k.weight[rows])
            m.bias.copy_(k.bias[rows])

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
               split="token", step=0, cells=None, pure=False, gate=None, gstats=None, soft=None, sky_cells=None, small_only=False):
        B, N, d = x.shape
        dev = x.device
        split = SPLIT_MODE or split
        if stream is None:
            stream = torch.zeros(B, N, dtype=torch.long, device=dev)
        cand = alive & is_cell
        if n_streams > 1:
            # stream s judges similarity with head s's keys (or its slice of the DTEM merge metric: G = streams)
            G = keys.shape[-1] // (d // heads)
            kh = F.normalize(keys.view(B, N, G, d // heads), dim=-1)
            Sh = torch.einsum("bnhd,bmhd->bhnm", kh, kh)
            S = Sh.gather(1, (stream % G)[:, None, :, None].expand(B, 1, N, N)).squeeze(1)
        else:
            kn = F.normalize(keys, dim=-1)
            S = kn @ kn.transpose(1, 2)
        if MERGE_GUIDE is not None:
            g = MERGE_GUIDE.to(dev)[:, None, :].expand(B, N, mask.shape[-1])
            on = mask > 0.5
            big = 1 << 20
            mn = torch.where(on, g, torch.full_like(g, big)).amin(-1)
            mx = torch.where(on, g, torch.full_like(g, -big)).amax(-1)
            grp = torch.where((mn == mx) & (mn >= 0) & on.any(-1), mn, torch.full_like(mn, -1))
            S = S + GUIDE_BONUS * ((grp[:, :, None] == grp[:, None, :]) & (grp[:, :, None] >= 0)).float()
        S0 = S
        same = stream[:, :, None] == stream[:, None, :]
        if pure:
            # only blobs of the same kind merge: kind 0 = all empty, 1 = all filled, 2 = mixed (never paired)
            kind = blob_kind(mask, cells)
            same = same & (kind[:, :, None] == kind[:, None, :]) & (kind[:, :, None] < 2)
            if sky_cells is not None:
                # tetromino merges: empty + empty only if a + b <= 4 cells or both multiples of 4, never the sky
                n = mask.sum(-1).round().long()
                sky = (mask * sky_cells.float()[:, None, :]).sum(-1) > 0.5
                emp = (kind == 0) & ~sky
                small = (n[:, :, None] + n[:, None, :]) <= 4
                four = ((n % 4) == 0)[:, :, None] & ((n % 4) == 0)[:, None, :]
                ee = (kind == 0)[:, :, None] & (kind == 0)[:, None, :]
                ok = emp[:, :, None] & emp[:, None, :] & (small | (four & (not small_only)))
                same = same & (torch.where(ee, ok, torch.zeros_like(ok)) if small_only else torch.where(ee, ok, torch.ones_like(ok)))
        if soft is not None:
            return S, cand, same
        r_all = torch.stack([(frac * (cand & (stream == st)).sum(1).float()).floor().long() if r_plan is None else r_plan[:, st]
                             for st in range(n_streams)], 1)                      # (B, streams): merges this step
        if split == "greedy":
            # no A/B sets: a greedy matching over all adjacent same-stream pairs (greedy_match)
            A = Bs = None
            eye = torch.eye(N, dtype=torch.bool, device=dev)[None]
            valid = cand[:, :, None] & cand[:, None, :] & adj & same & ~eye
            if gate is not None:
                # MergeTris: the learned score replaces the cosine; every matched pair with s > 0 merges
                gmod, ctx = gate
                bi, ii, jj = valid.nonzero(as_tuple=True)
                rr = torch.arange(mask.shape[-1], device=dev)
                sz = size.clamp(min=1.0)
                cr, ccol = (mask @ (rr // 10).float()) / sz, (mask @ (rr % 10).float()) / sz
                ls = sz.log()
                filled = (mask * cells.float().reshape(B, 1, -1)).sum(-1) > 0.5
                geo = torch.stack([(cr[bi, ii] - cr[bi, jj]).abs() / 4, (ccol[bi, ii] - ccol[bi, jj]).abs() / 4,
                                   (ls[bi, ii] + ls[bi, jj]) / 4, (ls[bi, ii] - ls[bi, jj]).abs() / 4,
                                   (filled[bi, ii] & filled[bi, jj]).float(), (filled[bi, ii] != filled[bi, jj]).float()], -1)
                sc = gmod(x, ctx, bi, ii, jj, geo, S[bi, ii, jj])
                S = torch.full_like(S, float("-inf")).index_put((bi, ii, jj), sc)
                r_all = torch.full_like(r_all, N)
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
            MERGE_HOOK(dict(extra=small_only, A=A, Bs=Bs, S=S0, dst=dst, sel=sel, mask=mask, size=size, cand=cand, stream=stream, first=first, keys=keys,
                            adj=adj, r=r_all, split=split))
        if gate is not None:
            # straight-through: forward the hard merge (s > 0), backward the weight acts as sigmoid(s)
            hard = sel & (score > 0)
            soft = torch.sigmoid(score.masked_fill(~sel, 0.0))
            mw = torch.where(sel, hard.float() + soft - soft.detach(), torch.zeros_like(soft))
            sel = hard
            gstats.append(mw.sum(1))
        else:
            mw = sel.float()
        xs = x * size[..., None]
        add = torch.zeros_like(xs).scatter_add(1, dst[..., None].expand(-1, -1, d), xs * mw[..., None])
        add_s = torch.zeros_like(size).scatter_add(1, dst, size * mw)
        new_size = size + add_s
        x = torch.where(sel[..., None], x, (xs + add) / new_size.clamp(min=1e-6)[..., None])
        new_size = new_size.detach()
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
    def _soft_merge(x, size, mask, box, cand, S, adj, same, stream, n_streams, r_all, elig, pres, tau):
        """DTEM's relaxed merge (training only), on our pairs: the allowed pairs are adjacent, same stream (and kind),
        both eligible board blobs; pair (i, j) with i > j moves weight from i into j (the hard greedy's direction).
        Soft greedy matching (DTEM _select): per stream r rounds of softmax(S / tau1 / tau2) over all its pairs,
        after each round log(1 - picked weight) added at both blobs of every pair (each blob is matched once);
        the weights are then scaled so no blob gives or takes more than 1 (DTEM's row clip, by a detached factor).
        Soft merge (DTEM _merge_train): the source keeps its feature, its mass and cell weights shrink by the share
        it gives; the partner's feature becomes the mass-weighted mean, its mass and cells grow. Nothing dies.
        Then each stream's eligible blobs keep only the (count - r) with the largest share still present (pres x keep;
        DTEM sorts by mass, the same when every token starts at size 1) for the later merges.
        (DTEM first keeps each A token's top k2 = 3 B partners; not needed here: our pairs are only adjacent ones.)
        Returns x, size, mask, box, elig, pres (the share of each blob still present after all soft merges so far)."""
        B, N, d = x.shape
        dev = x.device
        eps = torch.finfo(torch.float32).eps
        tau1, tau2 = tau
        lower = torch.ones(N, N, dtype=torch.bool, device=dev).tril(-1)          # i > j
        valid = cand[:, :, None] & cand[:, None, :] & adj & same & lower
        A = torch.zeros(B, N, N, device=dev)
        for st in range(n_streams):
            vs = valid & (stream == st)[:, :, None]
            L = (S.float() / tau1).masked_fill(~vs, float("-inf"))
            for t in range(int(r_all[:, st].max()) if r_all.numel() else 0):
                on = (t < r_all[:, st]) & vs.flatten(1).any(-1)                  # (B,): positions still merging
                flat = (L / tau2).flatten(1)
                P = torch.where(on[:, None], flat.softmax(-1), torch.zeros_like(flat)).nan_to_num(0.0).view(B, N, N)
                A = A + P
                used = P.sum(2) + P.sum(1)                                      # this round's pick at each blob
                lm = torch.log((1 - used).clamp(min=eps))
                L = L + lm[:, :, None] + lm[:, None, :]
        tot = (A.sum(2) + A.sum(1)).detach()
        div = (tot - 1).clamp(min=0) + 1
        A = A / torch.maximum(div[:, :, None], div[:, None, :])
        out = A.sum(2)                                                          # share each blob gives
        keep = 1 - out
        keep = keep + (keep.clamp(0.0, 1.0) - keep).detach()
        At = A.transpose(1, 2)                                                  # [j, i]: what j takes from i
        in_mass = (At @ size[..., None]).squeeze(-1)
        # the partner's mass-weighted mean, written as x + (inflow - in_mass x) / (mass + in_mass): a blob that takes
        # nothing stays exactly as it is (also one whose mass already went to 0, which the plain mean would zero)
        x_new = x + (At @ (size[..., None] * x) - in_mass[..., None] * x) / (size + in_mass).clamp(min=1e-6)[..., None]
        new_size = size * keep + in_mass
        new_mask = mask * keep[..., None] + At @ mask
        # boxes: a partner's box widens over the sources it took a real share from (> 1%)
        took = At > 0.01
        big = torch.full_like(box, 1000)
        bi = box[:, None, :, :].expand(B, N, N, 4)
        lo = torch.where(took[..., None], bi, big[:, None]).amin(2)
        hi = torch.where(took[..., None], bi, -big[:, None]).amax(2)
        box = torch.stack([torch.minimum(box[..., 0], lo[..., 0]), torch.maximum(box[..., 1], hi[..., 1]),
                           torch.minimum(box[..., 2], lo[..., 2]), torch.maximum(box[..., 3], hi[..., 3])], -1)
        pres = keep if pres is None else pres * keep
        # eligibility for the later merges: per stream the (count - r) blobs with the most still present (ties: mass)
        ne = torch.zeros_like(elig)
        for st in range(n_streams):
            cs = cand & (stream == st)
            k = (cs.sum(1) - r_all[:, st]).clamp(min=0)
            sc = (pres + 1e-6 * new_size).detach().masked_fill(~cs, float("-inf"))
            order = sc.argsort(-1, descending=True)
            rank = torch.empty_like(order).scatter_(1, order, torch.arange(N, device=dev).expand(B, N))
            ne = ne | (cs & (rank < k[:, None]))
        return x_new, new_size, new_mask, box, ne, pres

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
        pslot = torch.where(pvalid, pk, -1)
        nr = 0 if self.gate is None else self.gate.get("registers", 0)
        if nr:      # MergeTris registers: after the pieces, slot 10.. (par0 20..)
            xp = torch.cat([xp, self.registers.expand(B, nr, self.d)], 1)
            pslot = torch.cat([pslot, 10 + torch.arange(nr, device=dev).expand(B, nr)], 1)
            pvalid = torch.cat([pvalid, torch.ones(B, nr, dtype=torch.bool, device=dev)], 1)
        NP = xp.shape[1]
        x = torch.cat([phi + self.is_cell, xp], 1)                # (B, 207 + R, d)
        N = x.shape[1]
        size = torch.ones(B, N, device=dev)
        eye = torch.eye(200, device=dev)
        mask = torch.cat([eye.expand(B, 200, 200), torch.zeros(B, NP, 200, device=dev)], 1)
        r = torch.arange(200, device=dev) // 10
        c = torch.arange(200, device=dev) % 10
        cbox = torch.stack([r, r, c, c], -1).expand(B, 200, 4)
        box = torch.cat([cbox, torch.full((B, NP, 4), -1, device=dev, dtype=torch.long)], 1)
        alive = torch.cat([torch.ones(B, 200, dtype=torch.bool, device=dev), pvalid], 1)
        is_cell = torch.cat([torch.ones(B, 200, dtype=torch.bool, device=dev),
                             torch.zeros(B, NP, dtype=torch.bool, device=dev)], 1)
        # par0: the cells' checkerboard side; for the piece tokens (never merge candidates) 10 + their slot
        # (0 current, 1 hold, 2.. previews; registers 20..), which the merge gate's context reads
        par0 = torch.cat([((r + c) % 2).expand(B, 200), 10 + pslot], 1)
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
            if self.sky_gap:
                # the sky starts one row higher: its lowest row (the first empty row over the stack) becomes play cells
                rowi = (torch.arange(200, device=dev) // 10).expand(B, 200)
                low = torch.where(grp == 0, rowi, torch.full_like(rowi, 99)).amin(1, keepdim=True)
                grp = torch.where((grp == 0) & (rowi == low), torch.full_like(grp, -1), grp)
            sky_grp = grp
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
        sky_cells = None
        if self.tetro:
            sky_cells = (sky_grp == 0) if self.pretok else torch.zeros(B, 200, dtype=torch.bool, device=dev)
        merged_yet = False
        stream = torch.zeros(B, x.shape[1], dtype=torch.long, device=dev)
        attn, trace = [], []
        plan, step, n_steps = None, 0, sum(1 for f in self.merge if f > 0)
        gstats, n0, elig, pres = [], None, None, None         # pres: soft merges, the share of each blob still present
        for li, (blk, frac) in enumerate(zip(self.blocks, self.merge)):
            adj = adjacency(mask)
            if self.flat_bias:
                bias = self.bias(*M.pair_features(box, box, mask, mask))
            else:
                bias = self.bias(mask, size, is_cell, adj)
            h = blk.n1(x)
            y, a, keys = blk.att(h, h, bias, ~alive, torch.log(size.clamp(min=1e-4 if self.soft_merge and self.training else 1.0)))
            x = x + y
            attn.append(a)
            if frac > 0:
                if self.use_metric:
                    keys = self.metric[step](h.detach())                   # DTEM: the learned merge metric
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
                soft = self.soft_merge and self.training and self.use_metric
                if soft and elig is None:
                    elig = alive & is_cell
                r_plan, gate = None, None
                if self.gate is not None:
                    gm = self.gates[step]
                    if n0 is None:
                        cnt = torch.stack([(alive & is_cell & (stream == s_)).sum(1) for s_ in range(self.streams)], 1).float()
                        n0 = cnt.sum(1)
                        tg = self.gate["target"]
                        T = n0 * 0 if tg is None else torch.minimum(tg["base"] + ((cnt - tg["at"]) * tg["slope"]).clamp(min=0), cnt).sum(1)
                    gate = (gm, gm.context(x, alive, is_cell, torch.where(is_cell, -1, par0 - 10)))
                elif self.budget is not None or (li + 1) in self.frac_override or (li + 1) in self.r_scale:
                    cand = alive & is_cell & (elig if soft else True)
                    cnt = torch.stack([(cand & (stream == s_)).sum(1) for s_ in range(self.streams)], 1).float()   # (B, S)
                    if (li + 1) in self.frac_override:
                        r_plan = (cnt * self.frac_override[li + 1]).floor().long()
                    else:
                        if plan is None:
                            plan = self.budget_plan(cnt, n_steps)                  # (B, S, n_steps) planned counts
                        r_plan = (cnt - plan[..., step]).clamp(min=0).floor().long()
                    if (li + 1) in self.r_scale:
                        r_plan = (r_plan.float() * self.r_scale[li + 1]).floor().long()
                if soft:
                    assert r_plan is not None, "soft merging needs a merge budget (--tome-budget)"
                    S, cand, same = self._merge(x, size, mask, box, alive & elig, is_cell, keys, frac, not merged_yet, adj, par0, stream,
                                                self.streams, self.heads, r_plan, self.split, step, cells, self.pure, soft=True)
                    x, size, mask, box, elig, pres = self._soft_merge(x, size, mask, box, cand, S, adj, same, stream, self.streams,
                                                                      r_plan, elig, pres, self.soft_tau)
                else:
                    x, size, mask, box, alive, is_cell, par0, stream = self._merge(
                        x, size, mask, box, alive, is_cell, keys, frac, not merged_yet, adj, par0, stream, self.streams, self.heads, r_plan,
                        self.split, step, cells, self.pure, gate, gstats, sky_cells=sky_cells)
                    for _ in range(self.tetro_rounds if self.tetro else 0):
                        # completion rounds: only empty pairs with a + b <= 4, every matched pair merges (similarity:
                        # the keys / metric of the merged blobs' current features, this layer's projection)
                        full = torch.full((B, self.streams), x.shape[1], dtype=torch.long, device=dev)
                        n_before = int(alive.sum())
                        x, size, mask, box, alive, is_cell, par0, stream = self._merge(
                            x, size, mask, box, alive, is_cell, (self.metric[step](blk.n1(x).detach()) if self.use_metric else blk.att.k(blk.n1(x))), frac, False, adjacency(mask), par0, stream,
                            self.streams, self.heads, full, "greedy", step, cells, True, sky_cells=sky_cells, small_only=True)
                        if int(alive.sum()) == n_before:
                            break
                step += 1
                merged_yet = True
            x = x + blk.ff(blk.n2(x))
            if return_attn:
                trace.append({"mask": mask.detach(), "alive": alive.detach(), "size": size.detach(), "stream": stream.detach()})
        hb, pad = x, ~alive
        if nr:
            pad = pad | (~is_cell & (par0 >= 20))                     # registers: not in the cross-attention
        if self.gate is not None and n0 is not None:
            left = n0 - torch.stack(gstats, 0).sum(0)                  # board blobs left, soft (straight-through)
            self.aux_loss = self.gate["lambda"] * ((left - T) / n0.clamp(min=1.0)) ** 2     # (B,): per position
            self.gate_log.append((float(n0.sum()) / self.streams, float((alive & is_cell).sum()) / self.streams, B))

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
        cbias = self.c_bias(cfeats, cboth)
        if pres is not None:
            # soft merges: a hard merge removes the source from the cross-attention; here it fades by its share left
            cbias = cbias + torch.log(pres.clamp(min=1e-4))[:, None, None, :]
        y, ca = self.cross(y, cbias, pad, ctx=hb, ctx_norm=self.c_norm)
        logit = self.head(y).squeeze(-1).masked_fill(cpad, float("-inf"))
        out = {"logit": logit, "q1": self.q1(y).squeeze(-1), "qs": self.qs(y).squeeze(-1)} if heads else logit
        if return_attn:
            return out, {"self": attn, "cross": ca, "merges": trace}
        return out
