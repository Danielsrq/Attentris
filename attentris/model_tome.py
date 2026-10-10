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
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import model as M
import tokenizer as T

PIECE_FIRST = T.KINDS.index("current")       # current, hold, preview1..5 are contiguous in KINDS
N_PIECE_TOK = 7


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


class AtTenTrisToMe(nn.Module):
    def __init__(self, d=64, heads=4, layers=4, merge=(0.0, 0.45, 0.45, 0.45), n_teachers=2):
        super().__init__()
        assert len(merge) == layers
        self.d, self.merge = d, tuple(merge)
        self.random_merge = False      # analysis baseline: merge random adjacent pairs (same schedule)
        self.val = nn.Embedding(2, d)
        self.pos = M.Pos2D(d)
        self.slot = nn.Embedding(N_PIECE_TOK, d)                 # current / hold / preview1..5
        self.piece = nn.Embedding(M.N_PIECES, d)
        self.is_cell = nn.Parameter(torch.zeros(d))
        self.bias = RelBias(heads)
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
    @staticmethod
    def _merge(x, size, mask, box, alive, is_cell, keys, frac, first, adj):
        B, N, d = x.shape
        dev = x.device
        cand = alive & is_cell
        if first:
            idx = torch.arange(N, device=dev)
            par = torch.where(idx < 200, (idx // 10 + idx % 10) % 2, torch.zeros_like(idx)).expand(B, N)
        else:
            par = (torch.cumsum(cand.long(), 1) - 1) % 2
        A = cand & (par == 0)
        Bs = cand & (par == 1)
        kn = F.normalize(keys, dim=-1)
        S = kn @ kn.transpose(1, 2)
        valid = A[:, :, None] & Bs[:, None, :] & adj
        S = S.masked_fill(~valid, float("-inf"))
        score, dst = S.max(-1)                                   # (B, N): best adjacent B per A token
        ok = torch.isfinite(score)
        r = (frac * cand.sum(1).float()).floor().long()          # merges wanted per position
        order = score.masked_fill(~ok, float("-inf")).argsort(-1, descending=True)
        rank = torch.empty_like(order).scatter_(1, order, torch.arange(N, device=dev).expand(B, N))
        sel = ok & (rank < r[:, None])
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
        return g(x), g(size), g(mask), g(box), g(alive), g(is_cell)

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
        merged_yet = False
        attn, trace = [], []
        for blk, frac in zip(self.blocks, self.merge):
            adj = adjacency(mask)
            bias = self.bias(mask, size, is_cell, adj)
            h = blk.n1(x)
            y, a, keys = blk.att(h, h, bias, ~alive, torch.log(size.clamp(min=1.0)))
            x = x + y
            attn.append(a)
            if frac > 0:
                if self.random_merge:
                    keys = torch.randn_like(keys)
                x, size, mask, box, alive, is_cell = self._merge(x, size, mask, box, alive, is_cell, keys, frac, not merged_yet, adj)
                merged_yet = True
            x = x + blk.ff(blk.n2(x))
            if return_attn:
                trace.append({"mask": mask.detach(), "alive": alive.detach(), "size": size.detach()})
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
