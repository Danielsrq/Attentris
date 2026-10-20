"""AtTenTris model (DESIGN.md, "Model spec"): cell stage -> masked pooling into structure
tokens -> FLAT-2D-biased transformer -> candidate tokens cross-attending to the board.

Shapes: B batch, NT board-side tokens, NC candidates, d model width, H heads.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_KINDS = 40        # >= len(tokenizer.KINDS) (34 in v5); asserted by train.py. Pre-v5 checkpoints have 31
N_PIECES = 8        # IOTSZJL + none
Q_SCALE = 10.0      # value heads predict teacher scores / Q_SCALE


def unpack_masks(packed):
    """(…, 25) uint8 -> (…, 200) bool."""
    bits = np.unpackbits(packed, axis=-1)[..., :200]
    return bits.astype(bool)


class Pos2D(nn.Module):
    """Row / column embeddings, summed (cells) or concatenated over a box (tokens)."""

    def __init__(self, d):
        super().__init__()
        self.row = nn.Embedding(20, d)
        self.col = nn.Embedding(10, d)
        q = d // 4
        self.brow_lo, self.brow_hi = nn.Embedding(20, q), nn.Embedding(20, q)
        self.bcol_lo, self.bcol_hi = nn.Embedding(10, q), nn.Embedding(10, q)
        self.box_proj = nn.Linear(4 * q, d)

    def cells(self):
        dev = self.row.weight.device
        r = torch.arange(20, device=dev).repeat_interleave(10)
        c = torch.arange(10, device=dev).repeat(20)
        return self.row(r) + self.col(c)                       # (200, d)

    def box(self, box):
        """box: (…, 4) long, rows lo/hi, cols lo/hi; -1 = none (returns zeros there)."""
        ok = (box[..., 0] >= 0).unsqueeze(-1)
        b = box.clamp(min=0)
        e = torch.cat([self.brow_lo(b[..., 0]), self.brow_hi(b[..., 1]), self.bcol_lo(b[..., 2]), self.bcol_hi(b[..., 3])], -1)
        return self.box_proj(e) * ok


def pair_features(box_q, box_k, mask_q, mask_k):
    """FLAT-2D: four row-span and four column-span distances plus the overlap count.
    box: (B, Nq, 4) / (B, Nk, 4) long (-1 = no box); mask: (B, N, 200) float.
    Returns (B, Nq, Nk, 9) float scaled to about -1..1 and a (B, Nq, Nk) bool 'both have boxes'."""
    rq = box_q[..., :2].float(); rk = box_k[..., :2].float()
    cq = box_q[..., 2:].float(); ck = box_k[..., 2:].float()
    dr = (rq.unsqueeze(2).unsqueeze(-1) - rk.unsqueeze(1).unsqueeze(-2)).flatten(-2) / 20.0   # lo-lo, lo-hi, hi-lo, hi-hi
    dc = (cq.unsqueeze(2).unsqueeze(-1) - ck.unsqueeze(1).unsqueeze(-2)).flatten(-2) / 10.0
    ov = torch.einsum("bqc,bkc->bqk", mask_q, mask_k).unsqueeze(-1) / 50.0
    both = (box_q[..., 0] >= 0).unsqueeze(2) & (box_k[..., 0] >= 0).unsqueeze(1)
    return torch.cat([dr, dc, ov], -1), both


class Bias(nn.Module):
    """9 pair features -> one score per head; a learned constant per head for pairs
    without a box on both sides (piece tokens)."""

    def __init__(self, heads):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(9, 32), nn.ReLU(), nn.Linear(32, heads))
        self.nobox = nn.Parameter(torch.zeros(heads))

    def forward(self, feats, both):
        b = self.mlp(feats)                                      # (B, Nq, Nk, H)
        b = torch.where(both.unsqueeze(-1), b, self.nobox)
        return b.permute(0, 3, 1, 2)                             # (B, H, Nq, Nk)


class Attention(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.h, self.dh = heads, d // heads
        self.q, self.k, self.v, self.o = (nn.Linear(d, d) for _ in range(4))

    def forward(self, xq, xk, bias, key_pad):
        B, Nq, d = xq.shape
        Nk = xk.shape[1]
        q = self.q(xq).view(B, Nq, self.h, self.dh).transpose(1, 2)
        k = self.k(xk).view(B, Nk, self.h, self.dh).transpose(1, 2)
        v = self.v(xk).view(B, Nk, self.h, self.dh).transpose(1, 2)
        qk = q @ k.transpose(-1, -2) / math.sqrt(self.dh)        # content term
        s = qk + bias                                            # + FLAT-2D geometry term
        if getattr(self, "keep", False):                         # analysis only (attn_viz): keep both terms
            self.kept = (qk.detach(), (bias + torch.zeros_like(qk)).detach())
        s = s.masked_fill(key_pad[:, None, None, :], float("-inf"))
        a = s.softmax(-1)                                        # (B, H, Nq, Nk)
        out = a @ v                                              # (B, H, Nq, dh)
        hm = getattr(self, "head_mask", None)                    # analysis only: switch heads off
        if hm is not None:
            out = out * hm.view(1, -1, 1, 1)
        out = out.transpose(1, 2).reshape(B, Nq, d)
        return self.o(out), a


class Block(nn.Module):
    def __init__(self, d, heads, ff=4):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.att = Attention(d, heads)
        self.ff = nn.Sequential(nn.Linear(d, ff * d), nn.GELU(), nn.Linear(ff * d, d))

    def forward(self, x, bias, key_pad, ctx=None, ctx_norm=None):
        """Self-attention when ctx is None, else cross-attention from x to ctx."""
        h = self.n1(x)
        kv = h if ctx is None else ctx_norm(ctx)
        y, a = self.att(h, kv, bias, key_pad)
        x = x + y
        return x + self.ff(self.n2(x)), a


class AtTenTris(nn.Module):
    def __init__(self, d=64, heads=4, layers=4, n_extra=12, n_teachers=2, n_kinds=N_KINDS, cross_layers=1, cross_loops=1,
                 registers=0, flat=True):
        super().__init__()
        self.d = d
        # flat=False (ablation, train.py --no-flat): no FLAT-2D bias anywhere (self-attention and the
        # candidates' cross-attention); geometry reaches attention only through the box / cell position
        # embeddings. The Bias modules stay (unused) so checkpoints keep one layout.
        self.flat = flat
        # register tokens: R learned vectors appended to the board-side sequence (no mask, box or readout).
        # They read and write in every self-attention layer; candidates can attend to them in the cross block.
        # Geometry has no meaning for them, so their attention bias is a learned constant per head: as a key
        # (reg_k, reg_c) it starts at -8, so a model warm-started without registers behaves as before.
        self.registers = registers
        if registers:
            self.reg = nn.Parameter(torch.randn(registers, d) * 0.02)
            self.reg_k = nn.Parameter(torch.full((heads,), -8.0))     # any board-side query -> register key
            self.reg_q = nn.Parameter(torch.zeros(heads))             # register query -> board-token key
            self.reg_c = nn.Parameter(torch.full((heads,), -8.0))     # candidate query -> register key
        self.val = nn.Embedding(2, d)
        self.pos = Pos2D(d)
        self.Wk, self.Wv = nn.Linear(d, d, bias=False), nn.Linear(d, d, bias=False)
        self.kind_query = nn.Embedding(n_kinds, d)               # pooling query per token kind
        self.kind = nn.Embedding(n_kinds, d)
        self.piece = nn.Embedding(N_PIECES, d)
        self.extra = nn.Linear(n_extra, d)
        self.bias = Bias(heads)
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        self.norm = nn.LayerNorm(d)
        # candidates
        self.c_piece = nn.Embedding(N_PIECES, d)
        self.c_hold, self.c_spin, self.c_lines = nn.Embedding(2, d), nn.Embedding(3, d), nn.Embedding(5, d)
        self.c_attack = nn.Linear(1, d)
        self.style = nn.Embedding(n_teachers, d)
        self.c_bias = Bias(heads)
        self.cross = Block(d, heads)
        # v5.2: more cross-attention passes. cross_loops = K applies the same block (and bias) K times
        # (weight-tied, "looped"); cross_layers = K uses K independent blocks, each with its own bias.
        assert cross_layers == 1 or cross_loops == 1
        self.cross_loops = cross_loops
        self.cross_extra = nn.ModuleList(Block(d, heads) for _ in range(cross_layers - 1))
        self.c_bias_extra = nn.ModuleList(Bias(heads) for _ in range(cross_layers - 1))
        self.c_norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, 1)
        # value heads on the same candidate vector (v2): the teacher's 1-ply score of every
        # move (attack + V after it) and its 6-ply search score (moves the beam kept),
        # both in units of Q_SCALE lines of attack
        self.q1 = nn.Linear(d, 1)
        self.qs = nn.Linear(d, 1)

    def forward(self, batch, return_attn=False, heads=False):
        """Policy logits (B, NC); with heads=True a dict {logit, q1, qs} (Q in Q_SCALE units)."""
        cells = batch["cells"]                                   # (B, 200) long
        B = cells.shape[0]
        phi = self.val(cells) + self.pos.cells()                 # (B, 200, d)

        # --- region pooling (masked attention, one query per kind) ---
        kind = batch["tok_kind"]                                 # (B, NT) long, -1 pad
        pad = kind < 0
        k = kind.clamp(min=0)
        mask = batch["tok_mask"]                                 # (B, NT, 200) bool
        has = mask.any(-1)
        logits = torch.einsum("btd,bcd->btc", self.kind_query(k), self.Wk(phi)) / math.sqrt(self.d)
        logits = logits.masked_fill(~mask, float("-inf"))
        w = torch.where(has.unsqueeze(-1), logits.softmax(-1).nan_to_num(0.0), torch.zeros_like(logits))
        z = w @ self.Wv(phi)                                     # (B, NT, d)
        piece = batch["tok_piece"]
        z = z + torch.where((piece >= 0).unsqueeze(-1), self.piece(piece.clamp(min=0)), torch.zeros_like(z))

        box = batch["tok_box"]
        x = z + self.kind(k) + self.pos.box(box) + self.extra(batch["tok_extra"])
        maskf = mask.float()
        if self.flat:
            feats, both = pair_features(box, box, maskf, maskf)
            bias = self.bias(feats, both)
        else:
            bias = x.new_zeros(B, self.bias.nobox.shape[0], x.shape[1], x.shape[1])
        R = self.registers
        if R:
            NT = x.shape[1]
            x = torch.cat([x, self.reg.unsqueeze(0).expand(B, -1, -1)], 1)
            H = bias.shape[1]
            full = bias.new_zeros(B, H, NT + R, NT + R)
            full[:, :, :NT, :NT] = bias
            full[:, :, NT:, :NT] = self.reg_q.view(1, H, 1, 1)
            full[:, :, :, NT:] = self.reg_k.view(1, H, 1, 1)
            bias = full
            pad = torch.cat([pad, pad.new_zeros(B, R)], 1)
            # analysis only (register_usage.py): reg_knockout hides the registers from every query;
            # reg_mean = [layer -> (R, d)] replaces their states after each layer; reg_record collects them
            if getattr(self, "reg_knockout", False):
                pad = pad.clone()
                pad[:, NT:] = True
        attn = []
        for li, blk in enumerate(self.blocks):
            x, a = blk(x, bias, pad)
            if R:
                rm = getattr(self, "reg_mean", None)
                if rm is not None:
                    x = torch.cat([x[:, :NT], rm[li].to(x.dtype).unsqueeze(0).expand(B, -1, -1)], 1)
                rec = getattr(self, "reg_record", None)
                if rec is not None:
                    rec.setdefault(li, []).append(x[:, NT:].detach().cpu())
                    real = (~pad[:, :NT]).unsqueeze(-1).to(x.dtype)       # the mean of the real board tokens, for comparison
                    rec.setdefault(("board", li), []).append(((x[:, :NT] * real).sum(1) / real.sum(1).clamp(min=1)).detach().cpu())
            attn.append(a)
        hb = x                                                   # (B, NT, d), pre-norm residual

        # --- candidates ---
        cc = batch["cand_cells"]                                 # (B, NC, 4) long; -1 pad, -2 off-board
        cpad = cc[..., 0] == -1
        onb = cc >= 0
        idx = cc.clamp(min=0)
        cphi = torch.gather(phi.unsqueeze(1).expand(-1, cc.shape[1], -1, -1), 2,
                            idx.unsqueeze(-1).expand(-1, -1, -1, self.d))          # (B, NC, 4, d)
        cphi = (cphi * onb.unsqueeze(-1)).sum(2) / onb.sum(-1, keepdim=True).clamp(min=1)
        meta = batch["cand_meta"]                                # piece, hold, spin, lines, attack
        r, c = idx // 10, idx % 10
        big = torch.full_like(r, 99)
        cbox = torch.stack([torch.where(onb, r, big).min(-1).values, torch.where(onb, r, -big).max(-1).values,
                            torch.where(onb, c, big).min(-1).values, torch.where(onb, c, -big).max(-1).values], -1)
        cbox = torch.where((onb.any(-1) & ~cpad).unsqueeze(-1), cbox, torch.full_like(cbox, -1))
        y = (cphi + self.c_piece(meta[..., 0]) + self.c_hold(meta[..., 1]) + self.c_spin(meta[..., 2])
             + self.c_lines(meta[..., 3].clamp(max=4)) + self.c_attack(meta[..., 4:5].float() / 10.0)
             + self.pos.box(cbox) + self.style(batch["teacher"]).unsqueeze(1))
        cmask = torch.zeros(B, cc.shape[1], 200, device=cells.device)
        cmask.scatter_(2, idx, onb.float())
        if self.flat:
            cfeats, cboth = pair_features(cbox, box, cmask, maskf)
        passes = [(self.cross, self.c_bias)] * self.cross_loops + list(zip(self.cross_extra, self.c_bias_extra))
        cross_all = []
        for blk, cb in passes:
            cbias = cb(cfeats, cboth) if self.flat else y.new_zeros(B, cb.nobox.shape[0], y.shape[1], hb.shape[1])
            if R:
                cbias = torch.cat([cbias, self.reg_c.view(1, -1, 1, 1).expand(B, -1, cbias.shape[2], R)], -1)
            y, ca = blk(y, cbias, pad, ctx=hb, ctx_norm=self.c_norm)
            cross_all.append(ca)
        logit = self.head(y).squeeze(-1).masked_fill(cpad, float("-inf"))    # (B, NC)
        out = {"logit": logit, "q1": self.q1(y).squeeze(-1), "qs": self.qs(y).squeeze(-1)} if heads else logit
        if return_attn:
            return out, {"self": attn, "cross": ca, "cross_all": cross_all, "pool": w}
        return out
