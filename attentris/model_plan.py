"""Multi-move ("plan") variants of AtTenTris: does the network internalise search (several
moves from one encoder pass), and does predicting the next moves improve move 1?

Both extend model.AtTenTris (same encoder, same move-1 candidates, policy and Qs heads), so a
v5.1 checkpoint loads as is and plays move 1 exactly as before until trained (the new step
embeddings start at zero). Plan steps are k = 1..N, N <= MAX_STEPS = 6 (the current piece +
5 previews); one model serves any N up to the N it was trained with.

  autoreg  attentris-autoreg. The encoder runs once on board X. Step k (k >= 2) is one more
           pass of the SAME cross-attention block: its queries are the legal moves at plan step
           k (bot plan-cands, in X's frame), its keys the board tokens of X plus one token per
           earlier planned move (moves 1..k-1; teacher-forced in training, the model's own picks
           in play). The earlier-move tokens are embedded like candidates (their cells on X's
           board, piece, hold, spin, lines, attack) + a "planned move" vector + their step
           embedding, so the FLAT-2D bias sees candidate-vs-earlier-move geometry too.
           Always legal: it only ever picks among Rust's legal moves.
  plan     attentris-plan ("option C"). One pass: N step queries (learned per step) cross-attend
           to the board tokens (a block initialised from the trained cross block), one
           self-attention block over the N step tokens (a coherent plan), and a shared
           1760-way coordinate head: hold 2 x rotation class 4 x column 10 x lowest row 22
           (pack_plan.py's coord). A step's prediction can be illegal; play checks it with Rust.

Batch keys added to model.AtTenTris's (B = batch, K = MAX_STEPS - 1 steps after the first):
  autoreg: pc_cells (B, K, NC, 4) long, pc_meta (B, K, NC, 5) long, pc_n (B, K) long (legal
           moves at each step; 0 = no step), pc_target (B, K) long (the planned move's index)
  both:    chosen (B,) long (move 1, teacher-forced as the first earlier-move token: autoreg)
"""
import torch
import torch.nn as nn

import model as M

MAX_STEPS = 6
N_COORD = 2 * 4 * 10 * 22


class AtTenTrisPlan(M.AtTenTris):
    def __init__(self, *args, variant="autoreg", max_steps=MAX_STEPS, **kw):
        super().__init__(*args, **kw)
        assert variant in ("autoreg", "plan")
        d = self.d
        self.variant, self.max_steps = variant, max_steps
        self.step = nn.Embedding(max_steps, d)                  # step k's embedding (row k-1); zero init:
        nn.init.zeros_(self.step.weight)                        # move 1 starts exactly as the base model
        heads = self.cross.att.h
        if variant == "autoreg":
            self.prev_move = nn.Parameter(torch.zeros(d))       # marks the earlier-move key/value tokens
        else:
            self.plan_query = nn.Embedding(max_steps, d)
            nn.init.normal_(self.plan_query.weight, std=0.02)
            self.plan_cross = M.Block(d, heads)
            self.plan_c_bias = M.Bias(heads)
            self.plan_self = M.Block(d, heads)
            self.plan_norm = nn.LayerNorm(d)
            self.coord_head = nn.Linear(d, N_COORD)

    def init_from_base(self):
        """After loading a base (single-move) checkpoint: start attentris-plan's step queries'
        cross block from the trained cross block (and its FLAT-2D bias)."""
        if self.variant == "plan":
            self.plan_cross.load_state_dict(self.cross.state_dict())
            self.plan_c_bias.load_state_dict(self.c_bias.state_dict())

    # ---- the base model's forward, in parts ----
    def encode(self, batch):
        """Cells -> pooled structure tokens -> FLAT-2D self-attention. Returns the board side."""
        cells = batch["cells"]
        phi = self.val(cells) + self.pos.cells()
        kind = batch["tok_kind"]
        pad = kind < 0
        k = kind.clamp(min=0)
        mask = batch["tok_mask"]
        has = mask.any(-1)
        logits = torch.einsum("btd,bcd->btc", self.kind_query(k), self.Wk(phi)) / (self.d ** 0.5)
        logits = logits.masked_fill(~mask, float("-inf"))
        w = torch.where(has.unsqueeze(-1), logits.softmax(-1).nan_to_num(0.0), torch.zeros_like(logits))
        z = w @ self.Wv(phi)
        piece = batch["tok_piece"]
        z = z + torch.where((piece >= 0).unsqueeze(-1), self.piece(piece.clamp(min=0)), torch.zeros_like(z))
        box = batch["tok_box"]
        x = z + self.kind(k) + self.pos.box(box) + self.extra(batch["tok_extra"])
        maskf = mask.float()
        feats, both = M.pair_features(box, box, maskf, maskf)
        bias = self.bias(feats, both)
        attn = []
        for blk in self.blocks:
            x, a = blk(x, bias, pad)
            attn.append(a)
        return {"phi": phi, "hb": x, "box": box, "maskf": maskf, "pad": pad, "attn": attn, "pool": w}

    def embed_moves(self, phi, cc, meta, teacher):
        """Candidate (move) tokens: cells cc (B, N, 4) on X's board (-1 pad, -2 off-board), meta
        (B, N, 5). Returns the embeddings, their boxes and cell masks, and the padding."""
        cpad = cc[..., 0] == -1
        onb = cc >= 0
        idx = cc.clamp(min=0)
        cphi = torch.gather(phi.unsqueeze(1).expand(-1, cc.shape[1], -1, -1), 2,
                            idx.unsqueeze(-1).expand(-1, -1, -1, self.d))
        cphi = (cphi * onb.unsqueeze(-1)).sum(2) / onb.sum(-1, keepdim=True).clamp(min=1)
        r, c = idx // 10, idx % 10
        big = torch.full_like(r, 99)
        cbox = torch.stack([torch.where(onb, r, big).min(-1).values, torch.where(onb, r, -big).max(-1).values,
                            torch.where(onb, c, big).min(-1).values, torch.where(onb, c, -big).max(-1).values], -1)
        cbox = torch.where((onb.any(-1) & ~cpad).unsqueeze(-1), cbox, torch.full_like(cbox, -1))
        y = (cphi + self.c_piece(meta[..., 0]) + self.c_hold(meta[..., 1]) + self.c_spin(meta[..., 2])
             + self.c_lines(meta[..., 3].clamp(max=4)) + self.c_attack(meta[..., 4:5].float() / 10.0)
             + self.pos.box(cbox) + self.style(teacher).unsqueeze(1))
        cmask = torch.zeros(cc.shape[0], cc.shape[1], 200, device=cc.device)
        cmask.scatter_(2, idx, onb.float())
        return y, cbox, cmask, cpad

    def cross_pass(self, y, cbox, cmask, enc, extra=None):
        """The shared cross block: move tokens y attend to the board tokens (+ extra key/value
        tokens: (tokens, boxes, masks, pad))."""
        ctx, box_k, mask_k, pad_k = enc["hb"], enc["box"], enc["maskf"], enc["pad"]
        if extra is not None:
            t, b, m, p = extra
            ctx, box_k = torch.cat([ctx, t], 1), torch.cat([box_k, b], 1)
            mask_k, pad_k = torch.cat([mask_k, m], 1), torch.cat([pad_k, p], 1)
        feats, both = M.pair_features(cbox, box_k, cmask, mask_k)
        passes = [(self.cross, self.c_bias)] * self.cross_loops + list(zip(self.cross_extra, self.c_bias_extra))
        for blk, cb in passes:
            y, ca = blk(y, cb(feats, both), pad_k, ctx=ctx, ctx_norm=self.c_norm)
        return y, ca

    # ---- forward ----
    def forward(self, batch, return_attn=False, heads=False, n_steps=1, prev=None):
        """Move 1 exactly as model.AtTenTris (logit / q1 / qs over batch's candidates). With
        n_steps > 1 also the plan: autoreg -> "step_logits" (B, n_steps - 1, NC) over each later
        step's legal moves (batch pc_*), the earlier moves teacher-forced from batch["chosen"] and
        pc_target, or given by prev = list of (cells (B, 4), meta (B, 5)) for moves 1..n_steps-1;
        plan -> "coord_logits" (B, n_steps, 1760), steps 1..n_steps."""
        out, enc, ca = self.move1(batch)
        teacher = batch["teacher"]
        attn = {"self": enc["attn"], "cross": ca, "pool": enc["pool"]}
        if n_steps > 1 and self.variant == "autoreg":
            out["step_logits"], attn["step_cross"] = self.autoreg_steps(batch, enc, teacher, n_steps, prev)
        if n_steps > 1 and self.variant == "plan":
            out["coord_logits"], attn["plan_cross"] = self.plan_steps(enc, teacher, n_steps)
        if not heads and not (n_steps > 1):
            out = out["logit"]
        if return_attn:
            return out, attn
        return out

    def move1(self, batch):
        """Move 1 (logit / q1 / qs over batch's candidates) and the board encoding, which later
        plan steps reuse (autoreg_steps with prev given: play.py --commit)."""
        enc = self.encode(batch)
        y, cbox, cmask, cpad = self.embed_moves(enc["phi"], batch["cand_cells"], batch["cand_meta"], batch["teacher"])
        y = y + self.step.weight[0]
        y, ca = self.cross_pass(y, cbox, cmask, enc)
        logit = self.head(y).squeeze(-1).masked_fill(cpad, float("-inf"))
        return {"logit": logit, "q1": self.q1(y).squeeze(-1), "qs": self.qs(y).squeeze(-1)}, enc, ca

    def autoreg_steps(self, batch, enc, teacher, n_steps, prev):
        B = teacher.shape[0]
        ar = torch.arange(B, device=teacher.device)
        if prev is None:     # teacher forcing: move 1 = the label, step k = the plan's move
            prev = [(batch["cand_cells"][ar, batch["chosen"]], batch["cand_meta"][ar, batch["chosen"]])]
            for k in range(n_steps - 2):
                t = batch["pc_target"][:, k].clamp(min=0)
                prev.append((batch["pc_cells"][ar, k, t], batch["pc_meta"][ar, k, t]))
        toks, boxes, masks, pads = [], [], [], []
        for j, (pc, pm) in enumerate(prev[:n_steps - 1]):
            t, b, m, p = self.embed_moves(enc["phi"], pc.unsqueeze(1), pm.unsqueeze(1), teacher)
            toks.append(t + self.prev_move + self.step.weight[j])
            boxes.append(b), masks.append(m), pads.append(p)
        logits, cross = [], []
        for k in range(1, n_steps):          # plan step k + 1
            cc, cm = batch["pc_cells"][:, k - 1], batch["pc_meta"][:, k - 1]
            y, cbox, cmask, cpad = self.embed_moves(enc["phi"], cc, cm, teacher)
            y = y + self.step.weight[k]
            # an earlier move missing (no plan there) is padded out; with no keys left the step is masked anyway
            no_step = batch["pc_n"][:, k - 1] <= 0
            extra = (torch.cat(toks[:k], 1), torch.cat(boxes[:k], 1), torch.cat(masks[:k], 1),
                     torch.cat(pads[:k], 1) | no_step.unsqueeze(1))
            y, ca = self.cross_pass(y, cbox, cmask, enc, extra)
            logits.append(self.head(y).squeeze(-1).masked_fill(cpad, float("-inf")))
            cross.append(ca)
        return torch.stack(logits, 1), cross

    def plan_steps(self, enc, teacher, n_steps):
        B = teacher.shape[0]
        q = (self.plan_query.weight[:n_steps] + self.step.weight[:n_steps]).unsqueeze(0) + self.style(teacher).unsqueeze(1)
        qbox = torch.full((B, n_steps, 4), -1, dtype=torch.long, device=teacher.device)
        qmask = torch.zeros(B, n_steps, 200, device=teacher.device)
        feats, both = M.pair_features(qbox, enc["box"], qmask, enc["maskf"])
        q, ca = self.plan_cross(q, self.plan_c_bias(feats, both), enc["pad"], ctx=enc["hb"], ctx_norm=self.c_norm)
        no_pad = torch.zeros(B, n_steps, dtype=torch.bool, device=teacher.device)
        q, _ = self.plan_self(q, 0.0, no_pad)
        return self.coord_head(self.plan_norm(q)), ca


def load(path, variant=None, device="cpu"):
    """A plan checkpoint, or a base v5.1 checkpoint as the start of a plan model (variant given)."""
    ck = torch.load(path, map_location=device, weights_only=False)
    a = ck["args"]
    v = variant or a.get("plan_variant")
    net = AtTenTrisPlan(d=a["d"], layers=a["layers"], n_kinds=ck["model"]["kind.weight"].shape[0],
                        cross_layers=a.get("cross_layers", 1), cross_loops=a.get("cross_loops", 1),
                        n_teachers=ck["model"]["style.weight"].shape[0], variant=v)
    missing = net.load_state_dict(ck["model"], strict=False).missing_keys
    if not a.get("plan_variant"):
        net.init_from_base()
    return net, ck, missing
