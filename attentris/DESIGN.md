# AtTenTris

Oct 9, 2026 · @Daniel (revision 2: tokenizer v1 implemented, see `tokenizer.py`)

## Changes from revision 1

- **Width:** d = 64 (was 128), H = 4 heads of width 16, L = 4.
- **Sky:** every row above the highest filled cell, rather than every cell above each column's surface. Empty cells below the stack top now sit in stack tokens and in the negative-space overlays.
- **Stack:** split into the **clean zone** (clean wells) and the **top play area**. Clean wells replace "merge identical rows".
- **Lexicon:**
  - T structures come from the bot's own detectors (`bot structures`).
  - Two new families: rect12 (3×4 / 4×3 pits) and water-fill pits for negative space.
- **Overlay cap:** P_max = 8 (was 3). Measured n is 16.4 on average and 26 at most.
- **Worked example:** corrected. Cell (9,5) is an overhang zero inside R9, not Sky.
- **Placement head:** changed to candidate tokens (see Model spec). With the 4-cell dot product, every placement in open space would read one shared Sky vector.
- **Training data:** a concrete plan, using the gate4 weights under normal rules as the teacher.

## Goal

Build a Tetris board evaluator whose main transformer sees about 10 to 30 tokens, each a structure a human player would name (clean well, play row, slot, pit, overhang), instead of 200 cell tokens. Every token keeps its 2D footprint, so attention and attributions can still be drawn on the 20 × 10 board, chessformer-style.

Two success criteria:

1. When it places a T, the model's attention for that placement lands on the T structure it is cashing or building: TSD overhang, STSD, imperial cross, fractal.
2. In general, highlights follow the shape of the stack, not a flat wash.

Scope of version 1: no CNN, no in-forward merging, no learned tokenizer. Grouping is rule-based and fixed before the transformer runs.

## Ideas and where they come from

| Idea | Borrowed from | What we take |
| --- | --- | --- |
| Overlapping tokens with span positions | FLAT (Li et al., ACL 2020), Lattice-BERT | Base units plus lexicon-matched groups in one sequence; position as a span and pairwise span distances in the attention score |
| Small units pooled into big tokens by a local encoder | BLT (Meta, 2024), H-Net (Hwang et al., 2025) | Cells pooled into region tokens before the main transformer; our boundaries are rule-based |
| Variable-size patches with explicit 2D position | Quadformer (Ronen et al., 2023) | Coarse tokens for low-information regions (sky, clean wells), fine ones for the play area |
| Entities as tokens, scattered back to 2D | AlphaStar entity encoder | A token's vector is written back to the cells it covers |
| Tracking which inputs a token covers | ToMe (Bolya et al., 2023) | The mask matrix M that lets attention be drawn at cell level |

Key decisions:

- The tokenizer, not the network, decides what a token is, so the identity of every token is interpretable by construction.
- Every cell, filled or empty, is still embedded at the input; the transformer just never attends to cells one by one.
- Position enters twice: an absolute box embedding per token, and a relative 2D bias per pair.
- Only overlays may overlap; the base tokens partition the board exactly.

## Tokenizer spec (implemented: `tokenizer.py`)

**Base tokens (an exact partition, at least 2 for any non-empty board: Sky + stack).** h is the column heights, top = max(h).

1. **Sky**: rows top+1 to 20. One token; its extras carry h. It is empty only when the stack reaches row 20.
2. **Clean zone and wells**: from the bottom up, take each row with exactly one empty cell. Group them into runs with the same hole column. Each run is one **well** token (extras: length ℓ, hole column, covered).
   - ℓ = 1 is a cheese row.
   - Exception: when the zone's top run is a single open row (a TSD's stem row, a half-built row), it belongs to the play area instead.
3. **Top play area**: every row from the end of the clean zone up to top, one **play** token per row. Empty cells inside these rows (wells, overhangs, slots) are explicit zeros.
4. **Budget**: while board tokens exceed K = 12, merge the lowest adjacent pair of stack tokens into a **band**. Deep rows matter least.

**Overlays** (may overlap base tokens). At most P_max = 8, in priority order:

1. **T structures**, from the bot's slot detectors (the same templates its V uses, checked against the move generator in its tests):

   | Kind | Shape | Cells |
   | --- | --- | --- |
   | tsd | flat slot, nub down | the T |
   | fractal | flat slot with the same TSD two rows down | both T's |
   | imperial | vertical slot the T can turn 180° in | T, turned T, second T |
   | stsd | vertical slot; the TSD opens one row under the stem | T and second T |
   | cspin | vertical slot; the TST opens a TSD two rows under | T and second T |
   | vertical | vertical slot with no second T (TST) | the T |

   Mirror images are found by the same scan; both nub sides are searched.
2. **rect12**: an empty pit with a flat floor and walls on both sides (a board edge counts as a wall), 3 wide by 4 deep or 4 wide by 3 deep. The mask is the 12 cells. Brute force confirms that L+J+O, L+S+J and L+Z+J each tile both shapes exactly (2 ways each); L+L+O and J+J+O do too. No other 3-piece set tried does (L+J+T, L+S+Z, J+S+Z, O+S+Z, I+I+O).
3. **fit_s, fit_z, fit_t, fit_l, fit_j (piece-shaped recesses)**: added after the viewer showed S/Z placements that level the stack and that no token covered. A piece-shaped recess is a resting place for a piece with a shaped bottom where both of these hold:
   - The piece touches the surface under every one of its columns, so it leaves no gap.
   - It sits snugly: at least 2 of its cells' sides touch a filled cell or the board edge. A piece balanced on a peak touches only the step it rests on.
   - The surrounding columns (its own, ±1) end up no bumpier. A cliff just outside the recess can cancel the gain, so this only rules out towers.

   (Revised after the viewer: the first rule, "strictly less bumpy", missed the flat S in position 10 and the L/J hooks in 23 and 25.)

   The pieces counted:

   | Piece | Orientations |
   | --- | --- |
   | S, Z | flat and upright |
   | T | nub down |
   | L, J | three on top, and hook on top |

   The mask is the piece's own cells, so the recess is drawn as the S/Z/T/L/J-shaped hole. Flat-bottomed placements fit almost anywhere and are left out.
   - On 2,500 gate4 boards there are about 2.1 recesses per board.
   - Of the bot's non-clearing S, Z, L and J moves, 23–32% land exactly on a recess token; where a recess of that piece exists, 47–57%.
   - Most of the bot's other S/Z moves deliberately leave one gap or a bump (building overhangs and T-slots), which is construction rather than levelling.
4. **pit1–pit4 (negative space)**: every run of 1 to 4 columns lying below both its side walls (an edge counts as an infinitely high wall, but not on both sides). The pit is "water-filled" up to the lower wall.
   - pit1 is a well.
   - Pits nest: a 1-wide well inside a 3-wide pit is two tokens.
   - Priority goes by area. Pits cover only open space above the surface; overhang space is what the T-structure overlays cover.

**Piece tokens**: current, hold, previews 1–5. These have no footprint.

**Metadata per token** (written by the tokenizer, never learned): kind τ, box (row lo/hi, col lo/hi), a 200-bit mask, and extras:

| Kind | Extras |
| --- | --- |
| sky | h |
| well | ℓ, hole column, covered |
| play / band | rows, filled count, empty cells under the surface |
| pit | width, depth, level, area |
| T structures | lines, fill |

Stacking the masks gives M (n × 200).

**Measured on 2,500 gate4 boards under normal rules with garbage:**

| Measure | Mean | Median | Min | Max |
| --- | --- | --- | --- | --- |
| Tokens per board | 16.4 | 16 | 8 | 26 |
| Base tokens | 6.3 | 6 | 1 | 12 |
| Overlay tokens | 3.1 | 3 | 0 | 8 |

Board-level occurrences of each structure, counted per board, so one structure that persists over several placements counts on each:

| tsd | imperial | fractal | stsd | vertical | rect12 | cspin |
| --- | --- | --- | --- | --- | --- | --- |
| 1,639 | 256 | 93 | 55 | 13 | 10 | 6 |

Pits: pit2 2,077, pit3 1,354, pit1 1,241, pit4 887.

**Worked example** (TSD board, rows counted from the bottom; `test_tokenizer.py`):

```
row  col 0123456789     h = [9,9,9,8,7,10,10,10,10,9]
20-11    ..........     Sky (rows 11-20, 100 cells)
10       .....####.     P10
 9       ###...####     P9  ((9,5) is an overhang zero)
 8       ####.#####     P8  (one open hole: the TSD stem row, so play, not a cheese well)
 7-1     ###.######     W   (ℓ = 7, hole col 3, covered by (8,3))
```

The overlays are tsd {(9,3), (9,4), (9,5), (8,4)} with 2 lines, and pit2 {(9,3), (9,4), (8,4)}. Total n = 5 board + 2 overlay + 7 piece = 14.

## Model spec

Sizes: **d = 64, H = 4 (head width 16), L = 4**, n ≈ 8 to 31 (27 before v5). As built: 273k parameters (274.5k from v5); trained on the GPU since DAgger round 1 (CPU is fine for inference). Full diagram under "Architecture as implemented" below.

**1. Cell stage (no CNN).** Every cell gets its value and its absolute 2D position. Shape 200 × d.

```latex
\Phi = E_{\text{val}}[B] + \mathrm{PE}_{2D}(r, c) \in \mathbb{R}^{200 \times d}
```

**2. Region pooling.** Masked attention pooling with one learned query per token kind. W has shape n × 200 and each row sums to 1; Z has shape n × d.

```latex
w_{jc} = \frac{M_{jc}\, \exp(q_{\tau_j}^\top W_k \Phi_c)}{\sum_{c'} M_{jc'}\, \exp(q_{\tau_j}^\top W_k \Phi_{c'})}, \qquad Z = W \, \Phi W_v
```

Piece tokens have empty masks and skip pooling; their Z row is the piece embedding plus a slot embedding.

**3. Token embedding.** Add kind, absolute box position, and kind-specific extras. Tokens without a box get one learned vector instead of the box PE.

```latex
X = Z + e_{\text{type}}(\tau) + W_p\,[\,p_r(r^{lo}) \oplus p_r(r^{hi}) \oplus p_c(c^{lo}) \oplus p_c(c^{hi})\,] + e_{\text{extra}}
```

**4. FLAT-2D bias.** For each pair, nine integers: four row-span distances, four column-span distances, and the mask overlap count. A small MLP maps them to one scalar per head, computed once and shared across layers. Pairs involving a piece token get a learned constant per head.

**5. Transformer.** L pre-norm blocks over the n board, overlay and piece tokens.

**6. Placement head: candidate tokens (changed).** Each legal placement, around 30–80 including hold, becomes a query token.
- Its mask is the 4 cells it occupies.
- Its embedding is the pooled cells, the piece, a hold flag, spin type, lines cleared and its box PE.
- Its 2D bias to the board tokens uses the same FLAT-2D features.

A final cross-attention block (candidates attend to the board tokens, not to each other) followed by a linear layer gives one logit per placement. Cost is n × candidates and stays small.

*Why the change:* a placement's 4 cells are almost always empty cells in the Sky or a play row. With the original head (mean of q·H_cell over the 4 cells), every candidate in the Sky reads the same Sky vector plus a position term, so the head could only learn a position prior. Candidate tokens also give the explanation directly: **the candidate's cross-attention row, drawn through M, is "what the model looked at when it scored this placement."** That is exactly the T-overhang picture we want.

**7. Value heads (as built, v2 on).** Two more linear layers on each candidate's final vector, beside the policy logit: **Qs** predicts the teacher's search score of that move and **Q1** its 1-ply score (both in units of 10 lines of attack). From v3 on only Qs is trained (Q1 has weight 0).

### Architecture as implemented (v4 / v5 / v5.1, `model.py`)

273k parameters (274.5k with the v5 kind table of 40); one forward pass is about 13M multiply-adds. Shapes are for one position: NT ≤ 31 board-side tokens (12 base + 12 overlays + 7 pieces; 27 before v5), NC ≤ 96 candidate slots (about 46 legal moves on average, the rest padding). d = 64, 4 heads of width 16, feed-forward width 256. Everything right of the dividing line is learned and trained end to end from the same loss; everything left of it is rules (Rust and the Python tokenizer).

```
 RULES (no learning)                       MODEL (learned)
 ===================                       ===============

 board 20×10 (0/1) ──────────────────────► Φ = E_val[cell] + E_row[r] + E_col[c]               200 × 64
                                                 │
 tokenizer ── per token j: kind τ_j,             │  region pooling, one learned query per kind:
   mask M_j (cells), box (rows/cols lo-hi),      │    w_jc = softmax_c over M_j of  q_τj · (W_k Φ_c) / √64
   12 extras; piece tokens: piece, no mask       │    z_j  = Σ_c w_jc · (W_v Φ_c)
                                                 ▼
                                           x_j = z_j + E_kind[τ_j] + PE_box(box_j) + W_extra·extras_j
                                                     (+ E_piece[p] for current / hold / preview tokens)   NT × 64
                                                 │
 FLAT-2D pair features, token j vs token k:      │
   4 row-span offsets (lo/hi × lo/hi), 4 col-    │
   span offsets, |M_j ∩ M_k|  → 9 numbers ─────► B_self = MLP_self(9 → 32 → 4 heads)       4 × NT × NT
   (a pair without boxes, e.g. a piece token:     │   computed ONCE, shared by all 4 layers
    a learned constant per head)                  │
                                                 ▼
                         ┌──────────────── self-attention block × 4 (pre-norm) ────────────────┐
                         │ h = LN1(x)                                                          │
                         │ Q = W_q h,  K = W_k h,  V = W_v h          (per head: NT × 16)       │
                         │ A = softmax( Q Kᵀ / √16  +  B_self )       ← FLAT-2D bias goes HERE, │
                         │                                              into the scores         │
                         │ x = x + W_o · concat_heads(A V)            ← residual add after W_o  │
                         │ x = x + FF(LN2(x))      FF = Linear 64→256, GELU, Linear 256→64      │
                         └─────────────────────────────────────────────────────────────────────┘
                                                 │  t_1 … t_NT  (board tokens after self-attention)
                                                 │
 move generator (Rust `bot expand`) ── every     │
 legal placement of the current piece and of     │
 the hold piece (or next piece if hold empty):   │
   4 cells, piece, used hold?, spin             │
   (none / mini / full), lines, attack          │
        │                                        │
        ▼   candidate embedding (learned):       │
   c_i = mean of Φ at its 4 cells                │   (raw cell vectors, not encoder outputs)
       + E_cpiece[piece] + E_hold[0/1]           │
       + E_spin[0..2] + E_lines[0..4]            │
       + W_attack · (attack / 10)                │
       + PE_box(bounding box of its 4 cells)     │
       + E_style[teacher]                        │   NC × 64 (padding slots masked)
        │                                        │
 FLAT-2D pair features, candidate i vs           │
 token j (candidate box + its 4 cells as mask) ► B_cross = MLP_cross(9 → 32 → 4 heads)     4 × NC × NT
        │                                        │
        ▼                                        ▼
                         ┌──────────────── cross-attention block × 1 (pre-norm) ───────────────┐
                         │ h = LN1(c),  m = LN_ctx(t)                                          │
                         │ Q = W_q h      ← queries from the candidates only                   │
                         │ K = W_k m,  V = W_v m   ← keys and values from the board tokens only│
                         │ A = softmax_j( Q Kᵀ / √16  +  B_cross )    NC × NT per head         │
                         │       ← the row A_i is "where move i looked" (the viewer's bars)    │
                         │ c = c + W_o · concat_heads(A V)            ← residual add after W_o  │
                         │ c = c + FF(LN2(c))                                                  │
                         │ board tokens are read, not changed; candidates never see each other │
                         └─────────────────────────────────────────────────────────────────────┘
                                                 │  f_1 … f_NC   (one 64-vector per move)
                         ┌───────────────────────┼───────────────────────┐
                         ▼                       ▼                       ▼
                 policy: Linear 64→1      Qs: Linear 64→1         Q1: Linear 64→1
                 one logit per move       search score / 10       1-ply score / 10 (weight 0)
                 padding → −∞
                         │
                         ▼
                 softmax over the moves → probabilities; argmax (or top-k for search)
                 index i → the i-th placement in the move generator's list (no unembedding)
```

Notes:

- **Separate weights.** The 4 self-attention blocks, the cross block, the two bias MLPs (B_self, B_cross), the cell/token embeddings and the candidate embeddings each have their own parameters. W_q, W_k, W_v, W_o above are per block.
- **The bias is not a residual term.** FLAT-2D enters each attention layer only through the scores before the softmax: geometry (overlap, distance) tilts which tokens get weight; the residual stream only ever receives W_o times the weighted values.
- **Legality needs no mask.** Every candidate comes from the move generator, so illegal moves never exist as tokens; the only mask is for padding slots.
- **One pass scores every move.** All candidates go through the cross block together (one batched matrix multiply), each independently; they compete only in the final softmax. This is a pointer-network style output (Vinyals et al., 2015): the vocabulary is rebuilt for every position.

## Interpretability

1. **Attention footprint**: a candidate's (or token's) attention row times M, drawn on the board. Use the mass-conserving version (rows of M divided by token size) to compare tokens of different size.
2. **Attention rollout** in token space, multiplied by M once at the end.
3. **Gradient × input on Φ**: per-cell attribution, including inside merged tokens.

Caveats and controls (these decide whether criterion 1 means anything):

- Attention to a hand-labelled overlay shows the model used the label, not that it understands the concept. **Train a second model without T-structure overlays.** Check whether its gradient and rollout maps still land on the overhang cells for T placements. If they do, the model found the slot itself.
- Confirm with ablation: drop a token (zero its row of M and mask it out) and measure the change in the placement logit.
- **"Pointing game" metric:** for each T-spin placement, the fraction where the most-attended cell lies inside the slot's cells. Compare with a random baseline of equal area.

## Training data (new)

- **Teacher:** the gate4 weights (`bot/results/gate4/gate4.txt`) under normal rules with garbage; this is the "ambitious human" style. The full search runs at about 50–100 ms per piece, so 12 threads give about 0.5M positions per hour.
- **Records:** board, piece, queue, hold, the chosen placement, the full legal placement set, and the T structures. The replay format already has everything except the legal placement set.
- **Soft targets:** the beam search keeps each node's first move. A root's score is its best surviving descendant (or its 1-ply score if pruned at ply 1). A softmax over those scores gives a listwise target that carries more information than one-hot imitation.
- **Coverage:** STSD, C-spin and rect12 are rare (tens of boards per 2,500). Oversample boards where they appear, and add positions seeded from the STSD build-ups we drew earlier, so the "what does it consider for an STSD" question has data behind it.
- **Style tags:** each position carries the tag of the teacher that labelled it (`normal` = normal.txt, `gate4` = gate4.txt, `danger` = CEM-survival, `bot/results/survive/survive.txt`); the tag is a model input, chosen at play time (`play.py --style normal|gate4|danger|auto`).
- **The danger data is a mix, by choice (2026-10-12):** DAgger rounds 3-4 select the students' trouble positions (near deaths, danger boards, rising holes) but also "student unsure" positions on any board, and all of them get the danger tag. Board state of the selected positions (hot = stack >= 12 or >= 6 holes, the `--style auto` rule; calm = stack < 8 and < 3 holes):

  | | hot | in between | calm |
  |---|---|---|---|
  | round 3 (95,007; `danger.npz`) | 78.6% | 16.2% | 5.2% |
  | round 4 (96,968; `data/dagger4/labelled.jsonl`) | 72.3% | 20.9% | 6.8% (mostly "unsure") |

  Kept as a mix: the calm part teaches the danger style how CEM-survival plays once it has dug out (always-danger mode plays those boards). In a 70/30 fine-tune the danger style's hot-board share is about 30% x 0.75 = ~22-24% of each epoch. Alternative if needed: route by the auto rule (hot -> CEM-survival / `danger`, else -> normal.txt / `normal`), ~27k relabels per round.

## Implementation plan

1. **Tokenizer** ✔ (`tokenizer.py`, `test_tokenizer.py`, `survey.py`; Rust `bot structures`).
   - Partition asserted on 3,000 replay boards.
   - Worked example tested.
   - Text renders of every structure kind.
2. **Dataset dump** (Rust): a `bot dataset` command that plays games with the teacher. For each position it writes the record above, including legal placements with their cells and per-root scores.
3. **Batching**: pad to n_max = K + P_max + 7 = 27 board-side tokens plus up to about 80 candidates, with key-padding masks. Precompute the pair features.
4. **Model** (PyTorch, CPU): stages 1–7 with shape asserts and the permutation test.
5. **Training**: listwise cross-entropy on the soft targets, plus value regression.
6. **Visualisation**: footprint, rollout and gradient maps side by side for a chosen candidate, inside the pygame or Godot replay viewer.

Baselines: a plain per-cell chessformer (200 tokens), a row-token model (no overlays), and the same model without T-structure overlays (the control above). The questions are whether structure tokens match accuracy at lower cost, and whether they give clearer maps.

## Deferred options

| Option | What it adds | Cost / risk |
| --- | --- | --- |
| Shallow CNN at the cell stage | Local shape features in each cell vector | Leaks neighbour information into tokens; becomes the compute bottleneck at small n |
| In-forward merging (ToMe-style) | The model decides groups | Early layers still run at 200 tokens |
| Learned tokens (Perceiver / TokenLearner) | Discovers concepts missing from the lexicon | Token meaning must be reverse-engineered |
| Supervised learned tokens (DETR-style) | Queries trained to predict labelled structures, plus free queries | More machinery; free queries show what the lexicon missed |
| Partially filled rect12 | Pits already half-filled, completable by the queue | Needs a queue-aware matcher |

## References

- [FLAT: Chinese NER Using Flat-Lattice Transformer](https://arxiv.org/abs/2004.11795) (Li et al., ACL 2020)
- [Lattice-BERT](https://aclanthology.org/2021.naacl-main.137) (Lai et al., NAACL 2021)
- [Byte Latent Transformer](https://huggingface.co/papers/2412.09871) (Pagnoni et al., 2024)
- [H-Net: Dynamic Chunking](https://arxiv.org/abs/2507.07955) (Hwang, Wang, Gu, 2025)
- [Vision Transformers with Mixed-Resolution Tokenization](https://arxiv.org/abs/2304.00287) (Ronen et al., 2023)
- [Token Merging: Your ViT But Faster](https://arxiv.org/abs/2210.09461) (Bolya et al., ICLR 2023)
- [TokenLearner](https://arxiv.org/abs/2106.11297) (Ryoo et al., NeurIPS 2021)
- [Perceiver IO](https://arxiv.org/abs/2107.14795) (Jaegle et al., 2021)
- [Token Erasure as a Footprint of Implicit Vocabulary Items in LLMs](https://aclanthology.org/2024.emnlp-main.543) (Feucht et al., EMNLP 2024)
