//! Beam search over the known queue.
//!
//!   score(node) = sum of attack earned on the path + V(node's board)
//!
//! Reward is exact (the simulator says what a placement sent); V estimates what is
//! still to come. Each ply places exactly one piece, so comparing nodes at the same
//! ply compares equal piece budgets -- which is what makes attack-per-piece
//! efficiency (TSD vs quad) show up without being coded.
//!
//! Hold is a branch: place the current piece, or swap (an empty hold pulls the next
//! queue piece). After the known queue runs out, an optional CHANCE ply averages the
//! best reply over every piece the 7-bag can still deal, so a leaf that only works
//! for one lucky piece is marked down.

use std::collections::HashMap;

use crate::board::{Board, Placement};
use crate::eval::{features, Weights};
use crate::movegen::MoveGen;
use crate::rules::{clear_type, next_b2b, Piece, Rules, Shapes, Spin, BUFFER};

#[derive(Clone, Debug)]
pub struct SearchConfig {
    pub beam: usize,
    /// Plies to search; capped by the known queue.
    pub depth: usize,
    pub use_hold: bool,
    /// Expectimax ply over the unseen next piece, applied to the best `chance_top` leaves.
    pub chance: bool,
    pub chance_top: usize,
    /// Reward charged when a line dies (no legal placement / lock-out).
    pub death: f32,
    /// Also report every root placement with its scores (`Decision::roots`), for datasets.
    pub roots: bool,
    /// Root-diverse pruning (labelling only): every ply keeps the best `root_keep` lines of
    /// EVERY first move, plus the best beam/2 others, so each root move -- blunders too --
    /// ends with a search score. 0 = the normal beam.
    pub root_keep: usize,
}

impl Default for SearchConfig {
    fn default() -> Self {
        SearchConfig { beam: 64, depth: 6, use_hold: true, chance: true, chance_top: 12, death: -1000.0, roots: false, root_keep: 0 }
    }
}

/// Longest line a node remembers: the known queue is 6 pieces.
pub const MAX_PLAN: usize = 8;

#[derive(Clone, Copy, Debug)]
pub struct Step {
    pub use_hold: bool,
    pub placement: Placement,
}

#[derive(Clone, Copy)]
struct Node {
    board: Board,
    hold: Option<Piece>,
    qi: u8,
    b2b: i32,
    reward: f32,
    score: f32,
    dead: bool,
    /// The placements that led here from the root; path[0] is the move to play.
    path: [Step; MAX_PLAN],
    plen: u8,
    /// Index of the root placement this line starts with.
    root: u16,
}

/// Fill in `d`'s root search scores from a root-diverse search of the same position
/// (`SearchConfig::root_keep`): same board, same move generation, so the roots line up.
pub fn merge_root_best(d: &mut Decision, d2: &Decision) {
    if d2.roots.len() != d.roots.len() {
        return;
    }
    for (r1, r2) in d.roots.iter_mut().zip(d2.roots.iter()) {
        r1.best = match (r1.best, r2.best) {
            (Some(x), Some(y)) => Some(x.max(y)),
            (x, None) => x,
            (None, y) => y,
        };
    }
}

/// One legal first move and what the search thought of it (`SearchConfig::roots`).
#[derive(Clone, Copy, Debug)]
pub struct RootScore {
    pub step: Step,
    /// Attack the move itself sends.
    pub attack: f32,
    /// attack + V(board after the move): the 1-ply score.
    pub ply1: f32,
    /// Best line through this move that survived to the last ply (before the chance
    /// ply), or None if the beam pruned every line through it.
    pub best: Option<f32>,
}

#[derive(Clone, Debug)]
pub struct Decision {
    pub use_hold: bool,
    pub placement: Placement,
    /// The whole best line, starting with this move. Later steps are what the bot
    /// expects to play if nothing new changes its mind.
    pub plan: Vec<Step>,
    /// With `SearchConfig::roots`: every root placement, and which one was played.
    pub roots: Vec<RootScore>,
    pub chosen_root: usize,
}

pub struct Searcher {
    pub cfg: SearchConfig,
    pub weights: Weights,
    pub rules: Rules,
    shapes: Shapes,
    mg: MoveGen,
    buf: Vec<Placement>,
    pub nodes: u64,
    /// When not empty, only these first moves (hold flag, sorted cells) are searched: a
    /// shortlist from outside (AtTenTris's top-k, `bot play-ext --shortlist`).
    pub root_allow: Vec<(bool, [(i32, i32); 4])>,
}

/// Outcome of one placement on a board.
pub struct Outcome {
    pub board: Board,
    pub lines: u32,
    pub cleared: u64,
    pub attack: i32,
    pub b2b: i32,
    pub lock_out: bool,
    /// The spin as scored (a Mini promoted to Full in all-spin mode).
    pub spin: Spin,
    /// This clear repeated the previous clear's type and wounded a row.
    pub repeat: bool,
    /// Wound lines that healed on this clear.
    pub healed: u32,
    /// Burst rules: this clear's base attack was doubled.
    pub burst: bool,
}

pub fn apply(board: &Board, p: &Placement, b2b: i32, rules: &Rules, shapes: &Shapes) -> Outcome {
    let mut nb = *board;
    let cells = p.cells(shapes);
    // Lock-out (Game.lock_piece): the piece rests entirely in the buffer.
    let all_in_buffer = cells.iter().all(|&(_, dy)| (p.y as i32 + dy as i32) < BUFFER);
    let mut spin = p.spin;
    if rules.non_t_mini && p.piece != Piece::T && spin != Spin::None {
        spin = Spin::Mini;
    }
    if rules.promote_minis && spin == Spin::Mini {
        spin = Spin::Full;
    }
    let (lines, cleared) = nb.place(cells, p.x as i32, p.y as i32);
    let nb2 = next_b2b(b2b, spin, lines);
    let perfect = lines > 0 && nb.is_empty();
    let (attack, burst) = if rules.battle {
        nb.combo = if lines > 0 { board.combo.saturating_add(1) } else { -1 };
        (rules.battle_attack(spin, lines, nb.combo as i32, nb2, perfect), false)
    } else if rules.gate {
        let bonus = if rules.gate_b2b && nb2 >= 1 {
            let b = rules.b2b.bonus(nb2);
            rules.b2b_cap.map_or(b, |c| b.min(c))
        } else {
            0
        };
        let (credit, (left, len, pending)) =
            rules.gate_step_b2b((board.burst_armed, board.gate_len, board.gate_pending), spin, lines, bonus);
        nb.burst_armed = left;
        nb.gate_len = len;
        nb.gate_pending = pending;
        (credit, credit > 0)
    } else {
        let (burst, left) = rules.burst_step(board.burst_armed, spin, lines);
        nb.burst_armed = left;
        if rules.combo == crate::rules::ComboTable::None {
            (rules.attack_with(p.piece, spin, lines, nb2, perfect, burst), burst)
        } else {
            // the combo applies to base + B2B; the perfect clear bonus comes on top
            nb.combo = if lines > 0 { board.combo.saturating_add(1) } else { -1 };
            let a = rules.combo.apply(rules.attack_with(p.piece, spin, lines, nb2, false, burst), nb.combo as i32);
            (a + if perfect { rules.perfect_clear_bonus } else { 0 }, burst)
        }
    };
    let (mut repeat, mut healed) = (false, 0);
    if rules.repeat_penalty && lines > 0 {
        // Order per the rules: lines are gone, every wound counts down (healing the
        // oldest at 0), then a repeat wounds the new bottom in-play row.
        healed = nb.tick_wounds();
        let t = clear_type(p.piece, spin, lines);
        if nb.last_clear == t {
            // The real hole is chosen by the game with messiness; the search assumes the
            // previous wound's column (it only matters once the line turns to garbage).
            nb.add_wound(rules.wound_clears, nb.last_wound_hole);
            repeat = true;
        }
        nb.last_clear = t;
    }
    // TetrisBattle tops out when the top visible row holds anything after the lock.
    let lock_out = if rules.battle { nb.rows[BUFFER as usize] != 0 } else { all_in_buffer && lines == 0 };
    Outcome { board: nb, lines, cleared, attack, b2b: nb2, lock_out, spin, repeat, healed, burst }
}

impl Searcher {
    pub fn new(cfg: SearchConfig, weights: Weights, rules: Rules) -> Searcher {
        let mut mg = MoveGen::new();
        mg.all_spin = rules.all_spin;
        mg.battle = rules.battle;
        mg.no_180 = rules.no_180;
        Searcher { cfg, weights, rules, shapes: Shapes::for_rules(&rules), mg, buf: Vec::with_capacity(256), nodes: 0, root_allow: Vec::new() }
    }

    fn value(&self, b: &Board, b2b: i32) -> f32 {
        self.weights.value(&features(b, b2b, self.rules.b2b))
    }

    /// Children of `node` for one piece choice, appended to `out`.
    fn expand_piece(&mut self, node: &Node, piece: Piece, new_hold: Option<Piece>, next_qi: u8, used_hold: bool,
                    out: &mut Vec<Node>) {
        let mut buf = std::mem::take(&mut self.buf);
        self.mg.generate(&node.board, piece, &self.shapes, &mut buf);
        for p in &buf {
            let o = apply(&node.board, p, node.b2b, &self.rules, &self.shapes);
            if o.lock_out {
                continue;
            }
            self.nodes += 1;
            let reward = node.reward + o.attack as f32;
            let score = reward + self.value(&o.board, o.b2b);
            let mut path = node.path;
            let mut plen = node.plen;
            if (plen as usize) < MAX_PLAN {
                path[plen as usize] = Step { use_hold: used_hold, placement: *p };
                plen += 1;
            }
            out.push(Node { board: o.board, hold: new_hold, qi: next_qi, b2b: o.b2b, reward, score, dead: false, path, plen, root: node.root });
        }
        self.buf = buf;
    }

    fn expand(&mut self, node: &Node, queue: &[Piece], out: &mut Vec<Node>) {
        let qi = node.qi as usize;
        let before = out.len();
        let cur = queue[qi];
        self.expand_piece(node, cur, node.hold, qi as u8 + 1, false, out);
        if self.cfg.use_hold {
            if let Some(h) = node.hold {
                if h != cur {
                    self.expand_piece(node, h, Some(cur), qi as u8 + 1, true, out);
                }
            } else if qi + 1 < queue.len() {
                self.expand_piece(node, queue[qi + 1], Some(cur), qi as u8 + 2, true, out);
            }
        }
        if out.len() == before {
            // Nothing legal: this line tops out.
            let mut d = *node;
            d.dead = true;
            d.score = node.reward + self.cfg.death;
            out.push(d);
        }
    }

    /// Best reply to an unknown piece drawn from `bag`, averaged: E_p[max (r + V)].
    fn chance_value(&mut self, node: &Node, bag: &[Piece]) -> f32 {
        if bag.is_empty() {
            return node.score;
        }
        let mut total = 0.0;
        let mut kids = Vec::new();
        for &p in bag {
            kids.clear();
            // Either place the new piece, or (if hold is full) place hold and keep it.
            self.expand_piece(node, p, node.hold, node.qi, false, &mut kids);
            if self.cfg.use_hold {
                if let Some(h) = node.hold {
                    if h != p {
                        self.expand_piece(node, h, Some(p), node.qi, true, &mut kids);
                    }
                }
            }
            let best = kids.iter().map(|k| k.score).fold(f32::NEG_INFINITY, f32::max);
            total += if best.is_finite() { best } else { node.reward + self.cfg.death };
        }
        total / bag.len() as f32
    }

    /// `queue` = [current, previews...]; `bag` = pieces the first unseen draw can be.
    pub fn search(&mut self, board: &Board, queue: &[Piece], hold: Option<Piece>, b2b: i32, bag: &[Piece]) -> Option<Decision> {
        let empty = Step { use_hold: false, placement: Placement { piece: Piece::I, rot: 0, x: 0, y: 0, spin: crate::rules::Spin::None, kick: -1 } };
        let root = Node { board: *board, hold, qi: 0, b2b, reward: 0.0, score: 0.0, dead: false, path: [empty; MAX_PLAN], plen: 0, root: 0 };
        let mut beam = Vec::new();
        self.expand(&root, queue, &mut beam);
        if !self.root_allow.is_empty() {
            let shapes = &self.shapes;
            let allow = &self.root_allow;
            let kept: Vec<Node> = beam.iter().copied().filter(|n| {
                let mut c = n.path[0].placement.abs_cells(shapes);
                c.sort();
                !n.dead && allow.contains(&(n.path[0].use_hold, c))
            }).collect();
            if !kept.is_empty() {
                beam = kept;
            }
        }
        for (i, n) in beam.iter_mut().enumerate() {
            n.root = i as u16;
        }
        let mut roots: Vec<RootScore> = if self.cfg.roots {
            beam.iter()
                .filter(|n| !n.dead)
                .map(|n| RootScore { step: n.path[0], attack: n.reward, ply1: n.score, best: None })
                .collect()
        } else {
            Vec::new()
        };
        if beam.iter().all(|n| n.dead) {
            return None;
        }
        let plies = self.cfg.depth.max(1).min(queue.len());
        let mut seen: HashMap<u64, usize> = HashMap::new();
        for _ in 1..plies {
            self.prune_any(&mut beam, &mut seen);
            let mut next = Vec::with_capacity(beam.len() * 40);
            for node in &beam {
                if node.dead || node.qi as usize >= queue.len() {
                    // Finished early (hold consumed a queue piece) or dead: carry as a leaf.
                    next.push(*node);
                    continue;
                }
                self.expand(node, queue, &mut next);
            }
            beam = next;
        }
        self.prune_any(&mut beam, &mut seen);
        if self.cfg.roots {
            for n in &beam {
                if let Some(r) = roots.get_mut(n.root as usize) {
                    if !n.dead && r.best.map_or(true, |b| n.score > b) {
                        r.best = Some(n.score);
                    }
                }
            }
        }

        if self.cfg.chance && plies >= queue.len() {
            let k = self.cfg.chance_top.min(beam.len());
            for i in 0..k {
                if beam[i].dead {
                    continue;
                }
                let n = beam[i];
                beam[i].score = self.chance_value(&n, bag);
            }
            beam[..k].sort_by(|a, b| b.score.total_cmp(&a.score));
        }
        let best = &beam[0];
        let plan = best.path[..best.plen as usize].to_vec();
        let chosen_root = best.root as usize;
        Some(Decision { use_hold: plan[0].use_hold, placement: plan[0].placement, plan, roots, chosen_root })
    }
}

impl Searcher {
    fn prune_any(&self, beam: &mut Vec<Node>, seen: &mut HashMap<u64, usize>) {
        if self.cfg.root_keep > 0 {
            prune_diverse(beam, self.cfg.beam / 2, self.cfg.root_keep, seen);
        } else {
            prune(beam, self.cfg.beam, seen);
        }
    }
}

/// `SearchConfig::root_keep`: the best `keep` lines of every root move, then the best
/// `extra` of the rest. Transpositions are only merged within a root, so two first moves
/// that reach the same board each keep their line.
fn prune_diverse(nodes: &mut Vec<Node>, extra: usize, keep: usize, seen: &mut HashMap<u64, usize>) {
    nodes.sort_by(|a, b| b.score.total_cmp(&a.score));
    seen.clear();
    let key = |n: &Node| n.board.hash64()
        ^ (n.hold.map_or(7u64, |p| p as u64) << 56)
        ^ ((n.qi as u64) << 60)
        ^ ((n.b2b.clamp(-1, 30) + 1) as u64).wrapping_mul(0x2545_F491_4F6C_DD1D)
        ^ (n.root as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    let mut per_root: HashMap<u16, usize> = HashMap::new();
    let mut taken = vec![false; nodes.len()];
    let mut out = Vec::new();
    for (i, n) in nodes.iter().enumerate() {
        let c = per_root.entry(n.root).or_insert(0);
        if *c >= keep || seen.contains_key(&key(n)) {
            continue;
        }
        seen.insert(key(n), out.len());
        *c += 1;
        taken[i] = true;
        out.push(*n);
    }
    let mut added = 0;
    for (i, n) in nodes.iter().enumerate() {
        if added >= extra {
            break;
        }
        if taken[i] || seen.contains_key(&key(n)) {
            continue;
        }
        seen.insert(key(n), out.len());
        added += 1;
        out.push(*n);
    }
    // keep the beam sorted (the chance ply rescans the top of it)
    out.sort_by(|a, b| b.score.total_cmp(&a.score));
    *nodes = out;
}

/// Sort by score, drop transpositions (same board, hold, queue position and chain),
/// keep the top `width`.
fn prune(nodes: &mut Vec<Node>, width: usize, seen: &mut HashMap<u64, usize>) {
    nodes.sort_by(|a, b| b.score.total_cmp(&a.score));
    seen.clear();
    let mut out = Vec::with_capacity(width);
    for n in nodes.iter() {
        let key = n.board.hash64()
            ^ (n.hold.map_or(7u64, |p| p as u64) << 56)
            ^ ((n.qi as u64) << 60)
            ^ ((n.b2b.clamp(-1, 30) + 1) as u64).wrapping_mul(0x2545_F491_4F6C_DD1D);
        if seen.insert(key, out.len()).is_some() {
            continue;
        }
        out.push(*n);
        if out.len() >= width {
            break;
        }
    }
    *nodes = out;
}
