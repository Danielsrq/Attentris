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
}

impl Default for SearchConfig {
    fn default() -> Self {
        SearchConfig { beam: 64, depth: 6, use_hold: true, chance: true, chance_top: 12, death: -1000.0 }
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
}

#[derive(Clone, Debug)]
pub struct Decision {
    pub use_hold: bool,
    pub placement: Placement,
    /// The whole best line, starting with this move. Later steps are what the bot
    /// expects to play if nothing new changes its mind.
    pub plan: Vec<Step>,
}

pub struct Searcher {
    pub cfg: SearchConfig,
    pub weights: Weights,
    pub rules: Rules,
    shapes: Shapes,
    mg: MoveGen,
    buf: Vec<Placement>,
    pub nodes: u64,
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
}

pub fn apply(board: &Board, p: &Placement, b2b: i32, rules: &Rules, shapes: &Shapes) -> Outcome {
    let mut nb = *board;
    let cells = p.cells(shapes);
    // Lock-out (Game.lock_piece): the piece rests entirely in the buffer.
    let all_in_buffer = cells.iter().all(|&(_, dy)| (p.y as i32 + dy as i32) < BUFFER);
    let spin = if rules.promote_minis && p.spin == Spin::Mini { Spin::Full } else { p.spin };
    let (lines, cleared) = nb.place(cells, p.x as i32, p.y as i32);
    let nb2 = next_b2b(b2b, spin, lines);
    let perfect = lines > 0 && nb.is_empty();
    let attack = rules.attack(p.piece, spin, lines, nb2, perfect);
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
    Outcome { board: nb, lines, cleared, attack, b2b: nb2, lock_out: all_in_buffer && lines == 0, spin, repeat, healed }
}

impl Searcher {
    pub fn new(cfg: SearchConfig, weights: Weights, rules: Rules) -> Searcher {
        let mut mg = MoveGen::new();
        mg.all_spin = rules.all_spin;
        Searcher { cfg, weights, rules, shapes: Shapes::new(), mg, buf: Vec::with_capacity(256), nodes: 0 }
    }

    fn value(&self, b: &Board, b2b: i32) -> f32 {
        self.weights.value(&features(b, b2b))
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
            out.push(Node { board: o.board, hold: new_hold, qi: next_qi, b2b: o.b2b, reward, score, dead: false, path, plen });
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
        let root = Node { board: *board, hold, qi: 0, b2b, reward: 0.0, score: 0.0, dead: false, path: [empty; MAX_PLAN], plen: 0 };
        let mut beam = Vec::new();
        self.expand(&root, queue, &mut beam);
        if beam.iter().all(|n| n.dead) {
            return None;
        }
        let plies = self.cfg.depth.max(1).min(queue.len());
        let mut seen: HashMap<u64, usize> = HashMap::new();
        for _ in 1..plies {
            prune(&mut beam, self.cfg.beam, &mut seen);
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
        prune(&mut beam, self.cfg.beam, &mut seen);

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
        Some(Decision { use_hold: plan[0].use_hold, placement: plan[0].placement, plan })
    }
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
