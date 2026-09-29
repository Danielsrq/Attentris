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
use crate::rules::{next_b2b, Piece, Rules, Shapes, BUFFER};

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

#[derive(Clone, Copy)]
struct Node {
    board: Board,
    hold: Option<Piece>,
    qi: u8,
    b2b: i32,
    reward: f32,
    score: f32,
    root: u16,
    dead: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct Decision {
    pub use_hold: bool,
    pub placement: Placement,
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
}

pub fn apply(board: &Board, p: &Placement, b2b: i32, rules: &Rules, shapes: &Shapes) -> Outcome {
    let mut nb = *board;
    let cells = p.cells(shapes);
    // Lock-out (Game.lock_piece): the piece rests entirely in the buffer.
    let all_in_buffer = cells.iter().all(|&(_, dy)| (p.y as i32 + dy as i32) < BUFFER);
    let (lines, cleared) = nb.place(cells, p.x as i32, p.y as i32);
    let nb2 = next_b2b(b2b, p.spin, lines);
    let perfect = lines > 0 && nb.is_empty();
    let attack = rules.attack(p.piece, p.spin, lines, nb2, perfect);
    Outcome { board: nb, lines, cleared, attack, b2b: nb2, lock_out: all_in_buffer && lines == 0 }
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
    fn expand_piece(&mut self, node: &Node, piece: Piece, new_hold: Option<Piece>, next_qi: u8,
                    root_moves: Option<(&mut Vec<(bool, Placement)>, bool)>, out: &mut Vec<Node>) {
        let mut buf = std::mem::take(&mut self.buf);
        self.mg.generate(&node.board, piece, &self.shapes, &mut buf);
        let mut roots = root_moves;
        for p in &buf {
            let o = apply(&node.board, p, node.b2b, &self.rules, &self.shapes);
            if o.lock_out {
                continue;
            }
            self.nodes += 1;
            let reward = node.reward + o.attack as f32;
            let score = reward + self.value(&o.board, o.b2b);
            let root = match roots.as_mut() {
                Some((list, used_hold)) => {
                    list.push((*used_hold, *p));
                    (list.len() - 1) as u16
                }
                None => node.root,
            };
            out.push(Node { board: o.board, hold: new_hold, qi: next_qi, b2b: o.b2b, reward, score, root, dead: false });
        }
        self.buf = buf;
    }

    fn expand(&mut self, node: &Node, queue: &[Piece], roots: Option<&mut Vec<(bool, Placement)>>, out: &mut Vec<Node>) {
        let qi = node.qi as usize;
        let before = out.len();
        let cur = queue[qi];
        match roots {
            Some(list) => {
                self.expand_piece(node, cur, node.hold, qi as u8 + 1, Some((list, false)), out);
                if self.cfg.use_hold {
                    if let Some(h) = node.hold {
                        if h != cur {
                            self.expand_piece(node, h, Some(cur), qi as u8 + 1, Some((list, true)), out);
                        }
                    } else if qi + 1 < queue.len() {
                        self.expand_piece(node, queue[qi + 1], Some(cur), qi as u8 + 2, Some((list, true)), out);
                    }
                }
            }
            None => {
                self.expand_piece(node, cur, node.hold, qi as u8 + 1, None, out);
                if self.cfg.use_hold {
                    if let Some(h) = node.hold {
                        if h != cur {
                            self.expand_piece(node, h, Some(cur), qi as u8 + 1, None, out);
                        }
                    } else if qi + 1 < queue.len() {
                        self.expand_piece(node, queue[qi + 1], Some(cur), qi as u8 + 2, None, out);
                    }
                }
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
            self.expand_piece(node, p, node.hold, node.qi, None, &mut kids);
            if self.cfg.use_hold {
                if let Some(h) = node.hold {
                    if h != p {
                        self.expand_piece(node, h, Some(p), node.qi, None, &mut kids);
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
        let root = Node { board: *board, hold, qi: 0, b2b, reward: 0.0, score: 0.0, root: 0, dead: false };
        let mut roots: Vec<(bool, Placement)> = Vec::new();
        let mut beam = Vec::new();
        self.expand(&root, queue, Some(&mut roots), &mut beam);
        if roots.is_empty() {
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
                self.expand(node, queue, None, &mut next);
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
        let best = beam[0];
        let (use_hold, placement) = roots[best.root as usize];
        Some(Decision { use_hold, placement })
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
