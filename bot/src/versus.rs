//! In-process versus matches between two of our searchers, for self-play tuning (`bot tune --fitness
//! versus`) and `bot versus`. The rules are bot/tools/versus.py's: both players get the same piece
//! sequence and the same sequence of garbage hole columns and place one piece each per turn; a
//! placement's attack first cancels the player's own pending garbage, what is left of the two attacks
//! of a turn cancels out, and the rest goes to the opponent as one chunk (one hole column). Pending
//! garbage that was already waiting before this turn lands after the receiver's next placement that
//! clears nothing, at most `cap` rows per placement. A match ends when a player tops out (the other
//! wins) or both have placed `pieces` pieces (a draw).
use crate::board::Board;
use crate::game::{Bag, Rng};
use crate::rules::{Piece, Rules, Shapes, W};
use crate::search::{apply, Searcher};

struct Player {
    board: Board,
    hold: Option<Piece>,
    b2b: i32,
    bag: Bag,
    drawn: usize,
    /// [rows, hole column, turn it arrived]
    pending: Vec<[usize; 3]>,
    hole_i: usize,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct MatchResult {
    /// 0 = a won, 1 = b won, -1 = draw (both topped out the same turn, or the piece limit)
    pub winner: i8,
    pub turns: usize,
    /// attack after cancelling, i.e. garbage actually sent to the opponent
    pub sent: [u64; 2],
    pub received: [u64; 2],
    pub attack: [i64; 2],
}

impl MatchResult {
    /// Fitness of side `i`: 1 for a win, 0.5 for a draw, 0 for a loss, plus a margin term in
    /// (-0.25, 0.25): garbage sent minus received, so close losses and big wins are told apart.
    pub fn score(&self, i: usize) -> f64 {
        let result = if self.winner < 0 { 0.5 } else if self.winner as usize == i { 1.0 } else { 0.0 };
        let (s, r) = (self.sent[i] as f64, self.received[i] as f64);
        result + 0.25 * (s - r) / (s + r + 10.0)
    }
}

/// One piece for `p`: Some((attack, lines)) or None when it tops out.
fn place(p: &mut Player, s: &mut Searcher, rules: &Rules, shapes: &Shapes) -> Option<(i32, u32)> {
    let queue: Vec<Piece> = (0..6).map(|i| p.bag.get(p.drawn + i)).collect();
    let bag_next = p.bag.possible(p.drawn + 6);
    let spawned = queue[0];
    let d = s.search(&p.board, &queue, p.hold, p.b2b, &bag_next)?;
    p.drawn += 1;
    if d.use_hold {
        if p.hold.is_none() {
            p.drawn += 1;
        }
        p.hold = Some(spawned);
    }
    let o = apply(&p.board, &d.placement, p.b2b, rules, shapes);
    if o.lock_out {
        return None;
    }
    p.board = o.board;
    p.b2b = o.b2b;
    Some((o.attack.max(0), o.lines))
}

pub fn play_match(a: &mut Searcher, b: &mut Searcher, rules: &Rules, seed: u64, pieces: usize, cap: usize) -> MatchResult {
    let shapes = Shapes::for_rules(rules);
    let mut rng = Rng::new(seed.wrapping_mul(7919).wrapping_add(17));
    let holes: Vec<usize> = (0..4096).map(|_| rng.below(W as u64) as usize).collect();
    let new = || Player { board: Board::default(), hold: None, b2b: -1, bag: Bag::new(seed), drawn: 0, pending: Vec::new(), hole_i: 0 };
    let mut ps = [new(), new()];
    let mut res = MatchResult::default();
    for turn in 0..pieces {
        let ra = place(&mut ps[0], a, rules, &shapes);
        let rb = place(&mut ps[1], b, rules, &shapes);
        res.turns = turn + 1;
        let (la, lb) = match (ra, rb) {
            (None, None) => {
                res.winner = -1;
                return res;
            }
            (None, _) => {
                res.winner = 1;
                return res;
            }
            (_, None) => {
                res.winner = 0;
                return res;
            }
            (Some(x), Some(y)) => (x, y),
        };
        let locks = [la, lb];
        // cancel own pending garbage first
        let mut rem = [0u64; 2];
        for i in 0..2 {
            let mut atk = locks[i].0 as u64;
            res.attack[i] += locks[i].0 as i64;
            while atk > 0 && !ps[i].pending.is_empty() {
                let c = atk.min(ps[i].pending[0][0] as u64);
                ps[i].pending[0][0] -= c as usize;
                atk -= c;
                if ps[i].pending[0][0] == 0 {
                    ps[i].pending.remove(0);
                }
            }
            rem[i] = atk;
        }
        let m = rem[0].min(rem[1]); // simultaneous attacks cancel each other
        for i in 0..2 {
            let r = rem[i] - m;
            if r > 0 {
                let o = 1 - i;
                let hole = holes[ps[o].hole_i % holes.len()];
                ps[o].hole_i += 1;
                ps[o].pending.push([r as usize, hole, turn]);
                res.sent[i] += r;
            }
        }
        // land garbage that was waiting before this turn, after a placement that cleared nothing
        for i in 0..2 {
            if locks[i].1 > 0 {
                continue;
            }
            let mut room = cap;
            while room > 0 && !ps[i].pending.is_empty() && ps[i].pending[0][2] < turn {
                let c = room.min(ps[i].pending[0][0]);
                let hole = ps[i].pending[0][1] as i32;
                ps[i].board.push_garbage(&vec![hole; c]);
                ps[i].pending[0][0] -= c;
                room -= c;
                res.received[i] += c as u64;
                if ps[i].pending[0][0] == 0 {
                    ps[i].pending.remove(0);
                }
            }
        }
    }
    res.winner = -1;
    res
}
