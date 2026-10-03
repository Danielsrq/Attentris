//! Every placement a player can reach from spawn with left / right / soft drop /
//! cw / ccw / 180, including tucks and kicked spins -- unlike solver/enumerator.gd,
//! which only rotates at the top and drops.
//!
//! The walk is a BFS over (x, y, rot). Spin status is decided per ROTATION TRANSITION,
//! not per state, because the same resting state can be reached by a move (no spin) or
//! by different kicks (Mini vs Full via the kick-5 upgrade). For each distinct set of
//! cells we keep the best spin reachable, since a spin is never worse than the same
//! cells without one: its attack is >= and it keeps B2B where a plain clear breaks it.
//!
//! Not modelled: lock-delay move-reset limits (15) and gravity. With 20G-free handling
//! and the default lock rules, a human can reach everything this finds.

use crate::board::{Board, Placement};
use crate::rules::{kicks, Piece, Shapes, Spin};

const XOFF: i32 = 3;
const YOFF: i32 = 3;
const XS: i32 = 17;
const YS: i32 = 47;
const STATES: usize = (4 * XS * YS) as usize;
const TABLE: usize = 1024;

pub struct MoveGen {
    pub all_spin: bool,
    visited: Vec<u32>,
    stamp: u32,
    queue: Vec<(i8, i8, u8)>,
    keys: Vec<u64>,
    slot: Vec<u16>,
    key_stamp: Vec<u32>,
}

impl MoveGen {
    pub fn new() -> MoveGen {
        MoveGen {
            all_spin: true,
            visited: vec![0; STATES],
            stamp: 0,
            queue: Vec::with_capacity(1024),
            keys: vec![0; TABLE],
            slot: vec![0; TABLE],
            key_stamp: vec![0; TABLE],
        }
    }

    #[inline]
    fn idx(x: i32, y: i32, r: usize) -> usize {
        (r as i32 * XS * YS + (x + XOFF) * YS + (y + YOFF)) as usize
    }

    /// All reachable placements for `piece` on `board`. Empty when spawn is blocked
    /// (block-out).
    pub fn generate(&mut self, board: &Board, piece: Piece, shapes: &Shapes, out: &mut Vec<Placement>) {
        out.clear();
        self.stamp = self.stamp.wrapping_add(1);
        if self.stamp == 0 {
            self.visited.iter_mut().for_each(|v| *v = 0);
            self.key_stamp.iter_mut().for_each(|v| *v = 0);
            self.stamp = 1;
        }
        let stamp = self.stamp;
        let sh = &shapes.cells[piece as usize];
        let (sx, sy) = piece.spawn();
        if !board.fits(&sh[0], sx, sy) {
            return;
        }
        self.queue.clear();
        self.queue.push((sx as i8, sy as i8, 0));
        self.visited[Self::idx(sx, sy, 0)] = stamp;
        let mut head = 0;
        while head < self.queue.len() {
            let (x8, y8, r8) = self.queue[head];
            head += 1;
            let (x, y, r) = (x8 as i32, y8 as i32, r8 as usize);
            let cells = &sh[r];
            let grounded = !board.fits(cells, x, y + 1);
            if grounded {
                self.record(board, piece, shapes, r, x, y, Spin::None, -1, out);
            }
            // Moves: left, right, soft drop one cell.
            for (dx, dy) in [(-1, 0), (1, 0), (0, 1)] {
                let (nx, ny) = (x + dx, y + dy);
                if board.fits(cells, nx, ny) {
                    let i = Self::idx(nx, ny, r);
                    if self.visited[i] != stamp {
                        self.visited[i] = stamp;
                        self.queue.push((nx as i8, ny as i8, r as u8));
                    }
                }
            }
            // Rotations: cw, ccw, 180. First fitting kick wins (Game.try_rotate).
            for to in [(r + 1) % 4, (r + 3) % 4, (r + 2) % 4] {
                let tcells = &sh[to];
                for (k, &(kx, ky)) in kicks(piece, r, to).iter().enumerate() {
                    let (nx, ny) = (x + kx as i32, y + ky as i32);
                    if !board.fits(tcells, nx, ny) {
                        continue;
                    }
                    if !board.fits(tcells, nx, ny + 1) {
                        // Grounded right after a rotation: locking here keeps the
                        // rotation as the last action, so the spin counts.
                        let spin = detect_spin(board, piece, tcells, to, nx, ny, k as i32, self.all_spin);
                        self.record(board, piece, shapes, to, nx, ny, spin, k as i8, out);
                    }
                    let i = Self::idx(nx, ny, to);
                    if self.visited[i] != stamp {
                        self.visited[i] = stamp;
                        self.queue.push((nx as i8, ny as i8, to as u8));
                    }
                    break;
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn record(&mut self, _board: &Board, piece: Piece, shapes: &Shapes, r: usize, x: i32, y: i32,
              spin: Spin, kick: i8, out: &mut Vec<Placement>) {
        let cells = &shapes.cells[piece as usize][r];
        // Cells are pre-sorted by (y, x), so the absolute list is canonical and two
        // rotation states that cover the same squares (I, S, Z, O) produce one key.
        let mut key: u64 = 0;
        for &(dx, dy) in cells {
            let id = ((y + dy as i32) * 10 + (x + dx as i32)) as u64;
            key = key << 9 | id;
        }
        let mut h = (key.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 54) as usize & (TABLE - 1);
        loop {
            if self.key_stamp[h] != self.stamp {
                self.key_stamp[h] = self.stamp;
                self.keys[h] = key;
                self.slot[h] = out.len() as u16;
                out.push(Placement { piece, rot: r as u8, x: x as i8, y: y as i8, spin, kick });
                return;
            }
            if self.keys[h] == key {
                let p = &mut out[self.slot[h] as usize];
                if spin > p.spin {
                    *p = Placement { piece, rot: r as u8, x: x as i8, y: y as i8, spin, kick };
                }
                return;
            }
            h = (h + 1) & (TABLE - 1);
        }
    }
}

/// SpinDetect.detect, with the rotation already known to be the last action.
#[allow(clippy::too_many_arguments)]
pub fn detect_spin(board: &Board, piece: Piece, cells: &[(i8, i8); 4], rot: usize, x: i32, y: i32, kick: i32, all_spin: bool) -> Spin {
    if piece == Piece::T {
        let cx = x + 1;
        let cy = y + 1;
        let corners = [(-1, -1), (1, -1), (-1, 1), (1, 1)];
        let filled = corners.iter().filter(|&&(dx, dy)| board.blocked(cx + dx, cy + dy)).count();
        if filled >= 3 {
            let front: [(i32, i32); 2] = match rot {
                0 => [(-1, -1), (1, -1)],
                1 => [(1, -1), (1, 1)],
                2 => [(-1, 1), (1, 1)],
                _ => [(-1, -1), (-1, 1)],
            };
            let ff = front.iter().filter(|&&(dx, dy)| board.blocked(cx + dx, cy + dy)).count();
            if ff == 2 || kick == 4 {
                return Spin::Full;
            }
            return Spin::Mini;
        }
        // Fewer than three corners: falls through to the all-spin rule, exactly as
        // SpinDetect.detect does.
    }
    if !all_spin {
        return Spin::None;
    }
    // All-spin: immobile after the rotation (cannot move left, right, up or down).
    for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
        if board.fits(cells, x + dx, y + dy) {
            return Spin::None;
        }
    }
    Spin::Full
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rules::{B2bMode, Rules};

    /// `rows` are the bottom rows of the field, top first; 'X' = filled.
    fn board_from(rows: &[&str]) -> Board {
        let mut b = Board::default();
        let base = 40 - rows.len();
        for (i, r) in rows.iter().enumerate() {
            for (x, ch) in r.chars().enumerate() {
                if ch == 'X' {
                    b.rows[base + i] |= 1 << x;
                }
            }
        }
        b
    }

    fn placements(b: &Board, p: Piece, all_spin: bool) -> Vec<Placement> {
        let shapes = Shapes::new();
        let mut mg = MoveGen::new();
        mg.all_spin = all_spin;
        let mut out = Vec::new();
        mg.generate(b, p, &shapes, &mut out);
        out
    }

    #[test]
    fn empty_board_placement_counts() {
        let b = Board::default();
        // Distinct resting cell sets on a flat floor.
        assert_eq!(placements(&b, Piece::T, true).len(), 34);
        assert_eq!(placements(&b, Piece::I, true).len(), 17);
        assert_eq!(placements(&b, Piece::O, true).len(), 9);
        assert_eq!(placements(&b, Piece::S, true).len(), 17);
        assert_eq!(placements(&b, Piece::L, true).len(), 34);
    }

    #[test]
    fn tsd_slot_is_a_full_spin_double() {
        let b = board_from(&["...X......", "XXX...XXXX", "XXXX.XXXXX"]);
        let shapes = Shapes::new();
        for all_spin in [true, false] {
            let p = placements(&b, Piece::T, all_spin)
                .into_iter()
                .find(|p| {
                    let mut nb = b;
                    nb.place(p.cells(&shapes), p.x as i32, p.y as i32).0 == 2
                })
                .expect("TSD reachable");
            assert_eq!(p.spin, Spin::Full);
            assert_eq!(p.rot, 2);
        }
    }

    #[test]
    fn no_allspin_means_no_s_spins() {
        let b = board_from(&["X.........", "XX.XXXXXXX", "X..XXXXXXX"]);
        assert!(placements(&b, Piece::S, false).iter().all(|p| p.spin == Spin::None));
    }

    #[test]
    fn flat_drop_is_not_a_spin() {
        let b = board_from(&["XXXX.XXXXX"]);
        for p in placements(&b, Piece::I, true) {
            assert_eq!(p.spin, Spin::None, "{:?}", p);
        }
    }

    #[test]
    fn attack_table() {
        let r = Rules::default();
        assert_eq!(r.attack(Piece::T, Spin::None, 1, -1, false), 0);
        assert_eq!(r.attack(Piece::T, Spin::None, 2, -1, false), 1);
        assert_eq!(r.attack(Piece::I, Spin::None, 4, 0, false), 4);
        assert_eq!(r.attack(Piece::I, Spin::None, 4, 1, false), 5);
        assert_eq!(r.attack(Piece::T, Spin::Full, 2, 0, false), 4);
        assert_eq!(r.attack(Piece::T, Spin::Full, 3, 3, false), 7);
        assert_eq!(r.attack(Piece::T, Spin::Mini, 1, 0, false), 0, "a Mini sends nothing");
        assert_eq!(r.attack(Piece::S, Spin::Mini, 2, 5, false), 0, "   not even the B2B bonus");
        assert!(crate::rules::is_difficult(Spin::Mini, 1), "   but it keeps the chain alive");
        assert_eq!(r.attack(Piece::S, Spin::Full, 2, 0, false), 4);
        assert_eq!(r.attack(Piece::I, Spin::None, 4, 0, true), 14);
        // Fibonacci (default): +1 at chains 1, 4, 9, 25, 64, 169, 441, capped at +7.
        let fib = B2bMode::Fibonacci;
        assert_eq!([0, 1, 3, 4, 8, 9, 24, 25, 63, 64, 168, 169, 440, 441, 10_000].map(|c| fib.bonus(c)),
                   [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7]);
        let prime = B2bMode::Prime;
        assert_eq!([1, 3, 4, 9, 25, 48, 49, 121, 169, 289, 10_000].map(|c| prime.bonus(c)),
                   [0, 0, 1, 2, 3, 3, 4, 5, 6, 7, 7]);
        let sq = Rules { b2b: B2bMode::Squares, ..Rules::default() };
        assert_eq!(sq.attack(Piece::T, Spin::Full, 2, 4, false), 6);
        assert_eq!(sq.attack(Piece::T, Spin::Full, 2, 8, false), 6);
        assert_eq!(sq.attack(Piece::T, Spin::Full, 2, 9, false), 7);
    }
}
