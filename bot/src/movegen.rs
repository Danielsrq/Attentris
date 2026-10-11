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
    /// TetrisBattle rotation and spins (battle.rs) instead of SRS.
    pub battle: bool,
    /// Battle BFS: per (x, y, rot, tspin) the smallest lock-delay count seen, stamped.
    bvisit: Vec<u32>,
    bground: Vec<u8>,
    bqueue: Vec<(i8, i8, u8, u8, u8)>,
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
            battle: false,
            bvisit: vec![0; STATES * 3],
            bground: vec![0; STATES * 3],
            bqueue: Vec::with_capacity(4096),
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
        if self.battle {
            return self.generate_battle(board, piece, shapes, out);
        }
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

    /// TetrisBattle's search(): every state reachable from spawn with left / right / soft
    /// drop / cw / ccw (no 180), at most MAX_GROUND_OPS actions while resting; the T-spin
    /// flag is set by each rotation and kept through moves. Every resting state is a
    /// placement (the hard drop from above lands on one) with that state's flag.
    fn generate_battle(&mut self, board: &Board, piece: Piece, shapes: &Shapes, out: &mut Vec<Placement>) {
        use crate::battle::{collide, rotate, MAX_GROUND_OPS, SPAWN};
        out.clear();
        self.stamp = self.stamp.wrapping_add(1);
        if self.stamp == 0 {
            self.visited.iter_mut().for_each(|v| *v = 0);
            self.bvisit.iter_mut().for_each(|v| *v = 0);
            self.key_stamp.iter_mut().for_each(|v| *v = 0);
            self.stamp = 1;
        }
        let stamp = self.stamp;
        let (sx, sy) = SPAWN;
        if collide(board, piece, 0, sx, sy) {
            return;
        }
        let bidx = |x: i32, y: i32, r: usize, t: u8| Self::idx(x, y, r) * 3 + t as usize;
        self.bqueue.clear();
        self.bqueue.push((sx as i8, sy as i8, 0, 0, 0));
        let i0 = bidx(sx, sy, 0, 0);
        self.bvisit[i0] = stamp;
        self.bground[i0] = 0;
        let mut head = 0;
        while head < self.bqueue.len() {
            let (x8, y8, r8, t, g) = self.bqueue[head];
            head += 1;
            let (x, y, r) = (x8 as i32, y8 as i32, r8 as usize);
            if self.bvisit[bidx(x, y, r, t)] == stamp && self.bground[bidx(x, y, r, t)] < g {
                continue; // reached again with more lock delay left
            }
            let on_ground = collide(board, piece, r, x, y + 1);
            if on_ground {
                let spin = match t { 2 => Spin::Full, 1 => Spin::Mini, _ => Spin::None };
                self.record(board, piece, shapes, r, x, y, spin, if t > 0 { 0 } else { -1 }, out);
                if g >= MAX_GROUND_OPS {
                    continue;
                }
            }
            let g_base = if on_ground { g + 1 } else { 0 };
            let push = |s: &mut Self, nx: i32, ny: i32, nr: usize, nt: u8| {
                let ng = if collide(board, piece, nr, nx, ny + 1) { g_base } else { 0 };
                if !(-3..14).contains(&nx) || !(-3..44).contains(&ny) {
                    return;
                }
                let i = bidx(nx, ny, nr, nt);
                if s.bvisit[i] == stamp && s.bground[i] <= ng {
                    return;
                }
                s.bvisit[i] = stamp;
                s.bground[i] = ng;
                s.bqueue.push((nx as i8, ny as i8, nr as u8, nt, ng));
            };
            if !collide(board, piece, r, x - 1, y) {
                push(self, x - 1, y, r, t);
            }
            if !collide(board, piece, r, x + 1, y) {
                push(self, x + 1, y, r, t);
            }
            if !on_ground {
                push(self, x, y + 1, r, t);
            }
            for dir in [1, -1] {
                let (nx, ny, nr, nt) = rotate(board, piece, x, y, r, dir);
                push(self, nx, ny, nr, nt);
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

    fn mirrored(b: &Board) -> Board {
        let mut m = *b;
        for y in 0..40 {
            m.rows[y] = (0..10).filter(|x| b.rows[y] >> x & 1 != 0).map(|x| 1u16 << (9 - x)).sum();
        }
        m
    }

    /// The slot rows of the two vertical templates (rows 36..39; stem column 3, nub column
    /// 4 at row 37) with every fill of the cap/lane cells (columns 2..6 of rows 34..35;
    /// columns 0-1 and 7-9 there filled), in both orientations.
    fn template_boards(fixed: &[&str]) -> Vec<Board> {
        let vars: Vec<(usize, usize)> = (34..36).flat_map(|y| (2..7).map(move |x| (x, y))).collect();
        (0u32..1 << vars.len()).flat_map(|m| {
            let mut b = Board::default();
            for (i, r) in fixed.iter().enumerate() {
                for (x, ch) in r.chars().enumerate() {
                    if ch == 'X' { b.rows[36 + i] |= 1 << x; }
                }
            }
            for y in 34..36 { for x in [0, 1, 7, 8, 9] { b.rows[y] |= 1 << x; } }
            for (k, &(x, y)) in vars.iter().enumerate() {
                if m >> k & 1 != 0 { b.rows[y] |= 1 << x; }
            }
            [b, mirrored(&b)]
        }).collect()
    }

    const STSD_ROWS: [&str; 4] = ["XXX.XXXXXX", "XXX..XXXXX", "XXX..XXXXX", "XXXXXXXXX."];
    const CSPIN_ROWS: [&str; 4] = ["XXX.XXXXXX", "XXX..XXXXX", "XXX.XXXXXX", "XXXX.XXXXX"];

    /// V's vertical template accepts exactly the boards where SRS lands a FULL vertical T
    /// in the slot (through the fifth kick): 12 of the 1024 cap/lane fills, per template
    /// and orientation, the same 12 for the STSD (a double) and the C-spin (a TST).
    #[test]
    fn vertical_template_matches_movegen() {
        let shapes = Shapes::new();
        for (fixed, lines) in [(&STSD_ROWS, 2u32), (&CSPIN_ROWS, 3u32)] {
            let mut landed = 0;
            for (i, b) in template_boards(fixed).iter().enumerate() {
                let flip = i % 2 == 1;
                let mut want: Vec<(i32, i32)> = [(3, 36), (3, 37), (4, 37), (3, 38)]
                    .iter().map(|&(x, y)| (if flip { 9 - x } else { x }, y)).collect();
                want.sort();
                let lands = placements(b, Piece::T, true).iter().any(|p| {
                    let mut cells = p.abs_cells(&shapes).to_vec();
                    cells.sort();
                    let mut nb = *b;
                    p.spin == Spin::Full && cells == want && nb.place(p.cells(&shapes), p.x as i32, p.y as i32).0 == lines
                });
                let found = crate::eval::best_tslot(b, 34).map_or(false, |s| {
                    let mut cells = s.cells.to_vec();
                    cells.sort();
                    cells == want && s.lines == lines
                });
                assert_eq!(found, lands, "template {} lines, board {} (mirrored {})", lines, i / 2, flip);
                landed += lands as u32;
            }
            assert_eq!(landed, 24, "12 fills per orientation");
        }
    }

    /// The follow-up fill reaches 1 on the full chains: STSD -> TSD and TST (C-spin) -> TSD.
    #[test]
    fn followup_fill() {
        let follow = |b: &Board| {
            let s = crate::eval::scan_tslots(b, 34);
            crate::eval::cash_chain_from(b, s.best, s.vertical).1
        };
        for fixed in [STSD_ROWS, CSPIN_ROWS] {
            let mut rows = vec!["XXXX..XXXX", "XXX...XXXX"];
            rows.extend_from_slice(&fixed);
            let b = board_from(&rows);
            assert_eq!(follow(&b), 1.0);
            assert_eq!(follow(&mirrored(&b)), 1.0);
        }
        // Lower fill when the rows around the follow-up TSD are still open.
        let b = board_from(&["XXXX......", "XXX.......", "XXX.XXXXXX", "XXX..XXXXX", "XXX..XXXXX", "XXXXXXXXX."]);
        let f = follow(&b);
        assert!(f > 0.0 && f < 1.0, "{}", f);
    }

    /// A vertical slot with no cap cannot be entered, so V does not count it.
    #[test]
    fn uncapped_vertical_slot_is_not_counted() {
        let b = board_from(&["XXX.......", "XXXX......", "XXX..XXXXX", "XXXX.XXXXX", "XXXXXXXXX."]);
        let vertical = [(4, 36), (4, 37), (3, 37), (4, 38)];
        assert!(crate::eval::best_tslot(&b, 35).map_or(true, |s| s.cells != vertical));
        assert!(placements(&b, Piece::T, true).iter().all(|p| p.spin != Spin::Full));
    }

    /// The DT cannon core (Hard Drop wiki, "Double Triple Cannon Setups"): a flat TSD
    /// under the overhang, which leaves a vertical TST slot with cap and lane -- the
    /// C-spin template, mirrored. Both must be FULL spins, and V must see each in turn.
    #[test]
    fn dt_cannon() {
        let shapes = Shapes::new();
        let b = board_from(&[".....XX...", "......X...", "XXXXX.XXXX", "XXXX..XXXX", "XXXX...XXX", "XXXXX.XXXX", "XXXXX.XXXX"]);
        let top = 33;
        let tsd = placements(&b, Piece::T, true).into_iter().find(|p| {
            let mut nb = b;
            p.spin == Spin::Full && p.rot == 2 && nb.place(p.cells(&shapes), p.x as i32, p.y as i32).0 == 2
        });
        let tsd = tsd.expect("the DT's TSD is reachable as a full spin");
        let slots = crate::eval::scan_tslots(&b, top);
        println!("before: best {:?} vertical {:?}", slots.best.map(|s| (s.lines, s.cells)), slots.vertical.map(|s| (s.lines, s.cells)));
        assert_eq!(slots.best.map(|s| s.lines), Some(2), "V sees the TSD");
        let mut nb = b;
        nb.place(tsd.cells(&shapes), tsd.x as i32, tsd.y as i32);
        let after = crate::eval::scan_tslots(&nb, top + 2);
        println!("after:  best {:?} vertical {:?}", after.best.map(|s| (s.lines, s.cells)), after.vertical.map(|s| (s.lines, s.cells)));
        assert_eq!(after.vertical.map(|s| s.lines), Some(3), "then V sees the TST template slot");
        let tst = placements(&nb, Piece::T, true).into_iter().find(|p| {
            let mut n2 = nb;
            p.spin == Spin::Full && n2.place(p.cells(&shapes), p.x as i32, p.y as i32).0 == 3
        });
        let tst = tst.expect("the DT's TST is reachable as a full spin");
        assert!(tst.rot == 1 || tst.rot == 3);
    }

    /// Imperial cross (Hard Drop wiki): the basic setup, and the board the bot played
    /// one on (b2n_s6 placement 88). Both T positions -- twisted in, and turned 180 --
    /// are full hurdle doubles; V's template sees the twisted-in one.
    #[test]
    fn imperial_cross() {
        let shapes = Shapes::new();
        let wiki = board_from(&["..XX......", "...XXXXXXX", "XX.XXXXXXX", "X...XXXXXX", "XX.XXXXXXX"]);
        let played = board_from(&[".........X", "...XXXXXXX", "....XXXXXX", "XXX.XXXXXX", "XX...XXXXX", "XXX.XXXXXX", "XXX.XXXXXX"]);
        for (name, b, top) in [("wiki", wiki, 35), ("played", played, 33)] {
            let slots = crate::eval::scan_tslots(&b, top);
            let mut found = Vec::new();
            for p in placements(&b, Piece::T, true) {
                let mut nb = b;
                let lines = nb.place(p.cells(&shapes), p.x as i32, p.y as i32).0;
                if p.spin == Spin::Full && lines == 2 && (p.rot == 1 || p.rot == 3) {
                    let mut c = p.abs_cells(&shapes).to_vec();
                    c.sort();
                    found.push((p.rot, c));
                }
            }
            println!("{}: movegen full vertical doubles {:?}", name, found);
            println!("{}: V vertical slot {:?}", name, slots.vertical.map(|s| (s.lines, s.cells)));
            let v = slots.vertical.expect("V sees a vertical slot");
            let mut vc = v.cells.to_vec();
            vc.sort();
            assert!(found.iter().any(|f| f.1 == vc), "{}: V's slot is one the T can reach", name);
            assert_eq!(found.len(), 2, "{}: both orientations are full doubles", name);
        }
    }

    /// The known two-T structures cash as a chain: the slot left behind is ready
    /// (fill 1), and the structure's own cavities stop counting as holes.
    #[test]
    fn chains_cash_cleanly() {
        let mut stsd = vec!["XXXX..XXXX", "XXX...XXXX"];
        stsd.extend_from_slice(&STSD_ROWS);
        let mut cspin = vec!["XXXX..XXXX", "XXX...XXXX"];
        cspin.extend_from_slice(&CSPIN_ROWS);
        let cases: Vec<(&str, Vec<&str>)> = vec![
            ("fractal", vec!["...XX.....", "XXXX...XXX", "XXXXX.XXXX", "XXXX...XXX", "XXXXX.XXXX"]),
            ("DT", vec![".....XX...", "......X...", "XXXXX.XXXX", "XXXX..XXXX", "XXXX...XXX", "XXXXX.XXXX", "XXXXX.XXXX"]),
            ("imperial cross", vec!["..XX......", "...XXXXXXX", "XX.XXXXXXX", "X...XXXXXX", "XX.XXXXXXX"]),
            ("STSD -> TSD", stsd),
            ("C-spin", cspin),
        ];
        for (name, rows) in cases {
            let b = board_from(&rows);
            let top = 40 - rows.len() as i32;
            let slots = crate::eval::scan_tslots(&b, top);
            let slot = slots.best;
            let (after, follow) = crate::eval::cash_chain_from(&b, slot, slots.vertical);
            let (before_holes, _, _) = crate::eval::hole_stats_pub(&b);
            let (after_holes, after_rows, after_cover) = crate::eval::hole_stats_pub(&after);
            // The reference: cash, clear and rescan the whole board, from whichever first
            // slot the fast path used (best or the vertical one).
            let first = [slots.best, slots.vertical].into_iter().flatten()
                .map(|s| (s, crate::eval::cash_chain_dynamic(&b, s)))
                .max_by(|x, y| x.1 .1.total_cmp(&y.1 .1)).unwrap();
            let (dyn_board, dyn_follow) = first.1;
            let dyn_stats = crate::eval::hole_stats_pub(&dyn_board);
            let path = if first.0.second.is_some() || first.0.flip_second.is_some() { "static" } else { "dynamic" };
            println!("{:<15} first {:?}  {:<7} follow {:.2} (reference {:.2})  holes {} -> {} (reference {})",
                name, slot.map(|s| s.lines), path, follow, dyn_follow, before_holes, after_holes, dyn_stats.0);
            assert_eq!(follow, 1.0, "{}: the second slot is ready", name);
            assert_eq!(follow, dyn_follow, "{}: same follow-up as the reference", name);
            assert_eq!((after_holes, after_rows, after_cover), dyn_stats, "{}: same holes as the reference", name);
            assert!(after_holes < before_holes || before_holes == 0.0, "{}", name);
        }
    }
}
