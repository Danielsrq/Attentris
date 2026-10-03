//! Bitboard playfield. One u16 per row, bit x = column x filled, row 0 at the top --
//! the same orientation as core/board.gd, so coordinates cross the boundary untouched.

use crate::rules::{Piece, Shapes, Spin, H, W};

pub const FULL: u16 = (1 << W) - 1;

/// The playfield plus the all-spin repeat-penalty state, which is part of the position:
/// two boards with the same cells but different wounds or last clear are different
/// futures.
///
/// Wound lines (all-spin mode): a repeat inserts a solid, unclearable line at the very
/// bottom, pushing the whole stack up one row, exactly as a garbage line would. Each
/// wound counts down on every clear; at 0 it stays where it is as an ordinary garbage
/// line, its hole (chosen when it arrived) opening up. Nothing is taken from the
/// player's rows. Newer wounds enter beneath older ones, so the oldest -- highest --
/// turns into garbage first.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Board {
    pub rows: [u16; H as usize],
    /// Live wound lines on the board.
    pub wounds: u8,
    /// Per row: clears left before that row's wound turns into garbage; 0 = not a wound.
    pub wound_left: [u8; H as usize],
    /// Per row: the hole the wound line will have once it turns into garbage.
    pub wound_hole: [u8; H as usize],
    /// Hole of the most recent wound, for messiness (the chance the next one moves).
    pub last_wound_hole: u8,
    /// Type of the previous clear (see `rules::clear_type`), 0 = none yet. Only
    /// tracked when the repeat penalty is on.
    pub last_clear: u8,
    /// Burst rules: placements left in which a qualifying clear has its base attack
    /// doubled (set to the rules' window by a clear that arms, counts down on every other
    /// placement). Always 0 in other modes.
    pub burst_armed: u8,
    /// Gate rules: clears in the open chain, and the opener's value while it is alone.
    pub gate_len: u8,
    pub gate_pending: u8,
}

impl Default for Board {
    fn default() -> Self {
        Board {
            rows: [0; H as usize],
            wounds: 0,
            wound_left: [0; H as usize],
            wound_hole: [0; H as usize],
            last_wound_hole: 0,
            last_clear: 0,
            burst_armed: 0,
            gate_len: 0,
            gate_pending: 0,
        }
    }
}

impl Board {
    /// Board.is_blocked: walls, floor and ceiling count as solid.
    #[inline]
    pub fn blocked(&self, x: i32, y: i32) -> bool {
        if x < 0 || x >= W || y < 0 || y >= H {
            return true;
        }
        self.rows[y as usize] >> x & 1 != 0
    }

    #[inline]
    pub fn fits(&self, cells: &[(i8, i8); 4], x: i32, y: i32) -> bool {
        for &(dx, dy) in cells {
            let cx = x + dx as i32;
            let cy = y + dy as i32;
            if cx < 0 || cx >= W || cy < 0 || cy >= H {
                return false;
            }
            if self.rows[cy as usize] >> cx & 1 != 0 {
                return false;
            }
        }
        true
    }

    /// Lock the cells, clear full rows. Returns (lines, bitmask of cleared row indices
    /// on the pre-clear board, as a u64).
    pub fn place(&mut self, cells: &[(i8, i8); 4], x: i32, y: i32) -> (u32, u64) {
        let mut touched: u64 = 0;
        for &(dx, dy) in cells {
            let cx = x + dx as i32;
            let cy = (y + dy as i32) as usize;
            self.rows[cy] |= 1 << cx;
            touched |= 1 << cy;
        }
        let mut full: u64 = 0;
        let mut t = touched;
        while t != 0 {
            let r = t.trailing_zeros() as usize;
            t &= t - 1;
            if self.rows[r] == FULL && self.wound_left[r] == 0 {
                full |= 1 << r;
            }
        }
        if full != 0 {
            self.remove_rows(full);
        }
        (full.count_ones(), full)
    }

    fn remove_rows(&mut self, full: u64) {
        let mut out = [0u16; H as usize];
        let mut out_left = [0u8; H as usize];
        let mut out_hole = [0u8; H as usize];
        let mut write = H as usize;
        for y in (0..H as usize).rev() {
            if full >> y & 1 != 0 {
                continue;
            }
            write -= 1;
            out[write] = self.rows[y];
            out_left[write] = self.wound_left[y];
            out_hole[write] = self.wound_hole[y];
        }
        self.rows = out;
        self.wound_left = out_left;
        self.wound_hole = out_hole;
    }

    /// Shift everything up by `n` rows (the top `n` are lost), leaving the bottom `n`
    /// rows empty and not wounds.
    fn shift_up(&mut self, n: usize) {
        let h = H as usize;
        self.rows.copy_within(n..h, 0);
        self.wound_left.copy_within(n..h, 0);
        self.wound_hole.copy_within(n..h, 0);
        for y in h - n..h {
            self.rows[y] = 0;
            self.wound_left[y] = 0;
            self.wound_hole[y] = 0;
        }
        // A wound pushed off the top is gone.
        self.wounds = self.wound_left.iter().filter(|&&l| l > 0).count() as u8;
    }

    /// Board.push_garbage: shift the stack up, fill from the bottom; `holes` is
    /// bottom-most first. Anything pushed above row 0 is lost. Garbage enters at the very
    /// bottom, beneath any wound lines, pushing them up like everything else.
    pub fn push_garbage(&mut self, holes: &[i32]) {
        let n = holes.len();
        if n == 0 {
            return;
        }
        self.shift_up(n);
        for (i, &hole) in holes.iter().enumerate() {
            self.rows[H as usize - 1 - i] = FULL & !(1 << hole);
        }
    }

    /// Nothing on the board at all. A wound line is a garbage-like line, so while one is
    /// present a perfect clear is impossible.
    pub fn is_empty(&self) -> bool {
        self.rows.iter().all(|&r| r == 0)
    }

    /// A repeat: insert a solid wound line at the very bottom, pushing the stack up. It
    /// turns into garbage with a hole at `hole` after `clears` clears.
    pub fn add_wound(&mut self, clears: u8, hole: u8) {
        self.shift_up(1);
        let b = H as usize - 1;
        self.rows[b] = FULL;
        self.wound_left[b] = clears;
        self.wound_hole[b] = hole;
        self.last_wound_hole = hole;
        self.wounds += 1;
    }

    /// Called once per clear: every wound counts down; at 0 it becomes an ordinary garbage
    /// line in place, its hole opening. Returns how many turned into garbage.
    pub fn tick_wounds(&mut self) -> u32 {
        if self.wounds == 0 {
            return 0;
        }
        let mut healed = 0;
        for y in 0..H as usize {
            if self.wound_left[y] > 0 {
                self.wound_left[y] -= 1;
                if self.wound_left[y] == 0 {
                    self.rows[y] = FULL & !(1 << self.wound_hole[y]);
                    healed += 1;
                }
            }
        }
        self.wounds -= healed as u8;
        healed
    }

    /// Wound rows, top to bottom: (row, clears left, hole).
    pub fn wound_rows(&self) -> impl Iterator<Item = (usize, u8, u8)> + '_ {
        (0..H as usize).filter(|&y| self.wound_left[y] > 0).map(move |y| (y, self.wound_left[y], self.wound_hole[y]))
    }

    /// Column heights measured from the floor (0 = empty column).
    pub fn heights(&self) -> [i32; W as usize] {
        let mut h = [0i32; W as usize];
        let mut found: u16 = 0;
        for y in 0..H as usize {
            let new = self.rows[y] & !found;
            if new != 0 {
                let mut m = new;
                while m != 0 {
                    let x = m.trailing_zeros() as usize;
                    m &= m - 1;
                    h[x] = H - y as i32;
                }
                found |= new;
                if found == FULL {
                    break;
                }
            }
        }
        h
    }

    pub fn hash64(&self) -> u64 {
        // Rows are 10 bits; fold four at a time and mix. Only has to be good enough
        // to dedupe transpositions inside one search.
        let mut h: u64 = 0x9E37_79B9_7F4A_7C15;
        for chunk in self.rows.chunks(4) {
            let mut v: u64 = 0;
            for &r in chunk {
                v = v << 16 | r as u64;
            }
            h = (h ^ v).wrapping_mul(0xFF51_AFD7_ED55_8CCD);
            h ^= h >> 29;
        }
        if self.wounds > 0 || self.last_clear != 0 {
            let mut v = self.last_clear as u64;
            for (y, left, _) in self.wound_rows() {
                v = v.wrapping_mul(31) ^ ((y as u64) << 8 | left as u64);
            }
            h = (h ^ v ^ (self.wounds as u64) << 40).wrapping_mul(0xFF51_AFD7_ED55_8CCD);
            h ^= h >> 29;
        }
        if self.burst_armed > 0 {
            h = (h ^ 0xB5 ^ (self.burst_armed as u64) << 8 ^ (self.gate_len as u64) << 16 ^ (self.gate_pending as u64) << 24)
                .wrapping_mul(0xC4CE_B9FE_1A85_EC53);
        }
        h
    }
}

/// A final resting position, as the move generator found it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Placement {
    pub piece: Piece,
    pub rot: u8,
    pub x: i8,
    pub y: i8,
    pub spin: Spin,
    /// Kick index of the final rotation, -1 if the last action was not a rotation.
    pub kick: i8,
}

impl Placement {
    pub fn cells<'a>(&self, shapes: &'a Shapes) -> &'a [(i8, i8); 4] {
        &shapes.cells[self.piece as usize][self.rot as usize]
    }
    pub fn abs_cells(&self, shapes: &Shapes) -> [(i32, i32); 4] {
        let mut out = [(0, 0); 4];
        for (i, &(dx, dy)) in self.cells(shapes).iter().enumerate() {
            out[i] = (self.x as i32 + dx as i32, self.y as i32 + dy as i32);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rules::clear_type;

    const B: usize = H as usize - 1; // bottom row index

    #[test]
    fn wound_is_inserted_at_the_bottom_and_turns_into_garbage() {
        let mut b = Board::default();
        b.rows[B] = 0b0000011111; // the player's bottom row
        b.add_wound(10, 6);
        assert_eq!(b.wounds, 1);
        assert_eq!(b.rows[B], FULL, "wound line is solid");
        assert_eq!(b.rows[B - 1], 0b0000011111, "the stack was pushed up, nothing taken");
        assert!(!b.is_empty(), "a wound line is on the board: no perfect clear");
        for _ in 0..9 {
            assert_eq!(b.tick_wounds(), 0);
        }
        assert_eq!(b.tick_wounds(), 1, "turns into garbage on the 10th clear");
        assert_eq!(b.wounds, 0);
        assert_eq!(b.rows[B], FULL & !(1 << 6), "   an ordinary garbage line, hole at 6");
    }

    #[test]
    fn newer_wounds_enter_beneath_older_ones() {
        let mut b = Board::default();
        b.add_wound(10, 1);
        b.tick_wounds();
        b.add_wound(10, 2);
        assert_eq!((b.wound_left[B - 1], b.wound_left[B]), (9, 10), "older wound pushed up");
        for _ in 0..9 {
            b.tick_wounds();
        }
        assert_eq!(b.rows[B - 1], FULL & !(1 << 1), "the older (higher) one is garbage first");
        assert_eq!(b.rows[B], FULL, "   the newer one is still a wound");
        b.tick_wounds();
        assert_eq!(b.rows[B], FULL & !(1 << 2));
    }

    #[test]
    fn wound_lines_are_never_cleared() {
        let mut b = Board::default();
        b.add_wound(10, 0);
        // Fill the row above except one cell, then complete it: only that row clears.
        b.rows[B - 1] = FULL & !1;
        let (lines, _) = b.place(&[(0, 0), (0, -1), (0, -2), (0, -3)], 0, B as i32 - 1);
        assert_eq!(lines, 1);
        assert_eq!(b.wounds, 1, "the wound line survived the clear");
    }

    #[test]
    fn garbage_enters_beneath_wounds() {
        let mut b = Board::default();
        b.add_wound(10, 3);
        b.push_garbage(&[4]);
        assert_eq!(b.rows[B], FULL & !(1 << 4), "garbage at the very bottom");
        assert_eq!((b.rows[B - 1], b.wound_left[B - 1]), (FULL, 10), "wound pushed up with the stack");
    }

    #[test]
    fn clear_types() {
        use crate::rules::{Piece::*, Spin::*};
        // Non-spin clears are keyed by size only.
        assert_eq!(clear_type(I, None, 2), clear_type(O, None, 2));
        assert_ne!(clear_type(I, None, 1), clear_type(I, None, 2));
        assert_eq!(clear_type(I, None, 4), clear_type(I, None, 4));
        // Spins are keyed by piece and size.
        assert_ne!(clear_type(T, Full, 2), clear_type(Z, Full, 2));
        assert_ne!(clear_type(T, Full, 2), clear_type(T, Full, 3));
        assert_ne!(clear_type(T, Full, 2), clear_type(T, None, 2));
        assert_eq!(clear_type(T, Full, 0), 0);
    }
}
