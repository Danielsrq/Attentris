//! Bitboard playfield. One u16 per row, bit x = column x filled, row 0 at the top --
//! the same orientation as core/board.gd, so coordinates cross the boundary untouched.

use crate::rules::{Piece, Shapes, Spin, H, W};

pub const FULL: u16 = (1 << W) - 1;

/// One wound line can arrive per clear and each lasts 10 clears, so ten are ever live
/// at once; two spare for a longer `wound_clears`.
pub const MAX_WOUNDS: usize = 12;

/// The playfield plus the all-spin repeat-penalty state, which is part of the position:
/// two boards with the same cells but different wounds or last clear are different
/// futures.
///
/// Wound lines (all-spin mode) are the bottom `wounds` rows. They are SOLID in `rows`
/// (FULL, so nothing can enter them) but hold no pieces: the row's real contents are
/// parked in `wound_row` until it heals. The block is a FIFO: index 0 is the oldest and
/// sits at the TOP of the block, next to the play area; each new wound takes the
/// bottom in-play row and goes to the very bottom, pushing the older ones up, so the
/// oldest is always the next to heal and heals straight back into play.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Board {
    pub rows: [u16; H as usize],
    pub wounds: u8,
    /// Clears left before each wound heals, oldest first.
    pub wound_left: [u8; MAX_WOUNDS],
    /// Parked contents of each wound line, oldest first.
    pub wound_row: [u16; MAX_WOUNDS],
    /// Type of the previous clear (see `rules::clear_type`), 0 = none yet. Only
    /// tracked when the repeat penalty is on.
    pub last_clear: u8,
}

impl Default for Board {
    fn default() -> Self {
        Board { rows: [0; H as usize], wounds: 0, wound_left: [0; MAX_WOUNDS], wound_row: [0; MAX_WOUNDS], last_clear: 0 }
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
            if self.rows[r] == FULL {
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
        let mut write = H as usize;
        for y in (0..H as usize).rev() {
            if full >> y & 1 != 0 {
                continue;
            }
            write -= 1;
            out[write] = self.rows[y];
        }
        self.rows = out;
    }

    /// Board.push_garbage: shift the stack up, fill from the bottom; `holes` is
    /// bottom-most first. Anything pushed above row 0 is lost. Garbage enters at the
    /// bottom of the PLAY area, above any wound lines, so the wound block stays at the
    /// floor and every garbage row stays reachable.
    pub fn push_garbage(&mut self, holes: &[i32]) {
        let n = holes.len();
        if n == 0 {
            return;
        }
        let floor = H as usize - self.wounds as usize; // first wound row (or H)
        for y in 0..floor - n {
            self.rows[y] = self.rows[y + n];
        }
        for (i, &hole) in holes.iter().enumerate() {
            self.rows[floor - 1 - i] = FULL & !(1 << hole);
        }
    }

    /// No pieces in play. Wound lines hold no pieces, so they do not block a PC.
    pub fn is_empty(&self) -> bool {
        self.rows[..H as usize - self.wounds as usize].iter().all(|&r| r == 0)
    }

    /// The bottom in-play row becomes a wound line at the bottom of the block.
    pub fn add_wound(&mut self, clears: u8) {
        let n = self.wounds as usize;
        if n >= MAX_WOUNDS {
            return;
        }
        let r = H as usize - 1 - n;
        self.wound_row[n] = self.rows[r];
        self.wound_left[n] = clears;
        self.rows[r] = FULL;
        self.wounds += 1;
    }

    /// Called once per clear: every wound counts down; the oldest heals at 0, its
    /// parked contents returning as the bottom in-play row. Returns how many healed.
    pub fn tick_wounds(&mut self) -> u32 {
        let n = self.wounds as usize;
        for i in 0..n {
            self.wound_left[i] = self.wound_left[i].saturating_sub(1);
        }
        let mut healed = 0;
        while self.wounds > 0 && self.wound_left[0] == 0 {
            let top = H as usize - self.wounds as usize;
            self.rows[top] = self.wound_row[0];
            let n = self.wounds as usize;
            self.wound_left.copy_within(1..n, 0);
            self.wound_row.copy_within(1..n, 0);
            self.wounds -= 1;
            healed += 1;
        }
        healed
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
            for i in 0..self.wounds as usize {
                v = v.wrapping_mul(31) ^ self.wound_left[i] as u64;
            }
            h = (h ^ v ^ (self.wounds as u64) << 40).wrapping_mul(0xFF51_AFD7_ED55_8CCD);
            h ^= h >> 29;
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
    fn wound_takes_bottom_row_and_heals_after_ten_clears() {
        let mut b = Board::default();
        b.rows[B] = 0b0000011111; // row "A"
        b.add_wound(10);
        assert_eq!(b.wounds, 1);
        assert_eq!(b.rows[B], FULL, "wound line is solid");
        assert!(b.is_empty(), "wound lines hold no pieces, so a PC is possible");
        for _ in 0..9 {
            assert_eq!(b.tick_wounds(), 0);
        }
        assert_eq!(b.tick_wounds(), 1, "heals on the 10th clear");
        assert_eq!(b.wounds, 0);
        assert_eq!(b.rows[B], 0b0000011111, "contents come back");
    }

    #[test]
    fn fifo_model_a_order() {
        // Bottom-up rows A, B, C. Wound A, then B: B goes to the very bottom and A
        // rises; A heals first and comes back above B, so the rows return as B A C.
        let (a, bb, c) = (0b1, 0b10, 0b100);
        let mut b = Board::default();
        b.rows[B] = a;
        b.rows[B - 1] = bb;
        b.rows[B - 2] = c;
        b.add_wound(10);
        b.tick_wounds(); // one clear in between: A has 9 left
        b.add_wound(10);
        assert_eq!((b.wounds, b.rows[B], b.rows[B - 1]), (2, FULL, FULL));
        assert_eq!(b.rows[B - 2], c);
        for _ in 0..9 {
            b.tick_wounds();
        }
        assert_eq!(b.wounds, 1, "oldest (A) healed first");
        assert_eq!(b.rows[B - 1], a, "A returns next to the play area");
        b.tick_wounds();
        assert_eq!(b.wounds, 0);
        assert_eq!((b.rows[B], b.rows[B - 1], b.rows[B - 2]), (bb, a, c));
    }

    #[test]
    fn garbage_enters_above_wounds() {
        let mut b = Board::default();
        b.rows[B] = 0b11;
        b.add_wound(10);
        b.push_garbage(&[4]);
        assert_eq!(b.rows[B], FULL, "wound stays at the floor");
        assert_eq!(b.rows[B - 1], FULL & !(1 << 4), "garbage sits on top of it");
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
