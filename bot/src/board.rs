//! Bitboard playfield. One u16 per row, bit x = column x filled, row 0 at the top --
//! the same orientation as core/board.gd, so coordinates cross the boundary untouched.

use crate::rules::{Piece, Shapes, Spin, H, W};

pub const FULL: u16 = (1 << W) - 1;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Board {
    pub rows: [u16; H as usize],
}

impl Default for Board {
    fn default() -> Self {
        Board { rows: [0; H as usize] }
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
    /// bottom-most first. Anything pushed above row 0 is lost.
    pub fn push_garbage(&mut self, holes: &[i32]) {
        let n = holes.len();
        if n == 0 {
            return;
        }
        let h = H as usize;
        for y in 0..h - n {
            self.rows[y] = self.rows[y + n];
        }
        for (i, &hole) in holes.iter().enumerate() {
            self.rows[h - 1 - i] = FULL & !(1 << hole);
        }
    }

    pub fn is_empty(&self) -> bool {
        self.rows.iter().all(|&r| r == 0)
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
