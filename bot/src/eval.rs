//! V(board): a linear value over NEUTRAL board measurements. Nothing here says "build
//! a well" or "go for T-spins" -- it measures health (holes, surface, height) and
//! potential (how close rows are to an I or a T paying out). Which of those the bot
//! pursues is decided by the rewards the search actually collects and by the weights,
//! which `tune` fits by self-play.
//!
//! Two deliberate non-linearities, per the design notes:
//!   * `danger` is quadratic in stack height, so priorities can flip (attack when low,
//!     dig when high) with a single linear model.
//!   * a cavity that a T-spin would fill is NOT counted as a hole: holes are measured
//!     on the board as it would be after cashing the best T slot. Otherwise every
//!     T-slot overhang is punished as a hole and spins can never emerge.

use crate::board::{Board, FULL};
use crate::rules::{clear_type, Piece, Shapes, Spin, H, W};

pub const NAMES: [&str; 29] = [
    "holes",       // empty cells with something above them
    "hole_rows",   // rows containing at least one hole
    "cover",       // filled cells above each column's topmost hole
    "bump",        // |h[x] - h[x+1]|, the deepest column's two edges excluded
    "bump_sq",     // same, squared
    "row_trans",   // Dellacherie row transitions (walls count as filled)
    "col_trans",   // Dellacherie column transitions
    "max_height",  // tallest column
    "danger",      // max(0, max_height - 10)^2
    "well_rows",   // rows (from the deepest column's floor up, max 4) full except that column
    "extra_wells", // depth beyond 2 of every other 1-wide notch (I-dependencies)
    "t_single",    // best T slot would clear 1 row
    "t_double",    // ... 2 rows
    "t_triple",    // ... 3 rows
    "t_fill",      // fill fraction of the best T slot's rows (the gradient towards it)
    "b2b",         // back-to-back chain is live
    "b2b_level",   // min(chain, 8): for bonuses that grow with the chain
    // sqrt(chain + 1) while a chain is live: the shape of the floor(sqrt(chain)) bonus
    // every future difficult clear will earn, so a long chain is worth more to keep.
    "b2b_sqrt",
    "wounds",      // live wound lines (all-spin repeat penalty): a shorter field
    "ready_repeat", // a ready quad or spin slot would repeat the last clear type
    // All-spin slots for the other pieces, by family: the best resting, immobile spot
    // (the all-spin rule) that would clear 1 / 2 / 3 rows. Found by bitmask scan;
    // reachability is left to the search.
    "sz_single",
    "sz_double",
    "sz_triple",
    "lj_single",
    "lj_double",
    "lj_triple",
    "i_single",
    "i_double",
    "i_triple",
];
pub const NF: usize = NAMES.len();

#[derive(Clone, Debug)]
pub struct Weights {
    pub w: [f32; NF],
}

impl Default for Weights {
    /// Hand-set starting point, scaled so 1.0 ~ one line of attack. `tune` improves it.
    fn default() -> Self {
        Weights {
            w: [
                -1.6,  // holes
                -1.0,  // hole_rows
                -0.12, // cover
                -0.10, // bump
                -0.04, // bump_sq
                -0.12, // row_trans
                -0.15, // col_trans
                -0.03, // max_height
                -0.08, // danger
                0.45,  // well_rows
                -0.35, // extra_wells
                0.3,   // t_single
                1.6,   // t_double
                2.2,   // t_triple
                1.2,   // t_fill
                1.2,   // b2b
                0.0,   // b2b_level
                0.0,   // b2b_sqrt
                0.0,   // wounds
                0.0,   // ready_repeat
                0.0, 0.0, 0.0, // sz_single / double / triple
                0.0, 0.0, 0.0, // lj_*
                0.0, 0.0, 0.0, // i_*
            ],
        }
    }
}

impl Weights {
    pub fn parse(text: &str) -> Result<Weights, String> {
        let mut w = Weights::default();
        for (n, line) in text.lines().enumerate() {
            let line = line.split('#').next().unwrap().trim();
            if line.is_empty() {
                continue;
            }
            let mut it = line.split_whitespace();
            let name = it.next().unwrap();
            let val: f32 = it
                .next()
                .ok_or(format!("line {}: missing value", n + 1))?
                .parse()
                .map_err(|e| format!("line {}: {}", n + 1, e))?;
            let i = NAMES.iter().position(|&k| k == name).ok_or(format!("line {}: unknown feature {}", n + 1, name))?;
            w.w[i] = val;
        }
        Ok(w)
    }

    pub fn to_text(&self) -> String {
        let mut s = String::new();
        for i in 0..NF {
            s += &format!("{:<12} {:.4}\n", NAMES[i], self.w[i]);
        }
        s
    }

    #[inline]
    pub fn value(&self, f: &[f32; NF]) -> f32 {
        let mut v = 0.0;
        for i in 0..NF {
            v += self.w[i] * f[i];
        }
        v
    }
}

/// Row with walls: bit 0 and bit 11 are the walls, columns sit at bits 1..=10.
#[inline]
fn padded(b: &Board, y: i32) -> u16 {
    if y < 0 || y >= H {
        0xFFF
    } else {
        (b.rows[y as usize] << 1) | 0x801
    }
}

#[derive(Clone, Copy, Default, Debug)]
pub struct TSlot {
    pub lines: u32,
    pub fill: f32,
    /// Absolute cells the T would occupy, for the hole-exemption.
    pub cells: [(i32, i32); 4],
}

/// Best T-spin slot on the board: a resting T position that would register as a FULL
/// spin by the 3-corner rule (both front corners + one back corner). Reachability is
/// not checked -- this is a cheap estimate for V, the search checks the real thing.
pub fn best_tslot(b: &Board, top: i32) -> Option<TSlot> {
    let mut best: Option<TSlot> = None;
    let y0 = (top - 3).max(1);
    let pop = |y: i32| -> u32 { if (0..H).contains(&y) { b.rows[y as usize].count_ones() } else { 10 } };
    let consider = |lines: u32, fill: f32, cells: [(i32, i32); 4], best: &mut Option<TSlot>| {
        let better = match best {
            None => true,
            Some(s) => (lines, fill) > (s.lines, s.fill),
        };
        if better {
            *best = Some(TSlot { lines, fill, cells });
        }
    };
    for y in y0..H - 1 {
        let pm = padded(b, y - 1);
        let p0 = padded(b, y);
        let p1 = padded(b, y + 1);
        let e0 = !p0 & 0xFFF;
        let e1 = !p1 & 0xFFF;
        let em = !pm & 0xFFF;

        // Nub down (rot 2): the classic TSD. Cells (c-1,y) (c,y) (c+1,y) (c,y+1).
        let m2 = e0 & (e0 << 1) & (e0 >> 1) & e1 & ((p1 << 1) & (p1 >> 1)) & ((pm << 1) | (pm >> 1)) & 0x7FE;
        let mut m = m2;
        while m != 0 {
            let c = m.trailing_zeros() as i32;
            m &= m - 1;
            let x = c - 1;
            let a = pop(y) + 3;
            let bb = pop(y + 1) + 1;
            let lines = (a == 10) as u32 + (bb == 10) as u32;
            consider(lines, (a + bb) as f32 / 20.0, [(x - 1, y), (x, y), (x + 1, y), (x, y + 1)], &mut best);
        }
        if y < 1 {
            continue;
        }
        // Vertical T (rot 1 nub right / rot 3 nub left): the TST shape.
        // rot 1 cells (c,y-1) (c,y) (c+1,y) (c,y+1); front corners (c+1,y-1) (c+1,y+1).
        let base = em & e0 & e1;
        let m1 = base & (e0 >> 1) & (pm >> 1) & (p1 >> 1) & ((pm << 1) | (p1 << 1)) & 0x7FE;
        let m3 = base & (e0 << 1) & (pm << 1) & (p1 << 1) & ((pm >> 1) | (p1 >> 1)) & 0x7FE;
        for (mask, side) in [(m1, 1i32), (m3, -1i32)] {
            let mut m = mask;
            while m != 0 {
                let c = m.trailing_zeros() as i32;
                m &= m - 1;
                let x = c - 1;
                if x + side < 0 || x + side >= W {
                    continue;
                }
                let a = pop(y - 1) + 1;
                let bb = pop(y) + 2;
                let cc = pop(y + 1) + 1;
                let lines = (a == 10) as u32 + (bb == 10) as u32 + (cc == 10) as u32;
                consider(lines, (a + bb + cc) as f32 / 30.0, [(x, y - 1), (x, y), (x + side, y), (x, y + 1)], &mut best);
            }
        }
    }
    best
}

/// Holes, hole rows and cover on `b`.
fn hole_stats(b: &Board) -> (f32, f32, f32) {
    let mut acc: u16 = 0;
    let mut holes = 0u32;
    let mut hole_rows = 0u32;
    let mut first_hole_row = [-1i32; W as usize];
    for y in 0..H as usize {
        let r = b.rows[y];
        let h = acc & !r;
        if h != 0 {
            holes += h.count_ones();
            hole_rows += 1;
            let mut m = h;
            while m != 0 {
                let x = m.trailing_zeros() as usize;
                m &= m - 1;
                if first_hole_row[x] < 0 {
                    first_hole_row[x] = y as i32;
                }
            }
        }
        acc |= r;
    }
    let mut cover = 0u32;
    for x in 0..W as usize {
        let fy = first_hole_row[x];
        if fy > 0 {
            for y in 0..fy as usize {
                cover += (b.rows[y] >> x & 1) as u32;
            }
        }
    }
    (holes as f32, hole_rows as f32, cover.min(60) as f32)
}

static SHAPES: std::sync::OnceLock<Shapes> = std::sync::OnceLock::new();

fn shapes() -> &'static Shapes {
    SHAPES.get_or_init(Shapes::new)
}

/// Empty cells of row y as a u32 with the ten columns at bits 4..=13; walls, floor and
/// ceiling (everything outside) count as solid, as Board::fits sees them.
#[inline]
fn empty32(b: &Board, y: i32) -> u32 {
    if !(0..H).contains(&y) {
        0
    } else {
        (!(b.rows[y as usize] as u32) & 0x3FF) << 4
    }
}

/// Where a piece rotation fits with its box's top row at y: bit (X + 4) is set when the
/// box at column X fits. One AND per cell: the cell at (X+dx, y+dy) is empty exactly
/// when bit X+4 of (empty row y+dy >> dx) is set.
#[inline]
pub fn fits_mask(b: &Board, cells: &[(i8, i8); 4], y: i32) -> u32 {
    let mut m = 0x3FFFF; // X from -4 to 13
    for &(dx, dy) in cells {
        m &= empty32(b, y + dy as i32) >> dx;
    }
    m
}

/// A resting spot a piece cannot leave in any direction: the all-spin condition. If a
/// rotation ends here, it counts as a spin.
#[derive(Clone, Copy, Debug)]
pub struct SpinSlot {
    pub piece: Piece,
    pub rot: u8,
    pub x: i32,
    pub y: i32,
    pub lines: u32,
}

/// Immobility mask for one rotation at box row y, from the fits masks of rows y-1, y,
/// y+1: fits here, not one column left or right, not one row up or down.
#[inline]
pub fn immobile_mask(up: u32, here: u32, down: u32) -> u32 {
    here & !(here << 1) & !(here >> 1) & !up & !down
}

/// Every immobile spot of S, Z, L, J and I at or near the surface, with the rows it
/// would clear. Pure bit operations per row; the per-spot line count only runs on hits.
/// Reachability is NOT checked -- whether a piece can actually rotate in is the
/// search's job; this estimates potential for V.
pub fn scan_spin_slots(b: &Board, top: i32, mut hit: impl FnMut(SpinSlot)) {
    let sh = shapes();
    let y0 = (top - 4).max(-3);
    for piece in [Piece::S, Piece::Z, Piece::L, Piece::J, Piece::I] {
        for rot in 0..4u8 {
            let cells = &sh.cells[piece as usize][rot as usize];
            let mut up = fits_mask(b, cells, y0 - 1);
            let mut here = fits_mask(b, cells, y0);
            for y in y0..H {
                let down = fits_mask(b, cells, y + 1);
                let mut m = immobile_mask(up, here, down);
                while m != 0 {
                    let bit = m.trailing_zeros() as i32;
                    m &= m - 1;
                    let x = bit - 4;
                    // Rows the piece would complete: its cells in a row plus what is there.
                    let mut per_row = [0u32; 4];
                    for &(_, dy) in cells {
                        per_row[dy as usize] += 1;
                    }
                    let mut lines = 0;
                    for (dy, &k) in per_row.iter().enumerate() {
                        let r = y + dy as i32;
                        if k > 0 && (0..H).contains(&r) && b.rows[r as usize].count_ones() + k == W as u32 {
                            lines += 1;
                        }
                    }
                    hit(SpinSlot { piece, rot, x, y, lines });
                }
                up = here;
                here = down;
            }
        }
    }
}

pub fn features(b: &Board, b2b: i32) -> [f32; NF] {
    let mut f = [0f32; NF];
    let h = b.heights();
    let maxh = *h.iter().max().unwrap();
    let top = H - maxh;

    let slot = best_tslot(b, top);
    // Holes are measured as if the best slot were cashed, so the slot's own cavity is
    // not double-charged: it is an asset (t_*), not a hole.
    let (holes, hole_rows, cover) = match slot {
        Some(s) => {
            let mut filled = *b;
            for &(x, y) in &s.cells {
                filled.rows[y as usize] |= 1 << x;
            }
            hole_stats(&filled)
        }
        None => hole_stats(b),
    };
    f[0] = holes;
    f[1] = hole_rows;
    f[2] = cover;

    // Deepest column is the candidate well; its two edges are left out of bumpiness.
    let mut well = 0usize;
    for x in 1..W as usize {
        if h[x] < h[well] {
            well = x;
        }
    }
    let mut bump = 0.0;
    let mut bump_sq = 0.0;
    for x in 0..W as usize - 1 {
        if x == well || x + 1 == well {
            continue;
        }
        let d = (h[x] - h[x + 1]).abs() as f32;
        bump += d;
        bump_sq += d * d;
    }
    f[3] = bump;
    f[4] = bump_sq;

    let mut rt = 0u32;
    let mut ct = 0u32;
    for y in top.max(0)..H {
        let p = padded(b, y);
        rt += ((p ^ (p >> 1)) & 0x7FF).count_ones();
        let below = if y + 1 < H { b.rows[y as usize + 1] } else { FULL };
        ct += (b.rows[y as usize] ^ below).count_ones();
    }
    if top > 0 && top < H {
        // Transition from open air into the top row.
        ct += b.rows[top as usize].count_ones();
    }
    f[5] = rt as f32;
    f[6] = ct as f32;
    f[7] = maxh as f32;
    let d = (maxh - 10).max(0) as f32;
    f[8] = d * d;

    // Rows one I away: from the well's floor upward, full except the well column.
    let mut wr = 0;
    let mut y = H - 1 - h[well];
    let want = FULL & !(1 << well);
    while y >= 0 && wr < 4 && b.rows[y as usize] == want {
        wr += 1;
        y -= 1;
    }
    f[9] = wr as f32;

    let mut extra = 0;
    for x in 0..W as usize {
        if x == well {
            continue;
        }
        let l = if x == 0 { H } else { h[x - 1] };
        let r = if x + 1 == W as usize { H } else { h[x + 1] };
        let depth = l.min(r) - h[x];
        if depth > 2 {
            extra += depth - 2;
        }
    }
    f[10] = extra as f32;

    if let Some(s) = slot {
        match s.lines {
            1 => f[11] = 1.0,
            2 => f[12] = 1.0,
            3 => f[13] = 1.0,
            _ => {}
        }
        f[14] = s.fill;
    }
    f[15] = (b2b >= 0) as i32 as f32;
    f[16] = b2b.clamp(0, 8) as f32;
    f[17] = if b2b >= 0 { ((b2b + 1) as f32).sqrt() } else { 0.0 };
    // All-spin repeat penalty. Both are zero outside that mode (no wounds, no
    // last_clear), so they cost nothing in the other modes.
    f[18] = b.wounds as f32;
    if b.last_clear != 0 {
        let mut rep = 0.0;
        if wr >= 4 && b.last_clear == clear_type(Piece::I, Spin::None, 4) {
            rep += 1.0;
        }
        if let Some(s) = slot {
            if s.lines > 0 && b.last_clear == clear_type(Piece::T, Spin::Full, s.lines) {
                rep += 1.0;
            }
        }
        f[19] = rep;
    }
    // Best spin slot per family (S/Z, L/J, I): one-hot on the rows it would clear.
    let mut best: [Option<(u32, Piece)>; 3] = [None; 3];
    scan_spin_slots(b, top, |s| {
        if s.lines == 0 {
            return;
        }
        let fam = match s.piece {
            Piece::S | Piece::Z => 0,
            Piece::L | Piece::J => 1,
            _ => 2,
        };
        if best[fam].map_or(true, |(l, _)| s.lines > l) {
            best[fam] = Some((s.lines.min(3), s.piece));
        }
    });
    for (fam, bf) in best.iter().enumerate() {
        if let Some((lines, piece)) = *bf {
            f[20 + fam * 3 + lines as usize - 1] = 1.0;
            // A ready spin of the same type as the last clear would be a repeat.
            if b.last_clear != 0 && b.last_clear == clear_type(piece, Spin::Full, lines) {
                f[19] += 1.0;
            }
        }
    }
    f
}
