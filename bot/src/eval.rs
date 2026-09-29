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
use crate::rules::{H, W};

pub const NAMES: [&str; 18] = [
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
    f
}
