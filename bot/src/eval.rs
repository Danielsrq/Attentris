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
use crate::rules::{clear_type, B2bMode, Piece, Shapes, Spin, H, W};

pub const NAMES: [&str; 31] = [
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
    "b2b_next",    // B2B bonus the next difficult clear would earn under the active scaling
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
    // Burst rules: the last placement was a qualifying clear, so the next one is
    // doubled. A fact about the position, not a plan; 0 in every other mode.
    "burst_armed",
    // Fill of the TSD a vertical STSD / TST (C-spin) slot leaves behind: how ready the
    // second clear of a T-spin pair is. See scan_tslots.
    "t_followup",
];
pub const NF: usize = NAMES.len();

/// Normalisation: every feature is divided by a fixed, theory-based scale -- its realistic
/// range under the game's rules (the 20-row visible field, a slot's 0/1 flag, the B2B
/// bonus cap of 7) -- so each weight is in lines of attack per typical change and all
/// weights sit on one scale. For a linear V this changes nothing about play; it makes
/// the weights comparable and the tuner better conditioned.
pub const SCALE: [f32; NF] = [
    10.0,  // holes
    20.0,  // hole_rows: rows in the visible field
    20.0,  // cover
    20.0,  // bump
    100.0, // bump_sq
    40.0,  // row_trans: ~2 per row over 20 rows
    40.0,  // col_trans
    20.0,  // max_height: visible field
    100.0, // danger: (20 - 10)^2
    4.0,   // well_rows: 0..4
    10.0,  // extra_wells
    1.0, 1.0, 1.0, // t_single / t_double / t_triple: flags
    1.0,   // t_fill: 0..1
    1.0,   // b2b: flag
    8.0,   // b2b_level: capped at 8
    7.0,   // b2b_next: bonus capped at 7
    10.0,  // wounds
    2.0,   // ready_repeat
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, // spin-slot flags
    1.0,   // burst_armed: flag
    1.0,   // t_followup: 0..1
];

/// Features that carry no information in a mode, frozen at 0 by `tune --freeze` and in
/// Shapley runs: the non-T spin slots (Minis send nothing outside all-spin), max_height
/// (danger covers it), and the all-spin repeat-penalty pair.
pub const FROZEN_OUTSIDE_ALLSPIN: [&str; 12] = [
    "max_height", "wounds", "ready_repeat",
    "sz_single", "sz_double", "sz_triple", "lj_single", "lj_double", "lj_triple",
    "i_single", "i_double", "i_triple",
];

pub fn index_of(name: &str) -> Option<usize> {
    NAMES.iter().position(|&k| k == name)
}

pub const NORMALISED_MARKER: &str = "#! features normalised";

/// Experimental control: evaluate with raw (unscaled) features, as before normalisation.
pub static RAW_FEATURES: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

#[inline]
/// Weights in the active units (normalised unless --raw-features) to normalised units,
/// the units every log and weight file uses.
pub fn to_normalised(w: &[f32; NF]) -> [f32; NF] {
    let mut out = *w;
    for i in 0..NF {
        out[i] = w[i] / scale(i) * SCALE[i];
    }
    out
}

/// Normalised units to the active units.
pub fn from_normalised(w: &[f32; NF]) -> [f32; NF] {
    let mut out = *w;
    for i in 0..NF {
        out[i] = w[i] / SCALE[i] * scale(i);
    }
    out
}

fn scale(i: usize) -> f32 {
    if RAW_FEATURES.load(std::sync::atomic::Ordering::Relaxed) { 1.0 } else { SCALE[i] }
}

#[derive(Clone, Debug)]
pub struct Weights {
    pub w: [f32; NF],
}

impl Default for Weights {
    /// Hand-set starting point in raw units, 1.0 ~ one line of attack; converted to the
    /// active (normalised) units on construction. `tune` improves it.
    fn default() -> Self {
        let mut w = Weights {
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
                0.0,   // b2b_next
                0.0,   // wounds
                0.0,   // ready_repeat
                0.0, 0.0, 0.0, // sz_single / double / triple
                0.0, 0.0, 0.0, // lj_*
                0.0, 0.0, 0.0, // i_*
                0.0,   // burst_armed
                0.0,   // t_followup
            ],
        };
        for i in 0..NF {
            w.w[i] *= scale(i);
        }
        w
    }
}

impl Weights {
    /// Files written since normalisation carry NORMALISED_MARKER and hold weights for
    /// the normalised features. Older files hold weights for the raw features; they are
    /// converted on load (w * scale), so they evaluate every board exactly as before.
    pub fn parse(text: &str) -> Result<Weights, String> {
        let mut w = Weights::default();
        let normalised_file = text.lines().any(|l| l.trim() == NORMALISED_MARKER);
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
            // b2b_sqrt (sqrt(chain + 1)) was replaced by b2b_next; carry its weight over.
            let name = if name == "b2b_sqrt" { "b2b_next" } else { name };
            let i = NAMES.iter().position(|&k| k == name).ok_or(format!("line {}: unknown feature {}", n + 1, name))?;
            // Stored in the units of the active mode (normalised unless --raw-features).
            let raw = if normalised_file { val / SCALE[i] } else { val };
            w.w[i] = raw * scale(i);
        }
        Ok(w)
    }

    /// Always written for normalised features, with the marker.
    pub fn to_text(&self) -> String {
        let mut s = format!("{}\n", NORMALISED_MARKER);
        for i in 0..NF {
            s += &format!("{:<12} {:.4}\n", NAMES[i], self.w[i] / scale(i) * SCALE[i]);
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
    /// A vertical slot the T can also be turned 180 in (the cell behind the stem open,
    /// both corners there filled): the imperial cross's turn. Its cells, if so.
    pub flip: Option<[(i32, i32); 4]>,
    /// Static prophecy: the cells the SECOND T of a known two-T structure will fill, in
    /// today's board (no rows cleared yet). Vertical family (STSD, C-spin): the lane row
    /// plus the first empty cell under the lane side. Fractal: the flat slot two rows
    /// below. `flip_second`: the same for the 180 twin (imperial cross: the lane row
    /// plus the cell the twisted-in nub left).
    pub second: Option<[(i32, i32); 4]>,
    pub flip_second: Option<[(i32, i32); 4]>,
}

/// Everything the T-slot scan finds: the best slot, and the best follow-up fill over the
/// vertical template slots (see `scan_tslots`).
#[derive(Clone, Copy, Default, Debug)]
pub struct TSlots {
    pub best: Option<TSlot>,
    /// Best vertical template slot (STSD / TST / C-spin), for the diagnostics.
    pub vertical: Option<TSlot>,
}

thread_local! {
    /// Diagnostics: V evaluations on this thread, and how many found a vertical template
    /// slot / a ready one (2+ lines). Thread-local, so counting costs no synchronisation.
    static TEMPLATE_COUNTS: std::cell::Cell<[u64; 3]> = const { std::cell::Cell::new([0; 3]) };
}

/// This thread's (evaluations, vertical slot seen, vertical slot ready) so far.
pub fn template_counts() -> [u64; 3] {
    TEMPLATE_COUNTS.with(|c| c.get())
}

/// Best T-spin slot on the board.
pub fn best_tslot(b: &Board, top: i32) -> Option<TSlot> {
    scan_tslots(b, top).best
}

/// T-spin slots: resting T positions that would register as a FULL spin. Two shapes.
///
///   Flat (rot 2, nub down), the classic TSD: both front corners + one back corner.
///
///   Vertical (rot 1 nub right / rot 3 nub left), one template for the STSD, the TST and
///   the C-spin. Shown nub right; c = stem column, y = the nub's row:
///
///             c-1  c  c+1 c+2 c+3
///     y-3      ?   #   .   .   A     cap over the stem
///     y-2      ?   .   .   .   B     the lane the T kicks through
///     y-1      ?   T   #   ?   ?     roof over the nub
///     y        ?   T   T   ?   ?
///     y+1      ?   T   h   ?   ?     STSD: h empty -> the T clears y-1, y; a TSD opens at h
///     y+2      ?   ?   g   ?   ?     C-spin: h filled, g empty -> TST; a TSD opens at g
///
///   Three of the four corners around (c, y) must be filled (an STSD has only the roof
///   in front; the fifth-kick tuck makes it full), and A empty with B filled sends the
///   kick elsewhere. This is exactly the set of cap/lane fills where SRS lands the T in
///   the slot (tests: `vertical_template_matches_movegen`). Reachability from the spawn
///   is otherwise not checked -- the search checks the real thing.
///
/// Each vertical slot also carries its static prophecy (`TSlot::second`): once the first
/// clear is cashed, the lane row lands on the row holding h (or g), and the cap becomes
/// the new TSD's corner.
pub fn scan_tslots(b: &Board, top: i32) -> TSlots {
    scan_rows(b, (top - 3).max(1), H - 1)
}

/// `scan_tslots` over slot rows y_from..y_to only (the chain's banded rescan).
pub fn scan_rows(b: &Board, y_from: i32, y_to: i32) -> TSlots {
    let mut best: Option<TSlot> = None;
    let mut vertical: Option<TSlot> = None;
    let y0 = y_from.max(1);
    let pop = |y: i32| -> u32 { if (0..H).contains(&y) { b.rows[y as usize].count_ones() } else { 10 } };
    // Walls, floor and ceiling count as filled.
    let filled = |x: i32, y: i32| -> bool { b.blocked(x, y) };
    let consider = |slot: TSlot, best: &mut Option<TSlot>| {
        let better = match best {
            None => true,
            Some(s) => (slot.lines, slot.fill) > (s.lines, s.fill),
        };
        if better {
            *best = Some(slot);
        }
    };
    for y in y0..y_to.min(H - 1) {
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
            let cells = [(x - 1, y), (x, y), (x + 1, y), (x, y + 1)];
            consider(TSlot { lines, fill: (a + bb) as f32 / 20.0, cells, ..Default::default() }, &mut best);
        }
        if y < 3 {
            continue;
        }
        // Vertical template: stem empty, nub empty, roof over the nub (bitmask prefilter);
        // corners, cap and lane by lookup on the survivors.
        let base = em & e0 & e1;
        let m1 = base & (e0 >> 1) & (pm >> 1) & 0x7FE;
        let m3 = base & (e0 << 1) & (pm << 1) & 0x7FE;
        for (mask, s) in [(m1, 1i32), (m3, -1i32)] {
            let mut m = mask;
            while m != 0 {
                let c = m.trailing_zeros() as i32;
                m &= m - 1;
                let x = c - 1;
                if x + s < 0 || x + s >= W {
                    continue;
                }
                let corners = 1 + filled(x + s, y + 1) as u32 + filled(x - s, y - 1) as u32 + filled(x - s, y + 1) as u32;
                if corners < 3 {
                    continue;
                }
                let lane_open = !filled(x, y - 2) && !filled(x + s, y - 2) && !filled(x + 2 * s, y - 2)
                    && !filled(x + s, y - 3) && !filled(x + 2 * s, y - 3);
                if !filled(x, y - 3) || !lane_open || (!filled(x + 3 * s, y - 3) && filled(x + 3 * s, y - 2)) {
                    continue;
                }
                let a = pop(y - 1) + 1;
                let bb = pop(y) + 2;
                let cc = pop(y + 1) + 1;
                let lines = (a == 10) as u32 + (bb == 10) as u32 + (cc == 10) as u32;
                let cells = [(x, y - 1), (x, y), (x + s, y), (x, y + 1)];
                let flip = (!filled(x - s, y) && filled(x - s, y - 1) && filled(x - s, y + 1))
                    .then_some([(x, y - 1), (x, y), (x - s, y), (x, y + 1)]);
                let slot = TSlot { lines, fill: (a + bb + cc) as f32 / 30.0, cells, flip, ..Default::default() };
                consider(slot, &mut best);
                consider(slot, &mut vertical);
            }
        }
    }
    // The static prophecies, for the two slots a chain can start from only.
    TSlots { best: best.map(|s| annotate(b, s)), vertical: vertical.map(|s| annotate(b, s)) }
}

/// Adds a slot's static prophecy (`TSlot::second`, `flip_second`).
fn annotate(b: &Board, mut s: TSlot) -> TSlot {
    let filled = |x: i32, y: i32| -> bool { b.blocked(x, y) };
    let [c0, c1, c2, _] = s.cells;
    if c0.1 == c1.1 {
        // Flat (cells: left, centre, right, nub). Fractal: the same TSD shape two rows
        // down, its top corner supplied by the row above this one once this has cleared.
        let (x, y) = c1;
        if !filled(x - 1, y + 2) && !filled(x, y + 2) && !filled(x + 1, y + 2) && !filled(x, y + 3)
            && filled(x - 1, y + 3) && filled(x + 1, y + 3) && (filled(x - 1, y - 1) || filled(x + 1, y - 1))
        {
            s.second = Some([(x - 1, y + 2), (x, y + 2), (x + 1, y + 2), (x, y + 3)]);
        }
    } else {
        // Vertical (cells: stem top, stem middle, nub, stem bottom). The second T: a TSD in
        // the lane row, its nub in the first empty cell under the lane side -- h (STSD), g
        // (C-spin), or, turned 180, the cell the twisted-in nub left (imperial cross).
        let (x, y) = c1;
        let side = c2.0 - x;
        let lane = [(x, y - 2), (x + side, y - 2), (x + 2 * side, y - 2)];
        let hole = if !filled(x + side, y + 1) {
            Some(y + 1)
        } else if !filled(x + side, y + 2) {
            Some(y + 2)
        } else {
            None
        };
        s.second = hole.map(|hy| [lane[0], lane[1], lane[2], (x + side, hy)]);
        s.flip_second = s.flip.map(|_| [lane[0], lane[1], lane[2], (x + side, y)]);
    }
    s
}

/// Readiness (0..1) of a prophesied second T: the fill of its rows, counting the first
/// T's cells as filled; 1 = both rows complete once the two T's are in.
fn prophecy_fill(b: &Board, first: &[(i32, i32); 4], second: &[(i32, i32); 4]) -> f32 {
    let (mut have, mut need) = (0u32, 0u32);
    let mut rows: Vec<i32> = second.iter().map(|c| c.1).collect();
    rows.sort_unstable();
    rows.dedup();
    for y in rows {
        if !(0..H).contains(&y) {
            return 0.0;
        }
        let mine = second.iter().filter(|c| c.1 == y).count() as u32;
        have += b.rows[y as usize].count_ones() + first.iter().filter(|c| c.1 == y).count() as u32;
        need += W as u32 - mine;
    }
    have as f32 / need.max(1) as f32
}

/// The board after cashing the T-spin chain -- the best slot, then the best slot that
/// leaves -- and the readiness (fill, 0..1) of that second slot. Cells are filled and
/// full rows cleared, as the real placements would. Also tries the best vertical slot
/// (and its 180 twin): a covered flat slot can tie with it on lines and fill (an
/// imperial cross has one under its stem), but only the slot a chain starts from leaves
/// the second one. The chain with the better second slot wins.
pub fn cash_chain_from(b: &Board, first: Option<TSlot>, vertical: Option<TSlot>) -> (Board, f32) {
    let one = cash_one(b, first);
    match vertical {
        Some(v) if first.map_or(true, |f| f.cells != v.cells) => {
            let two = cash_one(b, Some(v));
            if two.1 > one.1 { two } else { one }
        }
        _ => one,
    }
}

fn cash_one(b: &Board, first: Option<TSlot>) -> (Board, f32) {
    let Some(s) = first else { return (*b, 0.0) };
    let fill = |nb: &mut Board, cells: &[(i32, i32); 4]| {
        let rel: [(i8, i8); 4] = std::array::from_fn(|i| (cells[i].0 as i8, cells[i].1 as i8));
        nb.place(&rel, 0, 0);
    };
    let chain = |cells: &[(i32, i32); 4], second: Option<[(i32, i32); 4]>| -> (Board, f32) {
        let mut nb = *b;
        if s.lines == 0 {
            // A slot that clears nothing starts no chain: just its own cavity is exempt.
            for &(x, y) in cells {
                nb.rows[y as usize] |= 1 << x;
            }
            return (nb, 0.0);
        }
        if let Some(sec) = second {
            // Static prophecy: both T's cells count as filled and the rows they complete
            // drop out, as after the real chain -- no rescan.
            for &(x, y) in cells.iter().chain(sec.iter()) {
                if (0..H).contains(&y) && (0..W).contains(&x) {
                    nb.rows[y as usize] |= 1 << x;
                }
            }
            let mut write = H as usize;
            for y in (0..H as usize).rev() {
                if nb.rows[y] == FULL && nb.wound_left[y] == 0 {
                    continue;
                }
                write -= 1;
                nb.rows[write] = nb.rows[y];
            }
            for y in 0..write {
                nb.rows[y] = 0;
            }
            return (nb, prophecy_fill(b, cells, &sec));
        }
        // Dynamic: cash it, then rescan the rows around it for the slot it leaves. Only
        // a ready double or triple starts a chain worth looking for (the gate and burst
        // rules open on those); a weaker slot just has its own cavity exempt.
        if s.lines < 2 {
            for &(x, y) in cells {
                nb.rows[y as usize] |= 1 << x;
            }
            return (nb, 0.0);
        }
        fill(&mut nb, cells);
        let lo = cells.iter().map(|c| c.1).min().unwrap() - 4;
        let hi = cells.iter().map(|c| c.1).max().unwrap() + 3;
        match scan_rows(&nb, lo, hi).best {
            Some(s2) => {
                fill(&mut nb, &s2.cells);
                (nb, s2.fill)
            }
            None => (nb, 0.0),
        }
    };
    let a = chain(&s.cells, s.second);
    match s.flip {
        // Whichever way round leaves the better second slot (an imperial cross is turned).
        Some(f) => {
            let b2 = chain(&f, s.flip_second);
            if b2.1 > a.1 { b2 } else { a }
        }
        None => a,
    }
}

/// The chain's health board and follow-up, computed the dynamic way only (cash, clear,
/// rescan the whole board) -- the reference the static prophecy is tested against.
pub fn cash_chain_dynamic(b: &Board, first: TSlot) -> (Board, f32) {
    let fill = |nb: &mut Board, cells: &[(i32, i32); 4]| {
        let rel: [(i8, i8); 4] = std::array::from_fn(|i| (cells[i].0 as i8, cells[i].1 as i8));
        nb.place(&rel, 0, 0);
    };
    let run = |cells: &[(i32, i32); 4]| {
        let mut nb = *b;
        fill(&mut nb, cells);
        let top = H - *nb.heights().iter().max().unwrap();
        match best_tslot(&nb, top) {
            Some(s2) => {
                fill(&mut nb, &s2.cells);
                (nb, s2.fill)
            }
            None => (nb, 0.0),
        }
    };
    let a = run(&first.cells);
    match first.flip {
        Some(f) => {
            let c = run(&f);
            if c.1 > a.1 { c } else { a }
        }
        None => a,
    }
}

/// `hole_stats`, for tests and diagnostics.
pub fn hole_stats_pub(b: &Board) -> (f32, f32, f32) {
    hole_stats(b)
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

pub fn features(b: &Board, b2b: i32, b2b_mode: B2bMode) -> [f32; NF] {
    let mut f = [0f32; NF];
    let h = b.heights();
    let maxh = *h.iter().max().unwrap();
    let top = H - maxh;

    let slots = scan_tslots(b, top);
    let slot = slots.best;
    TEMPLATE_COUNTS.with(|c| {
        let mut v = c.get();
        v[0] += 1;
        if let Some(s) = slots.vertical {
            v[1] += 1;
            v[2] += (s.lines >= 2) as u64;
        }
        c.set(v);
    });
    // Board health (holes, surface, transitions) is measured on the board as it would be
    // after cashing the T-spin chain: the best slot, then the slot that cashing it leaves
    // (a fractal's lower TSD, a DT's TST, a C-spin's TSD, an imperial cross's TSD...). A
    // cavity a T-spin chain will fill is an asset (t_*), not a hole or a jagged surface.
    let (health, follow) = cash_chain_from(b, slot, slots.vertical);
    let hb = &health;
    let (holes, hole_rows, cover) = hole_stats(hb);
    f[0] = holes;
    f[1] = hole_rows;
    f[2] = cover;

    // Deepest column is the candidate well; its two edges are left out of bumpiness.
    let hh = if slot.is_some() { hb.heights() } else { h };
    let htop = H - *hh.iter().max().unwrap();
    let mut hwell = 0usize;
    for x in 1..W as usize {
        if hh[x] < hh[hwell] {
            hwell = x;
        }
    }
    let mut bump = 0.0;
    let mut bump_sq = 0.0;
    for x in 0..W as usize - 1 {
        if x == hwell || x + 1 == hwell {
            continue;
        }
        let d = (hh[x] - hh[x + 1]).abs() as f32;
        bump += d;
        bump_sq += d * d;
    }
    f[3] = bump;
    f[4] = bump_sq;

    let mut rt = 0u32;
    let mut ct = 0u32;
    for y in htop.max(0)..H {
        let p = padded(hb, y);
        rt += ((p ^ (p >> 1)) & 0x7FF).count_ones();
        let below = if y + 1 < H { hb.rows[y as usize + 1] } else { FULL };
        ct += (hb.rows[y as usize] ^ below).count_ones();
    }
    if htop > 0 && htop < H {
        // Transition from open air into the top row.
        ct += hb.rows[htop as usize].count_ones();
    }
    f[5] = rt as f32;
    f[6] = ct as f32;

    // Deepest column of the real board, for the well features below.
    let mut well = 0usize;
    for x in 1..W as usize {
        if h[x] < h[well] {
            well = x;
        }
    }
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
    // The bonus the next difficult clear would earn: follows whichever scaling is active.
    f[17] = if b2b >= 0 { b2b_mode.bonus(b2b + 1) as f32 } else { 0.0 };
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
    f[29] = (b.burst_armed > 0) as i32 as f32;
    f[30] = follow;
    for i in 0..NF {
        f[i] /= scale(i);
    }
    f
}
