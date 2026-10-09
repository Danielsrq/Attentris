//! Game rules, ported one-to-one from the Godot project so the bot optimises the game
//! that actually ships:
//!
//!   core/piece_data.gd   -- shapes, boxes, spawn orientation
//!   core/piece_set.gd    -- the rotate_cw primitive (x, y) -> (box-1-y, x)
//!   core/kick_tables.gd  -- SRS + the SRS+ 180 table (written y-up, flipped here)
//!   core/spin_detect.gd  -- 3-corner T-spins with the kick-5 upgrade, all-spin by immobility
//!   core/attack_table.gd -- lines / spins / B2B / perfect clear
//!   core/game.gd         -- spawn position, B2B bookkeeping, lock-out
//!
//! Board space matches the Godot board: y = 0 is the TOP, 40 rows total, the top 20 are
//! the buffer. A piece position is the top-left of its bounding box.

pub const W: i32 = 10;
pub const H: i32 = 40;
pub const BUFFER: i32 = 20;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
#[repr(u8)]
pub enum Piece {
    I = 0,
    O,
    T,
    S,
    Z,
    J,
    L,
}

pub const ALL_PIECES: [Piece; 7] = [Piece::I, Piece::O, Piece::T, Piece::S, Piece::Z, Piece::J, Piece::L];

impl Piece {
    pub fn letter(self) -> char {
        b"iotszjl"[self as usize] as char
    }
    pub fn from_letter(c: char) -> Option<Piece> {
        match c.to_ascii_lowercase() {
            'i' => Some(Piece::I),
            'o' => Some(Piece::O),
            't' => Some(Piece::T),
            's' => Some(Piece::S),
            'z' => Some(Piece::Z),
            'j' => Some(Piece::J),
            'l' => Some(Piece::L),
            _ => None,
        }
    }
    pub fn box_size(self) -> i32 {
        match self {
            Piece::I => 4,
            Piece::O => 2,
            _ => 3,
        }
    }
    /// Spawn-state cells inside the bounding box, straight from PieceData.tetrominoes().
    fn base_cells(self) -> [(i32, i32); 4] {
        match self {
            Piece::I => [(0, 1), (1, 1), (2, 1), (3, 1)],
            Piece::O => [(0, 0), (1, 0), (0, 1), (1, 1)],
            Piece::T => [(1, 0), (0, 1), (1, 1), (2, 1)],
            Piece::S => [(1, 0), (2, 0), (0, 1), (1, 1)],
            Piece::Z => [(0, 0), (1, 0), (1, 1), (2, 1)],
            Piece::J => [(0, 0), (0, 1), (1, 1), (2, 1)],
            Piece::L => [(2, 0), (0, 1), (1, 1), (2, 1)],
        }
    }
    fn first_filled_row(self) -> i32 {
        if self == Piece::I { 1 } else { 0 }
    }
    /// Game.spawn_pos_for: centred, first filled row on the first visible row.
    pub fn spawn(self) -> (i32, i32) {
        ((W - self.box_size()) / 2, BUFFER - self.first_filled_row())
    }
}

/// cells[piece][rot] = the four occupied offsets, y-down.
pub struct Shapes {
    pub cells: [[[(i8, i8); 4]; 4]; 7],
}

impl Shapes {
    /// The piece set of a rule set: SRS, or TetrisBattle's under battle rules.
    pub fn for_rules(r: &Rules) -> Shapes {
        if r.battle { crate::battle::shapes() } else { Shapes::new() }
    }

    pub fn new() -> Shapes {
        let mut cells = [[[(0i8, 0i8); 4]; 4]; 7];
        for p in ALL_PIECES {
            let b = p.box_size();
            let mut cur = p.base_cells();
            for r in 0..4 {
                let mut sorted = cur;
                sorted.sort_by_key(|&(x, y)| (y, x));
                for i in 0..4 {
                    cells[p as usize][r][i] = (sorted[i].0 as i8, sorted[i].1 as i8);
                }
                // PieceSet.rotate_cw: a visually clockwise turn in y-down space.
                for c in cur.iter_mut() {
                    *c = (b - 1 - c.1, c.0);
                }
            }
        }
        Shapes { cells }
    }
}

/// Kick offsets in y-down board space, tried in order. Index 4 is the "1x2 tuck"
/// that upgrades a T-spin Mini to full.
pub fn kicks(piece: Piece, from: usize, to: usize) -> &'static [(i8, i8)] {
    // Tables are y-up as in kick_tables.gd; the y sign is flipped at the literal so
    // every entry below reads (x, -y_up).
    macro_rules! k {
        ($(($x:expr, $y:expr)),*) => { &[$(($x, -$y)),*] };
    }
    match piece {
        Piece::O => &[(0, 0)],
        Piece::I => match (from, to) {
            (0, 1) => k!((0, 0), (-2, 0), (1, 0), (-2, -1), (1, 2)),
            (1, 0) => k!((0, 0), (2, 0), (-1, 0), (2, 1), (-1, -2)),
            (1, 2) => k!((0, 0), (-1, 0), (2, 0), (-1, 2), (2, -1)),
            (2, 1) => k!((0, 0), (1, 0), (-2, 0), (1, -2), (-2, 1)),
            (2, 3) => k!((0, 0), (2, 0), (-1, 0), (2, 1), (-1, -2)),
            (3, 2) => k!((0, 0), (-2, 0), (1, 0), (-2, -1), (1, 2)),
            (3, 0) => k!((0, 0), (1, 0), (-2, 0), (1, -2), (-2, 1)),
            (0, 3) => k!((0, 0), (-1, 0), (2, 0), (-1, 2), (2, -1)),
            (0, 2) => k!((0, 0), (0, 1), (0, -1)),
            (2, 0) => k!((0, 0), (0, -1), (0, 1)),
            (1, 3) => k!((0, 0), (1, 0), (-1, 0)),
            (3, 1) => k!((0, 0), (-1, 0), (1, 0)),
            _ => &[(0, 0)],
        },
        _ => match (from, to) {
            (0, 1) => k!((0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)),
            (1, 0) => k!((0, 0), (1, 0), (1, -1), (0, 2), (1, 2)),
            (1, 2) => k!((0, 0), (1, 0), (1, -1), (0, 2), (1, 2)),
            (2, 1) => k!((0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)),
            (2, 3) => k!((0, 0), (1, 0), (1, 1), (0, -2), (1, -2)),
            (3, 2) => k!((0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)),
            (3, 0) => k!((0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)),
            (0, 3) => k!((0, 0), (1, 0), (1, 1), (0, -2), (1, -2)),
            (0, 2) => k!((0, 0), (0, 1), (1, 1), (-1, 1), (1, 0), (-1, 0)),
            (2, 0) => k!((0, 0), (0, -1), (-1, -1), (1, -1), (-1, 0), (1, 0)),
            (1, 3) => k!((0, 0), (1, 0), (1, 2), (1, 1), (0, 2), (0, 1)),
            (3, 1) => k!((0, 0), (-1, 0), (-1, 2), (-1, 1), (0, 2), (0, 1)),
            _ => &[(0, 0)],
        },
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub enum Spin {
    None = 0,
    Mini = 1,
    Full = 2,
}

impl Spin {
    pub fn name(self) -> &'static str {
        match self {
            Spin::None => "none",
            Spin::Mini => "mini",
            Spin::Full => "full",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum B2bMode {
    /// The original attack_table.gd: +1 while a chain is live.
    Flat,
    /// Legacy b2b_squares: floor(sqrt(chain)), uncapped.
    Squares,
    /// Default b2b_fibonacci: +1 at each chain that is a squared Fibonacci number
    /// (1, 4, 9, 25, 64, 169, 441), so +1..+7, capped at +7.
    Fibonacci,
    /// Power-up b2b_prime: +1 at each squared prime (4, 9, 25, 49, 121, 169, 289),
    /// capped at +7. Chains 1-3 earn nothing.
    Prime,
    /// TETR.IO's own table (TetrisBattle's b2b_level): +1 for chain 1-2, +2 for 3-7,
    /// +3 for 8-23, +4 for 24-66, +5 for 67-184, +6 for 185-503, +7 for 504-1369, +8 after.
    Tetrio,
}

/// TETR.IO B2B levels: (largest chain at this level, level).
pub const TETRIO_B2B: [(i32, i32); 8] = [(0, 0), (2, 1), (7, 2), (23, 3), (66, 4), (184, 5), (503, 6), (1369, 7)];

/// TetrisBattle's b2b_level.
pub fn tetrio_b2b_level(chain: i32) -> i32 {
    TETRIO_B2B.iter().find(|&&(limit, _)| chain <= limit).map_or(8, |&(_, level)| level)
}

pub const FIBONACCI_STEPS: [i32; 7] = [1, 4, 9, 25, 64, 169, 441];
pub const PRIME_STEPS: [i32; 7] = [4, 9, 25, 49, 121, 169, 289];

impl B2bMode {
    /// Bonus for a chain of length `chain` (ClearResult.b2b: -1 none, 0 first, >= 1).
    pub fn bonus(self, chain: i32) -> i32 {
        if chain < 1 {
            return 0;
        }
        match self {
            B2bMode::Flat => 1,
            B2bMode::Squares => (chain as f64).sqrt().floor() as i32,
            B2bMode::Fibonacci => FIBONACCI_STEPS.iter().filter(|&&s| chain >= s).count() as i32,
            B2bMode::Prime => PRIME_STEPS.iter().filter(|&&s| chain >= s).count() as i32,
            B2bMode::Tetrio => tetrio_b2b_level(chain),
        }
    }
    pub fn parse(s: &str) -> Option<B2bMode> {
        match s {
            "flat" => Some(B2bMode::Flat),
            "squares" | "sqrt" => Some(B2bMode::Squares),
            "fibonacci" | "fib" => Some(B2bMode::Fibonacci),
            "prime" => Some(B2bMode::Prime),
            "tetrio" => Some(B2bMode::Tetrio),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Rules {
    pub b2b: B2bMode,
    /// RuleSet.all_spin: immobility spins for S, Z, L, J, I (and O). Off = the
    /// "guideline" preset, T-spins only.
    pub all_spin: bool,
    pub perfect_clear_bonus: i32,
    /// All-spin mode (RuleSet.allspin_full): every Mini is promoted to a full spin.
    pub promote_minis: bool,
    /// All-spin mode (RuleSet.repeat_penalty): two consecutive clears of the same type
    /// wound the bottom in-play row for `wound_clears` clears.
    pub repeat_penalty: bool,
    pub wound_clears: u8,
    /// Chance a new wound line's hole moves from the previous wound's column
    /// (RuleSet.wound_messiness). High messiness turns expired wounds into cheese.
    pub wound_messiness: f64,
    /// Normal rules (RuleSet.non_t_spin_is_mini): non-T spins count only as Minis.
    pub non_t_mini: bool,
    /// Experimental: the B2B bonus never exceeds this (None = uncapped).
    pub b2b_cap: Option<i32>,
    /// Burst rules (RuleSet.burst): a qualifying clear straight after another doubles
    /// its base attack (see `burst_qualifies`); the B2B bonus is added after, undoubled.
    pub burst: bool,
    /// Base attack of a full T-spin triple (6; 10 under burst rules).
    pub tst_attack: i32,
    /// Burst rules: a qualifying clear is doubled when an arming clear happened within
    /// this many previous placements (1 = straight after).
    pub burst_window: u8,
    /// Burst rules: whether a full spin single arms the next clear. It can still be
    /// doubled itself either way.
    pub burst_tss_arms: bool,
    /// Gate rules (a training rule set): only chains score. See `gate_step`.
    pub gate: bool,
    /// gate_b2b: each spin clear of a paid gate chain also earns the B2B bonus of the
    /// chain count it reached; B2B runs (and plain clears break it) as in normal rules.
    pub gate_b2b: bool,
    /// TetrisBattle rules (a friend's RL bot's engine, external/tetrisbattle): its own
    /// rotation and kicks (battle.rs), T-spins only, TETR.IO attack with combos, and
    /// top-out when the top visible row is filled after a lock. See `battle_attack`.
    pub battle: bool,
}

/// Gate chain state: (placements left in the window, spin clears in the chain, value
/// held until the chain has two spin clears). Lives on the Board.
pub type GateState = (u8, u8, u8);

impl Default for Rules {
    fn default() -> Self {
        Rules {
            b2b: B2bMode::Fibonacci,
            all_spin: true,
            perfect_clear_bonus: 10,
            promote_minis: false,
            repeat_penalty: false,
            wound_clears: 10,
            wound_messiness: 0.5,
            non_t_mini: false,
            b2b_cap: None,
            burst: false,
            tst_attack: 6,
            burst_window: 1,
            burst_tss_arms: true,
            gate: false,
            gate_b2b: false,
            battle: false,
        }
    }
}

impl Rules {
    /// The all-spin preset: every piece spins (O excepted in practice), Minis count as
    /// full, and repeating a clear type wounds a row.
    /// Normal rules (TETR.IO-style), the default mode: every piece can spin, non-T spins
    /// are Minis, and a Mini sends nothing but keeps B2B alive.
    pub fn normal() -> Rules {
        Rules { all_spin: true, non_t_mini: true, ..Rules::default() }
    }

    /// Burst rules: normal rules, TST worth 10, and back-to-back *placements* of
    /// qualifying clears double the second one's base attack.
    pub fn burst() -> Rules {
        Rules { burst: true, tst_attack: 10, ..Rules::normal() }
    }

    /// Burst rules, relaxed and sharpened: spin singles no longer arm the doubling, and
    /// an arming clear arms the next TWO placements.
    pub fn burst2() -> Rules {
        Rules { burst_window: 2, burst_tss_arms: false, ..Rules::burst() }
    }

    /// Gate rules: normal rules, but a clear only scores as part of a chain. A full spin
    /// double or triple opens a gate; within `burst_window` placements, spin doubles /
    /// triples and quads join the chain and restart the window; anything else just uses
    /// up a placement. The chain pays the sum of its members (TSD 4, TST 10, quad 1) once
    /// it holds TWO spin clears: a quad keeps the gate open but cannot complete a chain,
    /// so a lone opener, or an opener plus quads, pays nothing. No B2B, no PC bonus.
    pub fn gate() -> Rules {
        Rules { gate: true, burst_window: 2, tst_attack: 10, ..Rules::normal() }
    }

    /// gate_b2b: the gate rules, plus the B2B bonus on every spin clear of a paid chain,
    /// so digging with plain clears (which break B2B) costs future chain value.
    /// TetrisBattle's rules (see the `battle` field).
    pub fn battle() -> Rules {
        Rules { battle: true, all_spin: false, b2b: B2bMode::Tetrio, perfect_clear_bonus: 10, ..Rules::default() }
    }

    /// TetrisBattle's clear_outcome: TETR.IO (Tetra League) attack. `combo_after` is the
    /// combo after this clear (0 = first clear of a streak), `b2b_after` the B2B count
    /// after it. Minis count as difficult and earn the B2B level; the level is added
    /// before the combo multiplier (1 + combo / 4), floored; a clear worth 0 sends
    /// floor(ln(1 + 1.25 combo)).
    pub fn battle_attack(&self, spin: Spin, lines: u32, combo_after: i32, b2b_after: i32, perfect: bool) -> i32 {
        if lines == 0 {
            return 0;
        }
        let base = match (spin, lines) {
            (Spin::None, 1) => 0,
            (Spin::None, 2) => 1,
            (Spin::None, 3) => 2,
            (Spin::Mini, 1) => 0,
            (Spin::Mini, 2) => 1,
            (_, 1) => 2,
            (_, 2) => 4,
            (_, 3) => 6,
            (_, l) => l as i32, // quads (spin or not) fall back to the line count: 4
        };
        let difficult = lines == 4 || spin != Spin::None;
        let level = if difficult { tetrio_b2b_level(b2b_after) } else { 0 };
        let mut attack = if base + level == 0 {
            (1.25 * combo_after as f64).ln_1p().floor() as i32
        } else {
            combo_after * (base + level) / 4 + base + level
        };
        if perfect {
            attack += self.perfect_clear_bonus;
        }
        attack
    }

    pub fn gate_b2b() -> Rules {
        Rules { gate_b2b: true, ..Rules::gate() }
    }

    /// Value of a clear that joins a gate chain (0 = it does not join).
    pub fn gate_value(&self, spin: Spin, lines: u32) -> i32 {
        match (spin, lines) {
            (Spin::Full, 2) => 4,
            (Spin::Full, 3) => self.tst_attack,
            (_, 4) => 1,
            _ => 0,
        }
    }

    /// One placement under gate rules: (attack credited now, the new state). A chain's
    /// payout is certain once it holds two spin clears, so it is credited then (and each
    /// later member as it joins) rather than when the window closes -- the same total,
    /// and the search sees it as soon as it is locked in.
    pub fn gate_step(&self, st: GateState, spin: Spin, lines: u32) -> (i32, GateState) {
        self.gate_step_b2b(st, spin, lines, 0)
    }

    /// `gate_step`, with `bonus` (the B2B bonus this clear earns) added to a spin clear's
    /// value under gate_b2b.
    pub fn gate_step_b2b(&self, st: GateState, spin: Spin, lines: u32, bonus: i32) -> (i32, GateState) {
        let (left, spins, held) = st;
        let base = self.gate_value(spin, lines);
        let v = if base > 0 && spin == Spin::Full { base + bonus } else { base };
        let is_spin = (spin == Spin::Full) as u8;
        if left > 0 {
            if v > 0 {
                let n = spins.saturating_add(is_spin);
                if spins >= 2 {
                    (v, (self.burst_window, n, 0))
                } else if n >= 2 {
                    (held as i32 + v, (self.burst_window, n, 0))
                } else {
                    (0, (self.burst_window, n, held.saturating_add(v as u8)))
                }
            } else if left == 1 {
                (0, (0, 0, 0))
            } else {
                (0, (left - 1, spins, held))
            }
        } else if v > 0 && is_spin == 1 {
            (0, (self.burst_window, 1, v as u8))
        } else {
            (0, (0, 0, 0))
        }
    }

    /// Does this clear arm the burst doubling for the next `burst_window` placements?
    pub fn burst_arms(&self, spin: Spin, lines: u32) -> bool {
        burst_qualifies(spin, lines) && (self.burst_tss_arms || !(spin == Spin::Full && lines == 1))
    }

    /// One placement of the burst state: `left` = placements still armed. Returns
    /// (this clear's base is doubled, the new `left`). Always (false, 0) without burst.
    pub fn burst_step(&self, left: u8, spin: Spin, lines: u32) -> (bool, u8) {
        if !self.burst {
            return (false, 0);
        }
        let doubled = left > 0 && burst_qualifies(spin, lines);
        let next = if self.burst_arms(spin, lines) { self.burst_window } else { left.saturating_sub(1) };
        (doubled, next)
    }

    pub fn allspin() -> Rules {
        Rules { all_spin: true, promote_minis: true, repeat_penalty: true, ..Rules::default() }
    }
}

/// Clear type for the repeat penalty: piece + lines + spin. All non-spin clears of the
/// same size share one type whatever the piece (double -> double repeats, single ->
/// double does not); spins are per piece (TSD -> ZSD does not repeat). 0 = no clear.
pub fn clear_type(piece: Piece, spin: Spin, lines: u32) -> u8 {
    if lines == 0 {
        0
    } else if spin == Spin::None {
        lines as u8
    } else {
        ((piece as u8 + 1) << 3) | ((spin as u8) << 6) | lines as u8
    }
}

impl Rules {
    /// AttackTable.attack_for. `b2b_after` is the chain value after this clear was
    /// counted (Game.lock_piece updates b2b before scoring).
    pub fn attack(&self, piece: Piece, spin: Spin, lines: u32, b2b_after: i32, perfect: bool) -> i32 {
        self.attack_with(piece, spin, lines, b2b_after, perfect, false)
    }

    /// `attack`, plus the burst state: `armed` = the previous placement was a
    /// qualifying clear. Only used when `self.burst` is on.
    pub fn attack_with(&self, piece: Piece, spin: Spin, lines: u32, b2b_after: i32, perfect: bool, armed: bool) -> i32 {
        if lines == 0 {
            return 0;
        }
        let l = lines as usize;
        let mut out = if spin == Spin::Mini {
            // Every piece's Mini sends nothing (AttackTable.t_spin_mini_attack).
            0
        } else if spin == Spin::Full && piece == Piece::T {
            [0, 2, 4, self.tst_attack, 0][l.min(4)]
        } else if spin != Spin::None {
            [0, 2, 4, 6, 8][l.min(4)]
        } else {
            [0, 0, 1, 2, 4][l.min(4)]
        };
        if self.burst && armed && burst_qualifies(spin, lines) {
            out *= 2;
        }
        // A Mini keeps the chain alive (is_difficult) but earns no bonus itself.
        if is_difficult(spin, lines) && b2b_after >= 1 && spin != Spin::Mini {
            let bonus = self.b2b.bonus(b2b_after);
            out += self.b2b_cap.map_or(bonus, |c| bonus.min(c));
        }
        if perfect {
            out += self.perfect_clear_bonus;
        }
        out
    }
}

/// ClearResult.is_difficult: quads and any spin that cleared lines keep B2B alive.
pub fn is_difficult(spin: Spin, lines: u32) -> bool {
    lines == 4 || (lines > 0 && spin != Spin::None)
}

/// Burst rules: clears that start or extend a burst -- full spins, quads, and Minis
/// that clear 2 or 3 rows (worth 0 themselves). Mini singles, plain clears and
/// placements that clear nothing break it.
pub fn burst_qualifies(spin: Spin, lines: u32) -> bool {
    match spin {
        Spin::Full => lines > 0,
        Spin::Mini => lines == 2 || lines == 3,
        Spin::None => lines == 4,
    }
}

/// Game.lock_piece's chain update: -1 none, 0 first difficult clear, >= 1 chain.
pub fn next_b2b(b2b: i32, spin: Spin, lines: u32) -> i32 {
    if lines == 0 {
        b2b
    } else if is_difficult(spin, lines) {
        b2b + 1
    } else {
        -1
    }
}

/// Human-readable clear name, as ClearResult.describe().
pub fn describe(piece: Piece, spin: Spin, lines: u32) -> String {
    let count = ["Zero", "Single", "Double", "Triple", "Quad"][lines.min(4) as usize];
    if spin != Spin::None {
        let mini = if spin == Spin::Mini { " Mini" } else { "" };
        format!("{}-Spin{} {}", piece.letter().to_ascii_uppercase(), mini, count)
    } else if lines > 0 {
        count.to_string()
    } else {
        String::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The worked examples from the burst-rules spec. B2B is off (chain 0) so the
    /// base-attack doubling is isolated, then checked with a live chain.
    #[test]
    fn burst_attack() {
        let r = Rules::burst();
        let (t, i, s) = (Piece::T, Piece::I, Piece::S);
        let at = |p, sp, l, armed| r.attack_with(p, sp, l, 0, false, armed);
        assert_eq!(at(t, Spin::Full, 3, false), 10, "TST is 10");
        assert_eq!(at(i, Spin::None, 4, false) + at(i, Spin::None, 4, true), 4 + 8, "quad -> quad");
        assert_eq!(at(t, Spin::Full, 2, false) + at(t, Spin::Full, 2, true), 4 + 8, "TSD -> TSD");
        assert_eq!(at(t, Spin::Full, 2, false) + at(t, Spin::Full, 3, true), 24, "TSD -> TST");
        assert_eq!(at(t, Spin::Full, 3, false) + at(t, Spin::Full, 2, true), 18, "TST -> TSD");
        assert_eq!(at(t, Spin::Full, 1, false) + at(t, Spin::Full, 2, true), 10, "TSS -> TSD");
        assert_eq!(at(t, Spin::Full, 2, false) + at(t, Spin::Full, 1, true), 8, "TSD -> TSS");
        assert_eq!(at(s, Spin::Mini, 2, true), 0, "a Mini is 0, doubled or not");
        assert_eq!(at(t, Spin::None, 2, true), 1, "plain clears are never doubled");
        // The B2B bonus is added after the doubling: TSD doubled (8) + chain-4 bonus (2).
        assert_eq!(r.attack_with(t, Spin::Full, 2, 4, false, true), 10);
        // And not in normal rules.
        assert_eq!(Rules::normal().attack_with(t, Spin::Full, 2, 0, false, true), 4);
        assert_eq!(Rules::normal().attack(t, Spin::Full, 3, 0, false), 6);

        assert!(burst_qualifies(Spin::Mini, 2) && burst_qualifies(Spin::Mini, 3));
        assert!(!burst_qualifies(Spin::Mini, 1), "Mini singles break the chain");
        assert!(!burst_qualifies(Spin::None, 3) && !burst_qualifies(Spin::Full, 0));
        assert!(burst_qualifies(Spin::None, 4) && burst_qualifies(Spin::Full, 1));
    }

    /// Runs a sequence of placements (spin, lines) through burst_step; returns which
    /// clears were doubled.
    fn doubled(r: &Rules, seq: &[(Spin, u32)]) -> Vec<bool> {
        let mut left = 0;
        seq.iter().map(|&(sp, l)| { let (d, n) = r.burst_step(left, sp, l); left = n; d }).collect()
    }

    #[test]
    fn burst_windows() {
        let (b1, b2) = (Rules::burst(), Rules::burst2());
        let (tsd, tss, quad, drop, plain) = ((Spin::Full, 2), (Spin::Full, 1), (Spin::None, 4), (Spin::None, 0), (Spin::None, 2));
        // Original burst: straight after only; a TSS arms.
        assert_eq!(doubled(&b1, &[tsd, tsd]), [false, true]);
        assert_eq!(doubled(&b1, &[tsd, drop, tsd]), [false, false, false]);
        assert_eq!(doubled(&b1, &[tss, quad]), [false, true]);
        // burst2: one placement in between is fine, two are not.
        assert_eq!(doubled(&b2, &[tsd, drop, tsd]), [false, false, true]);
        assert_eq!(doubled(&b2, &[tsd, plain, quad]), [false, false, true]);
        assert_eq!(doubled(&b2, &[tsd, drop, drop, tsd]), [false, false, false, false]);
        // A TSS does not arm, but is doubled when armed; it then uses up one placement.
        assert_eq!(doubled(&b2, &[tss, quad]), [false, false]);
        assert_eq!(doubled(&b2, &[quad, tss, tsd]), [false, true, true]);
        assert_eq!(doubled(&b2, &[quad, tss, drop, tsd]), [false, true, false, false]);
        // Chains: every arming clear re-arms the full window.
        assert_eq!(doubled(&b2, &[tsd, drop, tsd, drop, quad]), [false, false, true, false, true]);
        assert_eq!(doubled(&Rules::normal(), &[tsd, tsd]), [false, false]);
    }

    fn gate_credits(seq: &[(Spin, u32)]) -> Vec<i32> {
        let r = Rules::gate();
        let mut st = (0, 0, 0);
        seq.iter().map(|&(sp, l)| { let (c, n) = r.gate_step(st, sp, l); st = n; c }).collect()
    }

    #[test]
    fn gate_chains() {
        let (tsd, tst, tss, quad, drop, plain) =
            ((Spin::Full, 2), (Spin::Full, 3), (Spin::Full, 1), (Spin::None, 4), (Spin::None, 0), (Spin::None, 2));
        let mini2 = (Spin::Mini, 2);
        // C-spin: TST then TSD pays 10 + 4, credited once the chain has two members.
        assert_eq!(gate_credits(&[tst, tsd]), [0, 14]);
        assert_eq!(gate_credits(&[tst, drop, tsd]), [0, 0, 14], "one placement in between");
        assert_eq!(gate_credits(&[tst, drop, drop, tsd]), [0, 0, 0, 0], "the window ran out; the TSD opens anew");
        assert_eq!(gate_credits(&[tsd, quad, quad]), [0, 0, 0], "quads alone cannot complete a chain");
        assert_eq!(gate_credits(&[tsd, quad, tsd]), [0, 0, 9], "but they hold the gate open and add 1");
        assert_eq!(gate_credits(&[tsd, tsd, quad]), [0, 8, 1], "and add 1 to a complete chain");
        assert_eq!(gate_credits(&[tsd, quad, drop, drop, tsd]), [0, 0, 0, 0, 0], "a quad restarts the window, then it runs out");
        assert_eq!(gate_credits(&[quad, tsd]), [0, 0], "a quad does not open a gate");
        assert_eq!(gate_credits(&[tsd]), [0], "a lone opener pays nothing");
        assert_eq!(gate_credits(&[tss, tsd, plain, mini2, tsd]), [0, 0, 0, 0, 0], "TSS, plain, Mini: neutral, window spent");
        assert_eq!(gate_credits(&[tsd, plain, tsd, tss, tst]), [0, 0, 8, 0, 10], "joins restart the window");
        assert_eq!(gate_credits(&[tst, quad, quad, tsd]), [0, 0, 0, 16], "TST, two quads, TSD: 10 + 1 + 1 + 4");
        // Everything else scores nothing on its own.
        let r = Rules::gate();
        assert_eq!(r.gate_step((0, 0, 0), Spin::None, 4).0, 0);
    }

    /// gate_b2b: each spin clear of a paid chain adds its B2B bonus (given per clear by
    /// the caller); quads add none.
    #[test]
    fn gate_b2b_bonus() {
        let r = Rules::gate_b2b();
        let mut st = (0, 0, 0);
        let mut run = |sp, l, bonus| { let (c, n) = r.gate_step_b2b(st, sp, l, bonus); st = n; c };
        assert_eq!(run(Spin::Full, 2, 1), 0, "opener held: 4 + 1");
        assert_eq!(run(Spin::None, 4, 1), 0, "a quad holds the gate: +1, no bonus");
        assert_eq!(run(Spin::Full, 2, 2), 5 + 1 + 6, "pays (4+1) + 1 + (4+2)");
        assert_eq!(run(Spin::Full, 3, 2), 12, "a later TST: 10 + 2");
    }
}
