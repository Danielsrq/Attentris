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
    /// attack_table.gd b2b_bonus_for() with b2b_weight 1.0: floor(sqrt(chain)). Default.
    Sqrt,
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
}

impl Default for Rules {
    fn default() -> Self {
        Rules {
            b2b: B2bMode::Sqrt,
            all_spin: true,
            perfect_clear_bonus: 10,
            promote_minis: false,
            repeat_penalty: false,
            wound_clears: 10,
            wound_messiness: 0.5,
            non_t_mini: false,
            b2b_cap: None,
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
        if lines == 0 {
            return 0;
        }
        let l = lines as usize;
        let mut out = if spin == Spin::Mini {
            // Every piece's Mini sends nothing (AttackTable.t_spin_mini_attack).
            0
        } else if spin == Spin::Full && piece == Piece::T {
            [0, 2, 4, 6, 0][l.min(4)]
        } else if spin != Spin::None {
            [0, 2, 4, 6, 8][l.min(4)]
        } else {
            [0, 0, 1, 2, 4][l.min(4)]
        };
        // A Mini keeps the chain alive (is_difficult) but earns no bonus itself.
        if is_difficult(spin, lines) && b2b_after >= 1 && spin != Spin::Mini {
            let bonus = match self.b2b {
                B2bMode::Flat => 1,
                B2bMode::Sqrt => (b2b_after as f64).sqrt().floor() as i32,
            };
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
