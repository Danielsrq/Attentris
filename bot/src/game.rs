//! A full game driven by the searcher, plus the JSONL writer for the Godot replay
//! viewer (replay.gd / core/replay_log.gd). Each record is one placement: the board at
//! SPAWN time, the piece, queue and hold, and where the piece landed -- the same
//! schema as data/zenith_*.jsonl, so the viewer needs no simulation to show it.

use std::fmt::Write as _;

use crate::board::Board;
use crate::rules::{describe, Piece, Spin, ALL_PIECES, H, W};
use crate::search::{apply, Searcher};

pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    /// Raw generator state, for resumable runs.
    pub fn state(&self) -> u64 {
        self.0
    }
    pub fn from_state(s: u64) -> Rng {
        Rng(s)
    }
    pub fn next(&mut self) -> u64 {
        // xorshift64*
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    pub fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    pub fn chance(&mut self, p: f64) -> bool {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64 * 1.0 < p
    }
}

/// 7-bag with the guideline opening rule (first piece never S, Z or O), as
/// core/randomizer.gd.
pub struct Bag {
    rng: Rng,
    seq: Vec<Piece>,
}

impl Bag {
    pub fn new(seed: u64) -> Bag {
        Bag { rng: Rng::new(seed), seq: Vec::new() }
    }
    fn refill(&mut self) {
        let mut bag = ALL_PIECES;
        for i in (1..bag.len()).rev() {
            let j = self.rng.below(i as u64 + 1) as usize;
            bag.swap(i, j);
        }
        if self.seq.is_empty() && matches!(bag[0], Piece::S | Piece::Z | Piece::O) {
            let k = bag.iter().position(|p| !matches!(p, Piece::S | Piece::Z | Piece::O)).unwrap();
            bag.swap(0, k);
        }
        self.seq.extend_from_slice(&bag);
    }
    /// Piece number `i` of the whole game's sequence.
    pub fn get(&mut self, i: usize) -> Piece {
        while self.seq.len() <= i {
            self.refill();
        }
        self.seq[i]
    }
    /// Pieces that draw number `i` could be, given only the bag structure (what a
    /// player knows): the ones its bag has not dealt before position `i`.
    pub fn possible(&mut self, i: usize) -> Vec<Piece> {
        let start = i / 7 * 7;
        let mut left = ALL_PIECES.to_vec();
        for j in start..i {
            let p = self.get(j);
            left.retain(|&q| q != p);
        }
        left
    }
}

#[derive(Clone, Debug)]
pub struct GarbageConfig {
    /// Inject every `every` pieces (0 = off).
    pub every: usize,
    /// Rows per batch: `rows`, or a uniform draw from rows_min..=rows when rows_min < rows.
    pub rows: usize,
    pub rows_min: usize,
    /// Chance each subsequent row moves its hole (0 = clean, 1 = full cheese).
    pub messiness: f64,
    /// Chance a batch is cheese (every row a new hole) instead of clean ("mix" = 0.5).
    /// With an aligned well (batch_move < 1) a cheese batch leaves the clean column alone.
    pub cheese: f64,
    /// Chance a batch's first row moves from the previous batch's hole column (1 = every
    /// batch in a new column, as TETR.IO; lower keeps the garbage well aligned).
    pub batch_move: f64,
    /// Rows of cheese on the starting board.
    pub start_rows: usize,
}

impl Default for GarbageConfig {
    fn default() -> Self {
        GarbageConfig { every: 0, rows: 0, rows_min: 0, messiness: 0.0, cheese: 0.0, batch_move: 1.0, start_rows: 0 }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Stats {
    pub pieces: u64,
    pub attack: i64,
    pub lines: u64,
    pub died: bool,
    pub quads: u64,
    pub tspin_single: u64,
    pub tsd: u64,
    pub tst: u64,
    pub tspin_mini: u64,
    pub other_spins: u64,
    pub singles: u64,
    pub doubles: u64,
    pub triples: u64,
    pub perfect: u64,
    pub max_b2b: i32,
    pub garbage_in: u64,
    /// Every garbage batch this game: [piece number, rows, 1 if cheese else 0].
    pub garbage_log: Vec<[u32; 3]>,
    pub garbage_cleared: u64,
    pub max_height: i32,
    /// `bot play-ext`: replies that matched no legal move (the bot's own pick was played).
    pub ext_illegal: u64,
    pub nodes: u64,
    /// Wall-clock time of each search call, in microseconds.
    pub think_us: Vec<u32>,
    /// Moves where the previous search had a prediction, and how many it got right.
    pub plan_checks: u64,
    pub plan_kept: u64,
    /// `bot play --commit K`: moves played from the previous search's plan, without a search.
    pub committed: u64,
    /// All-spin mode: clears that repeated the previous type, and the most wound
    /// lines live at once.
    pub repeats: u64,
    pub max_wounds: u32,
    /// Spin clears of any piece by rows cleared (index 1..=3), and plain clears likewise.
    pub spin_by_lines: [u64; 5],
    /// Clears the bot believed were spins that the real rules scored as plain clears.
    pub phantom_spins: u64,
    pub plain_by_lines: [u64; 5],
    /// Per game, per piece: (attack, flags) with bit 0 = a burst-qualifying clear and
    /// bit 1 = its base attack was doubled. For window APP and burst metrics.
    pub seqs: Vec<Vec<(u16, u8)>>,
    /// Vertical T-slot template diagnostics. In search: V evaluations, how many saw a
    /// template slot, how many a ready one (2+ lines).
    pub tmpl_search: [u64; 3],
    /// On the real board before each move: a ready template slot was there; the bot then
    /// placed a T exactly into it; a T-spin double followed within 2 placements.
    pub tmpl_ready: u64,
    pub tmpl_cashed: u64,
    pub tmpl_followups: u64,
    /// Full vertical T-spins clearing 2+ rows the bot made, and how many of them the
    /// template had found on the board beforehand.
    pub vert_spins: u64,
    pub vert_spins_seen: u64,
}

impl Stats {
    pub fn add(&mut self, o: &Stats) {
        self.pieces += o.pieces;
        self.attack += o.attack;
        self.lines += o.lines;
        self.died |= o.died;
        self.quads += o.quads;
        self.tspin_single += o.tspin_single;
        self.tsd += o.tsd;
        self.tst += o.tst;
        self.tspin_mini += o.tspin_mini;
        self.other_spins += o.other_spins;
        self.singles += o.singles;
        self.doubles += o.doubles;
        self.triples += o.triples;
        self.perfect += o.perfect;
        self.max_b2b = self.max_b2b.max(o.max_b2b);
        self.garbage_in += o.garbage_in;
        self.garbage_cleared += o.garbage_cleared;
        self.max_height = self.max_height.max(o.max_height);
        self.nodes += o.nodes;
        self.think_us.extend_from_slice(&o.think_us);
        self.plan_checks += o.plan_checks;
        self.plan_kept += o.plan_kept;
        self.committed += o.committed;
        self.repeats += o.repeats;
        self.max_wounds = self.max_wounds.max(o.max_wounds);
        self.phantom_spins += o.phantom_spins;
        self.seqs.extend(o.seqs.iter().cloned());
        for i in 0..3 {
            self.tmpl_search[i] += o.tmpl_search[i];
        }
        self.tmpl_ready += o.tmpl_ready;
        self.tmpl_cashed += o.tmpl_cashed;
        self.tmpl_followups += o.tmpl_followups;
        self.vert_spins += o.vert_spins;
        self.vert_spins_seen += o.vert_spins_seen;
        for i in 0..5 {
            self.spin_by_lines[i] += o.spin_by_lines[i];
            self.plain_by_lines[i] += o.plain_by_lines[i];
        }
    }
    pub fn app(&self) -> f64 {
        self.attack as f64 / self.pieces.max(1) as f64
    }
}

/// Cell letters for the viewer; the bitboard alone cannot colour pieces.
struct Colors([u8; (W * H) as usize]);

impl Colors {
    fn row_string(&self, y: usize) -> String {
        self.0[y * W as usize..(y + 1) * W as usize].iter().map(|&c| c as char).collect()
    }
    fn clear_rows(&mut self, mask: u64) {
        let w = W as usize;
        let mut out = [b'.'; (W * H) as usize];
        let mut write = H as usize;
        for y in (0..H as usize).rev() {
            if mask >> y & 1 != 0 {
                continue;
            }
            write -= 1;
            out[write * w..(write + 1) * w].copy_from_slice(&self.0[y * w..(y + 1) * w]);
        }
        self.0 = out;
    }
    /// Garbage (and wound lines) enter at the very bottom, as Board::push_garbage.
    fn push_rows(&mut self, rows: &[[u8; W as usize]]) {
        let (w, h, n) = (W as usize, H as usize, rows.len());
        self.0.copy_within(n * w..h * w, 0);
        for (i, row) in rows.iter().enumerate() {
            let y = h - 1 - i;
            self.0[y * w..(y + 1) * w].copy_from_slice(row);
        }
    }
    fn push_garbage(&mut self, holes: &[i32]) {
        let rows: Vec<[u8; W as usize]> = holes.iter().map(|&h| garbage_row(h)).collect();
        self.push_rows(&rows);
    }
    /// Wound lines show as 'W'; once one turns into garbage it shows as 'G' with its hole.
    fn sync_wounds(&mut self, b: &Board) {
        let w = W as usize;
        for y in 0..H as usize {
            let is_w = self.0[y * w] == b'W';
            if is_w && b.wound_left[y] == 0 {
                self.0[y * w..(y + 1) * w].copy_from_slice(&garbage_row(b.wound_hole[y] as i32));
            }
        }
    }
}

fn garbage_row(hole: i32) -> [u8; W as usize] {
    let mut r = [b'G'; W as usize];
    r[hole as usize] = b'.';
    r
}

/// The colour grid must agree with the bitboard cell for cell ('W' rows are solid).
fn check_colors(c: &Colors, b: &Board) {
    for y in 0..H as usize {
        for x in 0..W as usize {
            let filled = b.rows[y] >> x & 1 != 0;
            let shown = c.0[y * W as usize + x] != b'.';
            assert_eq!(filled, shown, "colour grid out of sync at ({}, {})", x, y);
        }
    }
}

/// `bot play --commit K`: moves per search (1 = search every move).
pub static COMMIT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(1);

pub struct GameConfig {
    pub pieces: usize,
    pub seed: u64,
    pub garbage: GarbageConfig,
    pub label: String,
    /// Collect viewer JSONL.
    pub record: bool,
    /// The rules the game actually applies, when they differ from the rules the bot
    /// searches with (`--believe`). None = the bot knows the real rules.
    pub env_rules: Option<crate::rules::Rules>,
    /// Write AtTenTris dataset records (one per position, every root placement with its
    /// search scores) instead of viewer JSONL. Needs `SearchConfig::roots`.
    pub dataset: bool,
    /// Moves come from another process over stdin / stdout (`bot play-ext`); the searcher
    /// only lists the legal moves (`SearchConfig::roots`).
    pub external: bool,
    /// `bot play-ext --shortlist`: another process names the first moves worth searching
    /// (stdin: {"moves": [{"hold", "cells"}, ...]}), and this searcher picks among them.
    pub shortlist: bool,
    /// With `dataset`: also run a root-diverse search (keeping this many lines per first
    /// move) so every legal move's record gets a search score. 0 = survivors only.
    pub all_roots: usize,
    /// `bot play-ext --versus`: garbage comes from the opponent through the referee (stdout: a
    /// lock line after every placement; stdin: the garbage to add before the next piece).
    pub versus: bool,
}

pub fn play(searcher: &mut Searcher, cfg: &GameConfig) -> (Stats, String) {
    let env = cfg.env_rules.unwrap_or(searcher.rules);
    let mut bag = Bag::new(cfg.seed);
    let mut grng = Rng::new(cfg.seed ^ 0xC0FFEE);
    let mut board = Board::default();
    let mut colors = Colors([b'.'; (W * H) as usize]);
    let mut hold: Option<Piece> = None;
    let mut b2b = -1;
    let mut drawn = 0usize; // index of the next piece to spawn
    let mut stats = Stats::default();
    let mut out = String::new();
    let mut garbage_rows_total = 0u64;
    let mut garbage_hole = grng.below(W as u64) as i32;
    let mut garbage_left = 0u64; // garbage rows still on the board (approximate: counted by 'G' rows)

    let inject = |board: &mut Board, colors: &mut Colors, n: usize, grng: &mut Rng, hole: &mut i32, messiness: f64, cheese: bool| {
        // A batch starts in a new column with chance `batch_move` (1 = always, as TETR.IO:
        // one attack = one clean chunk); `messiness` is the chance each further row moves.
        let mut holes = Vec::with_capacity(n);
        for i in 0..n {
            // With batch_move < 1 (an aligned well), a batch that moves goes to an adjacent
            // column -- inward at a wall; otherwise any other column, as before.
            let aligned = cfg.garbage.batch_move < 1.0 && !cheese;
            let moves = cheese || if i == 0 { !aligned || grng.chance(cfg.garbage.batch_move) } else { grng.chance(messiness) };
            if moves && i == 0 && aligned && *hole >= 0 && *hole < W {
                *hole = if *hole == 0 {
                    1
                } else if *hole == W - 1 {
                    W - 2
                } else if grng.chance(0.5) {
                    *hole - 1
                } else {
                    *hole + 1
                };
            } else if moves {
                let mut h = grng.below(W as u64 - 1) as i32;
                if h >= *hole {
                    h += 1;
                }
                *hole = h;
            }
            holes.push(*hole);
        }
        colors.push_garbage(&holes);
        board.push_garbage(&holes);
    };
    if cfg.garbage.start_rows > 0 {
        let n = cfg.garbage.start_rows;
        // Starting cheese is always messy: every row gets its own hole.
        let mut holes = Vec::new();
        for _ in 0..n {
            let mut h = grng.below(W as u64 - 1) as i32;
            if h >= garbage_hole {
                h += 1;
            }
            garbage_hole = h;
            holes.push(h);
        }
        board.push_garbage(&holes);
        colors.push_garbage(&holes);
        garbage_rows_total += n as u64;
        garbage_left += n as u64;
        stats.garbage_in += n as u64;
    }

    let shapes = crate::rules::Shapes::for_rules(&env);
    let mut diverse = (cfg.dataset && cfg.all_roots > 0).then(|| {
        Searcher::new(crate::search::SearchConfig { root_keep: cfg.all_roots, ..searcher.cfg.clone() }, searcher.weights.clone(), searcher.rules)
    });
    let mut expected: Option<(Piece, [(i32, i32); 4])> = None;
    // `--commit K` (COMMIT): after a search, its plan's next K-1 moves are played without searching
    let commit = COMMIT.load(std::sync::atomic::Ordering::Relaxed);
    let mut committed: Vec<crate::search::Step> = Vec::new();
    let counts_at_start = crate::eval::template_counts();
    let mut followup_left = 0u8;
    let mut incoming: Vec<(usize, i32)> = Vec::new();
    for n in 0..cfg.pieces {
        if !incoming.is_empty() {
            // versus: the opponent's garbage, one hole column per attack
            for &(rows, hole) in &incoming {
                let holes = vec![hole.clamp(0, W - 1); rows];
                board.push_garbage(&holes);
                colors.push_garbage(&holes);
                garbage_rows_total += rows as u64;
                garbage_left += rows as u64;
                stats.garbage_in += rows as u64;
                stats.garbage_log.push([n as u32, rows as u32, 0]);
            }
            incoming.clear();
        }
        if cfg.garbage.every > 0 && n > 0 && n % cfg.garbage.every == 0 {
            let g = &cfg.garbage;
            let n_rows = if g.rows_min > 0 && g.rows_min < g.rows { g.rows_min + grng.below((g.rows - g.rows_min + 1) as u64) as usize } else { g.rows };
            let cheese = g.cheese > 0.0 && grng.chance(g.cheese);
            if cheese && g.batch_move < 1.0 {
                // Cheese over an aligned well: its holes do not move the well's column.
                let mut tmp = garbage_hole;
                inject(&mut board, &mut colors, n_rows, &mut grng, &mut tmp, 1.0, true);
            } else {
                let messiness = if g.cheese > 0.0 { if cheese { 1.0 } else { 0.0 } } else { g.messiness };
                inject(&mut board, &mut colors, n_rows, &mut grng, &mut garbage_hole, messiness, false);
            }
            garbage_rows_total += n_rows as u64;
            stats.garbage_log.push([n as u32, n_rows as u32, cheese as u32]);
            garbage_left += n_rows as u64;
            stats.garbage_in += n_rows as u64;
            // TetrisBattle checks the top visible row after garbage arrives, too.
            if env.battle && board.rows[crate::rules::BUFFER as usize] != 0 {
                stats.died = true;
                break;
            }
        }
        let spawned = bag.get(drawn);
        let queue: Vec<Piece> = (0..6).map(|i| bag.get(drawn + i)).collect();
        let bag_next = bag.possible(drawn + 6);
        let spawn_hold = hold;
        // Board as it stood at SPAWN, which is what the viewer draws the piece onto.
        let spawn_rows: Vec<String> =
            if cfg.record { (0..H as usize).map(|y| colors.row_string(y)).collect() } else { Vec::new() };
        // Wound lines at spawn, oldest first: board row and clears left before it heals.
        let wounds_json = board
            .wound_rows()
            .map(|(y, left, hole)| format!("{{\"row\":{},\"left\":{},\"hole\":{}}}", y, left, hole))
            .collect::<Vec<_>>()
            .join(",");

        if cfg.shortlist {
            searcher.root_allow = read_shortlist(n, &board, spawned, &queue[1..], hold, b2b);
        }
        let nodes_before = searcher.nodes;
        let t0 = std::time::Instant::now();
        // A committed move is played when nothing changed the board since the search that planned
        // it (no garbage this turn) and its piece is the one the plan expected (current or hold).
        let garbage_now = cfg.garbage.every > 0 && n > 0 && n % cfg.garbage.every == 0;
        let next_committed = if commit > 1 && !committed.is_empty() && !garbage_now && !cfg.external && !cfg.shortlist && !cfg.dataset {
            let s = committed.remove(0);
            let expect = if !s.use_hold { Some(spawned) } else if hold.is_some() { hold } else { Some(queue[1]) };
            if expect == Some(s.placement.piece) { Some(s) } else { None }
        } else {
            None
        };
        if next_committed.is_none() {
            committed.clear();
        }
        let mut d = match next_committed {
            Some(s) => {
                stats.committed += 1;
                crate::search::Decision { use_hold: s.use_hold, placement: s.placement, plan: committed.iter().copied().collect(),
                                          roots: Vec::new(), chosen_root: 0 }
            }
            None => match searcher.search(&board, &queue, hold, b2b, &bag_next) {
                Some(d) => {
                    if commit > 1 {
                        committed = d.plan.iter().skip(1).take(commit - 1).copied().collect();
                    }
                    d
                }
                None => {
                    stats.died = true;
                    break;
                }
            },
        };
        if cfg.external && !external_move(&mut d, n, &board, spawned, &queue[1..], hold, b2b, &bag_next, &shapes, &mut stats) {
            // the other process resigned (it sees no legal move: Cold Clear 2 topped out)
            stats.died = true;
            break;
        }
        stats.think_us.push(t0.elapsed().as_micros().min(u32::MAX as u128) as u32);
        stats.nodes += searcher.nodes - nodes_before;

        // Did the bot play what the previous search expected to play here? Only
        // comparable when nothing (garbage) changed the board in between.
        let injected = cfg.garbage.every > 0 && n > 0 && n % cfg.garbage.every == 0;
        if let Some((pp, cells)) = expected.take() {
            if !injected {
                stats.plan_checks += 1;
                if pp == d.placement.piece && cells == sorted_cells(&d.placement, &shapes) {
                    stats.plan_kept += 1;
                }
            }
        }
        if let Some(s) = d.plan.get(1) {
            expected = Some((s.placement.piece, sorted_cells(&s.placement, &shapes)));
        }
        let plan_json = if cfg.record { plan_to_json(&board, &d.plan, b2b, searcher, &shapes) } else { String::new() };
        if cfg.dataset {
            match diverse.as_mut().and_then(|dv| dv.search(&board, &queue, spawn_hold, b2b, &bag_next)) {
                Some(d2) => {
                    let mut dd = d.clone();
                    crate::search::merge_root_best(&mut dd, &d2);
                    dataset_record(&mut out, cfg, n, &board, spawned, &queue[1..], spawn_hold, b2b, &dd, &env, &shapes);
                }
                None => dataset_record(&mut out, cfg, n, &board, spawned, &queue[1..], spawn_hold, b2b, &d, &env, &shapes),
            }
        }

        // Advance the queue exactly as Game does: the spawned piece is consumed; an
        // empty-hold swap consumes one more.
        drawn += 1;
        if d.use_hold {
            if hold.is_none() {
                drawn += 1;
            }
            hold = Some(spawned);
        }
        let mut p = d.placement;
        if cfg.env_rules.is_some() && !env.battle {
            // The bot chose this move believing its own rules; the game scores it by
            // the real ones. Re-detect the spin: only a rotation can make one, and
            // under the real rules a non-T immobile spot may not count.
            p.spin = if p.kick < 0 {
                crate::rules::Spin::None
            } else {
                let cells = &shapes.cells[p.piece as usize][p.rot as usize];
                crate::movegen::detect_spin(&board, p.piece, cells, p.rot as usize, p.x as i32, p.y as i32, p.kick as i32, env.all_spin)
            };
        }
        let o = apply(&board, &p, b2b, &env, &shapes);
        // Template diagnostics on the real board.
        {
            let top = H - *board.heights().iter().max().unwrap();
            let pre = crate::eval::scan_tslots(&board, top).vertical;
            let mut slot_cells = pre.map(|s| s.cells);
            if let Some(c) = slot_cells.as_mut() {
                c.sort();
            }
            // Into the slot: the same stem (its 3-cell column), either nub side -- an imperial
            // cross is twisted in one way and turned 180 the other.
            let stem = |c: &[(i32, i32); 4]| -> Vec<(i32, i32)> {
                let x = (0..4).map(|i| c[i].0).max_by_key(|&x| c.iter().filter(|q| q.0 == x).count()).unwrap();
                c.iter().copied().filter(|q| q.0 == x).collect()
            };
            let played = sorted_cells(&p, &shapes);
            let into_slot = p.piece == Piece::T && slot_cells.map_or(false, |c| stem(&c) == stem(&played) && stem(&played).len() == 3);
            let ready = pre.map_or(false, |s| s.lines >= 2);
            stats.tmpl_ready += ready as u64;
            if ready && into_slot && o.lines >= 2 {
                stats.tmpl_cashed += 1;
                followup_left = 3;
            } else if followup_left > 0 && p.piece == Piece::T && o.spin == Spin::Full && o.lines == 2 && p.rot == 2 {
                stats.tmpl_followups += 1;
                followup_left = 0;
            }
            followup_left = followup_left.saturating_sub(1);
            if p.piece == Piece::T && o.spin == Spin::Full && (p.rot == 1 || p.rot == 3) && o.lines >= 2 {
                stats.vert_spins += 1;
                stats.vert_spins_seen += into_slot as u64;
            }
        }
        if o.lines > 0 && d.placement.spin != crate::rules::Spin::None && o.spin == crate::rules::Spin::None {
            stats.phantom_spins += 1;
        }
        if o.lock_out {
            stats.died = true;
            break;
        }
        // Colour grid follows the bitboard.
        for (x, y) in p.abs_cells(&shapes) {
            colors.0[(y * W + x) as usize] = p.piece.letter() as u8;
        }
        let g_before = count_garbage_rows(&colors, o.cleared);
        colors.clear_rows(o.cleared);
        // Wounds, in the order Board applied them: expired ones turn into garbage, then
        // a repeat inserts a new wound line at the bottom.
        let mut o_board = o.board;
        let mut wound_hole: Option<u8> = None;
        if o.repeat {
            // The real hole: messiness is the chance it moves from the previous wound's.
            let prev = board.last_wound_hole;
            let hole = if board.wounds == 0 && stats.repeats == 0 || grng.chance(env.wound_messiness) {
                let mut h = grng.below(W as u64 - 1) as u8;
                if h >= prev {
                    h += 1;
                }
                h
            } else {
                prev
            };
            o_board.wound_hole[H as usize - 1] = hole;
            o_board.last_wound_hole = hole;
            wound_hole = Some(hole);
            colors.push_rows(&[[b'W'; W as usize]]);
            stats.repeats += 1;
        }
        colors.sync_wounds(&o_board);
        check_colors(&colors, &o_board);
        stats.max_wounds = stats.max_wounds.max(o.board.wounds as u32);
        stats.garbage_cleared += g_before;
        garbage_left = garbage_left.saturating_sub(g_before);

        if stats.seqs.is_empty() {
            stats.seqs.push(Vec::new());
        }
        let qual = crate::rules::burst_qualifies(o.spin, o.lines) as u8;
        stats.seqs[0].push((o.attack.max(0) as u16, qual | (o.burst as u8) << 1));
        stats.pieces += 1;
        stats.attack += o.attack as i64;
        stats.lines += o.lines as u64;
        tally(&mut stats, p.piece, o.spin, o.lines);
        if o.lines > 0 && o.board.is_empty() {
            stats.perfect += 1;
        }
        stats.max_b2b = stats.max_b2b.max(o.b2b);
        let h = *o.board.heights().iter().max().unwrap();
        stats.max_height = stats.max_height.max(h);

        if cfg.record {
            write_record(&mut out, cfg, n, &spawn_rows,
                spawned, &queue[1..], spawn_hold, d.use_hold, &p, &shapes, &o, hold,
                (1..6).map(|i| bag.get(drawn + i - 1)).collect(), garbage_rows_total, h, garbage_left, &plan_json, &wounds_json, wound_hole);
        }
        board = o_board;
        b2b = o.b2b;
        if cfg.versus {
            match versus_lock(n, &o, &board, h, &shapes, &p) {
                Some(g) => incoming = g,
                None => break, // the referee ended the game (the opponent topped out)
            }
        }
    }
    let counts = crate::eval::template_counts();
    for i in 0..3 {
        stats.tmpl_search[i] = counts[i] - counts_at_start[i];
    }
    (stats, out)
}

/// One AtTenTris training position. Rows are bitmasks, top to bottom (bit x = column x),
/// as the replay boards; `cands` are the distinct legal first moves (same cells and hold
/// = one move): [hold, piece, rot, spin 0/1/2, lines, attack, ply1, best | null,
/// [x0, y0, ... x3, y3]]; `chosen` indexes the move the bot played.
pub fn dataset_record(out: &mut String, cfg: &GameConfig, n: usize, board: &Board, spawned: Piece, previews: &[Piece],
                  hold: Option<Piece>, b2b: i32, d: &crate::search::Decision, env: &crate::rules::Rules,
                  shapes: &crate::rules::Shapes) {
    use std::fmt::Write;
    let mut keys: Vec<(bool, [(i32, i32); 4])> = Vec::new();
    let mut cands: Vec<crate::search::RootScore> = Vec::new();
    let mut chosen = 0usize;
    for (i, r) in d.roots.iter().enumerate() {
        let key = (r.step.use_hold, sorted_cells(&r.step.placement, shapes));
        let k = match keys.iter().position(|q| *q == key) {
            Some(k) => {
                // a twin (another rotation / route to the same cells): keep the better scores
                let c = &mut cands[k];
                if r.ply1 > c.ply1 {
                    let best = c.best;
                    *c = *r;
                    c.best = omax(best, r.best);
                } else {
                    c.best = omax(c.best, r.best);
                }
                k
            }
            None => {
                keys.push(key);
                cands.push(*r);
                keys.len() - 1
            }
        };
        if i == d.chosen_root {
            chosen = k;
        }
    }
    let letter = |p: Piece| p.letter().to_ascii_uppercase();
    let _ = write!(out, "{{\"teacher\":\"{}\",\"seed\":{},\"n\":{},\"rows\":[{}],\"piece\":\"{}\",\"queue\":[{}],\"hold\":{},\"b2b\":{},\"cands\":[",
        cfg.label, cfg.seed, n,
        board.rows.iter().map(|r| r.to_string()).collect::<Vec<_>>().join(","),
        letter(spawned),
        previews.iter().take(5).map(|&p| format!("\"{}\"", letter(p))).collect::<Vec<_>>().join(","),
        hold.map_or("null".to_string(), |p| format!("\"{}\"", letter(p))), b2b);
    for (k, c) in cands.iter().enumerate() {
        let p = &c.step.placement;
        let o = crate::search::apply(board, p, b2b, env, shapes);
        let spin = match p.spin { Spin::None => 0, Spin::Mini => 1, _ => 2 };
        let cells = keys[k].1.iter().map(|(x, y)| format!("{},{}", x, y)).collect::<Vec<_>>().join(",");
        let _ = write!(out, "{}[{},\"{}\",{},{},{},{},{:.3},{},[{}]]", if k > 0 { "," } else { "" },
            c.step.use_hold as u8, letter(p.piece), p.rot, spin, o.lines, c.attack, c.ply1,
            c.best.map_or("null".to_string(), |b| format!("{:.3}", b)), cells);
    }
    let _ = writeln!(out, "],\"chosen\":{}}}", chosen);
}

fn write_state(n: usize, board: &Board, spawned: Piece, previews: &[Piece], hold: Option<Piece>, b2b: i32) {
    use std::io::Write;
    let up = |p: Piece| p.letter().to_ascii_uppercase();
    let mut o = std::io::stdout().lock();
    let _ = writeln!(o, "{{\"n\":{},\"rows\":[{}],\"piece\":\"{}\",\"queue\":[{}],\"hold\":{},\"b2b\":{}}}", n,
        board.rows.iter().map(|r| r.to_string()).collect::<Vec<_>>().join(","), up(spawned),
        previews.iter().take(5).map(|&p| format!("\"{}\"", up(p))).collect::<Vec<_>>().join(","),
        hold.map_or("null".to_string(), |p| format!("\"{}\"", up(p))), b2b);
    let _ = o.flush();
}

fn parse_move(m: &serde_json::Value) -> (bool, [(i32, i32); 4]) {
    let uh = m["hold"].as_i64().unwrap_or(0) != 0;
    let c: Vec<i32> = m["cells"].as_array().map(|a| a.iter().map(|x| x.as_i64().unwrap_or(-99) as i32).collect()).unwrap_or_default();
    let mut cells = [(-99, -99); 4];
    for (k, p) in c.chunks(2).take(4).enumerate() {
        if p.len() == 2 {
            cells[k] = (p[0], p[1]);
        }
    }
    cells.sort();
    (uh, cells)
}

/// `bot play-ext --shortlist`: send the state, read the shortlist of first moves.
fn read_shortlist(n: usize, board: &Board, spawned: Piece, previews: &[Piece], hold: Option<Piece>, b2b: i32) -> Vec<(bool, [(i32, i32); 4])> {
    use std::io::BufRead;
    write_state(n, board, spawned, previews, hold, b2b);
    let mut line = String::new();
    std::io::stdin().lock().read_line(&mut line).expect("read shortlist");
    let v: serde_json::Value = serde_json::from_str(&line).expect("shortlist JSON");
    v["moves"].as_array().map(|a| a.iter().map(parse_move).collect()).unwrap_or_default()
}

/// `bot play-ext --versus`: report the lock ({"lock": n, "attack", "lines", "spin" 0/1/2, "piece",
/// "b2b", "combo", "height", "cells"}) and read the garbage to add before the next piece
/// ({"garbage": [[rows, hole column], ...]}); None for {"stop": true} (the game ends, not as a death).
fn versus_lock(n: usize, o: &crate::search::Outcome, board: &Board, height: i32, shapes: &crate::rules::Shapes,
               p: &crate::board::Placement) -> Option<Vec<(usize, i32)>> {
    use std::io::{BufRead, Write};
    let spin = match o.spin { Spin::None => 0, Spin::Mini => 1, _ => 2 };
    {
        let mut out = std::io::stdout().lock();
        let cells = p.abs_cells(shapes).iter().map(|(x, y)| format!("{},{}", x, y)).collect::<Vec<_>>().join(",");
        // "rows": the board after this lock (before the referee's garbage), for replays (versus.py --record)
        let rows = board.rows.iter().map(|r| r.to_string()).collect::<Vec<_>>().join(",");
        let _ = writeln!(out, "{{\"lock\":{},\"attack\":{},\"lines\":{},\"spin\":{},\"piece\":\"{}\",\"b2b\":{},\"combo\":{},\"height\":{},\"cells\":[{}],\"rows\":[{}]}}",
            n, o.attack, o.lines, spin, p.piece.letter().to_ascii_uppercase(), o.b2b, board.combo, height, cells, rows);
        let _ = out.flush();
    }
    let mut line = String::new();
    std::io::stdin().lock().read_line(&mut line).expect("read garbage");
    let v: serde_json::Value = serde_json::from_str(&line).expect("garbage JSON");
    if v["stop"].as_bool().unwrap_or(false) {
        return None;
    }
    Some(v["garbage"].as_array().map(|a| a.iter().filter_map(|g| {
        let g = g.as_array()?;
        Some((g.first()?.as_u64()? as usize, g.get(1)?.as_i64()? as i32))
    }).filter(|&(r, _)| r > 0).collect()).unwrap_or_default())
}

/// `bot play-ext`: send the state, read the move, and play it if it is one of the legal moves
/// (several routes to the same cells: the highest-attack one). False when the other process
/// resigns ({"resign": true}).
fn external_move(d: &mut crate::search::Decision, n: usize, board: &Board, spawned: Piece, previews: &[Piece], hold: Option<Piece>,
                 b2b: i32, bag_next: &[Piece], shapes: &crate::rules::Shapes, stats: &mut Stats) -> bool {
    use std::io::{BufRead, Write};
    let up = |p: Piece| p.letter().to_ascii_uppercase();
    {
        let mut o = std::io::stdout().lock();
        // "bag": the pieces the draw after the last preview can be (7-bag state, for Cold Clear 2's start);
        // "combo": Board::combo (-1 none, 0 after the first clear of a streak; tracked under --combo / battle)
        let _ = writeln!(o, "{{\"n\":{},\"rows\":[{}],\"piece\":\"{}\",\"queue\":[{}],\"hold\":{},\"b2b\":{},\"combo\":{},\"bag\":[{}]}}", n,
            board.rows.iter().map(|r| r.to_string()).collect::<Vec<_>>().join(","), up(spawned),
            previews.iter().take(5).map(|&p| format!("\"{}\"", up(p))).collect::<Vec<_>>().join(","),
            hold.map_or("null".to_string(), |p| format!("\"{}\"", up(p))), b2b, board.combo,
            bag_next.iter().map(|&p| format!("\"{}\"", up(p))).collect::<Vec<_>>().join(","));
        let _ = o.flush();
    }
    let mut line = String::new();
    std::io::stdin().lock().read_line(&mut line).expect("read move");
    let v: serde_json::Value = serde_json::from_str(&line).expect("move JSON");
    if v["resign"].as_bool().unwrap_or(false) {
        return false;
    }
    let uh = v["hold"].as_i64().unwrap_or(0) != 0;
    let c: Vec<i32> = v["cells"].as_array().map(|a| a.iter().map(|x| x.as_i64().unwrap_or(-99) as i32).collect()).unwrap_or_default();
    let mut want: Vec<(i32, i32)> = c.chunks(2).filter(|p| p.len() == 2).map(|p| (p[0], p[1])).collect();
    want.sort();
    let pick = d.roots.iter()
        .filter(|r| r.step.use_hold == uh && sorted_cells(&r.step.placement, shapes).to_vec() == want)
        .max_by(|a, b| a.attack.total_cmp(&b.attack))
        .copied();
    match pick {
        Some(r) => {
            d.use_hold = r.step.use_hold;
            d.placement = r.step.placement;
            d.plan = vec![r.step];
        }
        None => stats.ext_illegal += 1,
    }
    true
}

fn omax(a: Option<f32>, b: Option<f32>) -> Option<f32> {
    match (a, b) {
        (Some(x), Some(y)) => Some(x.max(y)),
        (x, None) => x,
        (None, y) => y,
    }
}

fn sorted_cells(p: &crate::board::Placement, shapes: &crate::rules::Shapes) -> [(i32, i32); 4] {
    let mut c = p.abs_cells(shapes);
    c.sort();
    c
}

/// The planned line as JSON, every piece expressed in the rows of the board as it is
/// NOW. A planned piece that lands after an earlier planned clear sits on a board
/// whose rows have shifted down; mapping each row back to the row it came from puts
/// the forecast where it will visually end up, drawn over today's stack.
pub(crate) fn plan_to_json(board: &Board, plan: &[crate::search::Step], b2b: i32, searcher: &Searcher,
                shapes: &crate::rules::Shapes) -> String {
    let mut b = *board;
    let mut chain = b2b;
    // origin[y] = row of the current board that row y of the simulated board came
    // from; rows that scroll in from above the field get ids above the top (negative).
    let mut origin: Vec<i32> = (0..H).collect();
    let mut steps = Vec::new();
    for (k, s) in plan.iter().enumerate() {
        let p = &s.placement;
        let cells = p
            .abs_cells(shapes)
            .iter()
            .map(|&(x, y)| format!("[{},{}]", x, origin[y as usize]))
            .collect::<Vec<_>>()
            .join(",");
        let o = apply(&b, p, chain, &searcher.rules, shapes);
        let name = describe(p.piece, o.spin, o.lines);
        steps.push(format!(
            "{{\"k\":{},\"type\":\"{}\",\"hold\":{},\"cells\":[{}],\"clear\":\"{}\",\"attack\":{}}}",
            k, p.piece.letter(), s.use_hold, cells, name, o.attack
        ));
        if o.cleared != 0 {
            let top = *origin.iter().min().unwrap();
            let kept: Vec<i32> = (0..H as usize).filter(|&y| o.cleared >> y & 1 == 0).map(|y| origin[y]).collect();
            let n = H as usize - kept.len();
            origin = (0..n).map(|i| top - (n - i) as i32).chain(kept).collect();
        }
        if o.repeat {
            // A planned repeat inserts a wound line at the bottom, pushing every row up
            // one. The new row is below today's board; nothing is ever planned into it.
            origin.remove(0);
            origin.push(H);
        }
        b = o.board;
        chain = o.b2b;
    }
    format!("[{}]", steps.join(","))
}

fn count_garbage_rows(colors: &Colors, mask: u64) -> u64 {
    let mut n = 0;
    let mut m = mask;
    while m != 0 {
        let y = m.trailing_zeros() as usize;
        m &= m - 1;
        if colors.0[y * W as usize..(y + 1) * W as usize].contains(&b'G') {
            n += 1;
        }
    }
    n
}

fn tally(s: &mut Stats, piece: Piece, spin: Spin, lines: u32) {
    if lines > 0 {
        let l = lines.min(4) as usize;
        if spin != Spin::None {
            s.spin_by_lines[l] += 1;
        } else {
            s.plain_by_lines[l] += 1;
        }
    }
    if spin != Spin::None && lines > 0 {
        if piece == Piece::T {
            if spin == Spin::Mini {
                s.tspin_mini += 1;
            } else {
                match lines {
                    1 => s.tspin_single += 1,
                    2 => s.tsd += 1,
                    _ => s.tst += 1,
                }
            }
        } else {
            s.other_spins += 1;
        }
        return;
    }
    match lines {
        1 => s.singles += 1,
        2 => s.doubles += 1,
        3 => s.triples += 1,
        4 => s.quads += 1,
        _ => {}
    }
}

#[allow(clippy::too_many_arguments)]
fn write_record(out: &mut String, cfg: &GameConfig, n: usize, board_rows: &[String], spawned: Piece, previews: &[Piece],
                spawn_hold: Option<Piece>, used_hold: bool, p: &crate::board::Placement, shapes: &crate::rules::Shapes,
                o: &crate::search::Outcome, lock_hold: Option<Piece>, lock_queue: Vec<Piece>, garbage_rows: u64,
                height: i32, garbage_left: u64, plan: &str, wounds: &str, wound_hole: Option<u8>) {
    let q = |v: &[Piece]| v.iter().map(|p| format!("\"{}\"", p.letter())).collect::<Vec<_>>().join(",");
    let opt = |p: Option<Piece>| p.map_or("null".to_string(), |p| format!("\"{}\"", p.letter()));
    let cells = p.abs_cells(shapes).iter().map(|(x, y)| format!("[{},{}]", x, y)).collect::<Vec<_>>().join(",");
    let mut cleared = Vec::new();
    for y in 0..64 {
        if o.cleared >> y & 1 != 0 {
            cleared.push(y.to_string());
        }
    }
    let board = board_rows.iter().map(|r| format!("\"{}\"", r)).collect::<Vec<_>>().join(",");
    let name = describe(p.piece, o.spin, o.lines);
    let _ = writeln!(
        out,
        "{{\"replay\":\"{label}_s{seed}\",\"user\":\"bot\",\"mods\":\"{label}\",\"tier\":\"strict\",\"n\":{n},\"frame\":{frame},\
\"board\":[{board}],\"piece\":\"{piece}\",\"queue\":[{queue}],\"hold\":{hold},\"used_hold\":{used},\"holds\":0,\
\"place\":{{\"type\":\"{ptype}\",\"x\":{px},\"y\":{py},\"r\":{pr},\"spin\":\"{spin}\",\"kick\":{kick},\"soft\":false,\"cells\":[{cells}]}},\
\"cleared\":{lines},\"cleared_rows\":[{cleared}],\"spin\":\"{spin}\",\"lock_queue\":[{lq}],\"lock_hold\":{lh},\"lock_hold_locked\":{used},\
\"sealed\":0,\"height\":{height},\"hole_run\":0,\"q_depth\":{gleft},\"recovered\":false,\"garbage_rows\":{grows},\"q_last_zero\":-1,\
\"attack\":{attack},\"b2b\":{b2b},\"clear\":\"{name}\",\"repeat\":{repeat},\"burst\":{burst},\"wound_hole\":{whole},\"wounds\":[{wounds}],\"plan\":{plan}}}",
        label = cfg.label,
        seed = cfg.seed,
        n = n,
        frame = n * 30,
        board = board,
        piece = spawned.letter(),
        queue = q(previews),
        hold = opt(spawn_hold),
        used = used_hold,
        ptype = p.piece.letter(),
        px = p.x,
        py = p.y,
        pr = p.rot,
        spin = if o.lines > 0 || o.spin != Spin::None { o.spin.name() } else { "none" },
        kick = p.kick,
        cells = cells,
        lines = o.lines,
        cleared = cleared.join(","),
        lq = q(&lock_queue),
        lh = opt(lock_hold),
        height = height,
        gleft = garbage_left,
        grows = garbage_rows,
        attack = o.attack,
        b2b = o.b2b,
        name = name,
        plan = if plan.is_empty() { "[]" } else { plan },
        repeat = o.repeat,
        burst = o.burst,
        whole = wound_hole.map_or("null".to_string(), |h| h.to_string()),
        wounds = wounds,
    );
}
