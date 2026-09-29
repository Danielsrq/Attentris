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
    pub rows: usize,
    /// Chance each subsequent row moves its hole (0 = clean, 1 = full cheese).
    pub messiness: f64,
    /// Rows of cheese on the starting board.
    pub start_rows: usize,
}

impl Default for GarbageConfig {
    fn default() -> Self {
        GarbageConfig { every: 0, rows: 0, messiness: 0.0, start_rows: 0 }
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
    pub garbage_cleared: u64,
    pub max_height: i32,
    pub nodes: u64,
    /// Wall-clock time of each search call, in microseconds.
    pub think_us: Vec<u32>,
    /// Moves where the previous search had a prediction, and how many it got right.
    pub plan_checks: u64,
    pub plan_kept: u64,
    /// All-spin mode: clears that repeated the previous type, and the most wound
    /// lines live at once.
    pub repeats: u64,
    pub max_wounds: u32,
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
        self.repeats += o.repeats;
        self.max_wounds = self.max_wounds.max(o.max_wounds);
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

pub struct GameConfig {
    pub pieces: usize,
    pub seed: u64,
    pub garbage: GarbageConfig,
    pub label: String,
    /// Collect viewer JSONL.
    pub record: bool,
}

pub fn play(searcher: &mut Searcher, cfg: &GameConfig) -> (Stats, String) {
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

    let inject = |board: &mut Board, colors: &mut Colors, n: usize, grng: &mut Rng, hole: &mut i32| {
        // Each batch starts in a new column (one attack = one clean chunk, as in
        // TETR.IO); `messiness` is the chance each further row in the batch moves.
        let mut holes = Vec::with_capacity(n);
        for i in 0..n {
            if i == 0 || grng.chance(cfg.garbage.messiness) {
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

    let shapes = crate::rules::Shapes::new();
    let mut expected: Option<(Piece, [(i32, i32); 4])> = None;
    for n in 0..cfg.pieces {
        if cfg.garbage.every > 0 && n > 0 && n % cfg.garbage.every == 0 {
            inject(&mut board, &mut colors, cfg.garbage.rows, &mut grng, &mut garbage_hole);
            garbage_rows_total += cfg.garbage.rows as u64;
            garbage_left += cfg.garbage.rows as u64;
            stats.garbage_in += cfg.garbage.rows as u64;
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

        let nodes_before = searcher.nodes;
        let t0 = std::time::Instant::now();
        let d = match searcher.search(&board, &queue, hold, b2b, &bag_next) {
            Some(d) => d,
            None => {
                stats.died = true;
                break;
            }
        };
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

        // Advance the queue exactly as Game does: the spawned piece is consumed; an
        // empty-hold swap consumes one more.
        drawn += 1;
        if d.use_hold {
            if hold.is_none() {
                drawn += 1;
            }
            hold = Some(spawned);
        }
        let p = d.placement;
        let o = apply(&board, &p, b2b, &searcher.rules, &shapes);
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
            let hole = if board.wounds == 0 && stats.repeats == 0 || grng.chance(searcher.rules.wound_messiness) {
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
    }
    (stats, out)
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
fn plan_to_json(board: &Board, plan: &[crate::search::Step], b2b: i32, searcher: &Searcher,
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
\"attack\":{attack},\"b2b\":{b2b},\"clear\":\"{name}\",\"repeat\":{repeat},\"wound_hole\":{whole},\"wounds\":[{wounds}],\"plan\":{plan}}}",
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
        whole = wound_hole.map_or("null".to_string(), |h| h.to_string()),
        wounds = wounds,
    );
}
