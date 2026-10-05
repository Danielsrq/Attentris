//! tetris-bot: heuristic evaluation + beam search, on a simulator that follows the
//! Godot project's rules.
//!
//!   bot play   [opts] --out FILE.jsonl   play games, write viewer replays
//!   bot bench  [opts]                    compare greedy vs search on the same seeds
//!   bot tune   [opts] --save FILE        cross-entropy tuning of the weights by self-play
//!   bot parity --data FILE.jsonl         check move generation against real placements
//!
//! Common options:
//!   --games N --pieces N --seed N --threads N
//!   --beam N --depth N --no-hold --no-chance --chance-top N
//!   --weights FILE          (see `bot weights` for the format)
//!   --b2b fibonacci|squares|prime|flat   B2B bonus scaling (default fibonacci, capped +7)
//!   --no-allspin            T-spins only (RuleSet.guideline()); default is all-spin on
//!   --mode allspin          every spin full (Minis promoted), repeat-penalty wound lines
//!   --mode burst            normal + TST 10 + x2 base for back-to-back qualifying clears
//!   --mode burst2           burst, but spin singles do not arm and the window is 2 placements
//!   --mode gate             training rules: only chains opened by a spin double/triple score
//!   --mode gate_b2b         gate, plus the B2B bonus on every spin clear of a paid chain
//!   --wound-clears N        clears a wound line lasts (default 10)
//!   --garbage E,R,M         inject R rows every E pieces, hole moves with prob M
//!   --cheese N              start with N rows of messy garbage

mod board;
mod cpu;
mod eval;
mod game;
mod movegen;
mod opt;
mod parity;
mod rules;
mod search;
mod slots;

use std::collections::HashMap;
use std::time::Instant;

use eval::Weights;
use game::{GameConfig, GarbageConfig, Stats};
use rules::{B2bMode, Rules};
use search::{SearchConfig, Searcher};

struct Args {
    flags: HashMap<String, String>,
}

impl Args {
    fn parse(v: &[String]) -> Args {
        let mut flags = HashMap::new();
        let mut i = 0;
        while i < v.len() {
            let k = v[i].trim_start_matches("--").to_string();
            if i + 1 < v.len() && !v[i + 1].starts_with("--") {
                flags.insert(k, v[i + 1].clone());
                i += 2;
            } else {
                flags.insert(k, String::new());
                i += 1;
            }
        }
        Args { flags }
    }
    fn get<T: std::str::FromStr>(&self, k: &str, d: T) -> T {
        self.flags.get(k).and_then(|s| s.parse().ok()).unwrap_or(d)
    }
    fn has(&self, k: &str) -> bool {
        self.flags.contains_key(k)
    }
    fn str(&self, k: &str) -> Option<&str> {
        self.flags.get(k).map(|s| s.as_str())
    }
}

fn search_config(a: &Args) -> SearchConfig {
    let d = SearchConfig::default();
    SearchConfig {
        beam: a.get("beam", d.beam),
        depth: a.get("depth", d.depth),
        use_hold: !a.has("no-hold"),
        chance: !a.has("no-chance"),
        chance_top: a.get("chance-top", d.chance_top),
        death: d.death,
    }
}

fn rules(a: &Args) -> Rules {
    rules_for(a, a.str("mode"))
}

/// Rules for a mode name, with the shared flags (--b2b, --no-allspin, ...) applied.
fn rules_for(a: &Args, mode: Option<&str>) -> Rules {
    let mut r = match mode {
        Some("allspin") => Rules::allspin(),
        Some("allspin_nw") => Rules::allspin_nw(),
        Some("burst") => Rules::burst(),
        Some("burst2") => Rules::burst2(),
        Some("gate") => Rules::gate(),
        Some("gate_b2b") => Rules::gate_b2b(),
        Some("standard") => Rules { all_spin: false, ..Rules::default() },
        // Normal rules are the default, as in the Godot game.
        Some("normal") | None => Rules::normal(),
        Some("competitive") => Rules::default(),
        Some(m) => panic!("unknown --mode {} (normal | burst | burst2 | gate | gate_b2b | standard | competitive | allspin | allspin_nw)", m),
    };
    if a.has("no-allspin") {
        r.all_spin = false;
    }
    r.wound_clears = a.get("wound-clears", r.wound_clears);
    if let Some(c) = a.str("b2b-cap") {
        r.b2b_cap = c.parse().ok();
    }
    // --b2b fibonacci (default) | squares (legacy) | prime (power-up) | flat
    if let Some(m) = a.str("b2b") {
        r.b2b = B2bMode::parse(m).unwrap_or_else(|| panic!("unknown --b2b {} (fibonacci | squares | prime | flat)", m));
    }
    r
}

fn garbage(a: &Args) -> GarbageConfig {
    let mut g = GarbageConfig::default();
    if let Some(s) = a.str("garbage") {
        let p: Vec<&str> = s.split(',').collect();
        g.every = p.first().and_then(|x| x.parse().ok()).unwrap_or(0);
        // Rows: N, or MIN-MAX for a uniform draw per batch.
        match p.get(1).and_then(|x| x.split_once('-')) {
            Some((lo, hi)) => {
                g.rows_min = lo.parse().unwrap_or(1);
                g.rows = hi.parse().unwrap_or(g.rows_min);
            }
            None => g.rows = p.get(1).and_then(|x| x.parse().ok()).unwrap_or(1),
        }
        // Messiness: a chance, or "mix" for clean and cheese batches with even odds.
        match p.get(2) {
            Some(&"mix") => g.mix = true,
            m => g.messiness = m.and_then(|x| x.parse().ok()).unwrap_or(0.3),
        }
        // Optional 4th field: chance a batch moves from the previous batch's column.
        g.batch_move = p.get(3).and_then(|x| x.parse().ok()).unwrap_or(1.0);
    }
    g.start_rows = a.get("cheese", 0);
    g
}

fn weights(a: &Args) -> Weights {
    match a.str("weights") {
        Some(p) => {
            let t = std::fs::read_to_string(p).unwrap_or_else(|e| panic!("{}: {}", p, e));
            Weights::parse(&t).unwrap_or_else(|e| panic!("{}: {}", p, e))
        }
        None => Weights::default(),
    }
}

/// `bot probe --boards NAME=ROWS;NAME=ROWS --weights a.txt,b.txt [--detail b.txt]`
/// V of each board (rows top to bottom, '/'-separated, 'X' or '#' filled, bottom-aligned
/// on an empty field; no B2B chain) under each weights file, and the per-feature
/// breakdown (normalised feature x normalised weight) for the `--detail` file.
fn cmd_probe(a: &Args) {
    let boards: Vec<(String, board::Board)> = a.str("boards").expect("--boards").split(';').map(|spec| {
        let (name, rows) = spec.split_once('=').expect("NAME=ROWS");
        let rows: Vec<&str> = rows.split('/').collect();
        let mut b = board::Board::default();
        let base = rules::H as usize - rows.len();
        for (i, r) in rows.iter().enumerate() {
            for (x, ch) in r.chars().enumerate() {
                if ch == 'X' || ch == '#' {
                    b.rows[base + i] |= 1 << x;
                }
            }
        }
        (name.to_string(), b)
    }).collect();
    let files: Vec<&str> = a.str("weights").expect("--weights").split(',').collect();
    let load = |p: &str| Weights::parse(&std::fs::read_to_string(p).unwrap_or_else(|e| panic!("{}: {}", p, e))).unwrap();
    let mode = B2bMode::Fibonacci;
    print!("{:<34}", "V(board)");
    for (n, _) in &boards {
        print!(" {:>10}", n);
    }
    println!();
    for f in &files {
        let w = load(f);
        print!("{:<34}", f);
        for (_, b) in &boards {
            print!(" {:>10.3}", w.value(&eval::features(b, -1, mode)));
        }
        println!();
    }
    if let Some(d) = a.str("detail") {
        let w = load(d);
        let wn = eval::to_normalised(&w.w);
        let feats: Vec<[f32; eval::NF]> = boards.iter().map(|(_, b)| eval::features(b, -1, mode)).collect();
        println!("\nbreakdown under {}: raw feature value, then contribution (weight x normalised feature)", d);
        print!("{:<13} {:>8}", "feature", "weight");
        for (n, _) in &boards {
            print!(" {:>8} {:>8}", n, "contrib");
        }
        println!();
        for i in 0..eval::NF {
            if feats.iter().all(|f| f[i] == 0.0) || w.w[i] == 0.0 {
                continue;
            }
            print!("{:<13} {:>8.2}", eval::NAMES[i], wn[i]);
            for f in &feats {
                print!(" {:>8.2} {:>8.3}", f[i] * eval::SCALE[i], w.w[i] * f[i]);
            }
            println!();
        }
    }
}

/// `bot plan --board ROWS --queue LJTSZO [--hold T] --weights W [--mode M] [--beam N]`
/// The line the search would play from a given position: each step's piece, clear and
/// attack, the board after it, and V of the final board. ROWS as in `probe`.
/// Reads the live state the Godot pilot sends: the board, the queue, hold, and the B2B
/// chain, plus wounds -- which `--board` cannot express and which matter under all-spin
/// rules, where `RuleSet.allspin()` turns `repeat_penalty` on and the Godot board really
/// does grow unclearable rows. Both engines already represent a wound the same way (a
/// per-row countdown plus the hole it will leave), so it crosses the wire unchanged.
fn state_from_json(src: &str) -> Result<(board::Board, Vec<rules::Piece>, Option<rules::Piece>, i32), String> {
    let v: serde_json::Value = serde_json::from_str(src).map_err(|e| format!("state is not JSON: {e}"))?;
    let mut b = board::Board::default();
    let rows = v["board"].as_array().ok_or("state.board must be an array of row strings")?;
    // Bottom-aligned, as --board is: the caller may send only the occupied tail.
    let base = rules::H as usize - rows.len();
    for (i, row) in rows.iter().enumerate() {
        for (x, ch) in row.as_str().unwrap_or("").chars().enumerate() {
            if ch != '.' && ch != ' ' {
                b.rows[base + i] |= 1 << x;
            }
        }
    }
    for (i, w) in v["wound_left"].as_array().unwrap_or(&vec![]).iter().enumerate() {
        let left = w.as_u64().unwrap_or(0) as u8;
        if left > 0 && base + i < rules::H as usize {
            b.wound_left[base + i] = left;
            b.wounds += 1;
            b.wound_hole[base + i] = v["wound_hole"][i].as_u64().unwrap_or(0) as u8;
            b.last_wound_hole = b.wound_hole[base + i];
        }
    }
    // The gate chain, which lives on the BOARD here (search.rs reads burst_armed / gate_len /
    // gate_pending straight off it) and in Game.gate_state there, in the same order:
    // placements left, spin clears in the chain, the opener's value held.
    //
    // Omitting it is not a small loss. Under gate rules r.attack comes ONLY from a completed
    // chain, so a bot that always believes the gate is shut never values the second clear of
    // one and never completes any -- which is exactly how the gate4 weights scored 2.6 APP in
    // `bot play` and a flat 0.000 piloting Godot.
    if let Some(g) = v["gate"].as_array() {
        let at = |i: usize| g.get(i).and_then(|x| x.as_u64()).unwrap_or(0) as u8;
        b.burst_armed = at(0);
        b.gate_len = at(1);
        b.gate_pending = at(2);
    }
    let letter = |c: char| rules::Piece::from_letter(c).ok_or(format!("bad piece '{c}'"));
    let mut queue = Vec::new();
    for c in v["queue"].as_str().unwrap_or("").chars() {
        queue.push(letter(c)?);
    }
    let hold = match v["hold"].as_str().and_then(|h| h.chars().next()) {
        Some(c) => Some(letter(c)?),
        None => None,
    };
    let chain = v["b2b"].as_i64().unwrap_or(-1) as i32;
    Ok((b, queue, hold, chain))
}

/// One decision per line of stdin, one JSON reply per line of stdout, until EOF.
///
/// This is the mode the Godot pilot runs in, so the bot is a part of the game rather than a
/// process spawned and thrown away per piece. Spawning cost 95ms of the 112ms round trip and
/// capped piloting at roughly 8 pieces per second; holding the process open removes all of it
/// and leaves only the ~15ms search.
///
/// The weights and the search config are read ONCE here, not per request, which is the other
/// reason this is cheaper than repeated `plan` calls.
///
/// Errors are replies, never panics: the caller reads stdout, so a panic would put the reason
/// on stderr where it is discarded and tell the caller only that the pipe closed.
fn cmd_serve(a: &Args) {
    use std::io::{BufRead as _, Write as _};
    let r = rules(a);
    let mut sc = search_config(a);
    sc.chance = false;
    let mut s = Searcher::new(sc, weights(a), r);
    let shapes = rules::Shapes::new();
    // Announced, so the caller can tell the process came up and loaded its weights before it
    // sends anything -- otherwise a bad --weights path looks like a hang on the first request.
    println!("{{\"ready\":true,\"mode\":\"{}\"}}", a.str("mode").unwrap_or("standard"));
    let _ = std::io::stdout().flush();
    let stdin = std::io::stdin();
    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        let t = line.trim();
        if t.is_empty() {
            continue;
        }
        if t == "quit" {
            break;
        }
        match state_from_json(t) {
            Err(e) => println!("{{\"error\":{}}}", serde_json::to_string(&e).unwrap()),
            Ok((b, queue, hold, chain)) => {
                let before = s.nodes;
                match s.search(&b, &queue, hold, chain, &rules::ALL_PIECES) {
                    None => println!("{{\"error\":\"no legal move\"}}"),
                    Some(d) => {
                        let plan = game::plan_to_json(&b, &d.plan, chain, &s, &shapes);
                        // Nodes for THIS search: the counter accumulates across a session.
                        println!(
                            "{{\"use_hold\":{},\"nodes\":{},\"plan\":{}}}",
                            d.use_hold,
                            s.nodes.saturating_sub(before),
                            plan
                        );
                    }
                }
            }
        }
        let _ = std::io::stdout().flush();
    }
}

fn cmd_plan(a: &Args) {
    let piece = |c: char| rules::Piece::from_letter(c).unwrap_or_else(|| panic!("bad piece {}", c));
    // --state-file, not --state: the state is JSON, and a quoted JSON blob on a Windows
    // command line gets mangled before this process ever sees it. A file has no quoting.
    let from_file = a.str("state-file").map(|p| {
        std::fs::read_to_string(p).unwrap_or_else(|e| format!("{{\"err\":\"cannot read {p}: {e}\"}}"))
    });
    let state_src = from_file.as_deref().or_else(|| a.str("state"));
    let (b, queue, hold, chain) = match state_src {
        Some(src) => match state_from_json(src) {
            Ok(v) => v,
            Err(e) => {
                // Reported as JSON on stdout, not as a panic. The caller is another process
                // that reads stdout; a panic puts the reason on stderr where it is discarded,
                // and all it learns is "exit 101".
                println!("{{\"error\":{}}}", serde_json::to_string(&e).unwrap());
                return;
            }
        },
        None => {
            let rows: Vec<&str> = a.str("board").expect("--board or --state").split('/').collect();
            let mut b = board::Board::default();
            let base = rules::H as usize - rows.len();
            for (i, r) in rows.iter().enumerate() {
                for (x, ch) in r.chars().enumerate() {
                    if ch == 'X' || ch == '#' {
                        b.rows[base + i] |= 1 << x;
                    }
                }
            }
            let queue: Vec<rules::Piece> =
                a.str("queue").expect("--queue").chars().map(piece).collect();
            let hold = a.str("hold").and_then(|h| h.chars().next()).map(piece);
            let chain: i32 = a.str("chain").and_then(|c| c.parse().ok()).unwrap_or(-1);
            (b, queue, hold, chain)
        }
    };
    let r = rules(a);
    let mut sc = search_config(a);
    sc.chance = false;
    let mut s = Searcher::new(sc, weights(a), r);
    let shapes = rules::Shapes::new();
    let d = s.search(&b, &queue, hold, chain, &rules::ALL_PIECES).expect("no move");

    // Machine-readable mode for the Godot pilot: one JSON line, in the schema the replay
    // viewer already parses, so there is no second wire format to keep in step. The plan's
    // cells are what the pilot acts on -- it re-derives rotation and kick with Godot's own
    // kick tables rather than trusting this process's coordinate conventions.
    if a.str("json").is_some() {
        let plan = game::plan_to_json(&b, &d.plan, chain, &s, &shapes);
        println!("{{\"use_hold\":{},\"nodes\":{},\"plan\":{}}}", d.use_hold, s.nodes, plan);
        return;
    }
    let show = |b: &board::Board| {
        let top = (0..rules::H as usize).find(|&y| b.rows[y] != 0).unwrap_or(rules::H as usize).min(rules::H as usize - 1);
        for y in top.saturating_sub(1)..rules::H as usize {
            println!("      {}", (0..10).map(|x| if b.rows[y] >> x & 1 != 0 { '#' } else { '.' }).collect::<String>());
        }
    };
    let mut cur = b;
    // The chain the caller gave us, not a fresh board's -1: V and every attack below depend
    // on whether a B2B is already live.
    let mut chain = chain;
    let mut total = 0;
    println!("start (V {:.3})", s.weights.value(&eval::features(&cur, chain, r.b2b)));
    show(&cur);
    for (k, st) in d.plan.iter().enumerate() {
        let o = search::apply(&cur, &st.placement, chain, &r, &shapes);
        total += o.attack;
        println!("step {}: {}{} {}  attack {}  -> V {:.3}", k + 1, st.placement.piece.letter().to_ascii_uppercase(),
            if st.use_hold { " (hold)" } else { "" }, rules::describe(st.placement.piece, o.spin, o.lines), o.attack,
            s.weights.value(&eval::features(&o.board, o.b2b, r.b2b)));
        cur = o.board;
        chain = o.b2b;
        show(&cur);
    }
    println!("planned attack {}, nodes {}", total, s.nodes);
}

/// Plays `games` games (seeds seed..seed+games) across threads. Returns per-game stats
/// and the concatenated JSONL when recording.
/// `r` = the rules the bot searches with; `env` = the rules the game applies, if different.
fn run_games(sc: &SearchConfig, w: &Weights, r: Rules, env: Option<Rules>, g: &GarbageConfig, games: usize, pieces: usize,
             seed: u64, threads: usize, label: &str, record: bool) -> (Vec<Stats>, String) {
    let results: Vec<(usize, Stats, String)> = std::thread::scope(|s| {
        let mut handles = Vec::new();
        for t in 0..threads.min(games).max(1) {
            let (sc, w, g) = (sc.clone(), w.clone(), g.clone());
            let label = label.to_string();
            handles.push(s.spawn(move || {
                let mut out = Vec::new();
                let mut searcher = Searcher::new(sc, w, r);
                let mut i = t;
                while i < games {
                    let cfg = GameConfig { pieces, seed: seed + i as u64, garbage: g.clone(), label: label.clone(), record, env_rules: env };
                    let (st, js) = game::play(&mut searcher, &cfg);
                    out.push((i, st, js));
                    i += threads;
                }
                out
            }));
        }
        handles.into_iter().flat_map(|h| h.join().unwrap()).collect()
    });
    let mut results = results;
    results.sort_by_key(|x| x.0);
    let js = results.iter().map(|x| x.2.as_str()).collect::<String>();
    (results.into_iter().map(|x| x.1).collect(), js)
}

fn summary_header() {
    println!(
        "{:<22} {:>6} {:>5} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6} {:>5} {:>5} {:>4} {:>7}",
        "config", "APP", "died", "pcs", "quad%", "TSD%", "TST%", "TSS%", "mini%", "spin%", "1-3%", "gbclr", "b2b", "rep%", "wnd", "ms/pc"
    );
}

fn summary_row(name: &str, st: &[Stats], secs: f64) {
    let mut t = Stats::default();
    for s in st {
        t.add(s);
    }
    let died = st.iter().filter(|s| s.died).count();
    let per = |n: u64| 100.0 * n as f64 / t.pieces.max(1) as f64;
    println!(
        "{:<22} {:>6.3} {:>2}/{:<2} {:>6.0} {:>6.2} {:>6.2} {:>6.2} {:>6.2} {:>6.2} {:>6.2} {:>6.2} {:>6} {:>5} {:>5.2} {:>4} {:>7.2}",
        name,
        t.app(),
        died,
        st.len(),
        t.pieces as f64 / st.len().max(1) as f64,
        per(t.quads),
        per(t.tsd),
        per(t.tst),
        per(t.tspin_single),
        per(t.tspin_mini),
        per(t.other_spins),
        per(t.singles + t.doubles + t.triples),
        format!("{}/{}", t.garbage_cleared, t.garbage_in),
        t.max_b2b,
        per(t.repeats),
        t.max_wounds,
        1000.0 * secs / t.pieces.max(1) as f64 * threads_used() as f64,
    );
}

/// Burstiness: rolling APP over 7- and 14-piece windows (mean of each game's best
/// window, and the 90th percentile of all windows), the share of qualifying clears that
/// followed another one (pair rate), and the share of attack sent by burst pairs (both
/// clears of each doubled pair).
fn burst_line(seqs: &[Vec<(u16, u8)>]) -> String {
    let mut parts = Vec::new();
    for w in [7usize, 14] {
        let (mut best_sum, mut all) = (0.0, Vec::new());
        for s in seqs {
            let mut best = 0.0f64;
            for win in s.windows(w) {
                let app = win.iter().map(|x| x.0 as f64).sum::<f64>() / w as f64;
                best = best.max(app);
                all.push(app);
            }
            best_sum += best;
        }
        all.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p90 = if all.is_empty() { 0.0 } else { all[(all.len() - 1) * 9 / 10] };
        parts.push(format!("{}-pc window APP best {:.2} p90 {:.2}", w, best_sum / seqs.len().max(1) as f64, p90));
    }
    let (mut qual, mut doubled, mut total, mut pair_atk) = (0u64, 0u64, 0u64, 0u64);
    // Clears the rules in play actually doubled (burst2's window differs from "pairs").
    let (mut ruled, mut ruled_atk) = (0u64, 0u64);
    for s in seqs {
        for (i, &(atk, fl)) in s.iter().enumerate() {
            total += atk as u64;
            qual += (fl & 1) as u64;
            if fl & 2 != 0 {
                ruled += 1;
                ruled_atk += atk as u64;
            }
            // A pair: this placement and the previous one both qualifying clears -- the
            // clears burst rules double, counted the same way under any rules.
            if fl & 1 != 0 && i > 0 && s[i - 1].1 & 1 != 0 {
                doubled += 1;
                pair_atk += atk as u64;
                // The first clear of the pair, unless it closed a pair itself (counted already).
                if !(i > 1 && s[i - 2].1 & 1 != 0) {
                    pair_atk += s[i - 1].0 as u64;
                }
            }
        }
    }
    format!("burst -- {}; pair rate {:.1}% of {} qualifying clears; burst-pair attack share {:.1}%; doubled {} clears sending {:.1}% of attack",
        parts.join(", "), 100.0 * doubled as f64 / qual.max(1) as f64, qual, 100.0 * pair_atk as f64 / total.max(1) as f64,
        ruled, 100.0 * ruled_atk as f64 / total.max(1) as f64)
}

static mut THREADS: usize = 1;
fn threads_used() -> usize {
    unsafe { THREADS }
}

fn cmd_play(a: &Args) {
    let sc = search_config(a);
    let (games, pieces, seed) = (a.get("games", 4usize), a.get("pieces", 500usize), a.get("seed", 1u64));
    let label = a.str("label").map(|s| s.to_string()).unwrap_or(format!("beam{}_d{}{}", sc.beam, sc.depth, if sc.chance { "_ch" } else { "" }));
    let t0 = Instant::now();
    // --believe MODE: the bot searches as if the rules were MODE, while the game applies
    // --mode. Without it the bot knows the real rules.
    let real = rules(a);
    let (belief, env) = match a.str("believe") {
        Some(m) => (rules_for(a, Some(m)), Some(real)),
        None => (real, None),
    };
    let (st, js) = run_games(&sc, &weights(a), belief, env, &garbage(a), games, pieces, seed, threads_used(), &label, a.has("out"));
    summary_header();
    summary_row(&label, &st, t0.elapsed().as_secs_f64());
    let mut t: Vec<u32> = st.iter().flat_map(|s| s.think_us.iter().copied()).collect();
    if !t.is_empty() {
        t.sort_unstable();
        let pct = |p: f64| t[((t.len() - 1) as f64 * p) as usize] as f64 / 1000.0;
        let mean = t.iter().map(|&x| x as f64).sum::<f64>() / t.len() as f64 / 1000.0;
        println!("think time per move (wall clock, {} threads): mean {:.1} ms  p50 {:.1}  p95 {:.1}  p99 {:.1}  max {:.1}",
            threads_used(), mean, pct(0.5), pct(0.95), pct(0.99), pct(1.0));
    }
    {
        let mut t = Stats::default();
        for s in &st {
            t.add(s);
        }
        let per = |n: u64| 100.0 * n as f64 / t.pieces.max(1) as f64;
        let spin_lines: u64 = (1..5).map(|l| l as u64 * t.spin_by_lines[l]).sum();
        let all_lines: u64 = spin_lines + (1..5).map(|l| l as u64 * t.plain_by_lines[l]).sum::<u64>();
        println!("clears per 100 pieces -- spin singles {:.2}, doubles {:.2}, triples {:.2}; plain 1/2/3/4: {:.2}/{:.2}/{:.2}/{:.2}; attack per line {:.2}",
            per(t.spin_by_lines[1]), per(t.spin_by_lines[2]), per(t.spin_by_lines[3]),
            per(t.plain_by_lines[1]), per(t.plain_by_lines[2]), per(t.plain_by_lines[3]), per(t.plain_by_lines[4]),
            t.attack as f64 / all_lines.max(1) as f64);
        println!("{}", burst_line(&t.seqs));
        let [ev, seen, ready] = t.tmpl_search;
        println!("STSD/TST template -- in search: {:.2}% of {} V evals saw a slot, {:.2}% a ready one; on the board: ready before {} moves, \
cashed {} (followed by a TSD {}); vertical T-spin doubles/triples {}, {} of them found by the template beforehand",
            100.0 * seen as f64 / ev.max(1) as f64, ev, 100.0 * ready as f64 / ev.max(1) as f64,
            t.tmpl_ready, t.tmpl_cashed, t.tmpl_followups, t.vert_spins, t.vert_spins_seen);
        if let Some(path) = a.str("seq-out") {
            let mut out = String::from("game,n,attack,qualifying,doubled
");
            for (g, seq) in t.seqs.iter().enumerate() {
                for (n, &(atk, fl)) in seq.iter().enumerate() {
                    out += &format!("{},{},{},{},{}
", g, n, atk, fl & 1, fl >> 1 & 1);
                }
            }
            std::fs::write(path, out).expect("write --seq-out");
        }
        if t.phantom_spins > 0 {
            println!("phantom spins (believed spins scored as plain clears): {:.2} per 100 pieces", per(t.phantom_spins));
        }
    }
    let (checks, kept) = st.iter().fold((0, 0), |(c, k), s| (c + s.plan_checks, k + s.plan_kept));
    if checks > 0 {
        println!("plan stability: {:.1}% of moves were exactly what the previous search planned ({} / {})",
            100.0 * kept as f64 / checks as f64, kept, checks);
    }
    if let Some(p) = a.str("out") {
        write_out(p, &js, a.has("append"));
    }
}

fn write_out(path: &str, js: &str, append: bool) {
    use std::io::Write;
    if let Some(dir) = std::path::Path::new(path).parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let mut f = std::fs::OpenOptions::new().create(true).write(true).append(append).truncate(!append).open(path).unwrap();
    f.write_all(js.as_bytes()).unwrap();
    println!("wrote {} ({} placements)", path, js.lines().count());
}

/// The experiment that answers "does search help": identical seeds, identical
/// weights, only the search changes.
fn cmd_bench(a: &Args) {
    let (games, pieces, seed) = (a.get("games", 8usize), a.get("pieces", 500usize), a.get("seed", 1u64));
    let w = weights(a);
    let r = rules(a);
    let g = garbage(a);
    let base = search_config(a);
    let beam = base.beam;
    let configs: Vec<(String, SearchConfig)> = vec![
        ("greedy (1 ply)".into(), SearchConfig { depth: 1, chance: false, ..base.clone() }),
        (format!("beam{} d2", beam), SearchConfig { depth: 2, chance: false, ..base.clone() }),
        (format!("beam{} d4", beam), SearchConfig { depth: 4, chance: false, ..base.clone() }),
        (format!("beam{} d6", beam), SearchConfig { depth: 6, chance: false, ..base.clone() }),
        (format!("beam{} d6 +chance", beam), SearchConfig { depth: 6, chance: true, ..base.clone() }),
    ];
    let out = a.str("out");
    summary_header();
    let mut all_js = String::new();
    for (name, sc) in configs {
        let t0 = Instant::now();
        let label = name.replace([' ', '(', ')'], "").replace('+', "_");
        let (st, js) = run_games(&sc, &w, r, None, &g, games, pieces, seed, threads_used(), &label, out.is_some());
        summary_row(&name, &st, t0.elapsed().as_secs_f64());
        // Keep only the first game per config: enough to watch side by side.
        let first = format!("{}_s{}", label, seed);
        all_js += &js.lines().filter(|l| l.contains(&format!("\"replay\":\"{}\"", first))).map(|l| format!("{}\n", l)).collect::<String>();
    }
    println!("\ncolumns: APP = attack per piece; quad%/TSD%/... = clears of that type per 100 pieces;");
    println!("1-3% = plain singles/doubles/triples; gbclr = garbage rows cleared/received; b2b = longest chain");
    if let Some(p) = out {
        write_out(p, &all_js, false);
    }
}

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    if argv.is_empty() {
        eprintln!("usage: bot play|bench|tune|parity|weights|plan|serve [options]  (see src/main.rs)");
        std::process::exit(2);
    }
    let a = Args::parse(&argv[1..]);
    let t = a.get("threads", std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4));
    // Stay on the performance cores of a hybrid CPU (see cpu.rs); --any-cores opts out.
    if !a.has("any-cores") && argv[0] != "cpus" {
        let note = cpu::prefer_performance_cores();
        if !note.is_empty() {
            eprintln!("cpu: {}", note);
        }
    }
    // Experimental control: evaluate with the raw, unscaled features (before weights load).
    if a.has("raw-features") {
        eval::RAW_FEATURES.store(true, std::sync::atomic::Ordering::Relaxed);
    }
    unsafe {
        THREADS = t;
    }
    match argv[0].as_str() {
        "play" => cmd_play(&a),
        "bench" => cmd_bench(&a),
        "tune" => opt::cmd_tune(&a),
        "synth" => opt::cmd_synth(&a),
        "shapley" => opt::cmd_shapley(&a),
        "landscape" => {
            let what = argv.get(1).cloned().unwrap_or_default();
            opt::cmd_landscape(&Args::parse(&argv[2.min(argv.len())..]), &what);
        }
        "parity" => {
            let path = a.str("data").unwrap_or("../data/zenith_full.jsonl");
            if let Err(e) = parity::run(path, a.has("strict"), a.get("limit", usize::MAX)) {
                eprintln!("{}", e);
                std::process::exit(1);
            }
        }
        "pros" => {
            let path = a.str("data").unwrap_or("../data/zenith_full.jsonl");
            if let Err(e) = parity::pros(path, !a.has("no-allspin"), a.str("mode") == Some("allspin")) {
                eprintln!("{}", e);
                std::process::exit(1);
            }
        }
        "cpus" => {
            let sets = cpu::cpu_sets();
            let top = sets.iter().map(|&(_, c)| c).max().unwrap_or(0);
            println!("logical processor  efficiency class");
            for (i, c) in &sets {
                println!("{:>17}  {}{}", i, c, if *c == top { "   performance" } else { "   efficiency" });
            }
            match cpu::performance_mask() {
                Some(m) => println!("performance mask {:#x} ({} logical processors)", m, m.count_ones()),
                None => println!("not a hybrid CPU: no restriction"),
            }
        }
        "slots" => {
            let path = a.str("data").unwrap_or("../data/bot/allspin.jsonl");
            // --piece Z --lines 3 : find a Z-spin triple slot.
            let want = a.str("piece").and_then(|s| s.chars().next()).and_then(rules::Piece::from_letter).map(|p| (p, a.get("lines", 1u32)));
            if let Err(e) = slots::run(path, a.get("show", 3usize), a.get("limit", usize::MAX), want) {
                eprintln!("{}", e);
                std::process::exit(1);
            }
        }
        "weights" =>print!("{}", weights(&a).to_text()),
        "serve" => cmd_serve(&a),
        "probe" => cmd_probe(&a),
        "plan" => cmd_plan(&a),
        other => {
            eprintln!("unknown command {}", other);
            std::process::exit(2);
        }
    }
}
