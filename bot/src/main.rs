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
//!   --b2b sqrt|flat         sqrt = floor(sqrt(chain)) as attack_table.gd (default); flat = +1
//!   --no-allspin            T-spins only (RuleSet.guideline()); default is all-spin on
//!   --mode allspin          every spin full (Minis promoted), repeat-penalty wound lines
//!   --wound-clears N        clears a wound line lasts (default 10)
//!   --garbage E,R,M         inject R rows every E pieces, hole moves with prob M
//!   --cheese N              start with N rows of messy garbage

mod board;
mod eval;
mod game;
mod movegen;
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
    let mut r = match a.str("mode") {
        Some("allspin") => Rules::allspin(),
        Some("standard") => Rules { all_spin: false, ..Rules::default() },
        Some("competitive") | None => Rules::default(),
        Some(m) => panic!("unknown --mode {} (standard | competitive | allspin)", m),
    };
    if a.has("no-allspin") {
        r.all_spin = false;
    }
    r.wound_clears = a.get("wound-clears", r.wound_clears);
    if a.str("b2b") == Some("flat") {
        r.b2b = B2bMode::Flat;
    }
    r
}

fn garbage(a: &Args) -> GarbageConfig {
    let mut g = GarbageConfig::default();
    if let Some(s) = a.str("garbage") {
        let p: Vec<&str> = s.split(',').collect();
        g.every = p.first().and_then(|x| x.parse().ok()).unwrap_or(0);
        g.rows = p.get(1).and_then(|x| x.parse().ok()).unwrap_or(1);
        g.messiness = p.get(2).and_then(|x| x.parse().ok()).unwrap_or(0.3);
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

/// Plays `games` games (seeds seed..seed+games) across threads. Returns per-game stats
/// and the concatenated JSONL when recording.
fn run_games(sc: &SearchConfig, w: &Weights, r: Rules, g: &GarbageConfig, games: usize, pieces: usize, seed: u64,
             threads: usize, label: &str, record: bool) -> (Vec<Stats>, String) {
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
                    let cfg = GameConfig { pieces, seed: seed + i as u64, garbage: g.clone(), label: label.clone(), record };
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

static mut THREADS: usize = 1;
fn threads_used() -> usize {
    unsafe { THREADS }
}

fn cmd_play(a: &Args) {
    let sc = search_config(a);
    let (games, pieces, seed) = (a.get("games", 4usize), a.get("pieces", 500usize), a.get("seed", 1u64));
    let label = a.str("label").map(|s| s.to_string()).unwrap_or(format!("beam{}_d{}{}", sc.beam, sc.depth, if sc.chance { "_ch" } else { "" }));
    let t0 = Instant::now();
    let (st, js) = run_games(&sc, &weights(a), rules(a), &garbage(a), games, pieces, seed, threads_used(), &label, a.has("out"));
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
        let (st, js) = run_games(&sc, &w, r, &g, games, pieces, seed, threads_used(), &label, out.is_some());
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

/// Cross-entropy method over the weight vector. Fitness is attack per piece over a
/// fixed set of seeds (common random numbers, so candidates are compared on the same
/// piece sequences); topping out forfeits the remaining pieces' attack.
fn cmd_tune(a: &Args) {
    let gens = a.get("gens", 20usize);
    let pop = a.get("pop", 32usize);
    let elite = a.get("elite", 8usize);
    let games = a.get("games", 6usize);
    let pieces = a.get("pieces", 300usize);
    let sc = SearchConfig { beam: a.get("beam", 16), depth: a.get("depth", 3), chance: false, ..search_config(a) };
    let r = rules(a);
    let g = garbage(a);
    let save = a.str("save").unwrap_or("weights/tuned.txt").to_string();
    let mut mean = weights(a).w;
    let mut sd: Vec<f32> = mean.iter().map(|m| m.abs() * 0.5 + 0.2).collect();
    let mut rng = game::Rng::new(a.get("seed", 7u64));
    let mut gauss = move || {
        // Box-Muller.
        let u1 = ((rng.next() >> 11) as f64 + 1.0) / (1u64 << 53) as f64;
        let u2 = (rng.next() >> 11) as f64 / (1u64 << 53) as f64;
        ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32
    };
    println!("tuning: {} gens x {} candidates x {} games x {} pieces, search beam {} depth {}", gens, pop, games, pieces, sc.beam, sc.depth);
    let mut best_ever = (f64::NEG_INFINITY, Weights { w: mean });
    for g_i in 0..gens {
        let seed = 1000 + g_i as u64 * 101;
        let mut cands: Vec<Weights> = vec![Weights { w: mean }];
        while cands.len() < pop {
            let mut w = mean;
            for i in 0..eval::NF {
                w[i] += sd[i] * gauss();
            }
            cands.push(Weights { w });
        }
        let t0 = Instant::now();
        let fits: Vec<f64> = std::thread::scope(|s| {
            let per = cands.len().div_ceil(threads_used());
            let hs: Vec<_> = cands
                .chunks(per)
                .map(|chunk| {
                    let (sc, g) = (sc.clone(), g.clone());
                    s.spawn(move || {
                        chunk
                            .iter()
                            .map(|w| {
                                let mut searcher = Searcher::new(sc.clone(), w.clone(), r);
                                let mut atk = 0i64;
                                for k in 0..games {
                                    let cfg = GameConfig { pieces, seed: seed + k as u64, garbage: g.clone(), label: String::new(), record: false };
                                    atk += game::play(&mut searcher, &cfg).0.attack;
                                }
                                atk as f64 / (games * pieces) as f64
                            })
                            .collect::<Vec<f64>>()
                    })
                })
                .collect();
            hs.into_iter().flat_map(|h| h.join().unwrap()).collect()
        });
        let mut order: Vec<usize> = (0..cands.len()).collect();
        order.sort_by(|&x, &y| fits[y].total_cmp(&fits[x]));
        let el: Vec<&Weights> = order[..elite].iter().map(|&i| &cands[i]).collect();
        for i in 0..eval::NF {
            let m = el.iter().map(|w| w.w[i]).sum::<f32>() / elite as f32;
            let v = el.iter().map(|w| (w.w[i] - m).powi(2)).sum::<f32>() / elite as f32;
            mean[i] = m;
            // Noise floor that decays, so the search does not collapse early.
            sd[i] = v.sqrt() + 0.1 * (1.0 - g_i as f32 / gens as f32);
        }
        let top = fits[order[0]];
        if top > best_ever.0 {
            best_ever = (top, cands[order[0]].clone());
        }
        println!("gen {:>2}: best {:.3}  elite mean {:.3}  (incumbent mean {:.3})  {:.0}s",
            g_i, top, order[..elite].iter().map(|&i| fits[i]).sum::<f64>() / elite as f64, fits[0], t0.elapsed().as_secs_f64());
        if let Some(dir) = std::path::Path::new(&save).parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        let header = format!("# tuned by `bot tune`, gen {}, fitness (APP on training seeds) of distribution mean below\n", g_i);
        std::fs::write(&save, header + &Weights { w: mean }.to_text()).unwrap();
    }
    println!("\nsaved distribution mean to {} (best single candidate scored {:.3})", save, best_ever.0);
}

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    if argv.is_empty() {
        eprintln!("usage: bot play|bench|tune|parity|weights [options]  (see src/main.rs)");
        std::process::exit(2);
    }
    let a = Args::parse(&argv[1..]);
    let t = a.get("threads", std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4));
    unsafe {
        THREADS = t;
    }
    match argv[0].as_str() {
        "play" => cmd_play(&a),
        "bench" => cmd_bench(&a),
        "tune" => cmd_tune(&a),
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
        other => {
            eprintln!("unknown command {}", other);
            std::process::exit(2);
        }
    }
}
