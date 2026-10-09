//! Weight optimisation and the tools to look at it.
//!
//!   bot tune [--algo cem|cma] ...   CEM (default) or CMA-ES, resumable
//!   bot landscape slices ...         fitness along each weight around a weights file
//!   bot landscape plane ...          fitness on the plane through three weight files
//!   bot landscape trajectory ...     a tune's samples and clouds projected on a plane
//!
//! Everything is logged in NORMALISED weight units (see eval::SCALE), whatever units the
//! run evaluates in, so logs from different runs are directly comparable.

use std::io::Write;
use std::sync::atomic::{AtomicI64, AtomicUsize, Ordering};
use std::time::Instant;

use crate::eval::{self, Weights, NF};
use crate::game::{self, GameConfig, GarbageConfig};
use crate::rules::Rules;
use crate::search::{SearchConfig, Searcher};
use crate::{garbage, rules, search_config, threads_used, weights, Args};

// ---------------------------------------------------------------------------------
// Parallel fitness
// ---------------------------------------------------------------------------------

/// Attack per piece over `games` seeded games of `pieces` pieces. Every candidate plays
/// the same seeds (common random numbers), so differences come from the weights.
#[derive(Clone)]
pub struct Fitness {
    pub sc: SearchConfig,
    pub rules: Rules,
    pub garbage: GarbageConfig,
    pub games: usize,
    pub pieces: usize,
}

impl Fitness {
    /// Evaluates every candidate (active units) on seeds seed..seed+games. Work is split
    /// into (candidate, game) jobs pulled from a shared counter, so every thread stays
    /// busy and one slow candidate cannot hold up the rest. Results are identical to
    /// playing each candidate's games in turn: every game is deterministic.
    pub fn eval(&self, cands: &[[f32; NF]], seed: u64, threads: usize, on_job: &(dyn Fn(usize, usize) + Sync)) -> Vec<f64> {
        let total = cands.len() * self.games;
        let next = AtomicUsize::new(0);
        let done = AtomicUsize::new(0);
        let attack: Vec<AtomicI64> = (0..cands.len()).map(|_| AtomicI64::new(0)).collect();
        std::thread::scope(|s| {
            for _ in 0..threads.max(1).min(total.max(1)) {
                s.spawn(|| loop {
                    let j = next.fetch_add(1, Ordering::Relaxed);
                    if j >= total {
                        break;
                    }
                    let (ci, k) = (j / self.games, j % self.games);
                    let mut searcher = Searcher::new(self.sc.clone(), Weights { w: cands[ci] }, self.rules);
                    let cfg = GameConfig {
                        pieces: self.pieces,
                        seed: seed + k as u64,
                        garbage: self.garbage.clone(),
                        label: String::new(),
                        record: false,
                        env_rules: None,
                        dataset: false,
                        external: false,
                        shortlist: false,
                    };
                    attack[ci].fetch_add(game::play(&mut searcher, &cfg).0.attack, Ordering::Relaxed);
                    on_job(done.fetch_add(1, Ordering::Relaxed) + 1, total);
                });
            }
        });
        attack.iter().map(|a| a.load(Ordering::Relaxed) as f64 / (self.games * self.pieces) as f64).collect()
    }
}

fn fitness_from_args(a: &Args, default_games: usize) -> Fitness {
    Fitness {
        sc: SearchConfig { beam: a.get("beam", 24), depth: a.get("depth", 4), chance: false, ..search_config(a) },
        rules: rules(a),
        garbage: garbage(a),
        games: a.get("games", default_games),
        pieces: a.get("pieces", 300),
    }
}

// ---------------------------------------------------------------------------------
// Small linear algebra
// ---------------------------------------------------------------------------------

/// Eigen-decomposition of a symmetric n x n matrix (row-major) by cyclic Jacobi.
/// Returns (eigenvalues, eigenvectors as columns of a row-major n x n matrix).
pub fn jacobi(a: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut m = a.to_vec();
    let mut v = vec![0.0; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    for _sweep in 0..100 {
        let off: f64 = (0..n).flat_map(|i| (0..n).filter(move |&j| j != i).map(move |j| (i, j))).map(|(i, j)| m[i * n + j].powi(2)).sum();
        if off < 1e-22 {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                let apq = m[p * n + q];
                if apq.abs() < 1e-300 {
                    continue;
                }
                let theta = (m[q * n + q] - m[p * n + p]) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let t = if theta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let (mkp, mkq) = (m[k * n + p], m[k * n + q]);
                    m[k * n + p] = c * mkp - s * mkq;
                    m[k * n + q] = s * mkp + c * mkq;
                }
                for k in 0..n {
                    let (mpk, mqk) = (m[p * n + k], m[q * n + k]);
                    m[p * n + k] = c * mpk - s * mqk;
                    m[q * n + k] = s * mpk + c * mqk;
                }
                for k in 0..n {
                    let (vkp, vkq) = (v[k * n + p], v[k * n + q]);
                    v[k * n + p] = c * vkp - s * vkq;
                    v[k * n + q] = s * vkp + c * vkq;
                }
            }
        }
    }
    ((0..n).map(|i| m[i * n + i]).collect(), v)
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

fn gauss(rng: &mut game::Rng) -> f64 {
    // Box-Muller.
    let u1 = ((rng.next() >> 11) as f64 + 1.0) / (1u64 << 53) as f64;
    let u2 = (rng.next() >> 11) as f64 / (1u64 << 53) as f64;
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Active units <-> normalised units, per weight (1.0 unless --raw-features).
fn unit_factor() -> Vec<f64> {
    let ones = [1.0f32; NF];
    eval::to_normalised(&ones).iter().map(|&x| x as f64).collect()
}

// ---------------------------------------------------------------------------------
// The search distributions
// ---------------------------------------------------------------------------------

/// CMA-ES state, (mu/mu_w, lambda) with rank-one and rank-mu updates and cumulative
/// step-size adaptation (Hansen's tutorial defaults).
struct Cma {
    n: usize,
    lambda: usize,
    mu: usize,
    w: Vec<f64>,
    mueff: f64,
    cs: f64,
    ds: f64,
    cc: f64,
    c1: f64,
    cmu: f64,
    chin: f64,
    sigma: f64,
    c: Vec<f64>,
    ps: Vec<f64>,
    pc: Vec<f64>,
    b: Vec<f64>,
    d: Vec<f64>,
    generation: usize,
}

impl Cma {
    fn new(n: usize, lambda: usize, init_sd: &[f64]) -> Cma {
        let mu = lambda / 2;
        let raw: Vec<f64> = (0..mu).map(|i| (mu as f64 + 0.5).ln() - ((i + 1) as f64).ln()).collect();
        let sum: f64 = raw.iter().sum();
        let w: Vec<f64> = raw.iter().map(|x| x / sum).collect();
        let mueff = 1.0 / w.iter().map(|x| x * x).sum::<f64>();
        let nf = n as f64;
        let cs = (mueff + 2.0) / (nf + mueff + 5.0);
        let ds = 1.0 + 2.0 * (((mueff - 1.0) / (nf + 1.0)).sqrt() - 1.0).max(0.0) + cs;
        let cc = (4.0 + mueff / nf) / (nf + 4.0 + 2.0 * mueff / nf);
        let c1 = 2.0 / ((nf + 1.3).powi(2) + mueff);
        let cmu = (1.0 - c1).min(2.0 * (mueff - 2.0 + 1.0 / mueff) / ((nf + 2.0).powi(2) + mueff));
        let chin = nf.sqrt() * (1.0 - 1.0 / (4.0 * nf) + 1.0 / (21.0 * nf * nf));
        // Start with the same per-weight spread CEM starts with: sigma = 1, C = diag(sd^2).
        let mut c = vec![0.0; n * n];
        for i in 0..n {
            c[i * n + i] = init_sd[i] * init_sd[i];
        }
        let mut s = Cma { n, lambda, mu, w, mueff, cs, ds, cc, c1, cmu, chin, sigma: 1.0, c, ps: vec![0.0; n], pc: vec![0.0; n], b: vec![], d: vec![], generation: 0 };
        s.decompose();
        s
    }

    fn decompose(&mut self) {
        let n = self.n;
        // Keep C exactly symmetric before decomposing.
        for i in 0..n {
            for j in i + 1..n {
                let v = 0.5 * (self.c[i * n + j] + self.c[j * n + i]);
                self.c[i * n + j] = v;
                self.c[j * n + i] = v;
            }
        }
        let (vals, vecs) = jacobi(&self.c, n);
        self.d = vals.iter().map(|&x| x.max(1e-20).sqrt()).collect();
        self.b = vecs;
    }

    /// y = B D z (so x = m + sigma y).
    fn sample_y(&self, rng: &mut game::Rng) -> Vec<f64> {
        let n = self.n;
        let z: Vec<f64> = (0..n).map(|_| gauss(rng)).collect();
        (0..n).map(|i| (0..n).map(|k| self.b[i * n + k] * self.d[k] * z[k]).sum()).collect()
    }

    /// C^{-1/2} y = B D^{-1} B^T y.
    fn inv_sqrt_times(&self, y: &[f64]) -> Vec<f64> {
        let n = self.n;
        let bt: Vec<f64> = (0..n).map(|k| (0..n).map(|i| self.b[i * n + k] * y[i]).sum::<f64>() / self.d[k]).collect();
        (0..n).map(|i| (0..n).map(|k| self.b[i * n + k] * bt[k]).sum()).collect()
    }

    /// One update from the lambda steps `ys`, sorted best first.
    fn update(&mut self, mean: &mut [f64], ys_best_first: &[Vec<f64>]) {
        let n = self.n;
        let yw: Vec<f64> = (0..n).map(|i| (0..self.mu).map(|k| self.w[k] * ys_best_first[k][i]).sum()).collect();
        for i in 0..n {
            mean[i] += self.sigma * yw[i];
        }
        let iy = self.inv_sqrt_times(&yw);
        let a = (self.cs * (2.0 - self.cs) * self.mueff).sqrt();
        for i in 0..n {
            self.ps[i] = (1.0 - self.cs) * self.ps[i] + a * iy[i];
        }
        self.generation += 1;
        let psn = norm(&self.ps);
        let hs = psn / (1.0 - (1.0 - self.cs).powi(2 * self.generation as i32)).sqrt() < (1.4 + 2.0 / (n as f64 + 1.0)) * self.chin;
        let hsf = if hs { 1.0 } else { 0.0 };
        let ac = (self.cc * (2.0 - self.cc) * self.mueff).sqrt();
        for i in 0..n {
            self.pc[i] = (1.0 - self.cc) * self.pc[i] + hsf * ac * yw[i];
        }
        let keep = 1.0 - self.c1 - self.cmu + (1.0 - hsf) * self.c1 * self.cc * (2.0 - self.cc);
        for i in 0..n {
            for j in 0..n {
                let mut rank_mu = 0.0;
                for k in 0..self.mu {
                    rank_mu += self.w[k] * ys_best_first[k][i] * ys_best_first[k][j];
                }
                self.c[i * n + j] = keep * self.c[i * n + j] + self.c1 * self.pc[i] * self.pc[j] + self.cmu * rank_mu;
            }
        }
        self.sigma *= ((self.cs / self.ds) * (psn / self.chin - 1.0)).exp();
        self.decompose();
    }

    /// Covariance of the search distribution, sigma^2 C, in the active units.
    fn cov(&self) -> Vec<f64> {
        self.c.iter().map(|x| x * self.sigma * self.sigma).collect()
    }
}

// ---------------------------------------------------------------------------------
// Tune
// ---------------------------------------------------------------------------------

struct TuneState {
    generation: usize,
    mean: Vec<f64>,
    sd: Vec<f64>,
    cma: Option<Cma>,
    rng: game::Rng,
    best: f64,
    best_w: Vec<f64>,
}

fn row(v: &[f64]) -> String {
    v.iter().map(|x| format!("{:e}", x)).collect::<Vec<_>>().join(" ")
}

fn save_state(path: &str, s: &TuneState) {
    let mut t = format!("gen {}\nbest {:e}\nmean {}\nsd {}\nbest_w {}\nrng {}\n", s.generation, s.best, row(&s.mean), row(&s.sd), row(&s.best_w), s.rng.state());
    if let Some(c) = &s.cma {
        t += &format!("sigma {:e}\ncmac {}\nps {}\npc {}\ncmagen {}\n", c.sigma, row(&c.c), row(&c.ps), row(&c.pc), c.generation);
    }
    let tmp = format!("{}.tmp", path);
    std::fs::write(&tmp, t).unwrap();
    std::fs::rename(&tmp, path).unwrap();
}

fn load_state(path: &str, s: &mut TuneState) -> bool {
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(_) => return false,
    };
    let field = |k: &str| -> Vec<String> {
        text.lines().find(|l| l.starts_with(&format!("{} ", k))).map(|l| l.split_whitespace().skip(1).map(String::from).collect()).unwrap_or_default()
    };
    let nums = |k: &str| -> Vec<f64> { field(k).iter().map(|x| x.parse().unwrap()).collect() };
    s.generation = field("gen")[0].parse().unwrap();
    s.best = field("best")[0].parse().unwrap();
    s.mean = nums("mean");
    s.sd = nums("sd");
    s.best_w = nums("best_w");
    s.rng = game::Rng::from_state(field("rng")[0].parse().unwrap());
    if let Some(c) = s.cma.as_mut() {
        c.sigma = nums("sigma")[0];
        c.c = nums("cmac");
        c.ps = nums("ps");
        c.pc = nums("pc");
        c.generation = field("cmagen")[0].parse().unwrap();
        c.decompose();
    }
    true
}

/// `--freeze default` (the features that carry nothing outside all-spin, see
/// eval::FROZEN_OUTSIDE_ALLSPIN), `--freeze name,name,...`, or both (`default,b2b`):
/// weights pinned at 0.
pub fn frozen_mask(a: &Args) -> [bool; NF] {
    let mut m = [false; NF];
    if let Some(spec) = a.str("freeze") {
        // "default" expands to the features that carry nothing outside all-spin.
        let names: Vec<&str> = spec.split(',').flat_map(|n| if n == "default" { eval::FROZEN_OUTSIDE_ALLSPIN.to_vec() } else { vec![n] }).collect();
        for name in names {
            m[eval::index_of(name).unwrap_or_else(|| panic!("--freeze: unknown feature {}", name))] = true;
        }
    }
    m
}

pub fn cmd_tune(a: &Args) {
    let algo = a.str("algo").unwrap_or("cem").to_string();
    let gens = a.get("gens", 20usize);
    let pop = a.get("pop", 32usize);
    let elite = a.get("elite", 8usize);
    let f = Fitness { sc: SearchConfig { beam: a.get("beam", 16), depth: a.get("depth", 3), ..fitness_from_args(a, 6).sc }, ..fitness_from_args(a, 6) };
    let save = a.str("save").unwrap_or("weights/tuned.txt").to_string();
    if let Some(dir) = std::path::Path::new(&save).parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let (state_path, log_path, progress_path) = (format!("{}.state", save), format!("{}.log", save), format!("{}.progress", save));
    let (samples_path, dist_path) = (format!("{}.samples.csv", save), format!("{}.dist.csv", save));
    let u = unit_factor();
    let n = NF;

    let frozen = frozen_mask(a);
    // --fix name,...: weights held at their STARTING value (--freeze holds them at 0).
    let mut fixed = [false; NF];
    for name in a.str("fix").map(|s| s.split(',').collect::<Vec<_>>()).unwrap_or_default() {
        fixed[eval::index_of(name).unwrap_or_else(|| panic!("--fix: unknown feature {}", name))] = true;
    }
    let mut start_w: Vec<f64> = weights(a).w.iter().map(|&x| x as f64).collect();
    let mut init_sd: Vec<f64> = start_w.iter().map(|m| m.abs() * 0.5 + 0.2).collect();
    for i in 0..n {
        if frozen[i] {
            start_w[i] = 0.0;
        }
        if frozen[i] || fixed[i] {
            init_sd[i] = 0.0;
        }
    }
    let held = |i: usize| frozen[i] || fixed[i];
    assert!(algo == "cem" || !(0..n).any(held), "--freeze / --fix are only supported with --algo cem");
    if frozen.iter().any(|&x| x) {
        let names: Vec<&str> = (0..n).filter(|&i| frozen[i]).map(|i| eval::NAMES[i]).collect();
        println!("frozen at 0: {}", names.join(" "));
    }
    if fixed.iter().any(|&x| x) {
        let names: Vec<String> = (0..n).filter(|&i| fixed[i]).map(|i| format!("{} {:.2}", eval::NAMES[i], start_w[i] * unit_factor()[i])).collect();
        println!("fixed at start: {}", names.join(", "));
    }
    let mut st = TuneState {
        generation: 0,
        mean: start_w.clone(),
        sd: init_sd.clone(),
        cma: if algo == "cma" { Some(Cma::new(n, pop - 1, &init_sd)) } else { None },
        rng: game::Rng::new(a.get("seed", 7u64)),
        best: f64::NEG_INFINITY,
        best_w: start_w.clone(),
    };
    if !a.has("fresh") && load_state(&state_path, &mut st) {
        println!("resuming {} from generation {}", save, st.generation);
    }
    if st.generation == 0 {
        let _ = std::fs::write(&log_path, "gen best elite_mean incumbent seconds spread\n");
        let names = eval::NAMES.join(",");
        let _ = std::fs::write(&samples_path, format!("gen,idx,fitness,elite,{}\n", names));
        let _ = std::fs::write(&dist_path, "gen,kind,values (normalised units; cov is row-major n x n)\n");
    }
    println!("tuning ({}): {} gens x {} candidates x {} games x {} pieces, search beam {} depth {}{}", algo, gens, pop, f.games, f.pieces,
        f.sc.beam, f.sc.depth, if eval::RAW_FEATURES.load(Ordering::Relaxed) { "  [raw features]" } else { "" });

    let session = Instant::now();
    let gens_at_start = st.generation;
    let stop_after = a.get("stop-after", 0usize);
    let min_gain = a.get("min-gain", 0.005f64);
    let (mut best_elite, mut stale) = (f64::NEG_INFINITY, 0usize);
    while st.generation < gens {
        let g_i = st.generation;
        let seed = 1000 + g_i as u64 * 101;
        // Distribution at the START of this generation, for the trajectory plot.
        let cov: Vec<f64> = match &st.cma {
            Some(c) => c.cov(),
            None => {
                let mut m = vec![0.0; n * n];
                for i in 0..n {
                    m[i * n + i] = st.sd[i] * st.sd[i];
                }
                m
            }
        };
        log_dist(&dist_path, g_i, &st.mean, &cov, &u);

        // Candidate 0 is always the mean itself (the incumbent).
        let mut cands: Vec<Vec<f64>> = vec![st.mean.clone()];
        let mut ys: Vec<Vec<f64>> = vec![vec![0.0; n]];
        while cands.len() < pop {
            match &st.cma {
                Some(c) => {
                    let y = c.sample_y(&mut st.rng);
                    cands.push((0..n).map(|i| st.mean[i] + c.sigma * y[i]).collect());
                    ys.push(y);
                }
                None => cands.push((0..n).map(|i| st.mean[i] + st.sd[i] * gauss(&mut st.rng)).collect()),
            }
        }
        let cw: Vec<[f32; NF]> = cands.iter().map(|c| std::array::from_fn(|i| c[i] as f32)).collect();
        let t0 = Instant::now();
        let fits = f.eval(&cw, seed, threads_used(), &|done, total| {
            write_progress(&progress_path, g_i, gens, done, total, gens_at_start, session.elapsed().as_secs_f64());
        });

        // Rank: CEM ranks all candidates, CMA-ES ranks its lambda samples (not the mean).
        let first = if st.cma.is_some() { 1 } else { 0 };
        let mut order: Vec<usize> = (first..cands.len()).collect();
        order.sort_by(|&x, &y| fits[y].total_cmp(&fits[x]));
        let n_elite = if st.cma.is_some() { st.cma.as_ref().unwrap().mu } else { elite };
        let is_elite: Vec<bool> = (0..cands.len()).map(|i| order[..n_elite].contains(&i)).collect();
        log_samples(&samples_path, g_i, &cands, &fits, &is_elite, &u);

        let top = fits[order[0]];
        if top > st.best {
            st.best = top;
            st.best_w = cands[order[0]].clone();
        }
        let elite_mean = order[..n_elite].iter().map(|&i| fits[i]).sum::<f64>() / n_elite as f64;
        match st.cma.as_mut() {
            Some(c) => {
                let ranked: Vec<Vec<f64>> = order.iter().map(|&i| ys[i].clone()).collect();
                c.update(&mut st.mean, &ranked);
            }
            None => {
                for i in 0..n {
                    let m = order[..elite].iter().map(|&k| cands[k][i]).sum::<f64>() / elite as f64;
                    let v = order[..elite].iter().map(|&k| (cands[k][i] - m).powi(2)).sum::<f64>() / elite as f64;
                    st.mean[i] = m;
                    // Noise floor that decays, so the search does not collapse early.
                    st.sd[i] = v.sqrt() + 0.1 * (1.0 - g_i as f64 / gens as f64);
                    if frozen[i] || fixed[i] {
                        st.mean[i] = start_w[i];
                        st.sd[i] = 0.0;
                    }
                }
            }
        }
        st.generation += 1;

        // Average per-weight spread in normalised units, comparable across algorithms.
        let spread = match &st.cma {
            Some(c) => (0..n).map(|i| c.sigma * c.c[i * n + i].sqrt() * u[i]).sum::<f64>() / n as f64,
            None => (0..n).map(|i| st.sd[i] * u[i]).sum::<f64>() / n as f64,
        };
        let secs = t0.elapsed().as_secs_f64();
        let per_gen = session.elapsed().as_secs_f64() / (st.generation - gens_at_start) as f64;
        println!("gen {:>2}/{}: best {:.3}  elite mean {:.3}  (incumbent {:.3})  spread {:.3}  {:.0}s  eta {}",
            g_i + 1, gens, top, elite_mean, fits[0], spread, secs, fmt_dur(per_gen * (gens - st.generation) as f64));
        let mean_w: [f32; NF] = std::array::from_fn(|i| st.mean[i] as f32);
        let header = format!("# tuned by `bot tune --algo {}`, gen {}, distribution mean\n", algo, g_i);
        std::fs::write(&save, header + &Weights { w: mean_w }.to_text()).unwrap();
        save_state(&state_path, &st);
        if let Ok(mut fh) = std::fs::OpenOptions::new().append(true).create(true).open(&log_path) {
            let _ = writeln!(fh, "{} {:.4} {:.4} {:.4} {:.0} {:.4}", g_i, top, elite_mean, fits[0], secs, spread);
        }
        if elite_mean > best_elite + min_gain {
            best_elite = elite_mean;
            stale = 0;
        } else {
            stale += 1;
        }
        if stop_after > 0 && stale >= stop_after {
            println!("early stop: elite mean has not improved by more than {} for {} generations", min_gain, stop_after);
            break;
        }
    }
    // Final distribution, for the trajectory plot's last cloud.
    let cov: Vec<f64> = match &st.cma {
        Some(c) => c.cov(),
        None => (0..n * n).map(|k| if k / n == k % n { st.sd[k / n].powi(2) } else { 0.0 }).collect(),
    };
    log_dist(&dist_path, st.generation, &st.mean, &cov, &u);
    let _ = std::fs::write(&progress_path, "finished\n");
    println!("\nsaved distribution mean to {} (best single candidate scored {:.3})", save, st.best);
}

fn log_dist(path: &str, generation: usize, mean: &[f64], cov: &[f64], u: &[f64]) {
    let n = mean.len();
    // Skip if this generation is already logged (a resumed run re-logs nothing).
    if let Ok(t) = std::fs::read_to_string(path) {
        if t.lines().any(|l| l.starts_with(&format!("{},mean,", generation))) {
            return;
        }
    }
    let mn: Vec<String> = (0..n).map(|i| format!("{:.6}", mean[i] * u[i])).collect();
    let cv: Vec<String> = (0..n * n).map(|k| format!("{:.6e}", cov[k] * u[k / n] * u[k % n])).collect();
    if let Ok(mut fh) = std::fs::OpenOptions::new().append(true).create(true).open(path) {
        let _ = writeln!(fh, "{},mean,{}", generation, mn.join(","));
        let _ = writeln!(fh, "{},cov,{}", generation, cv.join(","));
    }
}

fn log_samples(path: &str, generation: usize, cands: &[Vec<f64>], fits: &[f64], elite: &[bool], u: &[f64]) {
    if let Ok(mut fh) = std::fs::OpenOptions::new().append(true).create(true).open(path) {
        for (i, c) in cands.iter().enumerate() {
            let w: Vec<String> = c.iter().zip(u).map(|(x, f)| format!("{:.5}", x * f)).collect();
            let _ = writeln!(fh, "{},{},{:.5},{},{}", generation, i, fits[i], elite[i] as u8, w.join(","));
        }
    }
}

fn write_progress(path: &str, g_i: usize, gens: usize, done: usize, total: usize, gens_at_start: usize, secs: f64) {
    let frac = (g_i as f64 + done as f64 / total as f64) / gens as f64;
    let run_frac = ((g_i - gens_at_start) as f64 + done as f64 / total as f64) / (gens - gens_at_start) as f64;
    let bar: String = (0..20).map(|i| if (i as f64) < frac * 20.0 { '#' } else { '-' }).collect();
    let eta = if run_frac > 0.0 { secs / run_frac * (1.0 - run_frac) } else { 0.0 };
    let _ = std::fs::write(path, format!("gen {}/{}  games {}/{}  [{}] {:.0}% of run  elapsed {}  eta {}\n",
        g_i + 1, gens, done, total, bar, frac * 100.0, fmt_dur(secs), fmt_dur(eta)));
}

pub fn fmt_dur(secs: f64) -> String {
    let m = (secs / 60.0).round() as u64;
    if m >= 60 { format!("{}h{:02}m", m / 60, m % 60) } else { format!("{}m", m) }
}

// ---------------------------------------------------------------------------------
// Landscape
// ---------------------------------------------------------------------------------

/// Evaluates normalised-unit points with a resumable cache (<dir>/<name>.cache: "idx fitness").
fn eval_cached(f: &Fitness, points: &[Vec<f64>], seed: u64, dir: &str, name: &str) -> Vec<f64> {
    let cache = format!("{}/{}.cache", dir, name);
    let mut got: Vec<Option<f64>> = vec![None; points.len()];
    if let Ok(t) = std::fs::read_to_string(&cache) {
        for l in t.lines() {
            let mut it = l.split_whitespace();
            if let (Some(i), Some(v)) = (it.next(), it.next()) {
                if let (Ok(i), Ok(v)) = (i.parse::<usize>(), v.parse::<f64>()) {
                    if i < got.len() {
                        got[i] = Some(v);
                    }
                }
            }
        }
    }
    let todo: Vec<usize> = (0..points.len()).filter(|&i| got[i].is_none()).collect();
    let session = Instant::now();
    let progress = format!("{}/{}.progress", dir, name);
    for (bi, batch) in todo.chunks(64).enumerate() {
        let cw: Vec<[f32; NF]> = batch.iter().map(|&i| eval::from_normalised(&std::array::from_fn(|k| points[i][k] as f32))).collect();
        let done_before = bi * 64;
        let fits = f.eval(&cw, seed, threads_used(), &|d, t| {
            let frac = (done_before as f64 + d as f64 / t as f64 * batch.len() as f64) / todo.len() as f64;
            let secs = session.elapsed().as_secs_f64();
            let _ = std::fs::write(&progress, format!("{}: {:.0}% of {} points  elapsed {}  eta {}\n", name, frac * 100.0, todo.len(),
                fmt_dur(secs), fmt_dur(if frac > 0.0 { secs / frac * (1.0 - frac) } else { 0.0 })));
        });
        let mut fh = std::fs::OpenOptions::new().append(true).create(true).open(&cache).unwrap();
        for (k, &i) in batch.iter().enumerate() {
            got[i] = Some(fits[k]);
            let _ = writeln!(fh, "{} {}", i, fits[k]);
        }
    }
    let _ = std::fs::write(&progress, "finished\n");
    got.into_iter().map(|x| x.unwrap()).collect()
}

fn normalised_weights_file(path: &str) -> Vec<f64> {
    let t = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {}", path, e));
    let w = Weights::parse(&t).unwrap_or_else(|e| panic!("{}: {}", path, e));
    eval::to_normalised(&w.w).iter().map(|&x| x as f64).collect()
}

pub fn cmd_landscape(a: &Args, what: &str) {
    let f = fitness_from_args(a, 12);
    let seed = a.get("seed", 7000u64);
    let dir = a.str("save").unwrap_or("results/landscape").to_string();
    let _ = std::fs::create_dir_all(&dir);
    match what {
        "slices" if a.str("only").is_some() => {
            // One weight set to the listed values (normalised units), the rest held at a
            // weights file: `--only t_vertical --values 0,1,2,...` -> slice_<name>.csv.
            let base = normalised_weights_file(a.str("weights").expect("--weights"));
            let name = a.str("only").unwrap();
            let i = eval::index_of(name).unwrap_or_else(|| panic!("--only: unknown feature {}", name));
            let values: Vec<f64> = a.str("values").expect("--values a,b,c").split(',').map(|v| v.parse().expect("--values")).collect();
            let points: Vec<Vec<f64>> = values.iter().map(|&v| { let mut p = base.clone(); p[i] = v; p }).collect();
            println!("slice {}: {} points x {} games x {} pieces (base value {:.3})", name, points.len(), f.games, f.pieces, base[i]);
            let tag = a.str("tag").unwrap_or("");
            let fits = eval_cached(&f, &points, seed, &dir, &format!("slice_{}{}", name, tag));
            let mut out = format!("{},fitness\n", name);
            for (v, ft) in values.iter().zip(&fits) {
                out += &format!("{:.4},{:.5}\n", v, ft);
                println!("  {:>8.3}  {:.4}", v, ft);
            }
            std::fs::write(format!("{}/slice_{}{}.csv", dir, name, tag), out).unwrap();
        }
        "slices" => {
            // One-dimensional cuts: every weight moved alone around a weights file.
            let base = normalised_weights_file(a.str("weights").expect("--weights"));
            let ks: Vec<i32> = (-3..=3).collect();
            let mut points = vec![base.clone()];
            let mut meta = vec![(usize::MAX, 0i32, 0.0f64)];
            for i in 0..NF {
                let step = (0.3 * base[i].abs()).max(1.0);
                for &k in &ks {
                    if k == 0 {
                        continue;
                    }
                    let mut p = base.clone();
                    p[i] += k as f64 * step;
                    points.push(p);
                    meta.push((i, k, base[i] + k as f64 * step));
                }
            }
            println!("slices: {} points x {} games x {} pieces", points.len(), f.games, f.pieces);
            let fits = eval_cached(&f, &points, seed, &dir, "slices");
            let mut out = String::from("feature,k,weight,fitness\n");
            for i in 0..NF {
                out += &format!("{},0,{:.5},{:.5}\n", eval::NAMES[i], base[i], fits[0]);
                for (j, m) in meta.iter().enumerate() {
                    if m.0 == i {
                        out += &format!("{},{},{:.5},{:.5}\n", eval::NAMES[i], m.1, m.2, fits[j]);
                    }
                }
            }
            std::fs::write(format!("{}/slices.csv", dir), out).unwrap();
            println!("wrote {}/slices.csv (base fitness {:.3})", dir, fits[0]);
        }
        "plane" => {
            // The plane through three weight files: A at (0,0), B at (1,0), C above.
            let (pa, pb, pc) = (
                normalised_weights_file(a.str("a").expect("--a")),
                normalised_weights_file(a.str("b").expect("--b")),
                normalised_weights_file(a.str("c").expect("--c")),
            );
            let uvec: Vec<f64> = (0..NF).map(|i| pb[i] - pa[i]).collect();
            let ul = norm(&uvec);
            let e1: Vec<f64> = uvec.iter().map(|x| x / ul).collect();
            let ca: Vec<f64> = (0..NF).map(|i| pc[i] - pa[i]).collect();
            let c1 = dot(&ca, &e1);
            let vvec: Vec<f64> = (0..NF).map(|i| ca[i] - c1 * e1[i]).collect();
            let vl = norm(&vvec).max(1e-9);
            let e2: Vec<f64> = vvec.iter().map(|x| x / vl).collect();
            let (cx, cy) = (c1 / ul, vl / ul);
            let g = a.get("grid", 9usize);
            let (x0, x1) = (cx.min(0.0) - 0.3, cx.max(1.0) + 0.3);
            let (y0, y1) = (-0.3f64.max(-cy - 0.3).min(-0.3), cy + 0.3);
            let mut points = vec![pa.clone(), pb.clone(), pc.clone()];
            let mut coords = vec![(0.0, 0.0), (1.0, 0.0), (cx, cy)];
            for iy in 0..g {
                for ix in 0..g {
                    let x = x0 + (x1 - x0) * ix as f64 / (g - 1) as f64;
                    let y = y0 + (y1 - y0) * iy as f64 / (g - 1) as f64;
                    points.push((0..NF).map(|i| pa[i] + x * ul * e1[i] + y * ul * e2[i]).collect());
                    coords.push((x, y));
                }
            }
            println!("plane: {} points x {} games x {} pieces", points.len(), f.games, f.pieces);
            let fits = eval_cached(&f, &points, seed, &dir, "plane");
            let mut out = String::from("kind,x,y,fitness\n");
            for (j, (x, y)) in coords.iter().enumerate() {
                let kind = ["A", "B", "C"].get(j).copied().unwrap_or("grid");
                out += &format!("{},{:.5},{:.5},{:.5}\n", kind, x, y, fits[j]);
            }
            std::fs::write(format!("{}/plane.csv", dir), out).unwrap();
            println!("wrote {}/plane.csv (A {:.3}, B {:.3}, C {:.3}; |B-A| = {:.2})", dir, fits[0], fits[1], fits[2], ul);
        }
        "trajectory" => cmd_trajectory(a, &f, seed, &dir),
        other => panic!("unknown landscape view {} (slices | plane | trajectory)", other),
    }
}

/// A tune's samples and clouds projected on one plane: e1 = start mean -> final mean,
/// e2 = the main direction the samples varied in besides e1 (PCA of the residuals).
/// Optionally evaluates fitness on a grid over that plane (--grid N) for the background.
fn cmd_trajectory(a: &Args, f: &Fitness, seed: u64, dir: &str) {
    let run = a.str("run").expect("--run <tune save path>");
    let samples = std::fs::read_to_string(format!("{}.samples.csv", run)).expect("samples log");
    let dist = std::fs::read_to_string(format!("{}.dist.csv", run)).expect("dist log");
    let nums = |s: &str| -> Vec<f64> { s.split(',').map(|x| x.parse().unwrap()).collect() };
    let mut means: Vec<(usize, Vec<f64>)> = vec![];
    let mut covs: Vec<(usize, Vec<f64>)> = vec![];
    for l in dist.lines().skip(1) {
        let mut parts = l.splitn(3, ',');
        let (g, kind, vals) = (parts.next().unwrap().parse::<usize>().unwrap(), parts.next().unwrap(), parts.next().unwrap());
        if kind == "mean" {
            means.push((g, nums(vals)));
        } else {
            covs.push((g, nums(vals)));
        }
    }
    let rows: Vec<(usize, f64, bool, Vec<f64>)> = samples
        .lines()
        .skip(1)
        .map(|l| {
            let v: Vec<&str> = l.split(',').collect();
            (v[0].parse().unwrap(), v[2].parse().unwrap(), v[3] == "1", v[4..].iter().map(|x| x.parse().unwrap()).collect())
        })
        .collect();
    let n = NF;
    let start = means.first().unwrap().1.clone();
    let end = means.last().unwrap().1.clone();
    let d: Vec<f64> = (0..n).map(|i| end[i] - start[i]).collect();
    let dl = norm(&d).max(1e-9);
    let e1: Vec<f64> = d.iter().map(|x| x / dl).collect();
    // PCA of the samples after removing their e1 component.
    let resid: Vec<Vec<f64>> = rows.iter().map(|r| {
        let rel: Vec<f64> = (0..n).map(|i| r.3[i] - start[i]).collect();
        let p = dot(&rel, &e1);
        (0..n).map(|i| rel[i] - p * e1[i]).collect()
    }).collect();
    let mu: Vec<f64> = (0..n).map(|i| resid.iter().map(|r| r[i]).sum::<f64>() / resid.len() as f64).collect();
    let mut cov = vec![0.0; n * n];
    for r in &resid {
        for i in 0..n {
            for j in 0..n {
                cov[i * n + j] += (r[i] - mu[i]) * (r[j] - mu[j]) / resid.len() as f64;
            }
        }
    }
    let (vals, vecs) = jacobi(&cov, n);
    let k = (0..n).max_by(|&x, &y| vals[x].total_cmp(&vals[y])).unwrap();
    let mut e2: Vec<f64> = (0..n).map(|i| vecs[i * n + k]).collect();
    let p = dot(&e2, &e1);
    for i in 0..n {
        e2[i] -= p * e1[i];
    }
    let l2 = norm(&e2).max(1e-9);
    for x in e2.iter_mut() {
        *x /= l2;
    }
    let proj = |w: &[f64]| -> (f64, f64) {
        let rel: Vec<f64> = (0..n).map(|i| w[i] - start[i]).collect();
        (dot(&rel, &e1), dot(&rel, &e2))
    };
    let proj_cov = |c: &[f64]| -> [f64; 3] {
        // [var1, cov12, var2] of e^T C e.
        let q = |a: &[f64], b: &[f64]| -> f64 { (0..n).map(|i| (0..n).map(|j| a[i] * c[i * n + j] * b[j]).sum::<f64>()).sum() };
        [q(&e1, &e1), q(&e1, &e2), q(&e2, &e2)]
    };

    let mut js = String::from("{\n");
    js += &format!("\"names\": [{}],\n", eval::NAMES.iter().map(|s| format!("\"{}\"", s)).collect::<Vec<_>>().join(","));
    js += &format!("\"e1\": [{}],\n\"e2\": [{}],\n", row_f(&e1), row_f(&e2));
    js += &format!("\"e2_variance_share\": {:.4},\n", vals[k] / vals.iter().map(|v| v.max(0.0)).sum::<f64>().max(1e-12));
    js += "\"gens\": [\n";
    let gens: Vec<usize> = means.iter().map(|m| m.0).collect();
    for (gi, g) in gens.iter().enumerate() {
        let m = &means[gi].1;
        let (mx, my) = proj(m);
        let c = proj_cov(&covs[gi].1);
        let pts: Vec<String> = rows.iter().filter(|r| r.0 == *g).map(|r| {
            let (x, y) = proj(&r.3);
            format!("[{:.4},{:.4},{:.4},{}]", x, y, r.1, r.2 as u8)
        }).collect();
        js += &format!("  {{\"gen\": {}, \"mean\": [{:.4},{:.4}], \"cov\": [{:.6},{:.6},{:.6}], \"samples\": [{}]}}{}\n",
            g, mx, my, c[0], c[1], c[2], pts.join(","), if gi + 1 < gens.len() { "," } else { "" });
    }
    js += "],\n";
    // Optional fitness grid over the plane, for the background.
    let g = a.get("grid", 0usize);
    if g >= 2 {
        let xs: Vec<f64> = rows.iter().map(|r| proj(&r.3).0).collect();
        let ys: Vec<f64> = rows.iter().map(|r| proj(&r.3).1).collect();
        let (x0, x1) = (xs.iter().cloned().fold(f64::INFINITY, f64::min), xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max));
        let (y0, y1) = (ys.iter().cloned().fold(f64::INFINITY, f64::min), ys.iter().cloned().fold(f64::NEG_INFINITY, f64::max));
        let mut points = vec![];
        let mut coords = vec![];
        for iy in 0..g {
            for ix in 0..g {
                let x = x0 + (x1 - x0) * ix as f64 / (g - 1) as f64;
                let y = y0 + (y1 - y0) * iy as f64 / (g - 1) as f64;
                points.push((0..n).map(|i| start[i] + x * e1[i] + y * e2[i]).collect::<Vec<f64>>());
                coords.push((x, y));
            }
        }
        let name = format!("traj_{}", std::path::Path::new(run).file_name().unwrap().to_string_lossy());
        println!("trajectory grid: {} points x {} games x {} pieces", points.len(), f.games, f.pieces);
        let fits = eval_cached(f, &points, seed, dir, &name);
        js += &format!("\"grid\": {{\"n\": {}, \"x\": [{:.4},{:.4}], \"y\": [{:.4},{:.4}], \"fitness\": [{}]}}\n", g, x0, x1, y0, y1,
            fits.iter().map(|v| format!("{:.4}", v)).collect::<Vec<_>>().join(","));
    } else {
        js += "\"grid\": null\n";
    }
    js += "}\n";
    let out = format!("{}/trajectory_{}.json", dir, std::path::Path::new(run).file_name().unwrap().to_string_lossy());
    std::fs::write(&out, js).unwrap();
    println!("wrote {}", out);
}

fn row_f(v: &[f64]) -> String {
    v.iter().map(|x| format!("{:.5}", x)).collect::<Vec<_>>().join(",")
}

// ---------------------------------------------------------------------------------
// Synthetic benchmark: CEM vs CMA-ES on standard functions at our budget
// ---------------------------------------------------------------------------------

/// Minimise f from x0 with `gens` x `pop` evaluations; returns the best TRUE value of
/// the final mean (noise affects selection only, as in tuning).
fn synth_run(algo: &str, f: &dyn Fn(&[f64]) -> f64, noise: f64, x0: &[f64], gens: usize, pop: usize, seed: u64) -> f64 {
    let n = x0.len();
    let mut rng = game::Rng::new(seed);
    let mut mean = x0.to_vec();
    let init_sd: Vec<f64> = mean.iter().map(|m| m.abs() * 0.5 + 0.2).collect();
    let mut sd = init_sd.clone();
    let mut cma = if algo == "cma" { Some(Cma::new(n, pop - 1, &init_sd)) } else { None };
    for g in 0..gens {
        let mut cands = vec![mean.clone()];
        let mut ys = vec![vec![0.0; n]];
        while cands.len() < pop {
            match &cma {
                Some(c) => {
                    let y = c.sample_y(&mut rng);
                    cands.push((0..n).map(|i| mean[i] + c.sigma * y[i]).collect());
                    ys.push(y);
                }
                None => cands.push((0..n).map(|i| mean[i] + sd[i] * gauss(&mut rng)).collect()),
            }
        }
        // Noisy fitness, as with a handful of games: relative noise on a positive scale.
        let fit: Vec<f64> = cands.iter().map(|c| { let v = f(c); v * (1.0 + noise * gauss(&mut rng)) }).collect();
        let first = if cma.is_some() { 1 } else { 0 };
        let mut order: Vec<usize> = (first..pop).collect();
        order.sort_by(|&a, &b| fit[a].total_cmp(&fit[b]));
        match cma.as_mut() {
            Some(c) => {
                let ranked: Vec<Vec<f64>> = order.iter().map(|&i| ys[i].clone()).collect();
                c.update(&mut mean, &ranked);
            }
            None => {
                let e = 8;
                for i in 0..n {
                    let m = order[..e].iter().map(|&k| cands[k][i]).sum::<f64>() / e as f64;
                    let v = order[..e].iter().map(|&k| (cands[k][i] - m).powi(2)).sum::<f64>() / e as f64;
                    mean[i] = m;
                    sd[i] = v.sqrt() + 0.1 * (1.0 - g as f64 / gens as f64);
                }
            }
        }
    }
    f(&mean)
}

pub fn cmd_synth(a: &Args) {
    let n = a.get("dims", NF);
    let gens = a.get("gens", 20usize);
    let pop = a.get("pop", 32usize);
    let reps = a.get("reps", 15u64);
    // A fixed random rotation for the non-separable variants.
    let mut rng = game::Rng::new(99);
    let mut s = vec![0.0; n * n];
    for i in 0..n {
        for j in i..n {
            let v = gauss(&mut rng);
            s[i * n + j] = v;
            s[j * n + i] = v;
        }
    }
    let (_, rot) = jacobi(&s, n);
    let rotate = |x: &[f64]| -> Vec<f64> { (0..n).map(|i| (0..n).map(|j| rot[j * n + i] * x[j]).sum()).collect() };
    let cond = |i: usize| 1e3f64.powf(i as f64 / (n - 1) as f64);
    let fs: Vec<(&str, Box<dyn Fn(&[f64]) -> f64>)> = vec![
        ("sphere (separable, well scaled)", Box::new(|x: &[f64]| x.iter().map(|v| v * v).sum())),
        ("ellipsoid (separable, 1:1000 scales)", Box::new(move |x: &[f64]| (0..n).map(|i| cond(i) * x[i] * x[i]).sum())),
        ("rotated ellipsoid (correlated, 1:1000)", Box::new(move |x: &[f64]| { let y = rotate(x); (0..n).map(|i| cond(i) * y[i] * y[i]).sum() })),
        ("rosenbrock (curved valley)", Box::new(|x: &[f64]| (0..x.len() - 1).map(|i| 100.0 * (x[i + 1] - x[i] * x[i]).powi(2) + (1.0 - x[i]).powi(2)).sum())),
    ];
    let x0 = vec![2.0; n];
    println!("{} dims, {} gens x {} candidates = {} evaluations per run, median of {} runs (final mean's true value, lower is better)", n, gens, pop, gens * pop, reps);
    println!("{:<40} {:>8} {:>12} {:>12} {:>8}", "function", "noise", "CEM", "CMA-ES", "winner");
    for (name, f) in &fs {
        for &noise in &[0.0, 0.1] {
            let med = |algo: &str| -> f64 {
                let mut v: Vec<f64> = (0..reps).map(|r| synth_run(algo, f.as_ref(), noise, &x0, gens, pop, 1 + r)).collect();
                v.sort_by(|a, b| a.total_cmp(b));
                v[v.len() / 2]
            };
            let (c, m) = (med("cem"), med("cma"));
            println!("{:<40} {:>7.0}% {:>12.4e} {:>12.4e} {:>8}", name, noise * 100.0, c, m, if c < m { "CEM" } else { "CMA-ES" });
        }
    }
    let start = fs[0].1(&x0);
    println!("(sphere at the start point: {:.1})", start);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jacobi_recovers_eigenpairs() {
        let n = 6;
        let mut rng = game::Rng::new(3);
        let mut a = vec![0.0; n * n];
        for i in 0..n {
            for j in i..n {
                let v = gauss(&mut rng);
                a[i * n + j] = v;
                a[j * n + i] = v;
            }
        }
        let (vals, vecs) = jacobi(&a, n);
        for k in 0..n {
            for i in 0..n {
                let av: f64 = (0..n).map(|j| a[i * n + j] * vecs[j * n + k]).sum();
                assert!((av - vals[k] * vecs[i * n + k]).abs() < 1e-9, "A v = lambda v");
            }
        }
    }

    /// CMA-ES must solve a rotated, ill-conditioned ellipsoid -- the kind of problem a
    /// diagonal method cannot rotate into.
    #[test]
    fn cma_solves_rotated_ellipsoid() {
        let n = 8;
        let mut rng = game::Rng::new(11);
        let mut s = vec![0.0; n * n];
        for i in 0..n {
            for j in i..n {
                let v = gauss(&mut rng);
                s[i * n + j] = v;
                s[j * n + i] = v;
            }
        }
        let (_, rot) = jacobi(&s, n);
        let f = |x: &[f64]| -> f64 {
            (0..n)
                .map(|i| {
                    let yi: f64 = (0..n).map(|j| rot[j * n + i] * x[j]).sum();
                    1e4f64.powf(i as f64 / (n - 1) as f64) * yi * yi
                })
                .sum()
        };
        let mut mean = vec![3.0; n];
        let mut cma = Cma::new(n, 12, &vec![1.0; n]);
        let mut best = f64::INFINITY;
        for _ in 0..600 {
            let ys: Vec<Vec<f64>> = (0..cma.lambda).map(|_| cma.sample_y(&mut rng)).collect();
            let mut fx: Vec<(f64, usize)> = ys
                .iter()
                .enumerate()
                .map(|(k, y)| {
                    let x: Vec<f64> = (0..n).map(|i| mean[i] + cma.sigma * y[i]).collect();
                    (f(&x), k)
                })
                .collect();
            fx.sort_by(|a, b| a.0.total_cmp(&b.0));
            best = best.min(fx[0].0);
            let ranked: Vec<Vec<f64>> = fx.iter().map(|&(_, k)| ys[k].clone()).collect();
            cma.update(&mut mean, &ranked);
        }
        assert!(best < 1e-8, "CMA-ES reached {:e}", best);
    }
}

// ---------------------------------------------------------------------------------
// Shapley values
// ---------------------------------------------------------------------------------

/// Monte Carlo Shapley values of each weight's contribution to fitness.
///
///   bot shapley --weights W --baseline B --save DIR/NAME [--perms 32] [--freeze default]
///
/// The players are the weights that differ between W and B (frozen ones are held at W's
/// value throughout). A coalition S plays with W's weights on S and B's elsewhere, so
/// v(empty) = fitness(B) and v(all) = fitness(W), and the values sum to the difference.
/// Each permutation adds the players one at a time and credits each with the change
/// it made; permutations come in antithetic pairs (an order and its reverse). Every
/// coalition plays the same seeded games (common random numbers). Resumable: coalition
/// fitnesses are cached in NAME.cache, the running estimate is NAME.csv.
pub fn cmd_shapley(a: &Args) {
    let f = fitness_from_args(a, 12);
    let seed = a.get("seed", 5000u64);
    let perms = (a.get("perms", 32usize) / 2 * 2).max(2);
    let save = a.str("save").expect("--save DIR/NAME").to_string();
    if let Some(dir) = std::path::Path::new(&save).parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let w = normalised_weights_file(a.str("weights").expect("--weights"));
    let mut b = normalised_weights_file(a.str("baseline").expect("--baseline"));
    let frozen = frozen_mask(a);
    for i in 0..NF {
        if frozen[i] {
            b[i] = w[i];
        }
    }
    let players: Vec<usize> = (0..NF).filter(|&i| !frozen[i] && (w[i] - b[i]).abs() > 1e-9).collect();
    let n = players.len();
    assert!(n > 0 && n < 64);
    println!("shapley: {} players, {} permutations, {} games x {} pieces each, beam {} depth {}", n, perms, f.games, f.pieces, f.sc.beam, f.sc.depth);
    println!("players: {}", players.iter().map(|&i| eval::NAMES[i]).collect::<Vec<_>>().join(" "));

    let cache_path = format!("{}.cache", save);
    let mut cache: std::collections::HashMap<u64, f64> = std::collections::HashMap::new();
    if let Ok(t) = std::fs::read_to_string(&cache_path) {
        for l in t.lines() {
            let mut it = l.split_whitespace();
            if let (Some(m), Some(v)) = (it.next(), it.next()) {
                if let (Ok(m), Ok(v)) = (u64::from_str_radix(m, 16), v.parse::<f64>()) {
                    cache.insert(m, v);
                }
            }
        }
        println!("resuming: {} coalitions cached", cache.len());
    }
    let point = |mask: u64| -> [f32; NF] {
        let mut p: [f32; NF] = std::array::from_fn(|i| b[i] as f32);
        for (k, &i) in players.iter().enumerate() {
            if mask >> k & 1 != 0 {
                p[i] = w[i] as f32;
            }
        }
        eval::from_normalised(&p)
    };

    // The permutations, fixed by the seed so a resumed run walks the same ones.
    let mut rng = game::Rng::new(a.get("perm-seed", 11u64));
    let mut orders: Vec<Vec<usize>> = Vec::new();
    while orders.len() < perms {
        let mut o: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            o.swap(i, rng.below(i as u64 + 1) as usize);
        }
        let r: Vec<usize> = o.iter().rev().copied().collect();
        orders.push(o);
        orders.push(r);
    }
    let full = (1u64 << n) - 1;
    let session = Instant::now();
    let batch = (a.get("batch", 4usize) / 2 * 2).max(2);
    let mut contrib: Vec<Vec<f64>> = vec![Vec::new(); n];
    for (bi, chunk) in orders.chunks(batch).enumerate() {
        let mut need: Vec<u64> = vec![0, full];
        for o in chunk {
            let mut m = 0u64;
            for &k in o {
                m |= 1 << k;
                need.push(m);
            }
        }
        need.sort_unstable();
        need.dedup();
        need.retain(|m| !cache.contains_key(m));
        if !need.is_empty() {
            let cw: Vec<[f32; NF]> = need.iter().map(|&m| point(m)).collect();
            let done_before = bi * batch;
            let fits = f.eval(&cw, seed, threads_used(), &|done, total| {
                let frac = (done_before as f64 + chunk.len() as f64 * done as f64 / total as f64) / perms as f64;
                let secs = session.elapsed().as_secs_f64();
                let bar: String = (0..20).map(|i| if (i as f64) < frac * 20.0 { '#' } else { '-' }).collect();
                let _ = std::fs::write(format!("{}.progress", save), format!("permutations {}/{}  [{}] {:.0}%  session {}\n",
                    done_before, perms, bar, frac * 100.0, fmt_dur(secs)));
            });
            let mut fh = std::fs::OpenOptions::new().append(true).create(true).open(&cache_path).unwrap();
            for (&m, &v) in need.iter().zip(&fits) {
                cache.insert(m, v);
                let _ = writeln!(fh, "{:x} {:.6}", m, v);
            }
        }
        for o in chunk {
            let mut m = 0u64;
            for &k in o {
                let before = cache[&m];
                m |= 1 << k;
                contrib[k].push(cache[&m] - before);
            }
        }
        // Running estimate: mean marginal contribution and its standard error, taken over
        // antithetic pairs (the pair mean is the independent sample).
        let done = (bi * batch + chunk.len()).min(perms);
        let mut csv = String::from("feature,phi,se,weight,baseline\n");
        let mut rows: Vec<(f64, String)> = Vec::new();
        for k in 0..n {
            let pairs: Vec<f64> = contrib[k].chunks(2).map(|p| p.iter().sum::<f64>() / p.len() as f64).collect();
            let mean = pairs.iter().sum::<f64>() / pairs.len() as f64;
            let var = if pairs.len() > 1 { pairs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (pairs.len() - 1) as f64 } else { 0.0 };
            let se = (var / pairs.len() as f64).sqrt();
            let i = players[k];
            csv += &format!("{},{:.5},{:.5},{:.4},{:.4}\n", eval::NAMES[i], mean, se, w[i], b[i]);
            rows.push((mean, format!("  {:<12} {:>+7.4} +- {:.4}", eval::NAMES[i], mean, se)));
        }
        csv += &format!("#v_empty,{:.5}\n#v_full,{:.5}\n#perms,{}\n", cache[&0], cache[&full], done);
        std::fs::write(format!("{}.csv", save), csv).unwrap();
        println!("after {} permutations (v(B) {:.3}, v(W) {:.3}, {} coalitions, {}):", done, cache[&0], cache[&full], cache.len(), fmt_dur(session.elapsed().as_secs_f64()));
        rows.sort_by(|x, y| y.0.total_cmp(&x.0));
        for (_, r) in rows {
            println!("{}", r);
        }
    }
    let _ = std::fs::write(format!("{}.progress", save), "finished\n");
}
