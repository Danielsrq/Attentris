//! Checks the move generator and spin rules against real placements: every record in
//! the harvested Zenith dataset carries a spawn-time board, the piece that was placed
//! and the cells it landed on. If our generator cannot reach those cells, or clears a
//! different number of lines, the simulator disagrees with the game.
//!
//! Spin labels in the dataset come from the vendored tetrp engine (a TETR.IO clone),
//! not from core/spin_detect.gd, so spin disagreement is reported rather than failed:
//! TETR.IO scores non-T all-spins as Mini, this project scores them as full.

use serde_json::Value;

use crate::board::Board;
use crate::movegen::MoveGen;
use crate::rules::{Piece, Shapes, Spin};

/// Pro baseline: the same statistics `bench` prints, computed from the dataset's
/// recorded clears and scored with OUR attack table. With `all_spin` off, non-T spins
/// are scored as plain clears (standard rules). Only the garbage-free `strict` climb
/// is used, so it compares like with like: no incoming garbage, the player's own stack.
pub fn pros(path: &str, all_spin: bool) -> Result<(), String> {
    use crate::rules::{next_b2b, Rules};
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {}", path, e))?;
    let rules = Rules { all_spin, ..Rules::default() };
    #[derive(Default, Clone)]
    struct Acc {
        pieces: u64,
        attack: i64,
        quad: u64,
        tsd: u64,
        tst: u64,
        tss: u64,
        mini: u64,
        other: u64,
        plain: u64,
        max_b2b: i32,
    }
    let mut per_user: std::collections::BTreeMap<String, Acc> = Default::default();
    let mut b2b: std::collections::HashMap<String, i32> = Default::default();
    for line in text.lines() {
        let v: Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(_) => continue,
        };
        if v["tier"].as_str() != Some("strict") {
            continue;
        }
        let replay = v["replay"].as_str().unwrap_or("").to_string();
        let user = v["user"].as_str().unwrap_or("?").to_string();
        let piece = match v["place"]["type"].as_str().and_then(|s| s.chars().next()).and_then(Piece::from_letter) {
            Some(p) => p,
            None => continue,
        };
        let lines = v["cleared"].as_u64().unwrap_or(0) as u32;
        let mut spin = match v["spin"].as_str() {
            Some("full") => Spin::Full,
            Some("mini") => Spin::Mini,
            _ => Spin::None,
        };
        if piece != Piece::T && !all_spin {
            spin = Spin::None;
        }
        let chain = b2b.entry(replay).or_insert(-1);
        *chain = next_b2b(*chain, spin, lines);
        let a = per_user.entry(user).or_default();
        a.pieces += 1;
        a.attack += rules.attack(piece, spin, lines, *chain, false) as i64;
        a.max_b2b = a.max_b2b.max(*chain);
        if lines > 0 {
            match (piece, spin, lines) {
                (_, Spin::None, 4) => a.quad += 1,
                (_, Spin::None, _) => a.plain += 1,
                (Piece::T, Spin::Mini, _) => a.mini += 1,
                (Piece::T, Spin::Full, 1) => a.tss += 1,
                (Piece::T, Spin::Full, 2) => a.tsd += 1,
                (Piece::T, Spin::Full, _) => a.tst += 1,
                _ => a.other += 1,
            }
        }
    }
    println!("pro baseline from {} (strict tier), scored with this project's attack table, all-spin {}",
        path, if all_spin { "on" } else { "off" });
    println!("{:<22} {:>6} {:>7} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6} {:>5}",
        "player", "APP", "pieces", "quad%", "TSD%", "TST%", "TSS%", "mini%", "spin%", "1-3%", "b2b");
    let mut all = Acc::default();
    let rows: Vec<(String, Acc)> = per_user.into_iter().collect();
    for (u, a) in rows.iter().map(|(u, a)| (u.as_str(), a)).chain(std::iter::once(("ALL", &Acc::default()))) {
        let a = if u == "ALL" { &all } else {
            all.pieces += a.pieces; all.attack += a.attack; all.quad += a.quad; all.tsd += a.tsd; all.tst += a.tst;
            all.tss += a.tss; all.mini += a.mini; all.other += a.other; all.plain += a.plain;
            all.max_b2b = all.max_b2b.max(a.max_b2b);
            a
        };
        let per = |n: u64| 100.0 * n as f64 / a.pieces.max(1) as f64;
        println!("{:<22} {:>6.3} {:>7} {:>6.2} {:>6.2} {:>6.2} {:>6.2} {:>6.2} {:>6.2} {:>6.2} {:>5}",
            u, a.attack as f64 / a.pieces.max(1) as f64, a.pieces, per(a.quad), per(a.tsd), per(a.tst), per(a.tss),
            per(a.mini), per(a.other), per(a.plain), a.max_b2b);
    }
    Ok(())
}

pub fn run(path: &str, strict_only: bool, limit: usize) -> Result<(), String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {}", path, e))?;
    let shapes = Shapes::new();
    let mut mg = MoveGen::new();
    let mut buf = Vec::new();
    let (mut total, mut reached, mut lines_ok) = (0u64, 0u64, 0u64);
    // confusion[recorded][ours], T and non-T separately, only for clears.
    let mut conf_t = [[0u64; 3]; 3];
    let mut conf_o = [[0u64; 3]; 3];
    let mut misses: Vec<String> = Vec::new();

    for line in text.lines() {
        if total as usize >= limit {
            break;
        }
        let v: Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(_) => continue,
        };
        if strict_only && v["tier"].as_str() != Some("strict") {
            continue;
        }
        let rows = match v["board"].as_array() {
            Some(r) if r.len() == 40 => r,
            _ => continue,
        };
        let mut b = Board::default();
        for (y, r) in rows.iter().enumerate() {
            for (x, ch) in r.as_str().unwrap_or("").chars().enumerate().take(10) {
                if ch != '.' {
                    b.rows[y] |= 1 << x;
                }
            }
        }
        let place = &v["place"];
        let piece = match place["type"].as_str().and_then(|s| s.chars().next()).and_then(Piece::from_letter) {
            Some(p) => p,
            None => continue,
        };
        let mut want: Vec<(i32, i32)> = place["cells"]
            .as_array()
            .map(|a| a.iter().map(|c| (c[0].as_i64().unwrap_or(-1) as i32, c[1].as_i64().unwrap_or(-1) as i32)).collect())
            .unwrap_or_default();
        want.sort();
        let rec_spin = match v["spin"].as_str() {
            Some("full") => Spin::Full,
            Some("mini") => Spin::Mini,
            _ => Spin::None,
        };
        let rec_lines = v["cleared"].as_u64().unwrap_or(0) as u32;
        total += 1;

        mg.generate(&b, piece, &shapes, &mut buf);
        let hit = buf.iter().find(|p| {
            let mut c: Vec<(i32, i32)> = p.abs_cells(&shapes).to_vec();
            c.sort();
            c == want
        });
        match hit {
            Some(p) => {
                reached += 1;
                let mut nb = b;
                let (lines, _) = nb.place(p.cells(&shapes), p.x as i32, p.y as i32);
                if lines == rec_lines {
                    lines_ok += 1;
                }
                if lines > 0 {
                    let conf = if piece == Piece::T { &mut conf_t } else { &mut conf_o };
                    conf[rec_spin as usize][p.spin as usize] += 1;
                }
            }
            None => {
                if misses.len() < 5 {
                    misses.push(format!("{} n={} piece={:?} cells={:?}", v["replay"], v["n"], piece, want));
                }
            }
        }
    }
    println!("records checked       {}", total);
    println!("placement reachable   {} ({:.2}%)", reached, 100.0 * reached as f64 / total.max(1) as f64);
    println!("lines cleared agree   {} ({:.2}% of reached)", lines_ok, 100.0 * lines_ok as f64 / reached.max(1) as f64);
    let names = ["none", "mini", "full"];
    for (title, conf) in [("T-piece clears", &conf_t), ("other-piece clears", &conf_o)] {
        println!("\n{}: rows = recorded spin, cols = ours", title);
        println!("          none   mini   full");
        for r in 0..3 {
            println!("  {:<5} {:>6} {:>6} {:>6}", names[r], conf[r][0], conf[r][1], conf[r][2]);
        }
    }
    if !misses.is_empty() {
        println!("\nfirst unreachable placements:");
        for m in misses {
            println!("  {}", m);
        }
    }
    Ok(())
}
