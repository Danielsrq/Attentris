//! `bot slots`: checks the spin-slot bitmask scan against the real move generator and
//! prints what it detects as ASCII.
//!
//! Soundness check: every reachable non-T spin that clears lines (move generator +
//! all-spin detection) must be found by the scan, which uses the same immobility test.
//! The scan may find MORE -- cavities no piece can rotate into -- and the ratio says how
//! optimistic the V estimate is.

use serde_json::Value;

use crate::board::Board;
use crate::eval::{fits_mask, immobile_mask, scan_spin_slots, SpinSlot};
use crate::movegen::MoveGen;
use crate::rules::{Piece, Shapes, Spin, H, W};

fn board_from_record(v: &Value) -> Option<Board> {
    let rows = v["board"].as_array()?;
    let mut b = Board::default();
    for (y, r) in rows.iter().enumerate().take(H as usize) {
        for (x, ch) in r.as_str().unwrap_or("").chars().enumerate().take(W as usize) {
            if ch != '.' {
                b.rows[y] |= 1 << x;
            }
        }
    }
    Some(b)
}

fn top_of(b: &Board) -> i32 {
    (0..H).find(|&y| b.rows[y as usize] != 0).unwrap_or(H)
}

fn cells_of(s: &SpinSlot, sh: &Shapes) -> Vec<(i32, i32)> {
    let mut c: Vec<(i32, i32)> =
        sh.cells[s.piece as usize][s.rot as usize].iter().map(|&(dx, dy)| (s.x + dx as i32, s.y + dy as i32)).collect();
    c.sort();
    c
}

/// `want`: only show examples containing a slot of this piece clearing at least this
/// many rows (e.g. a Z-spin triple).
pub fn run(path: &str, show: usize, limit: usize, want: Option<(Piece, u32)>) -> Result<(), String> {
    let matches = |s: &SpinSlot| match want {
        Some((p, l)) => s.piece == p && s.lines >= l,
        None => s.lines > 0,
    };
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {}", path, e))?;
    let sh = Shapes::new();
    let mut mg = MoveGen::new();
    let mut buf = Vec::new();
    let (mut boards, mut reachable, mut found, mut scanned, mut scanned_clear) = (0u64, 0u64, 0u64, 0u64, 0u64);
    let mut missed: Vec<String> = Vec::new();
    let mut examples: Vec<(Board, Vec<SpinSlot>, Vec<bool>)> = Vec::new();

    for line in text.lines().take(limit) {
        let v: Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let b = match board_from_record(&v) {
            Some(b) => b,
            None => continue,
        };
        boards += 1;
        let mut slots = Vec::new();
        scan_spin_slots(&b, top_of(&b), |s| slots.push(s));
        scanned += slots.len() as u64;
        scanned_clear += slots.iter().filter(|s| s.lines > 0).count() as u64;
        let slot_cells: Vec<(Piece, Vec<(i32, i32)>)> = slots.iter().map(|s| (s.piece, cells_of(s, &sh))).collect();
        let mut reach_flags = vec![false; slots.len()];
        for piece in [Piece::S, Piece::Z, Piece::L, Piece::J, Piece::I] {
            mg.generate(&b, piece, &sh, &mut buf);
            for p in buf.iter().filter(|p| p.spin == Spin::Full) {
                let mut c: Vec<(i32, i32)> = p.abs_cells(&sh).to_vec();
                c.sort();
                reachable += 1;
                match slot_cells.iter().position(|(sp, sc)| *sp == piece && *sc == c) {
                    Some(i) => {
                        found += 1;
                        reach_flags[i] = true;
                    }
                    None => {
                        if missed.len() < 5 {
                            missed.push(format!("{} n={} {:?} cells {:?}", v["replay"], v["n"], piece, c));
                        }
                    }
                }
            }
        }
        if examples.len() < show && slots.iter().enumerate().any(|(i, s)| matches(s) && (want.is_none() || reach_flags[i])) {
            examples.push((b, slots, reach_flags));
        }
    }

    println!("boards scanned                    {}", boards);
    println!("reachable non-T spins (movegen)   {}", reachable);
    println!("  also found by the bitmask scan  {} ({:.1}%)", found, 100.0 * found as f64 / reachable.max(1) as f64);
    println!("immobile spots found by the scan  {}  ({} would clear lines)", scanned, scanned_clear);
    println!("  of which reachable              {:.1}%  (the rest are cavities no piece can rotate into)",
        100.0 * found as f64 / scanned.max(1) as f64);
    for m in &missed {
        println!("  MISSED: {}", m);
    }
    for (i, (b, slots, reach)) in examples.iter().enumerate() {
        print_example(i + 1, b, slots, reach, &sh, &matches);
    }
    Ok(())
}

/// ASCII view of one board: the board, the raw masks for the best slot's rotation, and
/// every line-clearing slot drawn over the stack.
fn print_example(k: usize, b: &Board, slots: &[SpinSlot], reach: &[bool], sh: &Shapes, matches: &dyn Fn(&SpinSlot) -> bool) {
    let top = top_of(b);
    let from = (top - 2).max(0) as usize;
    let best = slots.iter().enumerate().filter(|(_, s)| matches(s)).max_by_key(|(i, s)| (reach[*i], s.lines)).unwrap();
    let (bi, bs) = (best.0, *best.1);
    println!("\n==== example {}: best slot {:?} rot {} at box ({}, {}), clears {}, {} ====",
        k, bs.piece, bs.rot, bs.x, bs.y, bs.lines, if reach[bi] { "reachable" } else { "NOT reachable" });

    // 1. Board with that slot's cells.
    let cells = cells_of(&bs, sh);
    println!("board ('#' filled, '{}' = slot cells):", bs.piece_letter());
    for y in from..H as usize {
        let mut s = String::new();
        for x in 0..W as usize {
            let c = if cells.contains(&(x as i32, y as i32)) {
                bs.piece_letter()
            } else if b.rows[y] >> x & 1 != 0 {
                '#'
            } else {
                '.'
            };
            s.push(c);
        }
        println!("  {:>2} |{}|", y, s);
    }

    // 2. Raw masks for this piece+rotation around the slot row. A mask row reads left to
    //    right as box column X = 0..9; '1' = the box fits there.
    let pc = &sh.cells[bs.piece as usize][bs.rot as usize];
    let bits = |m: u32| -> String { (0..W).map(|x| if m >> (x + 4) & 1 != 0 { '1' } else { '.' }).collect() };
    println!("masks for {:?} rot {} (column = box X 0..9):", bs.piece, bs.rot);
    println!("   box y |  fits(y)  | immobile(y) = fits & !left & !right & !up & !down");
    for y in (bs.y - 2).max(0)..=(bs.y + 1).min(H - 1) {
        let (u, h, d) = (fits_mask(b, pc, y - 1), fits_mask(b, pc, y), fits_mask(b, pc, y + 1));
        println!("   {:>5} |{}| {}", y, bits(h), bits(immobile_mask(u, h, d)));
    }

    // 3. Every line-clearing slot on this board.
    println!("line-clearing slots on this board:");
    for (i, s) in slots.iter().enumerate().filter(|(_, s)| s.lines > 0) {
        println!("   {:?} rot {} box ({:>2},{:>2})  clears {}  {}", s.piece, s.rot, s.x, s.y, s.lines,
            if reach[i] { "reachable" } else { "not reachable" });
    }
}

impl SpinSlot {
    fn piece_letter(&self) -> char {
        self.piece.letter().to_ascii_uppercase()
    }
}
