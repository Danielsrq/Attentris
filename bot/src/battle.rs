//! TetrisBattle's piece set, rotation and T-spin rule (external/tetrisbattle: tetris.py,
//! ai/search.py `fast_rotate`), so the bot can be tuned and played under the rules of a
//! friend's RL bot. Board coordinates are ours (y down, 40 rows, the top 20 the buffer);
//! his grid's visible top row is our row BUFFER. Quirks kept on purpose, for parity:
//!
//!   * No kick table. A rotation that collides is classified by the side the colliding
//!     cells are on (left / right / down, by the rotated piece's extremes), ties going to
//!     the side counted first in his x-major cell order; then a short list of shifts for
//!     that side is tried. A rotation that fails leaves the piece -- but still re-checks
//!     the T-spin.
//!   * A T-spin is set by the last rotation (the T resting right after it, both bottom
//!     corners and one top corner filled; full when both "front" corners are) and KEPT
//!     through later shifts, soft drops and the hard drop.
//!   * The corner check uses Python's negative indexing: a corner at column -1 reads
//!     column 9.
//!   * At most 3 actions while the piece rests (lock delay), then it must drop.

use crate::board::Board;
use crate::rules::{Piece, Shapes, BUFFER, H, W};

/// His OFFSETS per piece (our Piece order) and rotation, in his x-major cell order.
pub const BATTLE_CELLS: [[[(i8, i8); 4]; 4]; 7] = [
    [[(0, 2), (1, 2), (2, 2), (3, 2)], [(2, 0), (2, 1), (2, 2), (2, 3)], [(0, 1), (1, 1), (2, 1), (3, 1)], [(1, 0), (1, 1), (1, 2), (1, 3)]], // I
    [[(1, 1), (1, 2), (2, 1), (2, 2)], [(1, 1), (1, 2), (2, 1), (2, 2)], [(1, 1), (1, 2), (2, 1), (2, 2)], [(1, 1), (1, 2), (2, 1), (2, 2)]], // O
    [[(0, 2), (1, 1), (1, 2), (2, 2)], [(1, 1), (1, 2), (1, 3), (2, 2)], [(0, 2), (1, 2), (1, 3), (2, 2)], [(0, 2), (1, 1), (1, 2), (1, 3)]], // T
    [[(0, 2), (1, 1), (1, 2), (2, 1)], [(1, 1), (1, 2), (2, 2), (2, 3)], [(0, 2), (1, 1), (1, 2), (2, 1)], [(0, 0), (0, 1), (1, 1), (1, 2)]], // S
    [[(0, 1), (1, 1), (1, 2), (2, 2)], [(1, 1), (1, 2), (2, 0), (2, 1)], [(0, 1), (1, 1), (1, 2), (2, 2)], [(0, 2), (0, 3), (1, 1), (1, 2)]], // Z
    [[(0, 1), (0, 2), (1, 2), (2, 2)], [(1, 1), (1, 2), (1, 3), (2, 1)], [(0, 2), (1, 2), (2, 2), (2, 3)], [(0, 3), (1, 1), (1, 2), (1, 3)]], // J
    [[(0, 2), (1, 2), (2, 1), (2, 2)], [(1, 1), (1, 2), (1, 3), (2, 3)], [(0, 2), (0, 3), (1, 2), (2, 2)], [(0, 1), (1, 1), (1, 2), (1, 3)]], // L
];

/// Spawn: box at x = 4, two rows above the field, rotation 0 (Tetris.reset_pos).
pub const SPAWN: (i32, i32) = (4, BUFFER - 2);

/// Lock delay: actions allowed while resting (search.py MAX_GROUND_OPS).
pub const MAX_GROUND_OPS: u8 = 3;

/// His shapes as our Shapes (cells sorted by (y, x), as the move generator's keys need).
pub fn shapes() -> Shapes {
    let mut s = Shapes::new();
    for p in 0..7 {
        for r in 0..4 {
            let mut c = BATTLE_CELLS[p][r];
            c.sort_by_key(|&(x, y)| (y, x));
            s.cells[p][r] = c;
        }
    }
    s
}

#[inline]
fn filled(b: &Board, x: i32, y: i32) -> bool {
    b.rows[y as usize] >> x & 1 != 0
}

/// fast_collide: off the sides or below the floor collides; above the field does not.
#[inline]
pub fn collide(b: &Board, piece: Piece, r: usize, x: i32, y: i32) -> bool {
    for &(dx, dy) in &BATTLE_CELLS[piece as usize][r] {
        let (gx, gy) = (x + dx as i32, y + dy as i32);
        if gx < 0 || gx >= W || gy > H - 1 {
            return true;
        }
        if gy >= 0 && filled(b, gx, gy) {
            return true;
        }
    }
    false
}

/// A grid read with his Python indexing: column -1 is column 9, and a row above his
/// grid (which is the 20 visible rows plus one row per garbage line received) counts
/// from its bottom.
#[inline]
fn py_filled(b: &Board, x: i32, y: i32) -> bool {
    let x = if x < 0 { x + W } else { x };
    let excess = b.excess as i32;
    let y = if y < BUFFER - excess { y + (H - BUFFER) + excess } else { y };
    (0..H).contains(&y) && filled(b, x, y)
}

/// tspinCheck + tspin_kind for a T at (x, y) in rotation r: 0 none, 1 Mini, 2 full.
pub fn tspin(b: &Board, r: usize, x: i32, y: i32) -> u8 {
    if !collide(b, Piece::T, r, x, y + 1) || x + 2 >= W || y + 3 > H - 1 {
        return 0;
    }
    let (tl, tr, bl, br) = (py_filled(b, x, y + 1), py_filled(b, x + 2, y + 1), py_filled(b, x, y + 3), py_filled(b, x + 2, y + 3));
    if !((tl && bl && br) || (bl && br && tr)) {
        return 0;
    }
    let front = match r {
        0 => tl && tr,
        1 => tr && br,
        2 => bl && br,
        _ => tl && bl,
    };
    if front { 2 } else { 1 }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Side {
    Left,
    Right,
    Down,
}

const KICK_LEFT: [(i32, i32); 9] = [(0, 0), (0, 1), (0, -1), (1, 0), (1, 1), (1, -1), (2, 0), (2, 1), (2, -1)];
const KICK_RIGHT: [(i32, i32); 6] = [(-1, 0), (-1, 1), (-1, -1), (-2, 0), (-2, 1), (-2, -1)];
const KICK_DOWN: [(i32, i32); 5] = [(0, -1), (-1, -1), (1, -1), (-2, -1), (2, -1)];

/// fast_rotate: rotate by `dir` (+1 cw, -1 ccw). Returns (x, y, rotation, tspin).
pub fn rotate(b: &Board, piece: Piece, x: i32, y: i32, r: usize, dir: i32) -> (i32, i32, usize, u8) {
    let to = ((r as i32 + dir).rem_euclid(4)) as usize;
    let cells = &BATTLE_CELLS[piece as usize][to];
    let left_most = cells.iter().map(|c| c.0).min().unwrap();
    let right_most = cells.iter().map(|c| c.0).max().unwrap();
    let down_most = cells.iter().map(|c| c.1).max().unwrap();
    // Collision sides with their counts, in first-seen order (Counter.most_common ties).
    let mut sides: [(Side, u8); 3] = [(Side::Left, 0); 3];
    let mut n = 0usize;
    let bump = |s: Side, sides: &mut [(Side, u8); 3], n: &mut usize| {
        if let Some(e) = sides[..*n].iter_mut().find(|e| e.0 == s) {
            e.1 += 1;
        } else {
            sides[*n] = (s, 1);
            *n += 1;
        }
    };
    for &(dx, dy) in cells {
        let (gx, gy) = (x + dx as i32, y + dy as i32);
        if gx > W - 1 {
            bump(Side::Right, &mut sides, &mut n);
        }
        if gx < 0 {
            bump(Side::Left, &mut sides, &mut n);
        }
        if gy > H - 1 {
            bump(Side::Down, &mut sides, &mut n);
        }
        if (0..W).contains(&gx) && (BUFFER..H).contains(&gy) && filled(b, gx, gy) {
            if dx == left_most {
                bump(Side::Left, &mut sides, &mut n);
            } else if dx == right_most {
                bump(Side::Right, &mut sides, &mut n);
            } else if dy == down_most {
                bump(Side::Down, &mut sides, &mut n);
            }
        }
    }
    let (mut nx, mut ny, mut nr) = (x, y, to);
    if n > 0 {
        let best = sides[..n].iter().map(|e| e.1).max().unwrap();
        let side = sides[..n].iter().find(|e| e.1 == best).unwrap().0;
        let kicks: &[(i32, i32)] = match side {
            Side::Left => &KICK_LEFT,
            Side::Right => &KICK_RIGHT,
            Side::Down => &KICK_DOWN,
        };
        match kicks.iter().find(|&&(kx, ky)| !collide(b, piece, to, x + kx, y + ky)) {
            Some(&(kx, ky)) => {
                nx += kx;
                ny += ky;
            }
            None => nr = r,
        }
    }
    let t = if piece == Piece::T { tspin(b, nr, nx, ny) } else { 0 };
    (nx, ny, nr, t)
}

/// `bot battle-parity --data dump.jsonl` (bot/tools/battle_dump.py): for every recorded
/// TetrisBattle decision, our placement set (cells + best T-spin flag per cell set) must
/// equal his search()'s, and our TETR.IO attack, combo and B2B must equal his.
pub fn parity(path: &str) -> Result<(), String> {
    use crate::rules::Spin;
    use std::collections::BTreeMap;
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {}", path, e))?;
    let rules = crate::rules::Rules::battle();
    let shapes = shapes();
    let mut mg = crate::movegen::MoveGen::new();
    mg.battle = true;
    mg.all_spin = false;
    let mut out = Vec::new();
    let (mut n, mut sets_ok, mut attack_ok, mut b2b_ok, mut combo_ok, mut skipped) = (0, 0, 0, 0, 0, 0);
    let (mut missing, mut extra, mut tdiff) = (0usize, 0usize, 0usize);
    let mut shown = 0;
    let spin_of = |t: i64| match t {
        2 => Spin::Full,
        1 => Spin::Mini,
        _ => Spin::None,
    };
    for line in text.lines() {
        let v: serde_json::Value = serde_json::from_str(line).map_err(|e| e.to_string())?;
        n += 1;
        let excess = v["excess"].as_i64().unwrap() as i32;
        let shift = BUFFER - excess; // his row y -> our row y + shift
        let mut b = Board { excess: excess as u16, ..Board::default() };
        for (y, row) in v["rows"].as_array().unwrap().iter().enumerate() {
            let ry = y as i32 + shift;
            if ry < 0 {
                continue;
            }
            for (x, ch) in row.as_str().unwrap().chars().enumerate() {
                if ch == 'X' {
                    b.rows[ry as usize] |= 1 << x;
                }
            }
        }
        let piece = Piece::from_letter(v["piece"].as_str().unwrap().chars().next().unwrap()).unwrap();
        let standard = v["px"].as_i64() == Some(4) && v["py"].as_i64() == Some((excess - 2) as i64)
            && v["rot"].as_i64() == Some(0) && v["tspin"].as_i64() == Some(0);
        if standard {
            let sorted = |mut c: Vec<(i32, i32)>| {
                c.sort();
                c
            };
            let mut his: BTreeMap<Vec<(i32, i32)>, u8> = BTreeMap::new();
            for p in v["placements"].as_array().unwrap() {
                let cells: Vec<(i32, i32)> = p["cells"].as_array().unwrap().iter()
                    .map(|c| (c[0].as_i64().unwrap() as i32, c[1].as_i64().unwrap() as i32 + shift)).collect();
                let e = his.entry(sorted(cells)).or_insert(0);
                *e = (*e).max(p["t"].as_i64().unwrap() as u8);
            }
            mg.generate(&b, piece, &shapes, &mut out);
            let mut ours: BTreeMap<Vec<(i32, i32)>, u8> = BTreeMap::new();
            for p in &out {
                let t = match p.spin {
                    Spin::Full => 2,
                    Spin::Mini => 1,
                    Spin::None => 0,
                };
                ours.insert(sorted(p.abs_cells(&shapes).to_vec()), t);
            }
            let m = his.keys().filter(|k| !ours.contains_key(*k)).count();
            let x = ours.keys().filter(|k| !his.contains_key(*k)).count();
            let d = his.iter().filter(|(k, t)| ours.get(*k).is_some_and(|o| o != *t)).count();
            missing += m;
            extra += x;
            tdiff += d;
            if m == 0 && x == 0 && d == 0 {
                sets_ok += 1;
            } else if shown < 4 {
                shown += 1;
                eprintln!("decision {} ({:?}): {} missing, {} extra, {} spin flags differ ({} his vs {} ours)",
                    n, piece, m, x, d, his.len(), ours.len());
                if shown == 1 {
                    for y in BUFFER as usize..H as usize {
                        eprintln!("    {}  row {}", (0..W).map(|x| if b.rows[y] >> x & 1 != 0 { '#' } else { '.' }).collect::<String>(), y);
                    }
                    for (k, t) in his.iter().filter(|(k, t)| ours.get(*k).is_some_and(|o| o != *t)) {
                        eprintln!("    cells {:?}: his t {} ours t {}", k, t, ours[k]);
                    }
                }
            }
        } else {
            skipped += 1;
        }
        // Attack, combo and B2B of the placement his bot chose.
        let o = &v["outcome"];
        let lines = o["cleared"].as_i64().unwrap() as u32;
        let spin = spin_of(v["chosen"]["t"].as_i64().unwrap());
        let combo_after = o["combo_after"].as_i64().unwrap() as i32;
        let b2b_after = o["b2b_after"].as_i64().unwrap() as i32;
        let combo_before = v["combo_before"].as_i64().unwrap() as i32;
        let b2b_before = v["b2b_before"].as_i64().unwrap() as i32;
        combo_ok += ((if lines > 0 { combo_before + 1 } else { -1 }) == combo_after) as usize;
        b2b_ok += (crate::rules::next_b2b(b2b_before, spin, lines) == b2b_after) as usize;
        let atk = rules.battle_attack(spin, lines, combo_after, b2b_after, o["perfect"].as_bool().unwrap());
        if atk == o["attack"].as_i64().unwrap() as i32 {
            attack_ok += 1;
        } else if shown < 8 {
            shown += 1;
            eprintln!("decision {}: attack {} vs his {} (lines {}, {:?}, combo {}, b2b {})", n, atk, o["attack"], lines, spin, combo_after, b2b_after);
        }
    }
    let checked = n - skipped;
    println!("{} decisions: placement sets identical {}/{} ({} missing, {} extra, {} spin flags differ; {} non-spawn starts skipped)",
        n, sets_ok, checked, missing, extra, tdiff, skipped);
    println!("attack identical {}/{}, combo {}/{}, B2B {}/{}", attack_ok, n, combo_ok, n, b2b_ok, n);
    if sets_ok == checked && attack_ok == n && combo_ok == n && b2b_ok == n {
        Ok(())
    } else {
        Err("battle parity mismatches".into())
    }
}
