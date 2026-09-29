# Replay JSONL — the viewer's input contract

`replay.tscn` renders a post-processed JSONL file. It does **not** read `.ttr` files and
does **not** simulate: every record carries its own board, so step *n* draws `board(n)`
with the piece at its landing cells, and step *n+1*'s board is already the post-clear
result. Nothing can drift, because nothing is derived.

Any producer that emits this format can be watched in the viewer — the harvested TETR.IO
dataset and a bot's own games are the same file type. Reader: `core/replay_log.gd`.

## Shape

One JSON object per line. No wrapping array, no trailing commas, UTF-8, `\n` endings.
Records for a run must be contiguous and in placement order; several runs may share a
file and the viewer's dropdown lists them in first-appearance order.

## Required fields

| field | type | meaning |
|---|---|---|
| `replay` | string | run id. Groups records and names the dropdown entry. |
| `n` | int | placement number, 0-based. Shown as `#n`. |
| `board` | array of 40 strings, each 10 chars | the stack **before** this piece locks. |
| `piece` | string | the piece being placed: one of `i o t s z j l`. |
| `place` | object | where it landed — see below. |

`place` needs only two of its keys:

| key | type | meaning |
|---|---|---|
| `type` | string | piece letter, same alphabet as `piece`. |
| `cells` | array of 4 `[x, y]` | **absolute** board cells the piece occupies at lock. |

`cells` is absolute and so convention-free. The viewer recovers `(rotation, position)` by
finding which of its own four rotations reproduces the cell set, which is why a producer
does not have to match our kick tables or pivot convention. A record whose cells match no
rotation is skipped rather than drawn wrong.

## Board encoding

- 40 rows, `y = 0` at the **top**, index 39 is the floor. 10 columns, `x = 0` at the left.
- `.` empty.
- `G` garbage — **uppercase only**. Lowercase `g` is not a piece letter and silently reads
  as empty.
- `i o t s z j l` a settled cell of that piece, for colour. Case-insensitive.
- Rows equal to `..........` are skipped without parsing, so an empty board is cheap.

## Optional fields

Everything below is additive: the viewer works without it, and shows more with it.

| field | type | used for |
|---|---|---|
| `user`, `mods` | string | the info panel |
| `tier` | string | per-run counts in the dropdown; any string works as a bucket |
| `frame` | int | real-time playback pacing (60 Hz) |
| `lock_queue` | array of letters | NEXT panel — see the warning below |
| `lock_hold` | letter or null | HOLD panel at lock time |
| `lock_hold_locked` | bool | dims HOLD when it has been spent on this piece |
| `queue`, `hold`, `used_hold` | | spawn-time state; the learner's input, and the fallback when no `lock_*` is given |
| `spin`, `place.spin`, `place.kick` | | spin readout |
| `cleared`, `cleared_rows` | int, array | line-clear reporting |
| `height`, `sealed`, `hole_run` | int | timeline colouring and re-cutting a dataset |

### Emit `lock_queue` / `lock_hold`, not just `queue` / `hold`

`queue` and `hold` are the state when the piece **spawned** — that is what a learner
consumes (board + piece + queue + hold → action). But the viewer draws the board with the
piece at its **landing** position, and a hold swap happens between those two instants. Put
the spawn-time queue next to a placed piece and you are showing two different moments at
once, which looks like the piece on the board also sitting in NEXT. Emit both; the viewer
prefers `lock_*` for display and falls back to the spawn-time keys.

## Caveat on `replay`, `user`, `mods`, `tier`

These four are extracted by **string slicing, not JSON parsing** — at ~12k lines, parsing
every record just to group them is the whole load time. So they must be plain JSON strings
with no escapes: no `"` and no `\` in a run id.

## Minimal example

```json
{"replay":"bot_v1__0001","n":0,"board":["..........", "... 38 more ...",".........."],
 "piece":"t","place":{"type":"t","cells":[[3,39],[4,39],[5,39],[4,38]]}}
```

## Loading a file other than the default

`ReplayLog.DEFAULT_PATH` is `res://data/zenith_full.jsonl`; `load_file(path)` takes any
`res://` path. Loading is lazy — lines are kept as text and only the selected run is
parsed — so a large multi-run file is fine.
