# tetrp — the vendored reconstruction engine

Upstream: <https://github.com/jush0147/tetrp> (MIT). A third-party reimplementation of
TETR.IO's engine.

## What it is for

A `.ttr` replay contains **keypresses and nothing else** — no board state, per-frame or
otherwise. For the 5han run used throughout the analysis: 7,398 `keydown`, 7,396
`keyup`, 218 `ige`, and two markers, against 2,018 placements whose boards exist
nowhere in the file.

tetrp is what turns one into the other:

| file | role |
| --- | --- |
| `src/random.js` | MINSTD bag generator — the seed reproduces the exact piece sequence |
| `src/engine.js` | gravity, lock delay, sub-frame DAS/ARR, spawn, hold, clears |
| `src/board.js` | cell geometry and commits; `B.cells` is what `tools/verify.mjs` checks against |
| `src/attack.js` | attack values, the garbage queue, cancellation |
| `src/replay/{parser,profile,reconstruction}.js` | `.ttr` → canonical timeline → driven engine |

Our own `core/` engine cannot substitute. It implements guideline Tetris; TETR.IO
differs in sub-frame DAS/ARR, lock delay, `all-mini+` spin rules and the bag PRNG, and
any one mismatch desyncs within a few pieces. That our SRS tables can *represent* every
harvested placement (17,005 of 17,005 resolve, fit and clear correctly) is a check on
the conversion, not an independent derivation.

## The patch

`data/zenith-tetrp.patch` — 97 insertions across 6 files. Upstream supports 40L and
Tetra League and refuses Zenith outright. Every hunk is accommodation, not a change to
how the game is simulated:

- **`bagtype: "zenith"`** accepted as 7-bag. Verified downstream: the reconstructed
  piece order forms 1,745 consecutive valid 7-bag permutations across all runs.
- **Solo targeting.** Zenith emits `targets: []` and retargets as the tower's
  population changes; the 1v1 path requires exactly one fixed target.
- **`kev` events** (knockouts elsewhere in the tower) routed to metadata. They carry no
  `amt` and never touch this player's board.
- **Missing `cid`.** Solo garbage has no peer connection. The key must be *absent*
  rather than present-and-undefined, which `reconstruction.js` validates with `in`.
- **No `interaction_confirm`.** In 1v1 garbage is a two-step handshake; in Zenith the
  `interaction` *is* the arrival, so packets otherwise sit pending forever.
- **Placement resolution** routed through the TL path. Zenith is pseudo-versus and
  declares `hasgarbage: true`; without this nothing sets `nextWillTank`.
- **`gamemode: "zenithex"`** accepted alongside `"zenith"`. Zenith Expert is the same
  engine on a separate leaderboard — `zenith_expert` was already a known profile key
  upstream, and the only thing rejecting these replays was the mode string in
  `parser.js`. `prepareReplay`'s `isZenith` had to widen with it, or every `zenith_*`
  option would have been treated as a rules difference and failed the profile check.
- **Ack-cancellation disabled.** All tower garbage is labelled `from: 'P2'`, so the 1v1
  "attack in flight annihilates attack in flight" rule otherwise ate the first ~96
  packets.

To re-apply after a fresh clone:

    git clone --depth 1 https://github.com/jush0147/tetrp.git vendor/tetrp
    cd vendor/tetrp && git apply ../../data/zenith-tetrp.patch

## What this still cannot do

Zenith reconstructs **except for the garbage**. The tower's own garbage is generated
client-side and is absent from the replay: recorded packets total 12,956 lines against
22,858 actually received across the 51 replays — **57% coverage**, median 60%.

That is why `tools/harvest.mjs` suppresses garbage entirely and the dataset is cut at
the point the board stops recovering. See the header comments there.
