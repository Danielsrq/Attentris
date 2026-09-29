class_name EvalWeights
extends Resource

## Feature weights for BoardEvaluator. Negative means "worse board".
##
## THE CENTRAL PROBLEM this preset solves: clearing a line is structurally good by
## every other feature at once -- it lowers aggregate height, lowers max height, and
## removes a row's worth of transitions. Measured against these weights a single is
## worth roughly +14 before any penalty, so a small penalty does nothing and the
## solver clears singles all day. A penalty big enough to stop it (-18) instead made
## it refuse to clear at all and stack to death.
##
## The way out is not a better penalty number. It is to make small clears
## STRUCTURALLY IMPOSSIBLE by hard-reserving the well column: if one column is never
## filled, no row can ever be complete except by an I dropped into it. Quads stop
## being something the evaluator has to be bribed into and become the only clear
## available. `w_well_blocked` is therefore a near-ban rather than a nudge.
##
## That alone would stack to death when no I arrives, so `danger_height` releases
## both restrictions once the stack gets tall: above it, filling the well and taking
## a small clear become cheap again, and the solver digs out.

@export_group("Structure")
@export var w_aggregate_height: float = -0.20
@export var w_max_height: float = -1.20
## TRUE holes: empty cells with no 4-connected path to open air. Sealed. Severe.
@export var w_holes: float = -8.0
## Covered but still connected to the surface -- an overhang, not a hole. A T-spin
## slot is exactly this shape, so it must stay cheap or the solver can never build
## one. It is not free: an overhang still constrains what fits underneath.
## Default matches w_holes, which takes the cheap path and reproduces the naive
## behaviour. A quad solver WANTS overhangs priced as holes -- they are hole-shaped
## futures. Only a spin evaluator, which builds them deliberately, should lower this.
@export var w_overhang: float = -8.0
@export var w_covered_cells: float = -0.50
@export var w_bumpiness: float = -0.60
@export var w_row_transitions: float = -1.60
@export var w_col_transitions: float = -3.20
@export var w_landing_height: float = -2.20

@export_group("Well")
## Column reserved for the I piece. Fixed, not chosen per solve -- a well that moves
## between solves is never actually dug, which was the second bug.
## -1 picks the emptiest edge column each solve (unstable; kept for comparison).
@export var well_column: int = 9
## Paid per row that is genuinely complete except for the well.
@export var w_well_depth: float = 1.5
## Paid against how full those four rows are (0..1). This is the gradient the search
## climbs while building; without it the quad rate collapses, because a complete row
## is worth something only at the instant its last cell lands.
@export var w_well_progress: float = 14.0
## Paid once the well is 4+ deep, i.e. a quad is actually on offer.
@export var w_quad_ready: float = 12.0
## Per filled row in the reserved well. Large enough to be a ban, not a nudge.
@export var w_well_blocked: float = -60.0

@export_group("Clears")
@export var reward_quad: float = 80.0
## Spin clears, keyed by lines. Roughly 20 per point of attack, matching reward_quad
## at 80 for a quad's 4. Without these a TSD falls through to penalty_small_clear
## and the solver is punished for cashing the slot it just built.
## Charged when an I is spent on anything but a quad. The I is the only piece with
## no substitute -- a 1-wide 4-deep well accepts nothing else -- so a flat I is not
## just a mediocre placement, it is the loss of the next quad. Discounted when
## another I is already visible, since then the waste is recoverable.
@export var w_i_wasted: float = 0.0
## How much of that penalty survives when a spare I is in hold or the queue.
@export_range(0.0, 1.0, 0.05) var i_wasted_relief: float = 0.3
## Stack height (well excluded) above which spending an I on a non-quad stops being
## waste and starts being survival. Half the visible field by default. Below it the
## I is a future quad; above it, holding out for one is how you top out.
@export var i_waste_height: int = 10

@export var reward_spin_clear: Dictionary = {1: 40.0, 2: 80.0, 3: 120.0}
## Per line, for clears of 1-3.
@export var penalty_small_clear: float = -10.0
## Paid when a clear continues an existing back-to-back chain.
@export var w_b2b_kept: float = 6.0
## Charged when a clear BREAKS a live chain. Larger than the keep reward on purpose:
## losing a chain costs the 1.5x multiplier on everything that follows, so it should
## take a real reason. Scaled by `danger_relief`, because surviving beats the chain.
@export var w_b2b_broken: float = -25.0

@export_group("Dependencies and parity")
## Columns needing a vertical I to fill (notch depth >= 3), excluding the reserved
## well. You get one I per bag and it is already spoken for, so a SECOND I-dependency
## is a real stall.
@export var w_i_dependency: float = 0.0
## Notches exactly 2 deep: fillable, but only by a few specific pieces.
@export var w_deep_notch: float = 0.0
## Multiplier on w_i_dependency for a notch whose fixing piece IS available -- in
## hold or the visible queue. A 3-deep notch with an I coming is a plan, not a
## dependency; with no I in sight it is a stall. 1.0 means "no relief".
##
## The well's own claim is subtracted first: if the well is deep enough to cash, the
## next I is already spoken for and cannot also fix a notch elsewhere.
## Relief when the fixing piece is one step away in the queue. 1.0 means no relief.
@export_range(0.0, 1.0, 0.05) var dep_relief_if_available: float = 1.0
## Relief when the fixing piece is sitting in HOLD. Strongest, because hold can be
## summoned at any time -- it is not queued behind anything.
@export_range(0.0, 1.0, 0.05) var dep_relief_hold: float = 0.10
## Added to the relief factor per extra queue position, so a piece five away is
## barely reassuring. Relief of 1.0 is "no relief at all".
@export var dep_relief_falloff: float = 0.12
## Dellacherie's cumulative wells: total notch depth across the board.
@export var w_cumulative_well: float = 0.0
## |white - black| over filled cells on a checkerboard. Every tetromino except T
## covers 2-2 in every rotation and position; only T covers 3-1. So imbalance can
## only be resolved by T pieces, and T arrives once per bag.
@export var w_parity: float = 0.0

@export_group("T-spin slots")
## A slot whose two rows are otherwise full -- cashable right now. Scaled by whether
## a T is actually available, because a slot with no T is not an asset.
@export var w_slot_ready: float = 0.0
## Partial completeness of those rows, 0..1. The gradient to climb while building.
@export var w_slot_progress: float = 0.0
## A slot whose nub column IS the well: the T plugs the well itself, so both rows can
## complete. Slots anywhere else stay blocked while the well is open.
@export var w_slot_at_well: float = 0.0

@export_group("Survival")
## Max height (excluding the well) at which the solver stops protecting the well and
## starts digging. Too low and it never builds a quad; too high and it tops out.
@export var danger_height: int = 13
## How much of the well ban and small-clear penalty survives in danger. 0 would mean
## "abandon the strategy entirely"; a little residue keeps it from thrashing.
@export_range(0.0, 1.0, 0.05) var danger_relief: float = 0.15

static func quad() -> EvalWeights:
	return EvalWeights.new()

## Adds dependency counting and checkerboard parity on top of `quad`.
static func quad_v2() -> EvalWeights:
	var w := EvalWeights.new()
	w.w_i_dependency = -14.0
	w.w_deep_notch = -2.5
	w.w_cumulative_well = -0.8
	w.w_parity = -0.8
	return w

## The distillation hypothesis: most of the 20 terms are restating three ideas --
## holes, dependencies, and parity -- so the rest can go.
##
## Dropped as redundant: aggregate height, bumpiness, row transitions, column
## transitions, landing height, covered depth. Each is a proxy for surface roughness
## or burial, which explicit dependency and hole counting should already capture.
##
## Kept, and NOT because they are board judgements: the well terms and clear rewards
## define the OBJECTIVE (build one deliberate I-dependency, cash it as a quad). Those
## are what the solver is for, not heuristics about board quality.
##
## max_height is kept as the only survival term -- see minimal_no_height for why.
static func minimal() -> EvalWeights:
	var w := EvalWeights.new()
	w.w_aggregate_height = 0.0
	w.w_bumpiness = 0.0
	w.w_row_transitions = 0.0
	w.w_col_transitions = 0.0
	w.w_landing_height = 0.0
	w.w_covered_cells = 0.0
	w.w_i_dependency = -14.0
	w.w_deep_notch = -2.5
	w.w_cumulative_well = -0.8
	w.w_parity = -0.8
	return w

## minimal without any height term. Nothing then pushes the stack DOWN: a flat stack
## at height 15 scores the same as one at height 3, so the only downward pressure is
## the quad reward itself. Included to find out whether that is enough.
static func minimal_no_height() -> EvalWeights:
	var w := minimal()
	w.w_max_height = 0.0
	return w

## minimal without parity, to find out whether parity earns its place or whether the
## dependency terms were doing all the work.
static func minimal_no_parity() -> EvalWeights:
	var w := minimal()
	w.w_parity = 0.0
	return w

## The distilled heuristic set: holes + GRADED dependency counts + parity + max
## height, with no lookahead relief. Beat the 20-term version when it was measured.
##
## NAMED FOR ITS CONTENT, not its ranking. An earlier version of this was called
## best(), which stopped being true two features later and made a benchmark table
## read as if the good preset was losing.
static func distilled() -> EvalWeights:
	return minimal_count_only()

## distilled() plus lookahead-aware dependencies: a notch is discounted when a piece
## that can fill it is actually available, most strongly when it sits in hold.
## This is what main.gd runs.
static func lookahead() -> EvalWeights:
	var w := distilled()
	w.dep_relief_if_available = 0.25
	w.dep_relief_hold = 0.10
	w.dep_relief_falloff = 0.12
	return w

## distilled() without the small-clear penalty. The hypothesis: that penalty is already
## dead weight, because what actually stops singles is the well ban making them
## nearly impossible, not the -10 making them unattractive.
static func distilled_no_small_penalty() -> EvalWeights:
	var w := distilled()
	w.penalty_small_clear = 0.0
	return w

## Three dependency terms is itself redundant: cumulative_well is the SUM of notch
## depths, which already grades a 3-deep notch above a 2-deep one. These two presets
## ask whether the graded count and the sum are saying the same thing.
static func minimal_sum_only() -> EvalWeights:
	var w := minimal()
	w.w_i_dependency = 0.0
	w.w_deep_notch = 0.0
	w.w_cumulative_well = -3.0   # carries the whole signal, so weighted up
	return w

static func minimal_count_only() -> EvalWeights:
	var w := minimal()
	w.w_cumulative_well = 0.0
	return w

## distilled + lookahead + a penalty for spending the I on anything but a quad.
static func lookahead_i() -> EvalWeights:
	var w := lookahead()
	w.w_i_wasted = -25.0
	w.i_waste_height = 99       # never relaxes -- the flat version, for comparison
	return w

## lookahead_i with the penalty relaxed once the stack is past half height.
static func lookahead_ih() -> EvalWeights:
	var w := lookahead()
	w.w_i_wasted = -25.0
	w.i_waste_height = 10
	return w

## Hybrid: quad well kept, T-spin slots taken when the pieces allow.
##
## Overhangs are priced as holes exactly as in the quad presets -- a hang is only
## cheap if it belongs to a RECOGNISED slot, and TSpinEvaluator refunds it then.
## Pricing a generic overhang at some small negative would be weighting a thing we
## want with a negative number, which is incoherent.
static func tspin() -> EvalWeights:
	var w := lookahead()
	w.w_slot_ready = 55.0
	w.w_slot_progress = 18.0
	w.w_slot_at_well = 25.0
	return w

## tspin() with the well reservation dropped: slots anywhere, flat stack.
static func tspin_flat() -> EvalWeights:
	var w := tspin()
	w.w_well_blocked = -4.0
	w.w_quad_ready = 0.0
	w.w_well_progress = 0.0
	w.w_slot_at_well = 0.0
	return w

## The pre-fix behaviour, kept so the benchmark can show the difference.
static func line_greedy() -> EvalWeights:
	var w := EvalWeights.new()
	w.well_column = -1
	w.w_well_blocked = -8.0
	w.w_quad_ready = 3.0
	w.reward_quad = 26.0
	w.penalty_small_clear = -3.0
	return w
