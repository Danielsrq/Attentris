# Placeholder art

Stand-ins, drawn as SVG so they are legible and trivially replaceable. Every one is
referenced from a scene in `defence/scenes/`, so swapping in real art is a texture change
in the inspector and nothing else — no code knows what these look like.

Sizes are authored at the playfield's 30px cell so a tower body is one cell and scales by
footprint. `tower_barrel.svg` points **+x with its pivot at the left edge**, because the
node rotates it by the aim direction: a barrel drawn centred would orbit instead of turn.

| file | used by | notes |
|---|---|---|
| `tower_body.svg` | `tower.tscn` → `Body` | one cell; tiled per footprint cell |
| `tower_barrel.svg` | `tower.tscn` → `Barrel` | pivot at left, points +x |
| `enemy.svg` | `enemy.tscn` → `Body` | 24px = 2x `enemy_half_extents` |
| `shot.svg` | `shot.tscn` → `Body` | 12px, matching `projectile_radius * 2` |
