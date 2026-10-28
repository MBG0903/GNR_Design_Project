# 12 chat-tool GenAI tile proposals and combined review image

These are **new, actual image-tool generations**, one independent call per tile.
The tool does not expose its model name: **Sunburst identity is unverified**.
No user API key was used. This is an experiment for designer review, not an approved production raster.

- [Combined PNG](step_3_combined_review.png) / [BMP](step_3_combined_review.bmp): exactly **401 × 340**, pure red/white.
- [Before/after](before_after_review.png): original dense draft on the left, regenerated tile assembly on the right, enlarged with nearest-neighbour for inspection.
- [All 12 tiles](all_12_tiles_review.png): labelled contact sheet of disjoint cores.
- [Download review ZIP](all_tiles_and_combined_review.zip): individual core/context PNGs, combined files, comparison, seam guide, manifest and audit.
- [Audit](audit.json): native dimensions/palette, source hash, tile provenance, changed pixels and geometric contour screening.

The starting point was `examples/generated_dense_draft/step_3_dense_draft.png`,
not the rejected reference-fitting image. A 4-column/3-row grid partitions its native pixels;
each tile has up to 32 original pixels of surrounding context. Generation received the
square-padded, enlarged dense tile, matching sketch, full dense design and full sketch.
Each prompt requests native 3→2→1 (or 2→1 fallback) staircases, increasing run counts,
front/back anchor ordering, semantic separation and preservation of composition/internal white.
The designer reference was not supplied anew; the earlier dense starting draft was reference-guided.

Raw generations are retained in `raw_generated/`, and the exact prompts and input
transforms are recorded. The tool produced square 1254 × 1254 canvases despite the
1024 × 1024 request. Code restores the original native square with nearest-neighbour,
removes padding, projects to exact red/white, restores the original read-only halo,
and copies only disjoint cores. It applies no additional contour sorting, gap cutting,
morphology, blending or reference fitting. **4,852 pixels (3.56%)** differ from the dense
draft. Red coverage is **50.66%**; it was **50.64%** before.

The geometric screening still finds unresolved contour sequences. Its traces overlap,
include reverse/sub-arcs and may include exempt shapes, so counts are not semantic
pass rates. This experiment does **not** show that GenAI has enforced every staircase,
front/back ordering or two-pixel gap. Inspect complete curves and cross-tile seams.
Changes occur at 217 pixels within two pixels of a tile boundary; this is a review
location diagnostic, not a count of broken seams. No designer attestations or training
targets were created, and `examples/current_step3/` was not replaced.

Reassemble the recorded outputs without new model calls:

```bash
.venv/bin/python -m scripts.assemble_chat_tiles
```
