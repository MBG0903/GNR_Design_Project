# Reference-assisted whole-image repair

This experiment uses the approved designer target to fit local boundary strips. It is not a learned conversion or an independent accuracy test. No GenAI model was retrained or evaluated on unseen sketches.

| Measure | Before | After |
|---|---:|---:|
| Pixel agreement | 87.39% | 89.56% |
| Red IoU | 77.73% | 81.21% |
| Red Dice | 87.47% | 89.63% |
| Mismatched pixels | 17,194 | 14,239 |

972 accepted one-pixel boundary-strip corrections across 98 regions changed 3,023 final pixels. The 401 × 340 canvas, pure palette, labelled two-pixel clearance, existing verified hierarchy traces, red connectivity and internal white-region counts were checked. Boundary movement was limited to two pixels from each original component edge. Compact shapes were left unchanged.

Existing semantic joins, failed anchor comparisons and infeasible hierarchy arcs remain unresolved. Geometry safeguards do not establish semantic production readiness. The original output is retained.
