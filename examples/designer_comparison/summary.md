# Native comparison with designer Step 3

Both images are 401 × 340; no alignment or resizing was used.

| Measure | Result |
|---|---:|
| Exact pixel agreement | 87.39% |
| Red intersection-over-union | 77.73% |
| Red Dice overlap | 87.47% |
| Designer red coverage | 52.24% |
| Candidate red coverage | 48.43% |
| Differing pixels | 17,194 / 136,340 |
| Missing designer red | 11,190 |
| Excess candidate red | 6,004 |
| Mean nearest-boundary distance | 0.75 pixels |

Before rule edits, the dense draft had 88.13% pixel agreement and 79.31% red IoU; after the edits the latest values are 87.39% and 77.73%. No inference about improved rule compliance follows from either score.

Difference: blue means missing designer red; orange means excess candidate red. Red/white show agreement.

The designer image is identical to the reference already used for same-design density guidance. These results are reference-guided similarity, not independent evaluation or production approval. Nearest boundaries may belong to different semantic pieces.
