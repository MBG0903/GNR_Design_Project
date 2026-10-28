# GNR 3005: comparison with the newly supplied designer Step_3

Both AI attempts and the uploaded designer BMP are exactly **600 × 200** and
contain only pure red/white. Comparison uses their original native coordinates:
**no alignment, cropping, warping, resizing or palette conversion**.

| Measure | First AI attempt | Latest refinement |
| --- | ---: | ---: |
| Exact pixel agreement (red and white) | 61.52% | **55.24%** |
| Red-shape intersection-over-union (IoU) | 51.99% | **46.00%** |
| Red-shape Dice overlap | 68.41% | 63.01% |
| Red precision | 69.08% | 64.06% |
| Red recall | 67.76% | 62.00% |
| Red coverage | 60.33% | 59.53% |

Designer red coverage is **61.51%**. Similar coverage does not establish shape
agreement. The refinement reduces exact agreement by **6.28 percentage points**
and therefore does not improve similarity to the designer target.

Pixel agreement counts the same red or white colour at each of all 120,000
coordinates, including background. IoU is red intersection divided by red union;
it avoids counting matching white background as a success. Dice is twice the red
intersection divided by the sum of red pixel counts. These are different metrics,
not interchangeable estimates or percentages of business rules satisfied.

This uploaded GNR 3005 designer target was first available **after** both images
were generated and was not used to generate them. A different design supplied
style guidance. This single-design comparison does not measure accuracy across
the user's 1,600+ pairs, semantic correctness or production readiness.

- [Designer / first AI / latest AI / difference](designer_ai_comparison.png)
- [Latest native difference](latest_refinement_difference.png): red = both red;
  white = both white; blue = missing designer red; orange = excess AI red.
- [Exact latest metrics](latest_refinement_metrics.json)
- [Exact first-attempt metrics](first_attempt_metrics.json)
- [Review ZIP](comparison_review.zip)

The original designer BMP is preserved byte-for-byte. Metrics include input
hashes. Direct RGB agreement was recomputed independently to check the reported
percentage, and confusion counts sum to 120,000. Similarity is not a business-rule
acceptance gate: two-pixel gaps, complete staircase hierarchy and motif semantics
need their own designer review.
