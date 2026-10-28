# GNR 3005: full-image dense GenAI prompt experiment

Input: the uploaded `GNR 3005 ( FILE 2 ).bmp`, preserved byte-for-byte as
`source_step_2.bmp`. Its native size is **600 × 200**.

- [Latest review PNG](step_3_full_review.png) / [BMP](step_3_full_review.bmp)
- [Sketch and both attempts](sketch_and_both_attempts.png)
- [Native flower-detail comparison](native_flower_detail_comparison.png)
- [Latest image enlarged 4×](step_3_full_review_zoom4.png)
- [Review ZIP](gnr_3005_full_image_review.zip)
- [Audit](audit.json)

Two actual **full-image** calls used the image-generation tool available in chat.
Its model identity is not exposed; these are not verified Sunburst outputs.
There was no tiling, tile merge or seam repair. The independent design-362 sketch
and manually approved Step_3 supplied examples of semantic fill treatment, not an
approved target for this new design. No designer Step_3 for GNR 3005 is available,
so no reference similarity or conversion-accuracy score is claimed.

The [first prompt](prompt_full_image_dense_v2.txt) explicitly defines:

- Dense solid fill within existing motif envelopes, preserving intentional white.
- Two **native** white pixels of clearance between distinct semantic pieces,
  with genuine continuous forms and wider background spaces preserved.
- Contour run lengths 3→2→1 with n3<n2<n1, or 2→1 with n2<n1 for small shapes.
- Shorter inner/front anchors than outer/back anchors; dot/broad-petal exemptions.
- Unchanged floral/paisley composition, pure red/white and a logical 600×200 grid.

The [refinement prompt](prompt_full_image_refinement_v3.txt) additionally supplies
the native first draft enlarged with nearest-neighbour and asks for coarser native
pixel construction, less contour noise and fewer pinched channels. Both actual raw
generations, prompts, first-draft PNG/BMP and audits are retained for comparison.

**Verified:** both native PNG/BMP pairs are exactly 600×200, use only pure red and
white, and have pixel-identical lossless PNG/BMP round trips. ZIP integrity and
source preservation were checked. Latest red coverage is **59.525%**; the first
draft is **60.328%**. Coverage is diagnostic, not a business-rule acceptance test.

The tool returned 2167×725 and 2170×725 rasters despite the requested 3:1 ratio.
Postprocessing uses aspect-preserving nearest-neighbour containment, white canvas
padding and exact red/white projection. The first restores to 598×200 with one
white column on each side; the refinement restores to 599×200 with one white
column on the right. Final canvases remain exactly 600×200 without anisotropic
stretching. These conversions do not guarantee sketch-to-output registration.
No code-based contour, gap, morphology or reference-fitting corrections were applied.

**Not production approved.** Small flower clusters still contain narrow/pinched
channels and irregular contour steps. Two-pixel semantic separation, front/back
anchors, complete staircase hierarchy, deliberate internal white, motif fidelity
and repeated-family consistency need designer review. Geometry screening in the
audit includes overlapping/reversed/sub-arcs and potentially exempt shapes; its
counts are not semantic compliance rates. This experiment tests improved prompting
and does not establish that prompting alone enforces all rules.
