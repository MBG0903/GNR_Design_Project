# GNR Design Project

**Current review baseline:** [PNG](examples/current_step3/step_3_current.png) / [BMP](examples/current_step3/step_3_current.bmp). The later reference-assisted strip-fitting version was rejected by the designer and is disabled.

## New experiment: dense GenAI → 9/12 tiles → review → merge

Open `notebooks/Tile_Staircase_Review.ipynb` in Colab. It is standalone and embeds the existing dense draft and matching sketch. For a new design, run the generation stage of the primary notebook, then upload the sketch and dense result into this experiment. No new generation or automatic staircase corrections have been applied to the example tiles. The restored working baseline remains unchanged.

Default grid: **4 columns × 3 rows = 12 tiles**, with **32 native pixels of read-only context** around each core. A 3×3 grid gives 9 tiles. Cores partition the original image exactly, including uneven 401×340 dimensions; nothing is resized or blended. Only core edits are merged. Matching sketch context and a diagnostic ownership guide are supplied. Increase context or inspect the full component if either front/back edge or tip is cut off. Cross-core edits need coordinated neighbouring proposals and full-curve review after merging.

The notebook includes a red/white pixel editor and versioned tile proposals. Save requires reviewer identity, full affected contour/tip review, hierarchy/gap review and no unresolved notes. Halo edits, palette changes and dimension changes are rejected. **These attestations are not machine proof of semantic rules.** Merge output is always a review draft requiring global curve and seam inspection. The prior rejected strip fitter stays disabled; tiling does not make that method valid. Automated Astra contour correction and complete-curve validation are not implemented in this experiment.

`examples/tile_review12/` includes the manifest, raw context tiles, grid preview, no-edit roundtrip, and `tile_pack.zip`. Both 9- and 12-tile no-edit roundtrips were tested pixel-for-pixel, and the live editor load/merge API was exercised. The dense sample remains exactly 401×340. The blue grid is a diagnostic only and is absent from the raw rasters.

Regenerate the standalone notebook with `.venv/bin/python scripts/build_tile_notebook.py` when changing tile runtime code. The tile review archive retains proposal revisions and manifests. Download it before Colab storage is lost.

## Current primary workflow: Sunburst + Astra + Colab pixel corrections

Use `notebooks/GenAI_Textile_Colab.ipynb`. Photoshop is optional. The image model defaults to the documented API ID `gpt-image-2.5-sunburst`; enter the exact Astra API model ID available in your account for reasoning/review. Account-facing subscription names do not establish API model access or billing.

The original Step 2 width/height are the authoritative output size. Every saved Step 3 draft, correction and final raster must match exactly (the supplied example is **401 × 340**). Intermediate model resolutions are separate: the service may not support arbitrary native dimensions. The generation path preserves aspect ratio using a padded canvas, restores native dimensions and projects to exact red/white. It does not certify motif placement. Colab corrections compare against the ORIGINAL sketch dimensions and reject mismatches before saving.

Astra reviews the original and current raster, flags uncertainty and proposes bounded native-pixel red/white polygon fills. `textile.colab_edits.apply_plan` applies reviewed proposals using Pillow without antialiasing, records provenance/unresolved issues, and saves a new iteration rather than overwriting. Inspect/re-review the result and obtain designer sign-off. No global morphology, blanket gap cutting, or automatic hole filling is used.

The sample generation in `examples/generated_draft` is a draft, not a production approval. Exact palette and dimensions are verified; semantic fidelity, separation, taper hierarchy and consistency remain unresolved.

The earlier Windows/Photoshop integration below remains available as an optional finishing path.


Turn an artist's Step 2 saree sketch into a Step 3 textile raster using **GenAI**, then let a vision agent propose **Photoshop edits** for the business rules. A designer approves the final result. No conventional CV conversion model is trained.

## Run in Colab

Upload `notebooks/GenAI_Textile_Photoshop.ipynb` to Google Colab and run its cells in order. It embeds a matching copy of the runtime, so no GitHub push is required. CPU runtime is sufficient. Provide your API key through Colab Secrets as `TEXTILE_API_KEY` or the hidden prompt. API calls are billed by your provider; model access must be enabled.

The notebook mounts:

- `MyDrive/GNR Design Data-Set/Stage_2`
- `MyDrive/GNR Design Data-Set/Stage_3`

It pairs names such as `362 GNR 2.bmp` / `362 GNR 3.bmp`, audits the images, and requires `groups.csv` to group variants/crops by original design. The first audit generates a template. Whole-design reference/validation/test splits keep evaluated designs out of reference examples. Select 1–3 relevant approved pairs for generation; all 1,600 examples are not sent in one prompt. Automatic semantic retrieval and fine-tuning are not implemented.

## Pipeline

1. **GenAI generation:** image-edit API receives the sketch, rules and chosen example pairs. Default image model: `gpt-image-2.5-sunburst` (configurable).
2. **Native raster preparation:** aspect-preserving square padding, generation, native-size restoration, exact red/white projection. This enforces colours and dimensions, not semantic fidelity.
3. **Vision-agent critique:** your configured Astra API model compares original and current raster, reports rule issues, flags uncertainty and proposes bounded polygon edits.
4. **Photoshop edits:** validated proposals become a `.jsx` script. Run it via Photoshop's **File → Scripts → Browse**, selecting the exact reviewed PNG. It checks the file, duplicates the document, applies red/white fills without antialiasing on a correction layer, and saves PSD + PNG in a new folder.
5. **Closed-loop review:** return the actual Photoshop export to Colab, validate it and rerun agent critique. This prototype requires explicit desktop handoff; unattended Photoshop control is not implemented or verified.
6. **Designer sign-off:** interactive editor requires all 16 rule attestations, reviewer identity and no unresolved notes. Export PNG/BMP plus a JSON review record.

Generation/critique can misunderstand semantics, move motifs, or miss one-pixel geometry. Agent confidence is not proof. R1–R4, R6–R13, R15–R16 remain designer-reviewed. R5/R14 are deterministic checks. Production readiness additionally requires loom-specific format, weave mapping, repeat alignment and machine constraints, which are not yet specified.

## Local development

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m textile.app
```

On Windows use `.venv\Scripts\python` instead. The local editor is a fallback for final raster review, not an alternative conversion model. GenAI is the conversion path; the editor loads its output for correction.

Set `TEXTILE_API_KEY` securely and pass `OpenAI(api_key=os.environ['TEXTILE_API_KEY'])` to `generate`/`review` for local API calls. No credentials are stored in the project. API requests transmit source artwork and selected references to the provider. Input artwork/text and model replies are treated as data; Photoshop plans allow only validated bounded fills.

The provided BMPs are in `examples/`. Both are 401 × 340; Step 3 is exactly red/white. They are examples, not evidence of model quality. Run `python scripts/build_notebook.py` after changing runtime code to refresh the notebook's embedded runtime. Test the Photoshop script on your actual desktop version before using real production documents.

## Current validation scope

Tests cover native geometry/palette validation, review/export gates, lossless BMP/PNG round trips, pair grouping, bounded agent plans and mocked API response handling. Real GenAI quality, Drive-wide evaluation and desktop Photoshop execution require your connected accounts and desktop and have not been established here.

## Recommended: Mac orchestration + Windows Photoshop worker

Run Colab in the Mac browser. Keep Photoshop on the Windows PC, awake and signed in. Download the notebook's `photoshop_job.zip` and transfer it via Drive or Remote Desktop; unzip it into a **new folder per iteration**. On Windows, unpack this project and run:

```powershell
py -m venv .venv
.venv\Scripts\python -m pip install -r requirements-windows.txt
.venv\Scripts\python -m textile.windows_worker "C:\path\to\unpacked_job"
```

The worker accepts data-only jobs, validates the image hash and bounded edit plan, generates its own JSX, and calls the installed Photoshop through Windows COM. It saves PSD, PNG and a validation report under `results/`. Return the PNG to Colab for the next agent review. No public desktop server or inbound ports are needed. This worker is implemented but requires testing against your Windows Photoshop installation. File transfer and starting the worker are explicit; background watching/fully unattended orchestration is not implemented. Remote Desktop is useful for inspecting the edits, but not required by the worker.

## Downloadable example draft

The generated 401 × 340 draft files are versioned at `examples/generated_draft/step_3_draft.png` and `examples/generated_draft/step_3_draft.bmp`. They are unapproved drafts; `audit.json` records the checks and remaining semantic issues. The primary Colab notebook is `notebooks/GenAI_Textile_Colab.ipynb`.

## Dense-fill revision

Dense fill is now the default generation/review instruction: robust solid red inside existing motif envelopes, narrow clean separations, and preserved deliberate white channels/background. Coverage is diagnostic, not a universal target. No blanket dilation is applied.

The reference-guided dense sample is in `examples/generated_dense_draft/` (PNG/BMP, exactly 401 × 340). Its audit compares red coverage with the earlier draft and designer reference. The same-design approved target was used as guidance for this illustration; it does not demonstrate independent conversion accuracy. Some neighbouring pieces remain joined and semantic/taper checks still need designer review. The earlier sparse draft is retained.

## R2: two-pixel separation (partial pass)

The requested clearance is two native white pixels at closest approach, measured on the 8-neighbour/chessboard grid; existing wider background remains. `textile.separation.separate` accepts a semantic instance label raster (0 white, one positive ID per continuous piece). It removes only conflicting red edge pixels, preserves every labelled piece and its connectivity, and reports unsatisfiable constraints. It does not apply taper or other rule corrections.

The sample `examples/separation_2px_pass/` contains PNG/BMP, an audit, provisional labels and a blue-box review map. All 150 currently disconnected red regions have at least two pixels of clearance after the pass, with no unresolved geometric conflicts. **This does not complete semantic R2:** several joined leaf/stem and flower/stem groups share one region. Their boundaries must be labelled/reviewed before applying the same clearance routine. The audit explicitly retains `rule_complete: false`; do not proceed on the assumption that component counts prove semantic separation. The original dense draft is retained.

## Pixel hierarchy: contour staircases

The designer confirmed that front means the inner/concave edge and back means outer/convex. The front long run must be strictly shorter than the back long run. Following each long anchor toward a tip/turn, native contour runs progress through 3, then 2, then 1 pixels. Counts satisfy **n3 < n2 < n1**, for example `3,3,2,2,2,1,1,1,1,1`. This counts staircase runs, not stroke widths. Split traces at direction/octant transitions; all four directions and reflections are supported. Round dots and broad non-tapering petals retain their existing exemptions.

`textile.pixel_hierarchy` validates annotated grid-edge traces against the actual raster, checks the run/count/anchor rules, proposes endpoint-preserving counts, and can replace a selected contour while checking piece connectivity and two-pixel clearance. Infeasible/ambiguous cases are flagged rather than moving endpoints or silently skipping levels. Front/back labels must be assigned semantically.

`examples/pixel_hierarchy_pass/` contains a **partial** native 401 × 340 correction and a zoomed before/after diagnostic. One selected inner feather edge changes from `[anchor 9; 2,3,1,2,1,1]` to `[anchor 9; 3,2,2,1,1,1]` (counts 1 < 2 < 3), changing exactly two pixels. The outer long run is 16, so 9 < 16. This verifies the selected lower inner edge and anchor comparison; the opposite edge and all other curves are not certified. Another feather has inner/outer longest runs 38/12 and needs reshaping. The audit keeps image-wide approval false and preserves unresolved separation issues. Reproduce with `.venv/bin/python -m scripts.pixel_hierarchy_example`.

## Whole-image hierarchy with 2/1 fallback

The designer approved **2 → 1** when a component's size/shape cannot accommodate 3 → 2 → 1. The preferred counts remain n3 < n2 < n1; the fallback requires **n2 < n1**, with both levels present. There is no arbitrary jump/reversal and no all-1 fallback.

`textile.global_hierarchy.run_global` traces all outer and internal boundaries in both directions and orientations. It attempts 3/2/1 across the entire image first, protects already ordered 3/2/1 traces, then tries 2/1 for unresolved cases. Every fallback records why the preferred mode could not be applied under the size/envelope/clearance constraints. Native displacement is limited to two pixels, selected endpoints are fixed, and edits are rejected if they alter a neighbour, break a piece, change internal white-region count, invalidate a protected trace, violate labelled two-pixel separation or lose an existing passing geometric anchor relationship.

The run starts from `examples/pixel_hierarchy_pass/piece_labels.npy` and writes `examples/whole_image_hierarchy_fallback_pass/`. It scans **150 regions, 206 boundary loops and 3,191 candidate arcs**, making **359 corrections across 86 regions**: **155 use 3/2/1 and 204 use 2/1**. It also retains 150 already ordered fallback arcs after the preferred mode cannot be applied. Exactly **1,284 pixels** differ from the input. PNG/BMP are exactly **401 × 340**, pure red/white. Corrected and retained traces are reverified on the final raster, as is labelled two-pixel clearance.

**Whole-image processing does not mean every contour passes.** Final geometry-only re-extraction finds 1,783 arcs with at least three tail runs: 260 pass 3/2/1, 533 pass 2/1, and 990 do not pass either sequence check. These overlap and include reverse/sub-arcs and semantically exempt shapes; they are not counts of unique motifs. Compact shapes are not automatically edited, and their exemptions still need semantic confirmation. Short arcs, fixed endpoints, white channels, neighbours and concavity/anchor conflicts can prevent either pattern. Front/back inference is geometric and requires designer confirmation. Prior joined semantic pieces remain unresolved.

`audit.json` keeps `image_wide_rule_complete: false`. `contour_outcomes.csv` and compressed detailed audit distinguish corrected, already ordered, infeasible, overlapping and review cases. The result is an experiment for designer inspection, not production approval. No regenerated motif or global erosion/dilation is used.

Reproduce with `.venv/bin/python -m scripts.run_global_hierarchy`. Add `--strict-321` for a separate strict-only experiment. Before/after and blue changed-pixel previews are diagnostics; use the native PNG/BMP for pixel inspection. In Colab the new whole-image cell follows the two-pixel separation stage; update the piece labels if you edit the input afterward.

## Comparison with the designer Step 3

`examples/designer_comparison/` compares the latest whole-image fallback PNG with the uploaded designer BMP at their exact native 401 × 340 coordinates (no alignment, crop or resize). Exact pixel agreement is **87.39%**; red IoU is **77.73%**; red Dice is **87.47%**. There are **17,194 mismatched pixels**: 11,190 missing designer-red pixels and 6,004 excess generated-red pixels. The candidate is 48.43% red versus 52.24% in the designer image.

The mean symmetric nearest-boundary distance is 0.75 pixels and 98.31% of sampled boundary pixels are within 2 pixels of some opposite-image boundary. This does not establish correspondence of semantic pieces or correct topology.

The dense draft before rule edits had 88.13% exact agreement; the spacing/hierarchy passes reduced this to 87.39%. Rule compliance and reference similarity are separate objectives. The reference was already used to guide density for this same design: this is **not an independent unseen-image accuracy estimate** or production approval.

Difference legend: red = both red, white = both white, blue = missing designer red, orange = excess generated red. Reproduce with `.venv/bin/python -m scripts.compare_designer examples/step_3_reference.bmp examples/whole_image_hierarchy_fallback_pass/step_3_whole_image_hierarchy.png examples/designer_comparison`.

## Repair queue: closing the reference-similarity gap

Run `.venv/bin/python -m scripts.build_repair_queue` to build `examples/repair_queue/`: ranked disjoint 64px error tiles and six context sheets showing sketch, designer, current output and colour difference. This is a diagnostic, not an edit or improved raster. Every tile retains a pending semantic decision. Counts use core tiles; preview context overlaps.

Of 17,194 current mismatches, 14,660 (85.3%) persist from the GenAI draft and 2,534 were introduced during rule edits. Those edits also resolved 1,526 old mismatches. Thus most discrepancy requires better motif/contour reconstruction; global fill changes or more staircase sorting cannot alone solve it. The first priority is the lower tail/teardrop region, core box `[256,256,320,320]`, with 527 missing-red and 372 excess-red pixels.

The next repair loop should classify a crop semantically, propose local shape/edge corrections, validate native dimensions/palette, labelled two-pixel gaps, intentional white and protected hierarchy, then accept/reject and measure the result. Audit the designer reference against the same requested rules: reference similarity and rule compliance can disagree. Do not copy target error masks into the prediction and report that as improved inference. Future sketches will not have their target Step 3; semantic error detection must use the sketch/rules, and accuracy must be measured on held-out designs.

## Rejected strip-fitting experiment and restored baseline

The designer rejected `examples/reference_assisted_repair/`: straight boundary grow/trim strips introduced malformed steps and tips. Pixel agreement increased to 89.56%, but the business-rule result was worse. Preserving selected passing traces, connectivity, white-region counts and two-pixel separation was **insufficient** to validate entire affected curves. The experiment and its measured scores are retained only as failure evidence.

The public `fit_reference` and `scripts.run_reference_repair` now fail before applying changes. The Colab reference-fitting stage is removed and replaced with a rejection notice. Private historical code remains solely to reproduce/characterize the failure; its unit tests must not be interpreted as full business-rule validation.

**Current review files:** `examples/current_step3/step_3_current.png` and `.bmp`, restored byte-for-byte from the pre-repair whole-image hierarchy/fallback draft. Both remain 401 × 340, pure red/white. The previous known semantic joins, incomplete hierarchy and anchor issues remain; this rollback is not production certification.

The next repair method must work on whole semantically identified contours: paired inner/front and outer/back anchors, both tips and all transitions, preserving composition and gaps. A complete-curve hierarchy check must pass before a changed motif can be accepted. Reference similarity is a secondary diagnostic, never an acceptance gate by itself.
