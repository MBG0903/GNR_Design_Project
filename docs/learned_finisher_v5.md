# Updated learned-finisher notebook: v5 A/M/B staircase

[Open v5.1 in Colab](https://colab.research.google.com/github/MBG0903/GNR_Design_Project/blob/main/notebooks/GNR_Step3_v5_1_Conservative_Staircase.ipynb).

## v5.1 regression correction

The user showed a Photoshop draft closer to the designer than the pipeline final. The old broad gap-redrawing and pocket-filling stages were still active in v5 and could erase white detail/merge shapes. v5.1 disables both by default and starts the final directly from the original Photoshop draft, ignoring intermediate `RED_FINAL`/`RED_SEP` variables. It applies only bounded verified staircase edits, protects enclosed white regions and white far from red, and limits cumulative automatic changes to 1% of original pixels by default. This budget is a preservation guard, not a claim about remaining manual work.

The final cell records source-relative additions/removals, stage-by-stage changes, an isolated change visualization and exact native comparison scores when a designer reference exists. The reference never selects the result. If no feasible edit passes, the Photoshop pixels remain unchanged and unresolved rules are still reported. A saved learned finisher is only a proposal; broad/unsafe candidates are rejected. Every accepted corner is bounded against the original image too, preventing cumulative contour drift.

This updates the user-supplied `GNR_Step3_v4_Learned_Finisher.ipynb` against the latest illustrated rule. The original uploaded file is preserved. This update was tested without mounting the user's Drive, training models or making paid API calls. Old notebook outputs were cleared so they cannot be mistaken for v5 results.

## First run

1. Run **V5 SETUP** and **V5 SELF-TEST**. CPU is sufficient; no API key or training is required. The self-test checks the diagram, eight orientations, 2/3/4-pixel starts and invalid geometry.
2. Run the original **Cell 1: SETUP** to mount Drive. A following cell writes a new versioned `gnr_v5_lib_<hash>.py` into the notebook folder on Drive. The old `gnr_v4_lib.py` is never overwritten.
3. In **Cell 2**, set `DESIGN` to the design you want to test. The finisher requires an **existing AI-generated Step_3 draft**; its default mode does not generate one from the sketch. Use Cell 2's Photoshop prompt first if you do not have a draft yet.
4. In **Cell 3**, reuse a matching inbox file, set `AI_DRAFT_PATH` to a draft's full mounted Drive path, or upload one when the inbox draft is missing. Upload any filename; the notebook validates its native dimensions and copies its original bytes into the inbox under the design ID. Existing files are preserved. The designer reference is never chosen automatically as a fallback input.
5. Run **Cell 10 directly**. Cells 4–6 are optional diagnostic previews. Gap and pocket filling are OFF by default; even enabled previews are not used as the final baseline. Default execution does not retrain, load a finisher or make paid calls.
6. Inspect the full image, native pixel details and accompanying JSON report. Outputs are saved under `GNR_Step3_Photoshop_Pipeline/02_Review_Output_v5` as PNG/BMP plus JSON.

Cell 3 must finish successfully for the current design before Cells 4–6 or 10 can run. The updated notebook records the sketch/draft identity and invalidates it when a sketch is reloaded or an import fails. This prevents Colab from accidentally reusing older image variables after a missing-file error. Changing the design, draft or source file requires rerunning Cells 2 and 3.

Optional controls are in V5 SETUP. Enable `V5_USE_LEARNED_FINISHER` to use an existing saved finisher through Cell 9. Enable `V5_RUN_LLM_REVIEW` for the paid Cell 11. Training and the older paired-model experiments have separate switches. `V5_RUN_LEGACY_BATCH` uses the new generated v5 library in the supplied Cell 12. It requires the saved finisher and its existing PyTorch environment.

Later external Drive scripts were not uploaded and their implementations have not been upgraded. Their cells are disabled by default with `V5_RUN_EXTERNAL_CELLS=False`. Do not assume that enabling them makes their independent rule implementations v5-compatible. Adapt those scripts to use the updated library before relying on their exports.

## The implemented rule

The illustrated corner is:

`flat A → horizontal 3,2,2 → shared M (1 pixel) → vertical 2,2,3 → flat B`

Read either half from its flat toward M: `[3,2,2,1]`. Flat lengths are free. Starting run length can be 2, 3 or 4; successive lengths may hold or decrease one level, and end at the shared 1-pixel transition. There is no constraint on the relative **number** of 1-, 2- and 3-pixel steps. This supersedes the earlier count-dominance interpretation.

For this illustrated corner family, the validator requires one shared M that is a singleton in both the row and column runs. Other contour families or ambiguous junctions are unresolved; the code does not invent their intended interpretation. A/M/B come from the ordered boundary and orthogonal flat anchors, not a crop's centre or the midpoint of a straight chord. Mirroring/rotation are validated without resampling. Whole motifs are not made symmetric automatically.

## Constructor and checks

The embedded runtime is maintained in `textile/staircase_v5.py`:

- `corner_check` validates both axis-specific halves and reports their runs and A/M/B coordinates.
- `construct_corner` enumerates bounded 2/3/4-run sequences with fixed outer endpoints, nearby flat lengths and a maximum native Chebyshev boundary displacement (default 2). Search scope is explicitly bounded; no feasible result means designer review, not relaxed rules.
- `repair_curves` renders candidate contour replacements and independently checks that the constructed path really exists in the rendered raster. It re-extracts contours after each accepted edit, protects previously passing corners, rejects insufficient neighbour clearance and checks raster topology. These geometric guards do not infer semantic topology or component identity.
- `transition_guard` requires complete affected raster regions to pass after arbitrary cleanup or LLM replacements. Pre-existing unrelated issues remain in the audit. An explicit optional protected-white mask can block edits to known intentional white pixels.
- `curve_audit` reports passed, failed and unresolved detections, including tiny/unsupported contours. A percentage is a screening measure, not production certification.
- `export_review` checks dimensions against the original Step_2 file, round-trips PNG/BMP pixels and saves the curve/gap report. It never silently converts unresolved geometry into a production approval.

Existing `tidy`, separation heuristics, synthetic training and learned-finisher architecture remain available for optional experiments. The default final does not call global tidy or broad fill routines. LLM replacements must satisfy the same original-image budget, white protection and affected-contour checks. The geometry-only default gives a first inspection without those expensive experiments.

Input and reference dimension mismatches now stop instead of being silently stretched/resized. This can expose existing inbox/reference registration issues that the old version hid. The user's original files remain unchanged.

## What remains uncertain

Detecting a semantic curve and deciding which regions are intentional still need designer review. A disconnected red region is not necessarily one semantic piece. The gap audit uses a declared 8-neighbour/Chebyshev minimum-clearance metric between disconnected raster regions; it cannot detect two semantic pieces already joined inside one region. Wider intentional white spaces are not narrowed by the constructor.

Junctions, S-curves, very short anchors, other midpoint patterns, curves above the bounded-search scope or infeasible changes remain unresolved. Existing source defects can remain. All exports are review drafts even if detected geometry passes. This update does not establish the 10% manual-effort goal; record actual designer correction time/pixels on full designs to measure that goal.

## Rebuild and test

```bash
python scripts/update_learned_finisher_notebook.py /path/to/GNR_Step3_v4_Learned_Finisher.ipynb
python -m pip install -r requirements-staircase.txt
python -m unittest discover -s tests -p 'test_staircase_v5.py' -v
```

The updater reads the notebook as data, rewrites selected cells and embeds trusted runtime code. It never executes the uploaded notebook. Compatibility functions are extracted as definitions only. Test coverage includes exact diagram acceptance, all orientations, valid holds/starts, invalid jumps/growth, tiny unresolved contours, bounded repair after actual raster rendering, preservation of passing geometry, protected white, cleanup rejection, strict original dimensions, shared-library bindings and syntax/schema validation of the notebook.
