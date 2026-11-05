# Reuse the already-separated designer repository

[Open the component-reuse notebook](https://colab.research.google.com/github/MBG0903/GNR_Design_Project/blob/main/notebooks/Designer_Component_Reuse.ipynb).

The user has separated components from 1,800+ designer images and supplied its Drive folder in chat. This chat has not inspected or indexed that folder. The implementation below supports component image files and runs against the user's mounted Drive in Colab. An existing SQLite/JSON-only repository requires a schema adapter; its structure is not guessed.

## Why reuse can help

Copying a matching component at its native size preserves its existing staircase geometry, intentional white and pixel construction. It removes the need to regenerate those particular contours. It does not prove that every source component meets the latest rules or that a selected component is suitable for a new sketch. Semantic match, native-size coverage, repeated-family consistency and composition remain evaluation tasks.

## Recommended workflow

1. Add the shared folder as a shortcut in My Drive if needed, mount Drive in Colab, and set COMPONENTS_DIR to its actual mounted path. A folder URL/ID does not reveal its filesystem name. No public sharing is required.
2. Index each image file as one complete asset. Keep all disconnected parts in that file together. Optional trimming removes only empty outer margins and records the original offset; no motif pixels are scaled, rotated, smoothed or thresholded. Binary transparency can become output white; partial transparency and non-red/white visible colours are reported and excluded.
3. Build the local sketch/component similarity index once. Native width/height are exact-match filters; CLIP ranks candidate shapes. Duplicate rasters share embeddings while keeping their source provenance. Sidecar CSV/JSON schemas are retained but not automatically interpreted.
4. Load a new Step_2 and its Photoshop/GenAI Step_3 draft. The recommended `preserve_photoshop` mode treats the draft as layout and fallback. Source copies are kept in the run folder to avoid two uploads with the same filename overwriting the sketch.
5. For up to 20 largest source raster regions in the initial pilot, measure the actual native bounding box. This avoids relying on a vision model to estimate exact dimensions. A connected region remains only a proposal unit: it can be a merged motif, fragment or legitimate continuous form. The vision chooser sees the full sketch, draft, crop and candidate pieces; it can reject every candidate.
6. Replace only a complete selected region and only with an exact-width/height match. Keep neighbours untouched. Check overlap, protected white and a minimum two-white-pixel clearance to other raster regions. Preserve the source draft's enclosed white holes. A blocked, uncertain or unavailable match retains the original Photoshop pixels.
7. Inspect the whole-image result and the individual region proposals. The source component hash/path is recorded. PNG/BMP preserve Step_2 dimensions. Red/white lock masks are exported for later tools to honour. Do not run an unconstrained whole-image finisher over reused components; those masks do not automatically constrain unrelated software.

The optional `sketch_only` mode uses the earlier planning agent to assemble from Step_2 alone. Its exact-size measurements are estimates, and unmatched regions remain white. It is an exploratory partial-composition mode. The recommended fallback mode avoids making a good Photoshop draft blank wherever retrieval fails.

## Execution details

- Source: existing component images, not the original full Stage_3 folder by default. Set `INPUT_MODE='full_designs'` only for the older region-extraction mode, using a separate index.
- Search index: `MyDrive/GNR Component Search Index/separated_v1`. Keep it outside the source repository; source files are never modified. A changed source or index policy needs a new index version.
- Outputs: `MyDrive/GNR Textile Runs/Designer_Component_Reuse/trial_01`.
- Set the existing `OPENAI_API_KEY` Colab Secret. Choose an accessible vision reasoning model in the dropdown. No image-generation model is used to redraw the components and no weights are trained.
- Initial fallback-mode run: up to 20 paid candidate-choice requests, fewer when exact sizes are absent. Responses are cached; interruption handling prevents automatic repeat requests. Model-list availability alone does not establish vision endpoint access.
- Draft must be native-size pure red/white. The finisher Cell 3's `1 photoshop pure.png` is suitable. An explicit optional palette-projection setting creates a separate copy for other drafts; source assets themselves are not projected.
- Every accepted asset stays native; if a useful shape only fits after resizing, retain the Photoshop piece or obtain another approved native-size variant. There is no resizing fallback.

`report.json` separates reused, retained/no-size-match, retained/no-semantic-match, blocked and deferred regions. Reuse counts are not counts of improvements. `reused_red_lock.png` and `reused_white_lock.png` record the intended locked pixels. `production_ready` remains false.

Automated tests verify grouped assets are not split, source pixels/margins/transparency handling, immutable input snapshots, embedding reuse, exact native replacement, unchanged unmatched Photoshop regions, neighbour/white protection, PNG/BMP dimensions and API-response reuse. API responses and embeddings are mocked in these tests; live matching quality on the linked repository has not been measured.
