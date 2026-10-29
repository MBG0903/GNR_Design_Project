# Central component repository and reuse agent

[Open the Colab notebook](https://colab.research.google.com/github/MBG0903/GNR_Design_Project/blob/main/notebooks/Central_Component_Repository.ipynb).

This implements the requested batch-library approach: extract existing raster pieces from the full designer Stage_3 dataset, store them centrally, and search for suitable exact-size pieces when a new Step_2 is loaded. It does not require manually extracting every component. It does not generate, resize, rotate or redraw components.

## Run in Colab

1. Mount Drive. The notebook reads `MyDrive/GNR Design Data-Set/Stage_3` and creates `MyDrive/GNR Component Library/v1`.
2. Run batch extraction and inspect `index_summary.json` / `source_audit.csv`. Every valid red region, including single pixels, is retained. Non-red/white or unreadable originals are reported and excluded without changing the files. The library may therefore cover a subset until source problems are resolved. Step_2/Step_3 pairing and the earlier pairing audit are not prerequisites for this index.
3. Build the local CLIP similarity index. A GPU helps. All steps checkpoint on Drive, and identical rasters share embeddings while retaining their separate provenance. This first pass can be lengthy on a large dataset; subsequent runs reuse it. The model weights download from Hugging Face. CLIP ranks pieces only; it never converts sketches into designs.
4. Upload one Step_2 or set its Drive path. Its original width and height define the output canvas.
5. Enable the existing `OPENAI_API_KEY` Colab Secret and select a vision reasoning model from the account's available model IDs. Listing a model does not establish image-input/Responses API support. If the Astra API ID is accessible, select it; the notebook does not infer an API ID from a subscription name. Pause at this selector before continuing.
6. Run the agent. The initial pilot considers up to 20 agent-proposed pieces: one paid planning request, then up to 20 paid candidate-choice requests. No image-generation requests are made. Increase `MAX_PIECES` (up to 500) and use a new `RUN` folder for a larger experiment.
7. Review the PNG, BMP, diagnostic sketch overlay and `report.json`. Download the run ZIP or inspect the persistent files under `MyDrive/GNR Textile Runs/Component_Reuse/trial_01`.

The complete 1,700+ image index and live agent matching must run in the user's Colab session, where Drive and API credentials are available. They have not been run from this cloud workspace.

## What the agent does

- A vision model proposes complete semantic pieces and estimates tight integer bounding boxes on the original sketch grid. Uncertain interpretations are flagged.
- Retrieval filters by **exact native width and height** first. Sketch-crop embeddings rank the remaining candidates. Exact duplicate rasters are not repeated in the shortlist. The agent sees full-sketch context, the crop and the candidate images; it can reject every candidate.
- A selected candidate is translated to the proposed position without changing its native red pixels. Code blocks clipping, overlap, filling protected white and separation narrower than two white pixels on an 8-neighbour/Chebyshev grid. It does not move pieces to make them fit. Wider background spaces remain wider.
- Unmatched sizes, rejected shapes, ambiguous interpretations, blocked placements and pieces beyond the run limit remain unresolved. There is no generated filler. The resulting canvas can be partial, with unresolved areas white.
- Every placement records component ID/hash, original designer image, source design identifier, coordinates, candidate scores, agent response and spacing check. Production approval is never inferred.

## Important distinctions

**Extraction is not semantic segmentation.** Each 8-connected red region is a candidate: a touching leaf/stem may be one region; one intended motif can have disconnected parts; dots and noise are retained. No component count is treated as proof of topology. The agent is instructed to reject merged pieces, fragments and unrelated shapes, but its judgments require review.

**Exact size is measured against the agent's estimated sketch box.** A mistaken one-pixel estimate can produce no match. A matching rectangle does not establish matching shape. This strict version deliberately does not stretch a near match. The first experiment should establish retrieval coverage and estimate quality before adding more fitting mechanisms.

**Preserved geometry is not universal rule approval.** Reusing original pixels preserves the selected raster's staircases and internal geometry, including any existing defects. It does not certify the designer dataset. Closed white holes are protected using the source raster; open channels need semantic review. Complete continuous stems must be represented by one component: this prototype blocks joining separate placements.

**Whole-image completeness remains unverified.** Green overlay boxes indicate placed proposals; blue boxes indicate unresolved/deferred proposals. Motifs the model failed to identify may not have boxes. These colours exist only in the diagnostic overlay; the saved Step_3 is pure red/white. A low unresolved count is not a production pass rate or accuracy score.

**This is not fine-tuning.** There is no model-weight training, automatic full-dataset semantic annotation, or claim of 70–80% accuracy. The library becomes reusable immediately after indexing; quality depends on parsing, exact-size coverage, candidate selection and designer review.

## Persistence and validation

The central folder stores `library.sqlite3`, component PNGs, protected-white masks and source audits. Original source paths/file hashes and region coordinates are recorded. Sources are immutable within a library version; changing/removing originals requires a new snapshot. Added files can be indexed. Keep one indexing writer active per library. Do not edit SQLite records or stored component images manually.

Saved agent requests include hashes of model inputs. Completed responses are reused; interrupted requests block automatic repeat charges. Review provider usage before deliberately retrying in a new run. SDK automatic retries are disabled. Changing sketch/model/library/settings requires a different run folder.

Tests cover exact extraction and reconstruction, duplicate-raster embedding reuse, strict size filtering, original-pixel placement, protected holes, spacing, clipping/overlap rejection, uncertain/unmatched/deferred pieces, response reuse and interruption handling. Mocked API/embedding tests establish workflow correctness, not live model accuracy. Native source-image extraction is also exercised on the six designer BMPs already provided in this project.

The local six-image extraction check found **2,874 red regions** with no excluded files. Recombining their red masks at original coordinates reconstructed all six designer rasters exactly. This checks pixel preservation only; it does not establish 2,874 distinct semantic components. The full repository test suite passed 72 tests after this addition.
