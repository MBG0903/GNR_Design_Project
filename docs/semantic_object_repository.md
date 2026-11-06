# Central repository of named Stage_3 objects

[Open the Colab notebook](https://colab.research.google.com/github/MBG0903/GNR_Design_Project/blob/main/notebooks/Stage3_Semantic_Object_Repository.ipynb).

This extends the project from raster regions/already-separated component files to complete semantic objects: peacocks, elephants, flowers, leaves, feathers, paisleys, stems and other motifs. One complete object may contain many disconnected red islands and intentional white channels. Nested objects can retain an explicit parent relationship, such as a feather belonging to a peacock. They are not forced into an exclusive pixel partition.

## What runs

1. Colab mounts the user's existing `MyDrive/GNR Design Data-Set/Stage_3` folder. The new catalogue defaults to `MyDrive/GNR Object Repository/semantic_v1`, outside the sources and older indexes.
2. An available vision reasoning model, selected from the user's API account, proposes object categories, boxes and optional parent relationships. `OPENAI_API_KEY` is read from Colab Secrets. The source artwork is sent to the selected API at this stage. No image generation is performed.
3. Local pretrained `facebook/sam-vit-base` proposes masks inside those boxes. Its image embedding is reused for successive objects from the same source. SAM can miss disconnected parts or include neighbours; its predicted mask score is not semantic correctness or business-rule compliance.
4. Each candidate stores its original native crop, isolated red/white proposal, selection mask, source hash/path, original bounding box, label, category and hierarchy. Source images are never changed. Only selected source red is retained: no red is invented and original white cannot be filled. Objects remain pending review.
5. The offline HTML panel shows the original crop, proposed object and full-image context. The designer paints inclusion/exclusion with a native pixel brush, changes category if needed and explicitly approves complete object isolation or rejects the proposal. No public web server or Gradio sharing link is needed.
6. Upload the panel's review JSON to Colab. Imports validate identities, source hashes, native masks and every selected red pixel. Approved PNG/BMP assets remove only exterior white margins and retain the original source offset. Identical rasters within a category share asset storage while every source instance remains in the catalogue.
7. Search approved objects by category and exact native width/height. A clean export supplies only current approved PNGs plus provenance for the existing component indexer.

The new repository is not a trained image-generation model. Labels, masks and designer corrections are useful future supervision; this implementation stores them without starting training.

## First run

Run notebook Cells 1–5 in order. Use a Colab GPU for SAM. The default request budget is at most five new images and twenty new discovery requests per run. Review the first masks before extending the batch. Re-running Cell 5 skips completed unchanged images and advances through Stage_3.

For small leaves/petals, set `DETAIL_SCAN=True` **before creating the first catalogue**. Discovery then includes overlapping native views with full-image context. This increases paid requests and does not guarantee exhaustive detection. The default full-image pass can miss very small objects. The per-source `unassigned_overlay.png` colours red pixels outside all candidate masks blue; these are omission-review cues. Background decorations can legitimately remain unassigned, and an incorrect mask can cover the wrong object, so red coverage is not semantic recall.

Run Cell 6, download `review_gallery.html`, and open it in a browser. The gallery defaults to thirty pending objects at a time. Choose a category and a decision, enter the reviewer name, then download the reviewed masks JSON. Import it in Cell 7. Regenerate the gallery to review the next pending objects. Leave uncertain objects pending or reject them; do not approve a cropped animal missing its tail or an object that includes unrelated neighbouring leaves.

An object extending beyond its proposed box requires a corrected native box. The optional helper creates a new local SAM proposal for an already-catalogued source without another paid discovery request. Its mask still needs isolation review. Objects touching an original image edge can be genuinely clipped in the original and may be unsuitable reusable complete assets.

## Stored data

```text
GNR Object Repository/semantic_v1/
  objects.sqlite3
  objects.csv
  objects.json
  catalog_summary.json
  candidates/peacock/<instance-id>/
    source_crop.png
    object.png
    selection_mask.png
    metadata.json
  approved/peacock/
    <asset-id>.png
    <asset-id>.bmp
    <asset-id>_protected_white.png
  requests/<source-id>/view_0000/
  source_audits/<source-id>/
    audit.json
    unassigned_red.png
    unassigned_overlay.png
  review_gallery.html
  review_imports/
```

Categories include peacock, elephant, horse, deer, parrot, bird, animal, flower, leaf, petal, feather, stem, branch, tree, paisley, curl, vase, boat, house, bridge, sun, moon, ornament, border, dot and other. A free-text label keeps more specific identities such as lotus. Taxonomy changes require a new catalogue snapshot rather than silently reinterpreting an old one.

## Resumption, source integrity and export

The catalogue stores immutable source hashes and settings. Source changes/removals, scan-policy changes, model changes or prompt changes require a new library version. Successful discovery responses are cached. A request folder with no completed response blocks automatic repeat requests; inspect actual API usage before explicitly retrying with a new snapshot. Mask failures and invalid boxes are recorded rather than silently clipped, guessed or approved. Non-red/white, transparent and multiple-frame sources are excluded without repair.

Do not recursively index raw `approved/`: it includes white protection PNGs and historical asset versions. `export_current_approved` creates a new clean snapshot containing only current approved object PNGs and every source-instance record. Point `COMPONENTS_DIR` in `Designer_Component_Reuse.ipynb` to that export. Identical assets are deduplicated by category/raster; near-duplicates, mirrored and differently sized objects remain distinct. Source-family provenance is retained, but no training split is created here.

The existing draft-replacement agent targets connected draft regions, so it cannot yet automatically select/replace a complete multi-region peacock from this new repository. Whole-object layout matching and native mask composition are a subsequent step. No resizing/rotation fallback or guaranteed whole-sketch reconstruction is implied.

Isolation review confirms which source pixels belong to an object. It does not certify staircase geometry, two-pixel separation, paired front/back relationships or production readiness. Native copies preserve the designer's construction, including any original defects.

## Validation and execution status

All 99 project tests passed, including eight new catalogue tests covering disconnected whole-object preservation, internal white, source immutability, nested provenance, request budgets/resumption, exact masks, invalid batches, deduplication, approval/rejection and clean current-asset export. Vision responses and masks in these tests are fixtures, not evidence of live detection accuracy. An actual Transformers SAM forward/postprocessing smoke check with a tiny random model verified native mask dimensions and image-embedding reuse; pretrained mask quality was not tested. A Chromium browser check exercised native pixel painting, original-white protection, review-JSON download and backend import of the corrected object. The notebook contains current runtime code and no artwork, keys or saved results.

The full Drive dataset has not been inspected or processed in this workspace. Live API proposal quality, pretrained SAM mask quality on these drawings and exhaustive object recall remain to be measured in Colab. The notebook never publishes the user's source artwork or library to GitHub automatically.
