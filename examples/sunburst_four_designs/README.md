# Four supplied design pairs: Sunburst execution pending in Colab

The original eight BMP uploads are preserved byte-for-byte in separate folders:

| Folder | Step_2 and designer Step_3 native size |
| --- | --- |
| GNR_3001 | 400 × 720 |
| GNR_3002 | 360 × 600 |
| GNR_3003 | 360 × 600 |
| GNR_3004 | 480 × 190 |

Each folder contains `step_2.bmp`, `designer_step_3.bmp`, PNG previews and an input
status record. Every supplied target uses exact red/white and matches its sketch's
dimensions. `manifest.json` records input hashes and dimensions.

**No Sunburst images or comparison percentages have been generated yet.** The user
has an `OPENAI_API_KEY` secret inside Google Colab. That secret is not available
inside this managed cloud workspace. The supplied `userdata.get(...)` snippet is
an authentication recipe, not a transferable credential or an active connection.

Open the [standalone notebook in Colab](https://colab.research.google.com/github/MBG0903/GNR_Design_Project/blob/main/notebooks/Sunburst_Four_Design_Comparison.ipynb)
and run its cells. Only the version-matched Python runtime is embedded; the images
are saved on Drive and read from there:

1. Mount Drive in the first code cell. Inputs live under
   **MyDrive/GNR Data-Set/Sunburst_Four_Designs/** in four GNR_3001–GNR_3004 folders.
   Edit `DATA_ROOT` if your intended dataset folder has a different name.
2. Run setup and the input-loading cell. It copies matching Colab-uploaded files
   from `/content` (for example `GNR 3001 ( FILE 2 ).bmp`) and downloads any missing
   originals from pinned commit `7e938b6`. Targets are stored separately as
   `designer_step_3.bmp`. An independent reference pair is staged under
   `_references/design_362`. All ten original files are hash-verified before saving.
   Matching existing Drive files are reused; changed originals are not overwritten.
3. Enable notebook access to the existing **OPENAI_API_KEY** Colab Secret and
   verify access to the exact `gpt-image-2.5-sunburst` API model ID.
   Missing model access stops execution; another model is never selected silently.
4. Record the printed `RUN` folder and run the four paid image edits. Generation
   and comparison read from `INPUTS` on mounted Drive, not temporary runtime images.
5. Inspect/download the four folders and comparison CSV/ZIP. Set `RESUME_RUN` to
   the same folder to reuse completed drafts after an interruption.

Outputs are saved under `MyDrive/GNR Textile Runs/Sunburst_Four_Designs/run_...`.
Each output folder contains originals, actual raw generation, exact-native PNG/BMP,
prompt/provenance, native-coordinate metrics, comparison panel and difference map.
The root CSV reports exact pixel agreement, red IoU and Dice percentages for each
design. It reports completed designs only; absent images never receive fabricated
zero or placeholder accuracy scores.

None of these four designer targets is sent as a generation reference. An
independent design-362 sketch/approved-output pair supplies fill-treatment guidance.
Evaluation uses exact native coordinates, no alignment or target recolouring.
Prompts request dense red, two-native-pixel semantic gaps, intentional internal
white and 3/2/1 or 2/1 contour-run hierarchy. Only output palette and dimensions are
enforced by code. No target fitting, global gap cutting or staircase sorting is
applied; semantic business-rule compliance and production readiness stay unverified.

The notebook's image requests transmit query sketches and the independent reference
pair to OpenAI and use API billing. It stores no key in files or repository outputs.
Tests verify upload copying, real pinned-file downloads, hash checks, cached reruns
and preservation of conflicting Drive files using a temporary local Drive layout.
This development verification does not mount or modify the user's real Drive.
Mocked API tests cover native reconstruction, direct percentage checks, four-folder
exports, reference leakage rejection, interrupted-run resume and ZIP integrity.
These tests do not establish live Sunburst model quality or account access.

Rebuild after changing the batch runtime or inputs:

```bash
.venv/bin/python scripts/build_four_design_notebook.py
```
