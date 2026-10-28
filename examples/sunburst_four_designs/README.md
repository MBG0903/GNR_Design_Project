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
and run its cells. It embeds all eight inputs and the version-matched runtime:

1. Enable notebook access to the existing **OPENAI_API_KEY** Colab Secret.
2. Connect and verify access to the exact `gpt-image-2.5-sunburst` API model ID.
   Missing model access stops execution; another model is never selected silently.
3. Mount Drive, record the printed `RUN` folder and run the four paid image edits.
4. Inspect/download the four folders and comparison CSV/ZIP. Set `RESUME_RUN` to
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
Mocked tests cover native reconstruction, direct percentage checks, four-folder
exports, reference leakage rejection, interrupted-run resume and ZIP integrity.
These tests do not establish live Sunburst model quality or account access.

Rebuild after changing the batch runtime or inputs:

```bash
.venv/bin/python scripts/build_four_design_notebook.py
```
