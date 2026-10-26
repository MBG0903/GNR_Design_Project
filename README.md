# GNR Design Project

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
