"""Build standalone Colab split/review/merge experiment with the dense sample."""
from pathlib import Path
import base64,io,json,zipfile
root=Path(__file__).resolve().parents[1];buffer=io.BytesIO()
with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as archive:
    for p in sorted((root/'textile').glob('*.py')):archive.writestr(str(p.relative_to(root)),p.read_bytes())
    for name in ['requirements.txt','examples/step_2.bmp','examples/generated_dense_draft/step_3_dense_draft.png']:
        archive.writestr(name,(root/name).read_bytes())
payload=base64.b64encode(buffer.getvalue()).decode()
def md(s):return dict(cell_type='markdown',metadata={},source=s.splitlines(True))
def code(s):return dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=s.splitlines(True))
cells=[md('''# Dense Step 3 → tiles → staircase review → merge
This implements the proposed 9/12-tile experiment. It starts with the existing GenAI dense sample or your own sketch/dense output.
To generate a new dense proposal, run the generation stage of `GenAI_Textile_Colab.ipynb` first, then upload its draft here.

Tiling improves inspection context; it does not automatically solve hierarchy. This notebook supplies a manual pixel editor
and correction-data collection. Optional Sunburst proposals are available with API access; no autonomous rule compliance or training is claimed. The rejected strip fitter stays disabled.
Use a CPU runtime. No API key is needed for manual editing/data export; optional Sunburst proposals require separately enabled API billing/access.
'''),code('''import base64,io,pathlib,zipfile,sys,subprocess
runtime=pathlib.Path('/content/textile_tile_runtime') if pathlib.Path('/content').exists() else pathlib.Path.cwd()/'outputs/tile_runtime'
runtime.mkdir(parents=True,exist_ok=True)
payload='''+repr(payload)+'''
with zipfile.ZipFile(io.BytesIO(base64.b64decode(payload))) as archive:archive.extractall(runtime)
subprocess.check_call([sys.executable,'-m','pip','install','-r',str(runtime/'requirements.txt')])
sys.path.insert(0,str(runtime))
from PIL import Image
from IPython.display import display
'''),md('''## Choose inputs
The sample is the earlier dense GenAI draft, not the later rejected reference-fitting image.
Your own Step 2 and dense Step 3 must have exactly matching native dimensions.
Pure red/white Step 3 is required; inputs are never silently resized or recoloured.
'''),code('''USE_UPLOADED_IMAGES=False
SKETCH=runtime/'examples/step_2.bmp'
DENSE=runtime/'examples/generated_dense_draft/step_3_dense_draft.png'
if USE_UPLOADED_IMAGES:
    from google.colab import files
    print('Upload exactly one Step 2 image.')
    upload=files.upload()
    if len(upload)!=1:raise ValueError('Expected one sketch.')
    SKETCH=runtime/'uploaded_sketch'
    SKETCH.write_bytes(next(iter(upload.values())))
    print('Upload exactly one dense Step 3 image.')
    upload=files.upload()
    if len(upload)!=1:raise ValueError('Expected one dense draft.')
    DENSE=runtime/'uploaded_dense'
    DENSE.write_bytes(next(iter(upload.values())))
with Image.open(SKETCH) as image:print('Sketch:',image.size)
with Image.open(DENSE) as image:print('Dense:',image.size);display(image)
'''),md('''## Split at native resolution
12 pieces: ROWS=3, COLUMNS=4. For 9 pieces: ROWS=3, COLUMNS=3.
Each core owns distinct output pixels. The read-only context border is 32 pixels by default; increase it to see full curves.
If a motif/front/back/tip extends outside the context, inspect the full component before changing it.
By default, runs and correction data are retained in MyDrive/GNR Textile Reviews. Record the printed RUN path to resume later.
Tile-local coordinates map exactly to the manifest's global coordinates. No resampling or blending is used.
'''),code('''from textile.tiles import split,merge
import tempfile,json,uuid
ROWS=3
COLUMNS=4
OVERLAP=32
PERSIST_TO_DRIVE=True
RESUME_RUN=''  # Set to the existing experiment folder to continue prior designer edits.
DESIGN_GROUP_ID='sample_362' if not USE_UPLOADED_IMAGES else input('Original design/group ID (same for all variants): ').strip()
if PERSIST_TO_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
    WORK_ROOT=pathlib.Path('/content/drive/MyDrive/GNR Textile Reviews')
else:
    WORK_ROOT=pathlib.Path(tempfile.mkdtemp(prefix='textile_tiles_'))
if RESUME_RUN:
    from textile.tiles import read
    RUN=pathlib.Path(RESUME_RUN)
    manifest,_=read(RUN)
    SKETCH=RUN/'sketch.png';DENSE=RUN/'original.png'
else:
    RUN=WORK_ROOT/f'run_{uuid.uuid4().hex[:12]}'/'experiment'
    manifest=split(SKETCH,DENSE,RUN,rows=ROWS,columns=COLUMNS,overlap=OVERLAP)
print('Run:',RUN,'Tiles:',len(manifest['tiles']))
display(Image.open(RUN/'grid_preview.png'))
# Mandatory zero-edit reconstruction check before trying corrections.
roundtrip,report=merge(RUN)
import numpy as np
with Image.open(DENSE) as original:
    assert np.array_equal(np.asarray(original.convert('RGB')),np.asarray(roundtrip))
print('Unedited split/merge is pixel-identical:',roundtrip.size)
print(report)
'''),md('''## Review and edit each tile
The editor shows the sketch, AI input, designer target, a blue core guide and the full image. It also has an integer-magnified click grid for exact one-pixel changes.
Only the raw red/white pixels are saved; the blue guide is diagnostic.

- Inspect complete inner/front and outer/back curves, both tips and transitions.
- Prefer 3→2→1 with n3<n2<n1. Use 2→1 with n2<n1 when shape/size prevents 3→2→1.
- Preserve the short front/long back anchor relation, two-pixel gaps, intentional white and motif placement.
- Edit core pixels only. Halo edits are rejected. Coordinate adjacent-core changes and inspect the full curve after merging.
- Normalize the edited palette explicitly, inspect again, resolve issues, then sign off. Every save retains a new version.
- Enter a consistent original design/group ID, rule tags and a short correction explanation.
- Each save captures sketch/AI/designer pairs, changed/core masks, notes and review metadata. Revisions are retained.
- Export the correction dataset ZIP using the panel button. By default only the latest changed revision per tile is exported.
- The review checkboxes are attestations, not automatic compliance checks.

The Gradio relay is a temporary public URL protected by your chosen password. For private processing run locally.
'''),md('''## Connect Sunburst and optionally generate ALL tiles
For a new batch, enable API access and RUN_SUNBURST_BATCH below. This makes one paid image request per missing tile (12 with the default grid).
Completed proposals remain on Drive: resume the SAME RUN after an interruption to avoid generating completed tiles again.
The output includes all native tiles, a combined PNG/BMP matching Step 2 exactly, a before/after comparison and a ZIP.
This is an **unreviewed AI preview**; it does not bypass the designer review gates or establish staircase correctness.
API credentials are read from Colab Secrets/hidden input and are never saved. Chat subscriptions do not supply API billing.
'''),code('''import getpass
ENABLE_SUNBURST_PROPOSALS=False
RUN_SUNBURST_BATCH=False
IMAGE_MODEL='gpt-image-2.5-sunburst'
api_client=None
if ENABLE_SUNBURST_PROPOSALS or RUN_SUNBURST_BATCH:
    from google.colab import userdata
    from openai import OpenAI
    try:key=userdata.get('TEXTILE_API_KEY')
    except Exception:key=getpass.getpass('OpenAI API key (hidden): ')
    if not key:raise ValueError('Add TEXTILE_API_KEY to Colab Secrets and enable notebook access.')
    api_client=OpenAI(api_key=key)
    del key
if RUN_SUNBURST_BATCH:
    from textile.tile_batch import generate_all
    BATCH_OUTPUT,BATCH_REPORT=generate_all(RUN,api_client,model=IMAGE_MODEL,progress=print)
    print('Batch saved:',BATCH_OUTPUT)
    print(json.dumps(BATCH_REPORT,indent=2))
    display(Image.open(BATCH_OUTPUT/'combined_unreviewed.png'))
    display(Image.open(BATCH_OUTPUT/'before_left_after_right.png'))
    display(Image.open(BATCH_OUTPUT/'tile_seam_guide_diagnostic.png'))
    for tile in manifest['tiles']:
        print(tile['id'],'— generated context tile (only its core is merged)')
        display(Image.open(BATCH_OUTPUT/'tiles'/f"{tile['id']}_context.png"))
    from google.colab import files
    files.download(str(BATCH_OUTPUT/'all_tiles_and_combined_review.zip'))
    files.download(str(BATCH_OUTPUT/'combined_unreviewed.png'))
    files.download(str(BATCH_OUTPUT/'combined_unreviewed.bmp'))
'''),md('''## Open the designer panel
The batch preview contains pending AI candidates. Select each tile and use **Generate Sunburst proposal** only if you want a new paid alternative.
Review/edit the candidate and save a designer correction before using the reviewed merge. Batch previews carry no sign-off.
'''),code('''from textile.tile_app import build_tile_app
password=getpass.getpass('Choose a session password (12+ characters): ')
if len(password)<12:raise ValueError('Choose a longer password.')
app=build_tile_app(RUN,design_group_id=DESIGN_GROUP_ID,ai_client=api_client,image_model=IMAGE_MODEL)
app.launch(share=True,inline=True,auth=('designer',password),show_error=False)
'''),md('''## How the AI can learn from the corrections
Saving examples does NOT update Astra or Sunburst weights. The dataset is provider-neutral: original sketch,
AI proposal, designer result, changed-pixel mask, editable-core loss mask, rule notes and group IDs.
Use designer-approved examples as references first, and choose a verified supported training model/interface for fine-tuning later.
Do not assume that a ChatGPT subscription supplies API training or that Sunburst/Astra support arbitrary fine-tuning.
Keep all tiles/revisions/crops/variants from one original design in the SAME evaluation split.
Test on held-out complete designs before claiming improved accuracy. Human review remains required.

## Merge and inspect globally
Use the editor's merge/download button, or the cell below after saving proposals.
The merge copies each disjoint core exactly once. No averaging, antialiasing, scaling or global strip fitting is applied.
**Whole-image review remains required:** inspect every cross-tile contour, hierarchy, gap and seam at native resolution.
A tile can look correct while a curve assembled across tile boundaries is wrong. Do not call a format-valid merge production ready.
Colab files are temporary: download the output and the full review archive before ending the runtime.
'''),code('''from textile.tiles import merge_saved
MERGED,MERGE_REPORT=merge_saved(RUN)
MERGED.save(RUN/'merged_review_draft.png')
MERGED.save(RUN/'merged_review_draft.bmp')
(RUN/'merge_report.json').write_text(json.dumps(MERGE_REPORT,indent=2))
display(MERGED)
print(json.dumps(MERGE_REPORT,indent=2))
'''),code('''from google.colab import files
import shutil
archive=shutil.make_archive(str(RUN.parent/'tile_review_archive'),'zip',RUN)
files.download(archive)
files.download(str(RUN/'merged_review_draft.png'))
files.download(str(RUN/'merged_review_draft.bmp'))
''')]
notebook=dict(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'},'colab':{'name':'Tile_Staircase_Review.ipynb'}},nbformat=4,nbformat_minor=4)
(root/'notebooks/Tile_Staircase_Review.ipynb').write_text(json.dumps(notebook,indent=1))
print('Built Tile_Staircase_Review.ipynb')
