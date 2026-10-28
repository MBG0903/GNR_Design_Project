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
and review records, not an autonomous Astra staircase repair. The rejected strip fitter stays disabled.
Use a CPU runtime. No API key is needed for splitting, editing or merging.
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
Tile-local coordinates map exactly to the manifest's global coordinates. No resampling or blending is used.
'''),code('''from textile.tiles import split,merge
import tempfile,json
ROWS=3
COLUMNS=4
OVERLAP=32
RUN=pathlib.Path(tempfile.mkdtemp(prefix='textile_tiles_'))/'experiment'
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
The editor shows the sketch context, a blue core guide, the raw draft and the full image.
Only the raw red/white pixels are saved; the blue guide is diagnostic.

- Inspect complete inner/front and outer/back curves, both tips and transitions.
- Prefer 3→2→1 with n3<n2<n1. Use 2→1 with n2<n1 when shape/size prevents 3→2→1.
- Preserve the short front/long back anchor relation, two-pixel gaps, intentional white and motif placement.
- Edit core pixels only. Halo edits are rejected. Coordinate adjacent-core changes and inspect the full curve after merging.
- Normalize the edited palette explicitly, inspect again, resolve issues, then sign off. Every save retains a new version.
- The review checkboxes are attestations, not automatic compliance checks.

The Gradio relay is a temporary public URL protected by your chosen password. For private processing run locally.
'''),code('''import getpass
from textile.tile_app import build_tile_app
password=getpass.getpass('Choose a session password (12+ characters): ')
if len(password)<12:raise ValueError('Choose a longer password.')
app=build_tile_app(RUN)
app.launch(share=True,inline=True,auth=('designer',password),show_error=False)
'''),md('''## Merge and inspect globally
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
