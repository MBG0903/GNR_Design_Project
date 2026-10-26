"""Generate a standalone Colab notebook containing the version-matched runtime."""
from pathlib import Path
import base64
import io
import json
import zipfile
root=Path(__file__).resolve().parents[1]
buffer=io.BytesIO()
with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as archive:
    for p in sorted((root/'textile').glob('*.py')):
        archive.writestr(str(p.relative_to(root)),p.read_bytes())
    for name in ['requirements.txt','examples/step_2.bmp','examples/step_3_reference.bmp']:
        archive.writestr(name,(root/name).read_bytes())
payload=base64.b64encode(buffer.getvalue()).decode()
def md(s):return dict(cell_type='markdown',metadata={},source=s.splitlines(True))
def code(s):return dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=s.splitlines(True))
cells=[md('''# GenAI → Photoshop textile design agent
**Workflow:** Step 2 sketch → generative image edit → vision-agent rule review → Photoshop pixel edits → re-review → designer approval.
No conventional computer-vision model is trained. Your Drive pairs are a reference library and held-out evaluation set.
CPU Colab is sufficient: generation/review use hosted APIs. An OpenAI API key with image-model access and API billing is needed.
Calls send the sketch and selected references to the API. Model IDs are configurable and access depends on your account.
Photoshop runs on your desktop, not in Colab. This first version exports executable JSX edits; file transfer/execution is explicit.
Do not label any output production ready until a designer approves the rules and loom-specific requirements.
'''),code('''import base64, io, pathlib, zipfile, sys, subprocess
runtime = pathlib.Path('/content/textile_runtime') if pathlib.Path('/content').exists() else pathlib.Path.cwd() / 'outputs/colab_runtime'
runtime.mkdir(parents=True, exist_ok=True)
payload = '''+repr(payload)+'''
with zipfile.ZipFile(io.BytesIO(base64.b64decode(payload))) as archive:
    archive.extractall(runtime)
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', str(runtime/'requirements.txt')])
sys.path.insert(0,str(runtime))
from PIL import Image
from IPython.display import display
'''),md('''## Mount the paired reference library
The expected paths are `MyDrive/GNR Design Data-Set/Stage_2` and `Stage_3`.
Keep API keys out of files and source control. Add `TEXTILE_API_KEY` in Colab Secrets and allow notebook access, or use the hidden prompt.
'''),code('''from google.colab import drive, userdata
drive.mount('/content/drive')
from openai import OpenAI
import getpass
try:
    api_key = userdata.get('TEXTILE_API_KEY')
except Exception:
    api_key = getpass.getpass('OpenAI API key (hidden): ')
client=OpenAI(api_key=api_key)
del api_key
DATA=pathlib.Path('/content/drive/MyDrive/GNR Design Data-Set')
RUN=pathlib.Path('/content/drive/MyDrive/GNR Textile Runs/genai_001')
RUN.mkdir(parents=True,exist_ok=True)
IMAGE_MODEL='gpt-image-2.5-sunburst'
REVIEW_MODEL=input('Exact Astra API model ID available in your account: ').strip()
'''),md('''## Audit pairs and protect held-out evaluation
Names like `362 GNR 2.bmp` / `362 GNR 3.bmp` pair automatically. Ambiguous/unmatched files, invalid target palettes,
and mismatched dimensions are reported instead of silently modified.

Since the library contains variants/crops, **the first audit stops and writes `groups_template.csv`**.
Fill `design_id` for every row, using the SAME ID for all crops/variants of one original design.
Save as `groups.csv` in the run folder and rerun. Unrelated originals need distinct IDs.
This produces reference/validation/test splits by whole design and groups exact duplicates together.
This is evaluation isolation, not model training. Inspect the contact sheet to confirm pairing.
'''),code('''import json, random
from textile.data import audit_dataset, contact_sheet
report=audit_dataset(DATA/'Stage_2',DATA/'Stage_3',RUN,groups_csv=RUN/'groups.csv',require_groups=True)
print(json.dumps(report,indent=2))
rows=json.loads((RUN/'manifest.json').read_text())
sample=random.Random(42).sample(rows,min(12,len(rows)))
display(Image.open(contact_sheet(sample,RUN/'contact_sheet.png')))
'''),md('''## Select references and the query
Choose up to three **relevant, designer-approved reference pairs** from the reference split.
Do not send all 1,600 pairs in a prompt. This version uses manual reference selection; automatic semantic retrieval is a later improvement.
For evaluation use a test sketch, without sending its Step 3 target to the model. Replace QUERY with a new uploaded sketch for real work.
'''),code('''REFERENCE_KEYS=[]  # Fill with 1–3 canonical keys from the reference split after inspecting the library.
reference_rows=[r for r in rows if r['split']=='reference']
print('Available reference keys (first 30):',[r['key'] for r in reference_rows[:30]])
selected=[r for r in reference_rows if r['key'] in REFERENCE_KEYS]
if not 1<=len(selected)<=3 or len(selected)!=len(REFERENCE_KEYS):
    raise ValueError('Select 1–3 distinct valid reference keys, then rerun.')
references=[(r['source'],r['target']) for r in selected]
QUERY=next(r['source'] for r in rows if r['split']=='test')
display(Image.open(QUERY))
'''),md('''## Generate the Step 3 draft (paid API call)
GenAI interprets the design using the rules and reference examples.
Square padding preserves aspect ratio for generation; the result is restored to native dimensions and exact red/white.
The output can still change geometry or misunderstand white channels. Palette/dimensions are not semantic approval.
The raw generated image and transformation record are retained for diagnosis.
'''),code('''from textile.genai import generate, review
from textile.core import validate
DRAFT=generate(QUERY,RUN/'generation',client=client,references=references,model=IMAGE_MODEL)
display(Image.open(DRAFT))
with Image.open(QUERY) as source:
    print(validate(Image.open(DRAFT),source.size))
'''),md('''## Agent reviews rules and prepares Photoshop edits (paid API call)
The vision agent compares the ORIGINAL sketch with the CURRENT raster and proposes small red/white polygon fills.
Uncertain regions are flagged. Plans are bounded and validated before any JSX is emitted.
Large/redraw corrections need designer intervention; the agent cannot prove all business rules.
'''),code('''CURRENT=DRAFT  # Later iterations: set this to the actual Photoshop-exported PNG.
PLAN=review(QUERY,CURRENT,RUN/'agent_plan.json',client=client,model=REVIEW_MODEL)
print(json.dumps(PLAN,indent=2))
from textile.photoshop import write_jsx
with Image.open(CURRENT) as current:
    script=write_jsx(PLAN,current.size,RUN/'apply_agent_edits.jsx',CURRENT)
from google.colab import files
files.download(CURRENT)
files.download(script)
from textile.windows_worker import package_job
files.download(package_job(CURRENT, PLAN, RUN/'photoshop_job.zip'))
'''),md('''## Execute on your Windows PC, then close the loop
**Recommended for Mac + Windows:** keep Colab on the Mac and Photoshop on Windows.
Download `photoshop_job.zip`, transfer it to Windows (Drive or Remote Desktop), and unzip into a new folder.
Install this project and `requirements-windows.txt` on Windows, then run:
`python -m textile.windows_worker "C:\\path\\to\\unpacked_job"`
The local worker rebuilds validated JSX from data, drives Photoshop through Windows COM,
and writes `results/agent_corrected.psd`, `agent_corrected.png`, and `worker_report.json`.
No network server is opened. Return the PNG to this notebook for the next review.
The Windows integration has to be verified on your machine; it cannot be exercised in Colab.

**Manual fallback:**
1. On your desktop, open Photoshop → File → Scripts → Browse and select `apply_agent_edits.jsx`.
2. Select the exact PNG downloaded above. The script checks byte length/checksum and native dimensions.
3. It duplicates the image and applies agent polygon edits on a separate layer, with no antialiasing.
4. Choose a NEW output folder. It saves a layered PSD and `agent_corrected.png`; the source is retained.
5. Upload the actual exported PNG below, validate it, then rerun agent review with CURRENT pointing to that export.

This handoff is explicit: Colab does not remotely control your desktop. The Photoshop execution has to be tested on your installed version.
Never substitute a simulated raster for proof that Photoshop applied the edits.
Repeat review/edit for a small bounded number of iterations; send unresolved geometry to the designer.
'''),code('''uploaded=files.upload()
if len(uploaded)!=1:
    raise ValueError('Upload exactly one actual Photoshop-exported PNG.')
name,content=next(iter(uploaded.items()))
if pathlib.Path(name).suffix.lower()!='.png':
    raise ValueError('Expected a PNG export.')
CURRENT=str(RUN/'photoshop_return.png')
pathlib.Path(CURRENT).write_bytes(content)
with Image.open(QUERY) as source, Image.open(CURRENT) as corrected:
    checks=validate(corrected,source.size)
    display(corrected)
    print(checks)
    if not checks['automated_checks_passed']:
        raise ValueError('Photoshop export failed palette/size checks; inspect colour settings and geometry.')
'''),md('''## Designer approval and lossless raster export
Upload QUERY and CURRENT in the app and choose **Load model draft for current Step 2 sketch**.
Resolve uncertainties, inspect the pixel geometry, and sign off each rule before export.
The temporary Gradio relay is public and password protected. Use the local app instead for private processing.
PNG/BMP can be opened in Photoshop/PaintShop Pro; loom formats and weave mapping need a separate specification.
'''),code('''from textile.app import build_app
password=getpass.getpass('Choose a session password (12+ characters): ')
if len(password)<12: raise ValueError('Use a longer password.')
print('Sketch:',QUERY,'Current:',CURRENT)
app=build_app()
app.launch(share=True,inline=True,auth=('designer',password),show_error=False)
''')]
notebook=dict(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
                                  'language_info':{'name':'python'},'colab':{'name':'GenAI_Textile_Photoshop.ipynb'}},nbformat=4,nbformat_minor=4)
(root/'notebooks/GenAI_Textile_Photoshop.ipynb').write_text(json.dumps(notebook,indent=1))
print('Built GenAI notebook')
# Primary workflow: hosted Sunburst + Astra review + native pixel edits in Colab.
colab_cells=list(cells)
colab_cells[0]=md('''# Sunburst → Astra review → Colab corrections
Generate Step 3 with Sunburst, review it with your Astra reasoning model, apply bounded native-pixel corrections in Colab,
then repeat review and obtain designer approval. Photoshop is optional.
Your 1,600+ paired designs serve as a reference library and evaluation set; there is no conventional CV conversion model.
A CPU Colab runtime is sufficient. ChatGPT subscriptions and API billing/access are separate; confirm account access.
Artwork is sent to the API during generation/review. Do not embed API keys in the notebook.
''')
colab_cells[3]=code('''from google.colab import drive, userdata
drive.mount('/content/drive')
from openai import OpenAI
import getpass
try:
    api_key=userdata.get('TEXTILE_API_KEY')
except Exception:
    api_key=getpass.getpass('OpenAI API key (hidden): ')
client=OpenAI(api_key=api_key)
del api_key
DATA=pathlib.Path('/content/drive/MyDrive/GNR Design Data-Set')
RUN=pathlib.Path('/content/drive/MyDrive/GNR Textile Runs/sunburst_colab_001')
RUN.mkdir(parents=True,exist_ok=True)
IMAGE_MODEL='gpt-image-2.5-sunburst'
# Astra is an account-facing name. Supply your verified API ID; do not guess it.
REVIEW_MODEL=input('Exact Astra API model ID available in your account: ').strip()
if not REVIEW_MODEL: raise ValueError('A verified reasoning-model API ID is required for review.')
''')
colab_cells[10]=md('''## Astra review (paid API call)
Compare the original sketch to the current raster, flag unresolved regions and propose local corrections.
The model returns data, never executable code. Corrections are bounded pure red/white polygons at native pixel coordinates.
Visual model coordinates can be wrong: inspect each plan before applying. Semantic approval remains with the designer.
''')
colab_cells[11]=code('''from textile.genai import review
CURRENT=DRAFT  # For the next review, assign the previous iteration's corrected.png.
PLAN=review(QUERY,CURRENT,RUN/'agent_plan.json',client=client,model=REVIEW_MODEL)
print(json.dumps(PLAN,indent=2))
display(Image.open(QUERY),Image.open(CURRENT))
''')
colab_cells[12]=md('''## Apply confirmed corrections in Colab
Inspect the coordinates/reasons. Remove unsupported operations; retain their issues in `unresolved`.
Set the flag below only after checking the proposals. Code applies exact pixel fills without global erosion,
hole filling or morphology. Those global operations can break intentional white channels and genuine connections.
Each iteration saves PNG, BMP and an audit record in a NEW directory. Uncertainty remains unresolved until reviewed.
''')
colab_cells[13]=code('''from textile.colab_edits import apply_plan
CORRECTIONS_REVIEWED=False
ITERATION=1
if not CORRECTIONS_REVIEWED:
    raise RuntimeError('Inspect the plan and confirm each correction before enabling.')
CORRECTED=apply_plan(CURRENT,PLAN,RUN/f'corrections_{ITERATION:02d}',source=QUERY)
display(Image.open(CURRENT),Image.open(CORRECTED))
print((RUN/f'corrections_{ITERATION:02d}/audit.json').read_text())
# Next pass: set CURRENT=CORRECTED in the review cell; increase ITERATION.
# Stop after a few passes if issues persist; ask the designer instead of guessing.
''')
colab_cells[14]=md('''## Designer sign-off and lossless export
Upload QUERY and CORRECTED (or DRAFT if no changes were necessary) to the review app.
Inspect at native pixel resolution, resolve uncertainty, and sign off all R1–R16 before exporting.
Every exported raster must have exactly the original Step 2 width and height. Size mismatches block corrections/export. Palette and dimensions are deterministic checks; no automatic semantic production certification is implied.
The Gradio relay is public and password protected. Local mode is available for private processing.
Photoshop can open the final PNG/BMP if manual finishing is needed. Loom formats/weave mapping remain separate requirements.
''')
colab_cells[15]=code('''from textile.app import build_app
password=getpass.getpass('Choose a session password (12+ characters): ')
if len(password)<12: raise ValueError('Use a longer password.')
print('Sketch:',QUERY,'Draft:',DRAFT)
app=build_app()
app.launch(share=True,inline=True,auth=('designer',password),show_error=False)
''')
colab_notebook=dict(notebook,cells=colab_cells)
colab_notebook['metadata']=dict(notebook['metadata'],colab={'name':'GenAI_Textile_Colab.ipynb'})
(root/'notebooks/GenAI_Textile_Colab.ipynb').write_text(json.dumps(colab_notebook,indent=1))
print('Built Sunburst/Colab notebook')

# R2 pixel-rule stage: semantic labels must be verified before claiming completion.
rule_cells=[md("## R2 only: two-pixel separation\nNeighbouring semantic pieces need two native white pixels at closest approach; preserve wider background spaces.\nEach semantic instance needs its own positive integer label; a continuous stem/curl/body retains one label.\nConnected-region labels below are only a PROPOSAL: they cannot identify touching leaves versus stems.\nInspect and correct the label map for joined pieces. Never claim semantic R2 compliance from component counts.\nThe algorithm removes only conflicting edge pixels, preserves each labelled piece's connectivity and at least one anchor,\nand flags impossible constraints. It does not resize, add red, alter other rules or certify production readiness.\n"),code("import numpy as np\nfrom scipy import ndimage\nfrom textile.core import rgb, render, validate\nfrom textile.separation import separate\nwith Image.open(CURRENT) as opened:\n    current_rgb=rgb(opened)\nwith Image.open(QUERY) as original:\n    assert current_rgb.size == original.size, 'Source dimensions must match before R2.'\nred=np.all(np.asarray(current_rgb)==[255,0,0],axis=2)\npiece_labels,_=ndimage.label(red,np.ones((3,3)))\n# Replace piece_labels with reviewed semantic instance labels: label 0=white,\n# one positive integer per piece, preserving legitimate continuous forms.\nSEMANTIC_LABELS_REVIEWED=False\n# Safe to inspect a provisional pass, but not certify it.\nseparated,spacing=separate(piece_labels)\nspacing['semantic_approval']=SEMANTIC_LABELS_REVIEWED\nspacing['rule_complete']=bool(SEMANTIC_LABELS_REVIEWED and spacing['clearance_passed'])\ndisplay(render(separated>0))\nprint(json.dumps(spacing,indent=2))\nR2_OUT=RUN/'r2_separation'\nR2_OUT.mkdir(exist_ok=True)\nrender(separated>0).save(R2_OUT/'r2_draft.png')\n(R2_OUT/'audit.json').write_text(json.dumps(spacing,indent=2))\n# CURRENT=str(R2_OUT/'r2_draft.png') only after inspecting the result.\n")]
colab_notebook['cells']=colab_cells[:14]+rule_cells+colab_cells[14:]
(root/'notebooks/GenAI_Textile_Colab.ipynb').write_text(json.dumps(colab_notebook,indent=1))
# Explicit contour annotations are necessary; an unrun check is never a pass.
contour_cells=[md('''## Pixel hierarchy: contour runs (not stroke thickness)
Front = inner/concave; back = outer/convex. Front long anchor must be shorter. When size/shape prevents 3/2/1, the approved fallback is 2/1 with n2 < n1.
After each anchor, follow 3-pixel runs, then 2-pixel runs, then 1-pixel runs, with n3 < n2 < n1.
Select pixel-grid boundary traces from anchor toward tip/turn; split at direction reversals.
A trace is a list of integer [x,y] grid-edge vertices. These must match the actual red/white raster boundary.
Do not infer semantic front/back from connected components or apply the rule to dots/broad non-tapering petals.
The module can replace selected contours while preserving endpoints, connectivity and two-pixel clearance,
but infeasible cases need design review. No image-wide compliance follows from one passing contour.
'''),code('''from textile.pixel_hierarchy import check_pair
# Populate from inspected native-pixel boundary annotations; do not guess coordinates.
FRONT_TRACE=[]
BACK_TRACE=[]
LONG_AXIS='y'  # 'x' for a horizontal long anchor
if not FRONT_TRACE or not BACK_TRACE:
    print('Hierarchy check UNRUN: annotate the front and back contours first.')
else:
    with Image.open(CURRENT) as opened:
        red_mask=np.all(np.asarray(rgb(opened))==[255,0,0],axis=2)
    hierarchy=check_pair(FRONT_TRACE,BACK_TRACE,LONG_AXIS,mask=red_mask,allow_21=True)
    print(json.dumps(hierarchy,indent=2))
''')]
colab_notebook['cells']=colab_notebook['cells'][:-2]+contour_cells+colab_notebook['cells'][-2:]
(root/'notebooks/GenAI_Textile_Colab.ipynb').write_text(json.dumps(colab_notebook,indent=1))
# Whole-raster experiment follows the R2 stage and retains its instance labels.
whole_cells=[md('''## Whole-image pixel hierarchy experiment
This scans all outer/inner boundaries and both orientations, not just the annotated example.
It uses the labelled raster from the two-pixel separation stage. If you edited the image afterward,
update/review those labels first. Geometric concavity pairing and compact-shape exclusions need semantic review.

The pass applies feasible 3/2/1 reorderings within two native pixels, preserves arc endpoints and checks
piece connectivity, internal white-region count and two-pixel separation. Already accepted traces are protected.
It tries 3/2/1 across the image first, then uses 2/1 with n2 < n1 where size/shape or geometric constraints prevent 3/2/1. It flags cases that cannot fit either hierarchy instead of declaring the whole image compliant.
Larger images and many contour arcs can take substantially longer than the supplied 401×340 example.
'''),code('''from textile.global_hierarchy import run_global
GLOBAL_LABELS=separated  # Output labels from the R2 stage, not a resized preview.
GLOBAL_RESULT,GLOBAL_AUDIT=run_global(GLOBAL_LABELS,max_displacement=2,allow_21=True)
GLOBAL_OUT=RUN/'whole_image_hierarchy'
GLOBAL_OUT.mkdir(exist_ok=True)
render(GLOBAL_RESULT>0).save(GLOBAL_OUT/'step_3_whole_image_hierarchy.png')
render(GLOBAL_RESULT>0).save(GLOBAL_OUT/'step_3_whole_image_hierarchy.bmp')
np.save(GLOBAL_OUT/'piece_labels.npy',GLOBAL_RESULT)
import gzip
(GLOBAL_OUT/'contour_audit.json.gz').write_bytes(gzip.compress(json.dumps(GLOBAL_AUDIT,indent=2).encode()))
print(json.dumps({k:v for k,v in GLOBAL_AUDIT.items() if k not in ['arcs','pieces']},indent=2))
display(render(GLOBAL_LABELS>0),render(GLOBAL_RESULT>0))
# Inspect unresolved outcomes; image-wide approval remains false.
# CURRENT=str(GLOBAL_OUT/'step_3_whole_image_hierarchy.png') after visual review.
''')]
insert_at=next(i for i,c in enumerate(colab_notebook['cells']) if c['cell_type']=='markdown' and ''.join(c['source']).startswith('## Pixel hierarchy:'))
colab_notebook['cells'][insert_at:insert_at]=whole_cells
(root/'notebooks/GenAI_Textile_Colab.ipynb').write_text(json.dumps(colab_notebook,indent=1))
# The strip-fitting stage is disabled after designer-confirmed visual regression.
rejection_note=md('## Rejected contour-fitting stage — disabled\nThe designer rejected the reference-assisted boundary-strip version because it spoiled curve geometry and pixel hierarchy.\nA higher pixel-similarity score did not make it acceptable. Public fitting and its CLI are blocked.\nDo not apply local grow/trim strips to improve reference overlap.\nUse the pre-repair whole-image hierarchy result as the review baseline; the sample restored files are in `examples/current_step3/`.\nThat baseline still has known unresolved hierarchy/semantic issues. Further repairs require complete paired\ninner/outer contour construction and validation, including the tips and transitions; recognized passing arcs alone are insufficient.\n')
colab_notebook['cells']=colab_notebook['cells'][:-2]+[rejection_note]+colab_notebook['cells'][-2:]
(root/'notebooks/GenAI_Textile_Colab.ipynb').write_text(json.dumps(colab_notebook,indent=1))
