"""Standalone central component repository + automatic exact-size reuse agent."""
import base64
import io
import json
import zipfile
from pathlib import Path
root=Path(__file__).resolve().parents[1]
buf=io.BytesIO()
with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
    for name in ('__init__.py','core.py','data.py','genai.py','components.py','component_library.py','component_agent.py'):
        z.writestr('textile/'+name,(root/'textile'/name).read_bytes())
payload=base64.b64encode(buf.getvalue()).decode()
def md(s): return dict(cell_type='markdown',metadata={},source=s.splitlines(True))
def code(s): return dict(cell_type='code',metadata={},source=s.splitlines(True),outputs=[],execution_count=None)
cells=[md('''# Central component repository → agent search → paste unchanged
Index components from **all designer Step_3 files** in Drive once. For a new Step_2, a vision agent proposes pieces, searches exact-size library candidates and pastes suitable original pixels into a native-size review canvas.

This version follows the batch-repository approach: no manual extraction or approval form for every component, no image-generation call, no component scaling/rotation, and no conventional CV sketch-conversion model. Connected-region extraction only separates existing designer pixels; CLIP only ranks candidates; the vision agent proposes semantic matches.

**Limits:** disconnected red regions are extraction candidates, not verified semantic pieces. Touching leaves/stems may remain merged; fragments and dots remain separate. Invalid source palettes are listed and excluded without altering originals. Agent bounding boxes and matches are estimates. Unmatched, uncertain or colliding pieces remain unresolved; the exported composition can be partial. Existing production rules in the selected source raster are preserved, but global composition and semantic gaps need designer review.

A GPU runtime is recommended for the one-time similarity index. Indexing uses no OpenAI API calls. Matching uses your `OPENAI_API_KEY` Colab Secret and a **vision reasoning model** with Responses API access; Sunburst image generation is not used to redraw components.
'''),md('''## 1. Mount Drive
`LIBRARY` is the reusable central repository. Original designer files are never changed. Keep v1 to resume indexing; use a new library version if original files change or are removed. New files can be added.
'''),code('''from google.colab import drive
from pathlib import Path
drive.mount('/content/drive')
STAGE_3 = Path('/content/drive/MyDrive/GNR Design Data-Set/Stage_3')
LIBRARY = Path('/content/drive/MyDrive/GNR Component Library/v1')
RUN = Path('/content/drive/MyDrive/GNR Textile Runs/Component_Reuse/trial_01')
MAX_PIECES = 20  # First pilot: at most 20 agent-proposed pieces. Set up to 500 for a larger run.
TOP_K = 3
if not STAGE_3.is_dir(): raise FileNotFoundError(f'Correct STAGE_3: {STAGE_3}')
print('Central library:', LIBRARY)
print('Review outputs:', RUN)
'''),md('''## 2. Load runtime
Only code is embedded. Pretrained CLIP weights download from Hugging Face on first use. Similarity indexing runs locally in Colab.
'''),code('''import base64, io, json, sys, subprocess, zipfile
RUNTIME = Path('/content/gnr_component_runtime')
RUNTIME.mkdir(exist_ok=True)
PAYLOAD = '''+repr(payload)+'''
with zipfile.ZipFile(io.BytesIO(base64.b64decode(PAYLOAD))) as z: z.extractall(RUNTIME)
subprocess.check_call([sys.executable,'-m','pip','install',
    'numpy>=1.26,<3','Pillow>=10.4,<13','scipy>=1.12,<2',
    'openai>=2,<3','sentence-transformers>=3,<6','ipywidgets>=8,<9'])
sys.path.insert(0,str(RUNTIME))
for name in list(sys.modules):
    if name == 'textile' or name.startswith('textile.'): del sys.modules[name]
from PIL import Image
from IPython.display import display
from textile.component_library import index_designer_images, embed_library, EMBEDDER
from textile.component_agent import run_component_agent
'''),md('''## 3. Batch-extract every designer image
This reads Stage_3 directly; Step_2/Step_3 filename pairing is not needed for the library. Every 8-connected red region is stored, including single pixels. Other regions inside its bounding box are not incorporated into that component. The native red pixels and original source coordinates are retained.

The source audit distinguishes indexed and excluded images. Non-red/white source images are excluded and listed, never thresholded automatically. Closed white holes from the original are protected. Open internal white channels are not automatically classified.
Source records and component extraction are checkpointed after each designer image; reruns skip unchanged indexed files. Inspect `source_audit.csv` if excluded_files is nonzero: the resulting library covers only the valid subset.
'''),code('''index_report = index_designer_images(STAGE_3, LIBRARY, progress=print)
print(json.dumps(index_report,indent=2))
print('Full source audit:',LIBRARY/'source_audit.csv')
if not index_report['components']: raise ValueError('No components extracted. Inspect the source audit.')
'''),md('''## 4. Build the searchable index once
This can take time for tens of thousands of pieces. Identical raster shapes share an embedding, while all original source records remain stored. Each batch is saved; rerun to continue after a Colab interruption. Exact width/height are indexed separately and are mandatory match filters.
'''),code('''from sentence_transformers import SentenceTransformer
encoder = SentenceTransformer(EMBEDDER)
print(embed_library(LIBRARY,encoder=encoder,progress=print))
'''),md('''## 5. Load one new Step_2
Set QUERY_PATH to an existing Drive file, or leave it blank to upload one sketch. RUN must be a new folder for a different sketch or changed settings; keep the same folder to resume the same experiment.
'''),code('''QUERY_PATH = ''  # Example: /content/drive/MyDrive/GNR Design Data-Set/Stage_2/362 GNR 2.bmp
if not QUERY_PATH:
    from google.colab import files
    uploaded = files.upload()
    if len(uploaded)!=1: raise ValueError('Upload exactly one Step_2, or set QUERY_PATH explicitly.')
    QUERY_PATH = str(Path(next(iter(uploaded))).resolve())
with Image.open(QUERY_PATH) as opened:
    print('Authoritative native dimensions:',opened.size)
    display(opened.copy())
'''),md('''## 6. Connect the agent and select a vision reasoning model
Use the exact model ID accessible to your API project. The list below is filtered for likely reasoning/text models, but listing alone does not prove image-input or Responses API support. Select your Astra API model ID if present; otherwise select an accessible model you intend to use. Account subscription labels alone do not establish API IDs.

This cell displays a dropdown. Choose the model **before running the next cell**, rather than running the entire notebook unattended.
'''),code('''from google.colab import userdata
from openai import OpenAI
import ipywidgets as widgets
key = userdata.get('OPENAI_API_KEY')
if not key: raise ValueError('Enable OPENAI_API_KEY in Colab Secrets.')
client = OpenAI(api_key=key,timeout=900.0,max_retries=0)
del key
available = sorted(m.id for m in client.models.list())
likely = [m for m in available if ('astra' in m.lower() or m.startswith(('gpt-','o1','o3','o4')))
    and not any(t in m.lower() for t in ('image','audio','realtime','transcri','tts','embedding','codex','instruct','search'))]
if not likely: raise ValueError('No likely vision reasoning model listed. Check your API project/model access.')
model_selector = widgets.Dropdown(options=[('Select your vision reasoning model',None)]+[(m,m) for m in likely],
    value=None,description='Agent model:',layout=widgets.Layout(width='95%'))
display(model_selector)
'''),md('''## 7. Search and compose automatically
At most **1 planning request + MAX_PIECES candidate-selection requests** on the first run (up to 21 by default). Pieces without an exact-size match or with uncertain interpretation do not make selection calls. Artwork and candidate images are sent to the API for these paid reasoning calls; no paid image-generation requests are made.

Only candidates matching the agent-estimated native width AND height are considered. The vision model can reject all candidates. Accepted red pixels are pasted unchanged at integer coordinates; no stretching, rotations, morphological changes or generated filler. Overlap, clipping, intrusion into protected white, and spacing narrower than two white pixels are blocked. Wider meaningful spaces are not collapsed.

The spacing check uses the native 8-neighbour/Chebyshev grid. Continuous stems should be reused as complete components; joins between separate placements are blocked. Native component geometry is preserved, but motif fit and initial size estimates remain uncertain.

Completed agent responses are reused. Started requests without a saved response stop retries; check API usage before explicitly using a new RUN, because an interrupted request may have been billed. The first pilot processes up to 20 proposed pieces, so it is intentionally a partial composition. To attempt more, change MAX_PIECES and use a new RUN folder.
'''),code('''MODEL = model_selector.value
if not MODEL: raise ValueError('Select the intended vision reasoning model in the previous cell.')
report = run_component_agent(QUERY_PATH,LIBRARY,RUN,client,MODEL,
    encoder=encoder,max_pieces=MAX_PIECES,top_k=TOP_K,progress=print)
print('Placed for review:',report['placed'])
print('Agent-proposed pieces:',report['proposed_by_agent'])
print('Unresolved/deferred:',report['unresolved'])
print('Production ready:',report['production_ready'])
'''),md('''## 8. Review exact pixels and unresolved regions
The PNG/BMP preserve the Step_2 dimensions. Green boxes in the separate sketch overlay indicate placed pieces; blue boxes indicate unresolved/deferred pieces. The colours in this overlay are diagnostic only, absent from the output raster. Regions the agent failed to identify may not be boxed at all.
`report.json` lists each source component, original design, match candidates, agent reasoning and blocked/unmatched placements. Unplaced areas remain white; do not treat them as intentional background. The absence of unresolved records is not proof of whole-image completeness or production readiness.
'''),code('''from collections import Counter
print(dict(Counter(r['status'] for r in report['records'])))
display(Image.open(RUN/'step_3_review.png'))
display(Image.open(RUN/'placement_review_overlay.png'))
print('Review folder:',RUN)
'''),md('''## 9. Download the review pack
The central repository remains on Drive for the next sketch. Outputs are saved on Drive even if browser downloads fail. The ZIP contains this run only, including source sketch, candidate images, agent responses, PNG/BMP and provenance; it does not copy the entire central library.
'''),code('''import zipfile
archive_path = RUN/'component_reuse_review.zip'
with zipfile.ZipFile(archive_path,'w',zipfile.ZIP_DEFLATED) as z:
    for path in sorted(RUN.rglob('*')):
        if path.is_file() and path.suffix!='.zip': z.write(path,path.relative_to(RUN))
from google.colab import files
files.download(str(archive_path))
''')]
nb=dict(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
    'language_info':{'name':'python'},'colab':{'name':'Central_Component_Repository.ipynb'}},nbformat=4,nbformat_minor=4)
(root/'notebooks/Central_Component_Repository.ipynb').write_text(json.dumps(nb,indent=1))
print('Built central component repository notebook.')
