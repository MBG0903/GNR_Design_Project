"""Build a standalone Colab catalogue workflow; embed code, never artwork/keys."""
import base64
import io
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]
buffer=io.BytesIO()
with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as archive:
    for name in ('__init__.py','core.py','data.py','genai.py','components.py',
                 'component_library.py','component_agent.py','semantic_objects.py','object_review.py'):
        archive.writestr('textile/'+name,(ROOT/'textile'/name).read_bytes())
payload=base64.b64encode(buffer.getvalue()).decode()

def md(text):
    return dict(cell_type='markdown',metadata={},source=text.splitlines(True))

def code(text):
    return dict(cell_type='code',metadata={},source=text.splitlines(True),outputs=[],execution_count=None)

cells=[md('''# Build the central Stage_3 object repository
Catalogue complete **peacocks, elephants, flowers, leaves and other semantic objects** from your designer Stage_3 folder. A whole animal can include many disconnected red pieces. Nested feathers, flowers and leaves can be separate entries linked to a parent object.

Vision reasoning proposes labels and bounding boxes. Local pretrained **Segment Anything (SAM)** proposes masks. Exports copy selected **original native red pixels** and preserve white details: no image generation, resizing of assets, smoothing, filling, staircase repair or learned pixel correction.

These pretrained models can miss small motifs or isolate the wrong pieces in textile drawings. Initial candidates are pending review. The offline pixel editor lets the designer include/exclude source pixels and approve complete object isolation. Approval here does not certify business rules. Coverage maps show unassigned red; coverage is not semantic recall.

This chat has not processed the full Drive dataset. Run this notebook where Drive and your API key are available. GPU is recommended for SAM. The default pilot processes at most **5 new images and 20 new paid discovery requests** per run; identical cached requests are reused. Re-run discovery to advance through the dataset. Optional detail scanning adds overlapping native crop views for smaller parts, with more API requests. All sources remain unchanged and the catalogue checkpoints on Drive.
'''),md('''## 1. Mount Drive and choose separate source/output folders
STAGE_3 defaults to the folder already used in this project. Correct its actual mounted name if needed. Shared folders need a My Drive shortcut. Keep LIBRARY outside STAGE_3 and outside the older component index. Use a new version folder for changed source images, reasoning model or scan settings.
'''),code('''from google.colab import drive
from pathlib import Path
drive.mount('/content/drive')
STAGE_3=Path('/content/drive/MyDrive/GNR Design Data-Set/Stage_3')
LIBRARY=Path('/content/drive/MyDrive/GNR Object Repository/semantic_v1')
MAX_NEW_IMAGES=5
MAX_NEW_API_CALLS=20
DETAIL_SCAN=False  # Set before first discovery. Adds overlapping views for tiny leaves/petals.
WINDOW_SIZE=512
OVERLAP=128
if not STAGE_3.is_dir():
    raise FileNotFoundError(f'Stage_3 folder missing: {STAGE_3}. Set its actual mounted Drive path.')
print('Designer sources:',STAGE_3)
print('Object repository:',LIBRARY)
'''),md('''## 2. Install and load the embedded runtime
No image or API key is embedded in this notebook. SAM weights download from Hugging Face on first use. Model preprocessing resizes internally; masks are returned to native coordinates and output pixels are copied from the original images.
'''),code('''import base64,io,json,sys,subprocess,zipfile
RUNTIME=Path('/content/gnr_semantic_object_runtime');RUNTIME.mkdir(exist_ok=True)
PAYLOAD='''+repr(payload)+'''
with zipfile.ZipFile(io.BytesIO(base64.b64decode(PAYLOAD))) as z:z.extractall(RUNTIME)
subprocess.check_call([sys.executable,'-m','pip','install',
    'numpy>=1.26,<3','Pillow>=10.4,<13','scipy>=1.12,<2',
    'openai>=2,<3','transformers==4.57.1','ipywidgets>=8,<9'])
sys.path.insert(0,str(RUNTIME))
for name in list(sys.modules):
    if name=='textile' or name.startswith('textile.'):del sys.modules[name]
from PIL import Image
from IPython.display import display
from textile.semantic_objects import (SamSegmenter,build_object_catalog,catalog_summary,
    list_objects,import_reviews,export_manifest,export_current_approved,add_manual_candidate)
from textile.object_review import write_review_gallery
'''),md('''## 3. Select your available vision reasoning model
Use your existing OPENAI_API_KEY Colab Secret. Choose the intended reasoning model available through your API account. This step lists models; it does not generate images or start extraction. Source artwork will be sent to the selected API for semantic discovery when you run Cell 5. SAM segmentation runs locally in Colab.
'''),code('''from google.colab import userdata,files
from openai import OpenAI
import ipywidgets as widgets
api_key=userdata.get('OPENAI_API_KEY')
if not api_key:raise ValueError('OPENAI_API_KEY is missing from Colab Secrets.')
client=OpenAI(api_key=api_key)
available=sorted(record.id for record in client.models.list())
choices=[m for m in available if ('astra' in m.lower() or m.startswith(('gpt-','o1','o3','o4')))
    and not any(t in m.lower() for t in ('image','audio','realtime','transcri','tts','embedding','codex','instruct','search'))]
if not choices:raise ValueError('Confirm an API model supporting vision and JSON responses is available.')
model_selector=widgets.Dropdown(options=[('Select intended vision model',None)]+[(m,m) for m in choices],
    value=None,description='Vision model:',layout=widgets.Layout(width='95%'))
display(model_selector)
'''),md('''## 4. Load the local pretrained segmentation model
GPU: Runtime → Change runtime type → GPU. CPU works but can be slow. SAM's mask score is not a semantic correctness or textile-rule score. It can omit disconnected feathers or include a neighbouring motif; review remains necessary.
'''),code('''import torch
segmenter=SamSegmenter('facebook/sam-vit-base')
print('SAM device:',segmenter.device)
print('Weights:',segmenter.signature)
'''),md('''## 5. Discover objects and save candidates to Drive
Start with the bounded pilot, inspect the masks, then increase MAX_NEW_IMAGES and the explicit request budget if useful. Re-running this cell skips completed unchanged images and advances through Stage_3. In-progress views resume from cached JSON; a started request without a response blocks automatic retry to avoid accidental repeat charges. Inspect usage and choose a new library version before explicitly retrying such a request.

Full-view detection is always used. DETAIL_SCAN adds overlapping native views for small leaves/petals; objects cut by a view should be flagged instead of exported as complete. The complete source is supplied as context. No exhaustive object-recall guarantee is made.

Invalid source colours, transparency or multiple-frame images are excluded and recorded. They are never silently converted. A library is an immutable source snapshot: keep sources present and unchanged.
'''),code('''MODEL=model_selector.value
if not MODEL:raise ValueError('Select the intended vision model in Cell 3 first.')
summary=build_object_catalog(STAGE_3,LIBRARY,client,MODEL,segmenter,
    max_new_images=MAX_NEW_IMAGES,max_new_calls=MAX_NEW_API_CALLS,
    detail_scan=DETAIL_SCAN,window=WINDOW_SIZE,overlap=OVERLAP,progress=print)
export_manifest(LIBRARY)
print(json.dumps(summary,indent=2))
print('Objects CSV:',LIBRARY/'objects.csv')
print('Per-image omission maps:',LIBRARY/'source_audits')
'''),md('''## 6. Review complete objects with the offline pixel panel
This gallery needs no public server and sends no artwork elsewhere. Download the HTML, open it in a browser and inspect each source context. Left: original crop. Right: isolated object. Zoom 4×/8×/12× and paint **Include red / Exclude red** with a 1-native-pixel brush. Only existing source red can be selected; original white cannot be filled. Complete disconnected parts can be selected together.

Choose the category and Approve complete object isolation, or Reject. Enter the designer name and click Download reviewed masks JSON. Upload that JSON in Cell 7. Objects left Not reviewed remain pending. If parts extend outside the proposed box, reject it and use the optional corrected-box helper below. Isolation approval is separate from staircase/production approval.
'''),code('''GALLERY_LIMIT=30
gallery=write_review_gallery(LIBRARY,limit=GALLERY_LIMIT)
print('Saved offline editor:',gallery)
files.download(str(gallery))
'''),md('''## 7. Import the designer decisions
Upload the review JSON saved by the panel. Source hashes and candidate identities must match. Masks are native-size selections of source red; changed sources, out-of-bounds runs, invented red and empty approvals are rejected. Approved assets are tightly cropped by removing white exterior margins only; the exact source offset is retained. Identical approved rasters share one asset file while every source instance remains recorded.
'''),code('''uploaded=files.upload()
reviews=[name for name in uploaded if name.lower().endswith('.json')]
if len(reviews)!=1:raise ValueError('Upload exactly one object-review JSON.')
review_path=LIBRARY/'review_imports'/Path(reviews[0]).name
review_path.parent.mkdir(exist_ok=True)
import hashlib
payload=uploaded[reviews[0]]
review_path=review_path.with_name(review_path.stem+'_'+hashlib.sha256(payload).hexdigest()[:12]+'.json')
review_path.write_bytes(payload)
print(json.dumps(import_reviews(LIBRARY,review_path),indent=2))
'''),md('''## 8. Search by object category and exact native size
Search returns current approved source instances and provenance. Change CATEGORY to elephant, flower, leaf, paisley, etc. Set width/height when an exact native-size match is needed. No resizing/rotation fallback is applied. Width/height refer to the approved tight footprint.
'''),code('''CATEGORY='peacock'
WIDTH=None
HEIGHT=None
hits=list_objects(LIBRARY,category=CATEGORY,status='approved',width=WIDTH,height=HEIGHT)
print('Source instances:',len(hits),'— unique pixel assets:',len({h['asset_id'] for h in hits}))
for hit in hits[:12]:
    print(hit['label'],hit['approved_size'],'—',Path(hit['source_path']).name,hit['export_bbox'])
    display(Image.open(LIBRARY/hit['approved_png']))
'''),md('''## 9. Export a clean snapshot for the existing component search notebook
The raw approved folder includes historical asset versions and white-mask PNGs. **Do not index that folder directly.** This export contains only current approved object PNGs, deduplicated by category/asset, plus every source-instance record. It can then be indexed with INPUT_MODE='component_files' in Designer_Component_Reuse.ipynb. Use a new EXPORT folder for each snapshot.

The existing draft replacement agent targets connected draft regions. It does not yet automatically match or place a complete multi-region peacock/elephant. This catalogue supplies those whole-object assets and labels; whole-object composition is a separate next step.
'''),code('''EXPORT=Path('/content/drive/MyDrive/GNR Object Repository/exports/approved_v1')
print(export_current_approved(LIBRARY,EXPORT))
print('Set COMPONENTS_DIR in Designer_Component_Reuse.ipynb to:',EXPORT)
'''),md('''## Optional: re-propose an object using a corrected box
Run only after setting SOURCE, CATEGORY and BBOX to an already-catalogued source and its original integer [x,y,width,height]. This makes a new mask proposal, not an approved object. Then regenerate the review gallery. It uses local SAM without another paid discovery call.
'''),code('''ADD_CORRECTED_BOX=False
if ADD_CORRECTED_BOX:
    SOURCE=STAGE_3/'CHANGE_TO_SOURCE.bmp'
    CATEGORY='peacock'
    BBOX=[0,0,100,200]  # EDIT using the original source pixel grid.
    key=add_manual_candidate(LIBRARY,SOURCE,CATEGORY,BBOX,segmenter)
    print('New pending object:',key)
'''),md('''## Output layout and limits
`objects.sqlite3`, `objects.csv` and `objects.json` hold the catalogue. `candidates/<category>/<id>/` stores original crop, proposed object, selection mask and metadata. `approved/<category>/` holds reviewed native PNG/BMP assets and white protection masks. `requests/` caches API responses. `source_audits/` records failed/ambiguous proposals and unassigned red maps. Gallery/review JSON records can later support supervised learning, but this notebook does not train a model.

Hierarchy and categories are proposals until reviewed; all-small-object detection remains unmeasured. Native copies retain source geometry, including any designer-source defects. No automatic production certification or designer-match percentage is inferred. Source artwork stays on Drive unless you explicitly export it elsewhere; this notebook never pushes source assets to GitHub.
''')]
notebook=dict(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
    'language_info':{'name':'python'},'colab':{'name':'Stage3_Semantic_Object_Repository.ipynb'}},nbformat=4,nbformat_minor=4)
(ROOT/'notebooks/Stage3_Semantic_Object_Repository.ipynb').write_text(json.dumps(notebook,indent=1))
print('Built Stage3_Semantic_Object_Repository.ipynb without artwork, credentials or saved outputs.')
