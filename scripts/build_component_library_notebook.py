"""Standalone notebook for the user's already-separated designer component library."""
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
def md(s):return dict(cell_type='markdown',metadata={},source=s.splitlines(True))
def code(s):return dict(cell_type='code',metadata={},source=s.splitlines(True),outputs=[],execution_count=None)
cells=[md('''# Reuse your designer's existing component repository
Use the components already separated from 1,800+ designer images. **Each image file is one asset**, including all its disconnected red parts. No re-segmentation, scaling, rotation, staircase repair or generated filler is applied to library assets.

Your supplied Drive folder has not been inspected from this chat. This notebook reads the folder through your mounted Drive; public sharing is not required. If it is in Shared with me, add a shortcut in My Drive and set COMPONENTS_DIR to that shortcut's actual path.

**Recommended mode: preserve_photoshop.** Your Photoshop draft supplies the layout and fallback. Native raster regions provide measured dimensions; a vision agent selects suitable exact-size designer assets. Replacements preserve the asset's pixels and neighbours. Unmatched/rejected pieces keep their original Photoshop pixels, so they do not become blank.

**Optional mode: sketch_only.** The agent estimates regions directly from Step_2 and assembles a blank canvas using exact-size assets. Estimates/matches may fail, leaving a partial composition. It is an exploratory path, not a complete production promise.

Existing SQLite/JSON-only repositories need a schema adapter. CSV/JSON metadata alongside images is preserved but not interpreted automatically. The source audit reports invalid colours/partial alpha without changing originals. Source components preserve their geometry, but source-rule compliance, semantic suitability, repeated-family consistency and global layout still need review.

Similarity indexing uses local pretrained CLIP (GPU recommended); matching uses your OPENAI_API_KEY and an accessible vision reasoning model. No Sunburst image-generation call or weight training is required.
'''),md('''## 1. Mount Drive and select the existing repository
Edit COMPONENTS_DIR to the actual folder/shortcut name visible in My Drive. The URL's folder ID does not establish its mounted filesystem name. The source and search-index folders must be separate.
'''),code('''from google.colab import drive
from pathlib import Path
drive.mount('/content/drive')
COMPONENTS_DIR=Path('/content/drive/MyDrive/GNR Component Repository')  # EDIT: your folder/shortcut name
INPUT_MODE='component_files'  # Existing separated files. Alternative: 'full_designs'.
STAGE_3=Path('/content/drive/MyDrive/GNR Design Data-Set/Stage_3')
LIBRARY=Path('/content/drive/MyDrive/GNR Component Search Index/separated_v1')
RUN=Path('/content/drive/MyDrive/GNR Textile Runs/Designer_Component_Reuse/trial_01')
COMPOSITION_MODE='preserve_photoshop'  # Recommended. Or 'sketch_only'.
MAX_PIECES=20  # Initial pilot: 20 largest Photoshop regions, or first 20 sketch proposals.
TOP_K=3
TRIM_WHITE_MARGIN=True  # Remove empty outer margins only; native motif pixels are not resized.
RESERVE_WHITE_FOOTPRINT=True  # Protect all non-red pixels within the asset's native footprint.
print('Source:',COMPONENTS_DIR)
print('Persistent search index:',LIBRARY)
print('Persistent review run:',RUN)
'''),md('''## 2. Load the runtime
Only Python code is embedded. No artwork or API keys are embedded. CLIP weights download from Hugging Face on first use and run locally in Colab.
'''),code('''import base64,io,json,sys,subprocess,zipfile
RUNTIME=Path('/content/gnr_component_runtime');RUNTIME.mkdir(exist_ok=True)
PAYLOAD='''+repr(payload)+'''
with zipfile.ZipFile(io.BytesIO(base64.b64decode(PAYLOAD))) as z:z.extractall(RUNTIME)
subprocess.check_call([sys.executable,'-m','pip','install',
    'numpy>=1.26,<3','Pillow>=10.4,<13','scipy>=1.12,<2',
    'openai>=2,<3','sentence-transformers>=3,<6','ipywidgets>=8,<9'])
sys.path.insert(0,str(RUNTIME))
for name in list(sys.modules):
    if name=='textile' or name.startswith('textile.'):del sys.modules[name]
from PIL import Image
from IPython.display import display
from textile.component_library import index_component_files,index_designer_images,embed_library,load_piece,connect,EMBEDDER
from textile.component_agent import run_component_agent,run_draft_reuse
'''),md('''## 3. Index existing component files without splitting them
A file may contain a complete leaf, a flower, a branch or multiple disconnected parts belonging together. This mode keeps them together. It supports strict red/white visible pixels and binary transparency; partially transparent or other-colour edges are excluded, not thresholded.

TRIM_WHITE_MARGIN removes only empty exterior margins and records the offset. False preserves the full file canvas. RESERVE_WHITE_FOOTPRINT protects white/internal gaps throughout the tight asset footprint; this conservative policy can reject crowded interleaving. It never fills those white pixels to make an asset fit.

Reruns reuse unchanged sources. Source changes/removals or policy changes require a new LIBRARY version. Each source path/hash and native dimensions are recorded. Inspect index_summary.json and source_audit.csv before the paid matching stage.
'''),code('''if INPUT_MODE=='component_files':
    if not COMPONENTS_DIR.is_dir():
        visible=[p.name for p in Path('/content/drive/MyDrive').iterdir() if p.is_dir()]
        print('Visible My Drive folders:',visible[:60])
        raise FileNotFoundError('Set COMPONENTS_DIR to the actual component folder/shortcut path. No source files were modified.')
    index_report=index_component_files(COMPONENTS_DIR,LIBRARY,
        trim_white_margin=TRIM_WHITE_MARGIN,reserve_white_footprint=RESERVE_WHITE_FOOTPRINT,progress=print)
elif INPUT_MODE=='full_designs':
    index_report=index_designer_images(STAGE_3,LIBRARY,progress=print)
else:raise ValueError('Choose component_files or full_designs.')
print(json.dumps(index_report,indent=2))
print('Full source audit:',LIBRARY/'source_audit.csv')
if not index_report['components']:raise ValueError('No valid components. Inspect source_audit.csv.')
db=connect(LIBRARY)
preview_ids=[r[0] for r in db.execute('SELECT id FROM pieces ORDER BY id LIMIT 6')];db.close()
for key in preview_ids:
    im,_,meta=load_piece(LIBRARY,key)
    print('Native size:',im.size,'— source:',meta['source_path'])
    preview=im.copy();preview.thumbnail((180,180));display(preview)  # Preview only; asset is unchanged.
'''),md('''## 4. Build the searchable index once
Exact width/height are mandatory search filters. CLIP ranks matching-size shapes. Identical raster shapes share embeddings, while all source records remain. Batches checkpoint on Drive; resume with the same library settings.
'''),code('''from sentence_transformers import SentenceTransformer
encoder=SentenceTransformer(EMBEDDER)
print(embed_library(LIBRARY,encoder=encoder,progress=print))
'''),md('''## 5. Load the new sketch and optional Photoshop draft
Set file paths or upload one file at each prompt. The recommended preserve_photoshop mode needs a Step_3 draft at the exact Step_2 dimensions and pure red/white. You can use the `1 photoshop pure.png` saved by the finisher's Cell 3. The Step_2 sketch is the semantic reference; the draft is the layout/fallback. A designer target is not needed for generation.

NORMALIZE_DRAFT_PALETTE is an explicit option for a nonbinary draft. It saves a projected copy only, never modifies the original; review that copy because projection can move edges. Components themselves are never normalized/resized.
'''),code('''QUERY_PATH=''
PHOTOSHOP_DRAFT_PATH=''
NORMALIZE_DRAFT_PALETTE=False
from google.colab import files
if not QUERY_PATH:
    print('Upload ONE new Step_2 sketch:')
    uploaded=files.upload()
    if len(uploaded)!=1:raise ValueError('Select exactly one Step_2.')
    QUERY_PATH=str(Path(next(iter(uploaded))).resolve())
RUN.mkdir(parents=True,exist_ok=True)
def preserve_input(path,role):
    source=Path(path);data=source.read_bytes()
    folder=RUN/'inputs';folder.mkdir(exist_ok=True)
    destination=folder/(role+source.suffix.lower())
    if destination.exists() and destination.read_bytes()!=data:
        raise ValueError('Input changed in this RUN. Choose a new run folder to preserve earlier results.')
    if not destination.exists():destination.write_bytes(data)
    return str(destination)
QUERY_PATH=preserve_input(QUERY_PATH,'step_2')  # Protect it from a later upload with the same filename.
with Image.open(QUERY_PATH) as im:print('Native Step_2 size:',im.size);display(im.copy())
if COMPOSITION_MODE=='preserve_photoshop':
    if not PHOTOSHOP_DRAFT_PATH:
        print('Upload ONE Photoshop/GenAI Step_3 draft:')
        uploaded=files.upload()
        if len(uploaded)!=1:raise ValueError('Select exactly one AI draft.')
        PHOTOSHOP_DRAFT_PATH=str(Path(next(iter(uploaded))).resolve())
    PHOTOSHOP_DRAFT_PATH=preserve_input(PHOTOSHOP_DRAFT_PATH,'photoshop_original')
    if NORMALIZE_DRAFT_PALETTE:
        from textile.core import exact_palette
        RUN.mkdir(parents=True,exist_ok=True)
        with Image.open(PHOTOSHOP_DRAFT_PATH) as im:normalized=exact_palette(im)
        normalized_path=RUN/'photoshop_palette_copy.png'
        if normalized_path.exists():
            with Image.open(normalized_path) as saved:
                if saved.convert('RGB').tobytes()!=normalized.tobytes() or saved.size!=normalized.size:
                    raise ValueError('Normalized draft changed. Use a new RUN folder.')
        else:normalized.save(normalized_path)
        PHOTOSHOP_DRAFT_PATH=str(normalized_path)
    from textile.components import binary
    draft,_=binary(PHOTOSHOP_DRAFT_PATH)
    with Image.open(QUERY_PATH) as im:
        if draft.size!=im.size:raise ValueError('Draft and sketch must have identical native dimensions.')
    display(draft)
elif COMPOSITION_MODE!='sketch_only':raise ValueError('Choose preserve_photoshop or sketch_only.')
'''),md('''## 6. Connect the agent and select a vision reasoning model
Enable OPENAI_API_KEY in Colab Secrets. Select your accessible vision model, including the exact Astra API ID if it is listed. Model listing alone does not prove image-input/Responses API access. Pause at this dropdown before continuing; do not Run all unattended through model selection.
'''),code('''from google.colab import userdata
from openai import OpenAI
import ipywidgets as widgets
key=userdata.get('OPENAI_API_KEY')
if not key:raise ValueError('Enable OPENAI_API_KEY in Colab Secrets.')
client=OpenAI(api_key=key,timeout=900.0,max_retries=0);del key
available=sorted(m.id for m in client.models.list())
likely=[m for m in available if ('astra' in m.lower() or m.startswith(('gpt-','o1','o3','o4')))
    and not any(t in m.lower() for t in ('image','audio','realtime','transcri','tts','embedding','codex','instruct','search'))]
if not likely:raise ValueError('Confirm vision reasoning API access in your account.')
model_selector=widgets.Dropdown(options=[('Select intended vision reasoning model',None)]+[(m,m) for m in likely],
    value=None,description='Agent model:',layout=widgets.Layout(width='95%'))
display(model_selector)
'''),md('''## 7. Retrieve native pieces and compose for review
In preserve_photoshop mode, measured raster regions supply exact bounds, avoiding the vision model's one-pixel coordinate estimates. Connected regions are candidate units, not semantic truth: a merged source region may need manual review. The agent compares the sketch, Photoshop piece and exact-size library candidates. It can reject every match.

Only the selected source region is removed, and only after its replacement passes overlap/protected-white/minimum-two-white-pixel-clearance checks. Other pixels remain. Missing sizes, unsuitable assets, blocked placements and regions beyond the run limit retain Photoshop pixels. No resizing, rotation, hole filling or global density expansion is performed.

Default pilot: at most 20 paid choice calls, fewer when sizes are absent. Sketch-only mode uses one planning call plus up to 20 choice calls and can leave missing areas white. Artwork/candidates are sent to the API for reasoning. Responses are cached; interrupted started requests block automatic repeat charges. Use a new RUN for different inputs, model, library or limits.
'''),code('''MODEL=model_selector.value
if not MODEL:raise ValueError('Select the intended vision model first.')
if COMPOSITION_MODE=='preserve_photoshop':
    report=run_draft_reuse(QUERY_PATH,PHOTOSHOP_DRAFT_PATH,LIBRARY,RUN,client,MODEL,
        encoder=encoder,max_pieces=MAX_PIECES,top_k=TOP_K,progress=print)
    print('Replaced:',report['replaced'],'— retained Photoshop regions:',report['retained'])
    print('Changed native pixels:',report['changed_pixels_vs_photoshop'])
else:
    report=run_component_agent(QUERY_PATH,LIBRARY,RUN,client,MODEL,
        encoder=encoder,max_pieces=MAX_PIECES,top_k=TOP_K,progress=print)
    print('Placed:',report['placed'],'— unresolved/deferred:',report['unresolved'])
print('Production ready:',report['production_ready'])
'''),md('''## 8. Inspect before/after and provenance
The output stays at the exact Step_2 size. Every replacement references its original component file and hash. Review semantic match, layout, spacing and repeated-family consistency. Native source staircases are preserved, including any source defects; no rule-compliance percentage is inferred from copying.

Green overlay boxes indicate reused pieces; blue indicates retained/unresolved regions. The overlay is diagnostic only. Full Step_3 contains only red/white. If there are no accepted replacements in preserve_photoshop mode, it remains identical to the draft.
'''),code('''from collections import Counter
print(dict(Counter(r['status'] for r in report['records'])))
if (RUN/'photoshop_baseline.png').exists():
    print('Photoshop baseline:');display(Image.open(RUN/'photoshop_baseline.png'))
print('Repository reuse review:');display(Image.open(RUN/'step_3_review.png'))
display(Image.open(RUN/'placement_review_overlay.png'))
print('Full records:',RUN/'report.json')
'''),md('''## 9. Download the review pack
This ZIP contains this run's outputs/proposals/provenance, not the entire component library. All files remain on Drive if browser downloads fail. Source components are never uploaded to GitHub automatically.
'''),code('''archive_path=RUN/'component_reuse_review.zip'
with zipfile.ZipFile(archive_path,'w',zipfile.ZIP_DEFLATED) as z:
    for path in sorted(RUN.rglob('*')):
        if path.is_file() and path.suffix!='.zip':z.write(path,path.relative_to(RUN))
files.download(str(archive_path))
''')]
nb=dict(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
    'language_info':{'name':'python'},'colab':{'name':'Designer_Component_Reuse.ipynb'}},nbformat=4,nbformat_minor=4)
for name in ('Designer_Component_Reuse.ipynb','Central_Component_Repository.ipynb'):
    (root/'notebooks'/name).write_text(json.dumps(nb,indent=1))
print('Built existing-component reuse notebook and legacy link alias.')
