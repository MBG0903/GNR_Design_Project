"""Build a standalone Drive-backed reference-retrieval pilot notebook."""
import base64
import io
import json
import zipfile
from pathlib import Path

root = Path(__file__).resolve().parents[1]
buffer = io.BytesIO()
with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
    for name in ('__init__.py', 'core.py', 'data.py', 'genai.py', 'comparison.py', 'retrieval_experiment.py'):
        path = root/'textile'/name
        archive.writestr(str(path.relative_to(root)), path.read_bytes())
payload = base64.b64encode(buffer.getvalue()).decode()


def md(text):
    return dict(cell_type='markdown', metadata={}, source=text.splitlines(True))


def code(text):
    return dict(cell_type='code', metadata={}, source=text.splitlines(True), execution_count=None, outputs=[])


cells = [md('''# Learn from approved examples: first GenAI pilot
Compare **rules only** with **rules + two similar approved sketch/output pairs**, on the same five validation designs.
This retrieves examples; it does not train or fine-tune model weights. CLIP only selects similar sketches locally; the image API creates the design.

- Original dataset: `MyDrive/GNR Design Data-Set/Stage_2` and `Stage_3`.
- Approximately 80/10/10 reference/validation/test **families**, with exact duplicates grouped transitively. Pair counts may differ from those percentages.
- The final test split is reserved; this notebook only generates validation drafts.
- Filename-based family suggestions are provisional. Check `groups.csv`: crops/variants of the same design must share a `design_id`, even if filenames use different numbers. Also verify sketch/target alignment; matching dimensions alone do not prove it.
- Default pilot: at most **10 paid image requests**, one per method per design. Fewer when fewer validation families exist. No automatic API retries.
- Query designer targets are never sent to generation. Only training/reference examples are sent alongside the new sketch.
- Dimensions and palette are enforced; two-pixel gaps and staircase rules are requested, not certified. Final images remain review drafts.

Use your **OPENAI_API_KEY** Colab Secret. A GPU runtime speeds up the one-time local similarity index; CPU also works.
'''), md('''## 1. Mount Drive and choose the dataset
Keep the default folder for the 1,700+ pairs, or change DATA_ROOT. Original BMPs are read, never modified.
Keep the same EXPERIMENT path to resume. Choose a new name when changing groups, query count, model or rules.
'''), code('''from google.colab import drive
from pathlib import Path
drive.mount('/content/drive')
DATA_ROOT = Path('/content/drive/MyDrive/GNR Design Data-Set')
STAGE_2, STAGE_3 = DATA_ROOT/'Stage_2', DATA_ROOT/'Stage_3'
EXPERIMENT = Path('/content/drive/MyDrive/GNR Textile Runs/Reference_Pilot_v1')
GROUPS = DATA_ROOT/'groups.csv'
GROUPS_REVIEWED = False  # True only after checking family IDs AND paired-image alignment.
PILOT_DESIGNS = 5
EXAMPLES_PER_QUERY = 2
IMAGE_MODEL = 'gpt-image-2.5-sunburst'
for folder in (STAGE_2, STAGE_3):
    if not folder.is_dir():
        raise FileNotFoundError(f'Dataset folder missing: {folder}. Correct DATA_ROOT above.')
EXPERIMENT.mkdir(parents=True, exist_ok=True)
print('Inputs:', DATA_ROOT)
print('Persistent outputs:', EXPERIMENT)
'''), md('''## 2. Install the runtime
Only Python code is embedded. No artwork or API credentials are embedded.
The first similarity run downloads pretrained CLIP weights from Hugging Face. It runs locally in Colab; it does not generate textile pixels.
'''), code('''import base64, io, json, sys, subprocess, zipfile
RUNTIME = Path('/content/gnr_reference_pilot_runtime')
RUNTIME.mkdir(exist_ok=True)
PAYLOAD = '''+repr(payload)+'''
with zipfile.ZipFile(io.BytesIO(base64.b64decode(PAYLOAD))) as archive:
    archive.extractall(RUNTIME)
subprocess.check_call([sys.executable, '-m', 'pip', 'install',
    'numpy>=1.26,<3', 'Pillow>=10.4,<13', 'scipy>=1.12,<2',
    'openai>=2,<3', 'sentence-transformers>=3,<6'])
sys.path.insert(0, str(RUNTIME))
from PIL import Image
from IPython.display import display
from textile.data import audit_dataset, contact_sheet
from textile.retrieval_experiment import suggest_groups, embed_sources, plan_pilot, preview_plan, run_pilot
'''), md('''## 3. Pair, audit and split the dataset
Both naming styles are supported: `362 GNR 2.bmp` and `GNR 3001 ( FILE 2 ).bmp`, with matching Stage_3 names.
Ambiguous names, unpaired files, invalid designer colours or mismatched dimensions stop the audit. Inspect `audit/audit.json` to resolve them; the notebook does not silently resize or recolour the designer truth.

`groups.csv` is created only if absent and preserved on reruns. Filename numbers suggest original families; that cannot discover every crop or renamed variant. You can run the initial exploratory pilot with GROUPS_REVIEWED=False. Its scores are **provisional**, not evidence of generalization. Review the CSV and sample pairs before claiming held-out accuracy. After changing groups, use a new EXPERIMENT folder and rerun this audit.
'''), code('''suggest_groups(STAGE_2, GROUPS)
print('Family mapping:', GROUPS)
report = audit_dataset(STAGE_2, STAGE_3, EXPERIMENT/'audit',
    groups_csv=GROUPS, require_groups=True, seed=42, hold_fraction=0.10)
print(json.dumps(report, indent=2))
rows = json.loads((EXPERIMENT/'audit/manifest.json').read_text())
# Final test designs are not indexed or used for retrieval/generation in this pilot.
pilot_rows = [r for r in rows if r['split'] != 'test']
contact_sheet(pilot_rows, EXPERIMENT/'paired_input_preview.png', count=8)
display(Image.open(EXPERIMENT/'paired_input_preview.png'))
if not GROUPS_REVIEWED:
    print('PROVISIONAL: filename groups/alignment still need review; related designs may cross splits.')
'''), md('''## 4. Select similar approved examples
Compute sketch embeddings once and cache them on Drive. Rank only reference-split sketches and choose distinct design families. Designer targets are never used to rank examples.
The preview shows each validation sketch alongside its selected reference pairs; it does not show the query answer to the generation model.
'''), code('''vectors = embed_sources(pilot_rows, EXPERIMENT/'sketch_embeddings.npz')
plan = plan_pilot(pilot_rows, vectors, count=PILOT_DESIGNS,
    references=EXAMPLES_PER_QUERY, groups_reviewed=GROUPS_REVIEWED)
plan_path = EXPERIMENT/'selection_plan.json'
if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
    raise ValueError('Selection changed. Use a new EXPERIMENT folder to preserve this run.')
plan_path.write_text(json.dumps(plan, indent=2))
preview_plan(pilot_rows, plan, EXPERIMENT/'selected_examples.png')
display(Image.open(EXPERIMENT/'selected_examples.png'))
print('Validation queries:', len(plan['jobs']))
print('Maximum image requests:', 2*len(plan['jobs']))
print('Grouping status:', plan['grouping_status'])
'''), md('''## 5. Connect the image API
The API key remains in memory. Listing a model does not prove image-edit permissions or available credit. If this exact model ID is unavailable, use the API model ID confirmed for your account; there is no silent substitution.
'''), code('''from google.colab import userdata
from openai import OpenAI
api_key = userdata.get('OPENAI_API_KEY')
if not api_key:
    raise ValueError('Add OPENAI_API_KEY in Colab Secrets and enable notebook access.')
client = OpenAI(api_key=api_key, timeout=900.0, max_retries=0)
del api_key
available = {m.id for m in client.models.list()}
if IMAGE_MODEL not in available:
    print('Visible image models:', sorted(m for m in available if 'image' in m or 'sunburst' in m))
    raise ValueError('Confirm the exact image API model ID/access before generation.')
print('Selected model:', IMAGE_MODEL)
'''), md('''## 6. Run the paired experiment
Running this cell sends the validation sketches and selected reference pairs to the image provider. Each missing method/design makes one paid request.
Both methods receive the same rules and generation settings. One output per method is retained; no best-of selection. Native width/height and pure red/white are restored after generation, with no target-guided corrections.

Each design folder includes both outputs, BMP exports, raw responses, prompts, native coloured differences and a comparison panel. Blue means missing red; orange means extra red. Targets are used only for scoring.
Completed drafts are reused. An interrupted request is marked before submission and blocks automatic repetition: inspect provider usage and the incomplete method folder before deleting just that folder to explicitly retry. The original request may already have been billed.
'''), code('''summary = run_pilot(pilot_rows, plan, EXPERIMENT/'results', client,
    model=IMAGE_MODEL, progress=print)
print(json.dumps(summary, indent=2))
'''), md('''## 7. Review percentages and designs
Scores are exact native-coordinate pixel agreement, red IoU and red Dice, with no target registration or resizing. Averages give every completed design equal weight. Method differences use only designs completed by BOTH methods.
The report counts designs reaching 70% and 80% red IoU as an explicit, stricter overlap measure alongside pixel agreement. This is not a business-rule pass rate.
Inspect semantic motif fidelity, two-pixel separation, front/back anchors, staircase step counts, deliberate white areas and repeated pieces. Record unresolved issues; do not infer production readiness from a similarity percentage.
One five-design run is an initial signal. Evaluate more validation families/repeated runs before deciding whether reference prompting helps, then use the untouched test split only after fixing the method.
'''), code('''import csv
from IPython.display import HTML
from html import escape
with (EXPERIMENT/'results/scores.csv').open(newline='') as handle:
    scores = list(csv.DictReader(handle))
headers = ['Design', 'Method', 'Pixel agreement', 'Red IoU', 'Red Dice']
body = ''.join('<tr><td>'+ '</td><td>'.join([
    escape(r['query']), escape(r['method']),
    *[f"{float(r[k]):.2f}%" for k in ('pixel_agreement_pct','red_iou_pct','red_dice_pct')]])+'</td></tr>' for r in scores)
display(HTML('<table><tr><th>'+'</th><th>'.join(headers)+'</th></tr>'+body+'</table>'))
for folder in sorted((EXPERIMENT/'results').glob('design_*')):
    print(folder.name)
    if (folder/'comparison.png').exists(): display(Image.open(folder/'comparison.png'))
review_path = EXPERIMENT/'designer_review.csv'
if not review_path.exists():
    with review_path.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['query','method','motifs_preserved','two_pixel_separation',
            'front_back_anchors','staircase_counts','internal_white_preserved','family_consistency',
            'designer_minutes_to_fix','notes'])
        for r in scores: writer.writerow([r['query'],r['method']]+['unreviewed']*6+['',''])
print('Optional designer scorecard:', review_path)
print('All images and reports remain on Drive:', EXPERIMENT)
'''), md('''## 8. Download or share the review ZIP
The ZIP contains generation outputs and comparison reports. The selection preview, family mapping and similarity cache remain in the experiment/dataset folders on Drive. Downloads are optional; Drive preserves files if the browser blocks them.
No generated result is published to GitHub automatically. Share the scores and comparison panels for review of whether to expand the experiment or investigate paired-image training support.
'''), code('''from google.colab import files
files.download(str(EXPERIMENT/'results/pilot_review.zip'))
''')]
notebook = dict(cells=cells, metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                'language_info': {'name': 'python'}, 'colab': {'name': 'Reference_Guided_Pilot.ipynb'}},
                nbformat=4, nbformat_minor=4)
(root/'notebooks/Reference_Guided_Pilot.ipynb').write_text(json.dumps(notebook, indent=1))
print('Built Reference_Guided_Pilot.ipynb')
