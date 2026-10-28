"""Standalone four-design batch using the user's Colab OPENAI_API_KEY secret."""
import base64
import io
import json
import zipfile
import hashlib
from pathlib import Path
root=Path(__file__).resolve().parents[1]
buffer=io.BytesIO()
with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as archive:
    for name in ['__init__.py','core.py','genai.py','comparison.py','design_batch.py','drive_inputs.py']:
        path=root/'textile'/name;archive.writestr(str(path.relative_to(root)),path.read_bytes())
payload=base64.b64encode(buffer.getvalue()).decode()
manifest=json.loads((root/'examples/sunburst_four_designs/manifest.json').read_text())
raw_base='https://raw.githubusercontent.com/MBG0903/GNR_Design_Project/7e938b6/'
assets=[]
for record in manifest:
    key=record['design_id'];number=key.removeprefix('GNR_')
    for name,role,hash_key in [('step_2.bmp','2','source_sha256'),('designer_step_3.bmp','3','designer_sha256')]:
        assets.append(dict(path=f'{key}/{name}',sha256=record[hash_key],upload_name=f'GNR {number} ( FILE {role} ).bmp',url=raw_base+f'examples/sunburst_four_designs/{key}/{name}'))
for role,source in [('step_2.bmp','examples/step_2.bmp'),('designer_step_3.bmp','examples/step_3_reference.bmp')]:
    assets.append(dict(path=f'_references/design_362/{role}',sha256=hashlib.sha256((root/source).read_bytes()).hexdigest(),url=raw_base+source))
def md(text):return dict(cell_type='markdown',metadata={},source=text.splitlines(True))
def code(text):return dict(cell_type='code',metadata={},source=text.splitlines(True),execution_count=None,outputs=[])
cells=[md('''# Sunburst: GNR 3001–3004 → four folders → designer comparisons
This notebook mounts Drive first, saves your four sketches and matching designer targets under **MyDrive / GNR Data-Set / Sunburst_Four_Designs**, then reads the inputs from Drive.
It copies matching Colab-uploaded BMPs when available and downloads missing originals from the pinned project version. No image bytes are embedded in notebook cells.
It uses your existing **OPENAI_API_KEY** Colab Secret.
It generates one full Step_3 per Step_2 (four paid image edits), preserving native size and pure red/white.
The four query designer targets are used **only after generation for comparison**, never sent as generation references.
An independent design-362 sketch/approved-output pair illustrates dense fill treatment.

Prompts request dense solid fill, two native white pixels between distinct pieces, intentional internal white,
front/back anchor ordering and native 3→2→1 (n3<n2<n1) / 2→1 (n2<n1) staircases.
Only palette/dimensions are enforced by code. Semantic gaps/hierarchy, motif fidelity and production readiness require review.
No rejected strip fitting, global morphology, tile stitching or target copying is applied.
No live Sunburst output or accuracy score is claimed until you run this notebook successfully.
CPU runtime is sufficient. Add OPENAI_API_KEY in the key-shaped Secrets panel and enable notebook access.
'''),md('''## 1. Mount Google Drive
The four sketches are stored in GNR_3001–GNR_3004 folders, alongside separate designer targets.
Edit DATA_ROOT if your intended Drive folder has a different name. Existing inputs are verified and reused, never silently replaced.
'''),code('''from google.colab import drive
import pathlib
drive.mount('/content/drive')
DATA_ROOT=pathlib.Path('/content/drive/MyDrive/GNR Data-Set')
INPUTS=DATA_ROOT/'Sunburst_Four_Designs'
print('Drive dataset folder:',INPUTS)
'''),md('''## 2. Load the runtime code
Only Python runtime code is embedded in this setup cell. Images are stored and read on mounted Drive.
'''),code('''import base64,io,json,pathlib,sys,subprocess,zipfile
runtime=pathlib.Path('/content/gnr_four_design_runtime')
runtime.mkdir(parents=True,exist_ok=True)
payload='''+repr(payload)+'''
with zipfile.ZipFile(io.BytesIO(base64.b64decode(payload))) as archive:archive.extractall(runtime)
subprocess.check_call([sys.executable,'-m','pip','install','numpy>=1.26,<3','Pillow>=10.4,<13','openai>=2,<3','scipy>=1.12,<2'])
sys.path.insert(0,str(runtime))
from PIL import Image
from IPython.display import display
'''),md('''## 3. Copy the four images to Drive and read them back
Already-uploaded files named `GNR 3001 ( FILE 2 ).bmp` through `GNR 3004 ( FILE 2 ).bmp` in `/content` are copied first.
Matching FILE 3 designer targets are handled separately for comparison. Any missing originals and the independent reference pair are downloaded from the pinned GitHub version.
Hashes verify the original bytes before saving. Reruns reuse matching Drive files without downloading again.
All generation/comparison inputs below are read from Drive, including the independent reference pair.
'''),code('''from textile.drive_inputs import stage_files
manifest='''+repr(manifest)+'''
assets='''+repr(assets)+'''
manifest_path=INPUTS/'manifest.json'
if manifest_path.exists() and json.loads(manifest_path.read_text())!=manifest:
    raise ValueError('This dataset manifest differs. Choose a new DATA_ROOT to preserve existing files.')
staged=stage_files(INPUTS,assets,uploads=pathlib.Path('/content'))
manifest_path.write_text(json.dumps(manifest,indent=2))
(INPUTS/'staging_report.json').write_text(json.dumps(staged,indent=2))
SHOW_INPUT_PREVIEWS=False
for record in manifest:
    source_path=INPUTS/record['design_id']/'step_2.bmp'
    with Image.open(source_path) as image:
        assert image.size==tuple(record['source_size'])
        print(record['design_id'],image.size,'— reading from',source_path)
        if SHOW_INPUT_PREVIEWS:display(image.copy())
print('Ready: four Step_2 sketches, four designer targets and one independent reference pair on Drive.')
'''),md('''## Connect using your existing Colab Secret
This uses the same authentication method you supplied. The key stays in memory and is never written into the review files.
`models.list()` establishes model-list access; the selected image model and edit endpoint need their own access.
If the exact Sunburst model ID is unavailable, the notebook stops rather than silently choosing another model.
Do not paste an API key into chat or commit it to GitHub.
'''),code('''from google.colab import userdata
from openai import OpenAI
api_key=userdata.get('OPENAI_API_KEY')
if not api_key:raise ValueError('OPENAI_API_KEY is missing. Add it in Colab Secrets and enable notebook access.')
client=OpenAI(api_key=api_key,timeout=900.0,max_retries=0)
del api_key
IMAGE_MODEL='gpt-image-2.5-sunburst'
model_records=list(client.models.list())
print('OpenAI API connection successful. Model records:',len(model_records))
if IMAGE_MODEL not in {m.id for m in model_records}:
    visible_images=[m.id for m in model_records if 'image' in m.id.lower() or 'sunburst' in m.id.lower()]
    print('Visible image model IDs:',visible_images)
    raise ValueError('The selected Sunburst model ID is not listed for this API project. Confirm the exact ID/access before paid generation.')
print('Selected model:',IMAGE_MODEL)
'''),md('''## Save to Drive and select a resumable run folder
Outputs appear under **MyDrive / GNR Textile Runs / Sunburst_Four_Designs / run_...**.
Record the printed RUN path. To resume, paste it into RESUME_RUN and rerun setup/connection before the batch.
Completed native drafts are reused; changed model/inputs/policy require a new run.
This makes four paid generation requests, one per missing design. Sending the artwork to the provider is part of generation.
SDK automatic retries are disabled. If a connection drops after the provider accepts a request but before a response is saved,
the provider may still charge; inspect your API usage before retrying that missing design.
'''),code('''from datetime import datetime,timezone
RESUME_RUN=''  # Paste the previously printed full RUN path here to resume.
WORK_ROOT=pathlib.Path('/content/drive/MyDrive/GNR Textile Runs/Sunburst_Four_Designs')
RUN=pathlib.Path(RESUME_RUN) if RESUME_RUN else WORK_ROOT/('run_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S'))
print('Save this RUN path:',RUN)
'''),md('''## Generate all four full images and compare
Run once, then wait for progress. Each design receives its own folder:
step_2.bmp, designer_step_3.bmp, generation/generated_raw.png, generation/generation.json,
ai_step_3.png, ai_step_3.bmp, comparison.png, difference_native.png and comparison.json.
The source's native width/height are preserved without stretching. Square padding is removed after generation.
Comparison uses exact native coordinates without alignment, resizing or palette changes to the designer target.
Percentages measure single-design raster similarity, not business-rule compliance.
'''),code('''from textile.design_batch import run_batch
REFERENCES=[(INPUTS/'_references/design_362/step_2.bmp',INPUTS/'_references/design_362/designer_step_3.bmp')]
results=run_batch(INPUTS,RUN,client,model=IMAGE_MODEL,references=REFERENCES,progress=print)
print('Finished all four designs. Results folder:',RUN)
'''),code('''from IPython.display import HTML
header='<tr><th>Design</th><th>Native size</th><th>Pixel agreement</th><th>Red IoU</th><th>Red Dice</th></tr>'
body=''.join(f"<tr><td>{r['design_id']}</td><td>{r['width']} x {r['height']}</td><td>{r['pixel_agreement_pct']:.2f}%</td><td>{r['red_iou_pct']:.2f}%</td><td>{r['red_dice_pct']:.2f}%</td></tr>" for r in results)
display(HTML('<table>'+header+body+'</table>'))
for row in results:
    folder=RUN/row['design_id']
    print(row['design_id'],'— full native AI image')
    display(Image.open(folder/'ai_step_3.png'))
    print('Sketch / Designer / Sunburst / Difference:')
    display(Image.open(folder/'comparison.png'))
'''),md('''## Download the four folders and comparison percentages
The ZIP contains all four design folders, raw generations, prompts and comparison records.
The same outputs remain in Drive if browser downloads are blocked. Use Drive to download them manually.
No generated files are automatically pushed to GitHub from Colab. Share the review ZIP back in chat for further inspection.
'''),code('''from google.colab import files
files.download(str(RUN/'comparison_summary.csv'))
files.download(str(RUN/'all_designs_review.zip'))
print('Drive folder:',RUN)
''')]
notebook=dict(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'},'colab':{'name':'Sunburst_Four_Design_Comparison.ipynb'}},nbformat=4,nbformat_minor=4)
(root/'notebooks/Sunburst_Four_Design_Comparison.ipynb').write_text(json.dumps(notebook,indent=1))
print('Built standalone four-design Sunburst notebook.')
