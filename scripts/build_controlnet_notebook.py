"""Standalone paired native-pixel ControlNet fine-tuning notebook."""
import base64
import io
import json
import zipfile
from pathlib import Path
root=Path(__file__).resolve().parents[1]
buf=io.BytesIO()
with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
    for name in ('__init__.py','core.py','data.py','comparison.py','genai.py','retrieval_experiment.py','paired_training.py'):
        z.writestr('textile/'+name,(root/'textile'/name).read_bytes())
    for name in ('train_controlnet.py','LICENSE','NOTICE.md'):
        z.writestr('vendor/controlnet/'+name,(root/'vendor/controlnet'/name).read_bytes())
payload=base64.b64encode(buf.getvalue()).decode()
def md(s):return dict(cell_type='markdown',metadata={},source=s.splitlines(True))
def code(s):return dict(cell_type='code',metadata={},source=s.splitlines(True),execution_count=None,outputs=[])
cells=[md('''# Train on paired textile sketches: Stable Diffusion 1.5 + Lineart ControlNet
This experiment **updates ControlNet model weights** using your Step_2 inputs and designer Step_3 targets. Stable Diffusion's image backbone, VAE and text encoder stay frozen. It is generative paired-image training, not example retrieval or conventional sketch thresholding.

- Pretrained base: `stable-diffusion-v1-5/stable-diffusion-v1-5`.
- Pretrained conditioning model: `lllyasviel/control_v11p_sd15_lineart`.
- Approximately **80% training / 20% testing by original design family**. Related crops/variants and exact duplicates stay together. With 1,700 independent pairs this would be about 1,360 / 340; actual counts depend on grouping and valid files.
- Native 512×512 crops/padding preserve the training target's pixel grid. No scaling, interpolation or random geometric augmentation. Larger images provide aligned patches; test designs stay whole and outside training.
- Whole-image inference uses native dimensions padded to multiples of 64, then removes padding; it does not generate and stitch separate tiles.
- Final red/white projection and size checks are deterministic. Learned gap widths, staircase counts and semantics are not guaranteed. The model's latent VAE compresses spatial detail, so one-pixel rules remain a known challenge.

Use a **GPU runtime** (T4 16GB is the initial target; larger GPUs give more headroom). Actual GPU training and model downloads have not been validated in this cloud workspace. This notebook uses Hugging Face model downloads and Colab compute, not your OpenAI API key. Runtime duration and memory fit must be measured on your dataset.

The trainer is vendored from Diffusers v0.35.1 under Apache-2.0, with local dataset loading and strict no-resize preprocessing. Model cards describe the separate pretrained model licences.
'''),md('''## 1. Mount Drive and configure the experiment
Resolve the earlier dataset audit issues before training. Keep original files unchanged. Existing groups.csv is preserved; verify that related crops/variants share one design_id and paired images align. Scores remain provisional until this is checked.
'''),code('''from google.colab import drive
from pathlib import Path
drive.mount('/content/drive')
DATA_ROOT=Path('/content/drive/MyDrive/GNR Design Data-Set')
JOB=Path('/content/drive/MyDrive/GNR Textile Training/Native_ControlNet_v1')
GROUPS=DATA_ROOT/'groups.csv'
GROUPS_REVIEWED=False  # Set True only after checking family assignments and pair alignment.
RESOLUTION=512
MAX_TRAIN_STEPS=2000  # Optimizer updates, not epochs. Keep fixed for this experiment.
LEARNING_RATE=1e-5
GRADIENT_ACCUMULATION=4
JOB.mkdir(parents=True,exist_ok=True)
print('All data/checkpoints/results saved under:',JOB)
'''),md('''## 2. Install and check the GPU
The notebook uses Colab's installed PyTorch/torchvision CUDA build. It installs the pinned Diffusers trainer dependencies without replacing that CUDA build. Restart the session and rerun setup if pip reports that an already imported dependency needs a restart.
'''),code('''import base64,io,json,sys,subprocess,zipfile
RUNTIME=Path('/content/gnr_controlnet_runtime');RUNTIME.mkdir(exist_ok=True)
PAYLOAD='''+repr(payload)+'''
with zipfile.ZipFile(io.BytesIO(base64.b64decode(PAYLOAD))) as z:z.extractall(RUNTIME)
subprocess.check_call([sys.executable,'-m','pip','install',
    'diffusers==0.35.1','transformers>=4.44,<5','accelerate>=1.2,<2',
    'datasets>=3,<4','bitsandbytes>=0.45,<0.49','ftfy','tensorboard',
    'Pillow>=10.4,<13','numpy>=1.26,<3','scipy>=1.12,<2','safetensors>=0.4'])
sys.path.insert(0,str(RUNTIME))
for name in list(sys.modules):
    if name=='textile' or name.startswith('textile.'):del sys.modules[name]
import torch
if not torch.cuda.is_available():raise RuntimeError('Select Runtime → Change runtime type → GPU, then rerun.')
print('GPU:',torch.cuda.get_device_name(0))
print('VRAM GiB:',round(torch.cuda.get_device_properties(0).total_memory/2**30,1))
from PIL import Image
from IPython.display import display
from textile.data import audit_dataset,audit_diagnostics,contact_sheet
from textile.retrieval_experiment import suggest_groups
from textile.paired_training import prepare_pairs,file_hash,BASE_MODEL,CONTROLNET_MODEL,CAPTION
'''),md('''## 3. Audit the real pairs
If the audit fails, read the printed category breakdown and `audit/audit_issues.csv`. Invalid targets, ambiguous names, unpaired files or incomplete groups are not silently discarded. No paid API is involved. Do not train on misregistered targets: equal dimensions alone do not establish alignment.
'''),code('''suggest_groups(DATA_ROOT/'Stage_2',GROUPS)
try:
    audit=audit_dataset(DATA_ROOT/'Stage_2',DATA_ROOT/'Stage_3',JOB/'audit',
        groups_csv=GROUPS,require_groups=True,seed=42)
except ValueError as exc:
    if str(exc).startswith('Audit failed'):
        print(audit_diagnostics(json.loads((JOB/'audit/audit.json').read_text())))
        print('Full issues:',JOB/'audit/audit_issues.csv')
    raise
print('Audited pairs:',audit['pairs'])
if not GROUPS_REVIEWED:print('PROVISIONAL grouping: review family IDs and alignment before claiming independent test accuracy.')
'''),md('''## 4. Freeze the 80:20 split and prepare native training patches
The audit's preliminary three-way labels are replaced here by the requested train/test split, preserving its transitive family/duplicate groups. Approximately 20% of families are held out before making any training patches. Test patches/targets never enter the training dataset.

Smaller designs are padded to 512×512; larger ones provide overlapping 512×512 native crops. Shapes can be truncated at patch boundaries, so this is a local-pattern baseline with limited large-motif context. Test inference attempts the whole image without resizing or stitching. Larger designs contribute more patches; this weighting is recorded in the patch manifest.
'''),code('''PREPARED=JOB/'prepared'
summary=prepare_pairs(JOB/'audit/manifest.json',PREPARED,size=RESOLUTION,
    seed=42,groups_reviewed=GROUPS_REVIEWED,progress=print)
print(json.dumps(summary,indent=2))
rows=json.loads((PREPARED/'split.json').read_text())
contact_sheet([r for r in rows if r['split']=='train'],JOB/'training_pair_preview.png',count=8)
display(Image.open(JOB/'training_pair_preview.png'))
from datasets import Dataset,Image as HFImage
patches=[json.loads(line) for line in (PREPARED/'train_patches.jsonl').read_text().splitlines()]
HF_TRAIN=Path('/content/gnr_hf_train')  # Local cache rebuilt from persistent prepared pairs.
import shutil
if HF_TRAIN.exists():shutil.rmtree(HF_TRAIN)
dataset=Dataset.from_list([{k:r[k] for k in ('image','conditioning_image','text')} for r in patches])
dataset=dataset.cast_column('image',HFImage()).cast_column('conditioning_image',HFImage())
dataset.save_to_disk(str(HF_TRAIN))
print('Training patches:',len(dataset),'— test pairs:',summary['test_pairs'])
'''),md('''## 5. Pin pretrained revisions and training settings
Model files download from Hugging Face. These public model IDs may still require account/network access; follow the model-card access requirements if a download fails. No model fallback is selected automatically.
The first run records exact model revisions. Reruns reuse them and reject changed training settings. To change settings or datasets, choose a new JOB folder.
'''),code('''from huggingface_hub import model_info,snapshot_download
import hashlib
TRAIN_OUTPUT=JOB/'trained_controlnet'
TRAIN_CONFIG=JOB/'training_config.json'
settings=dict(base_model=BASE_MODEL,controlnet_model=CONTROLNET_MODEL,resolution=RESOLUTION,
    max_train_steps=MAX_TRAIN_STEPS,learning_rate=LEARNING_RATE,gradient_accumulation=GRADIENT_ACCUMULATION,
    seed=42,caption=CAPTION,preparation_hash=hashlib.sha256((PREPARED/'preparation.json').read_bytes()).hexdigest(),
    trainer_hash=hashlib.sha256((RUNTIME/'vendor/controlnet/train_controlnet.py').read_bytes()).hexdigest())
if TRAIN_CONFIG.exists():
    training_config=json.loads(TRAIN_CONFIG.read_text())
    if training_config['settings']!=settings:raise ValueError('Training settings changed. Use a new JOB folder.')
else:
    training_config=dict(settings=settings,base_revision=model_info(BASE_MODEL).sha,
        controlnet_revision=model_info(CONTROLNET_MODEL).sha)
    TRAIN_CONFIG.write_text(json.dumps(training_config,indent=2))
BASE_REVISION=training_config['base_revision']
PRETRAINED_CONTROLNET=snapshot_download(CONTROLNET_MODEL,revision=training_config['controlnet_revision'],
    allow_patterns=['config.json','*.safetensors','*.bin'])
print('Base revision:',BASE_REVISION)
print('ControlNet revision:',training_config['controlnet_revision'])
'''),md('''## 6. Fine-tune ControlNet
This is real weight training. The image backbone, VAE and text encoder stay frozen. Defaults: batch size 1, effective batch 4, fp16 mixed precision, gradient checkpointing and 8-bit Adam. No OpenAI API calls or automatic model publication.

Training progress appears below. Checkpoints are saved on Drive every 250 optimizer updates; the latest two resumable checkpoints and final model are retained. After a Colab disconnect, rerun earlier cells with the same JOB to resume the latest checkpoint. Training can take hours; no duration or convergence claim is made before a live run. An out-of-memory error needs a GPU with more headroom or a separately configured smaller-patch experiment; do not resize designer targets silently.

Keep the held-out test set out of checkpoint selection. The default experiment evaluates the final fixed-step checkpoint. If changing hyperparameters later, use a validation subset within the training 80% and keep the test 20% untouched.
'''),code('''complete=JOB/'training_complete.json'
if complete.exists():
    saved=json.loads(complete.read_text())
    if saved['training_config']!=training_config:raise ValueError('Completed training configuration differs.')
    if any(not (TRAIN_OUTPUT/name).exists() or file_hash(TRAIN_OUTPUT/name)!=checksum for name,checksum in saved['artifacts'].items()):
        raise ValueError('Saved model files changed; preserve this run and use a new JOB.')
    print('Completed run retained:',TRAIN_OUTPUT)
else:
    command=[sys.executable,str(RUNTIME/'vendor/controlnet/train_controlnet.py'),
        '--pretrained_model_name_or_path',BASE_MODEL,'--revision',BASE_REVISION,
        '--controlnet_model_name_or_path',PRETRAINED_CONTROLNET,
        '--train_data_dir',str(HF_TRAIN),'--image_column','image',
        '--conditioning_image_column','conditioning_image','--caption_column','text',
        '--output_dir',str(TRAIN_OUTPUT),'--resolution',str(RESOLUTION),
        '--train_batch_size','1','--gradient_accumulation_steps',str(GRADIENT_ACCUMULATION),
        '--learning_rate',str(LEARNING_RATE),'--lr_scheduler','constant','--lr_warmup_steps','0',
        '--max_train_steps',str(MAX_TRAIN_STEPS),'--mixed_precision','fp16',
        '--gradient_checkpointing','--use_8bit_adam','--dataloader_num_workers','0',
        '--checkpointing_steps','250','--checkpoints_total_limit','2','--seed','42',
        '--proportion_empty_prompts','0','--report_to','tensorboard']
    if any(TRAIN_OUTPUT.glob('checkpoint-*')):command+=['--resume_from_checkpoint','latest']
    subprocess.check_call(command)
    if not (TRAIN_OUTPUT/'config.json').exists():raise RuntimeError('Final ControlNet model was not saved.')
    artifacts={p.name:file_hash(p) for p in TRAIN_OUTPUT.iterdir() if p.is_file() and p.suffix in ('.json','.safetensors','.bin')}
    if not any(name.endswith(('.safetensors','.bin')) for name in artifacts):raise RuntimeError('Final learned weights are missing.')
    complete.write_text(json.dumps({'status':'training_process_completed','optimizer_steps':MAX_TRAIN_STEPS,
        'quality_verified':False,'training_config':training_config,'artifacts':artifacts},indent=2))
print('Trained ControlNet folder:',TRAIN_OUTPUT)
'''),md('''## 7. Compare pretrained versus fine-tuned on the held-out 20%
This evaluates **all test pairs** with identical seeds, prompt, scheduler and inference settings. Both methods use Step_2 only as their image input. Designer targets are used afterward for scoring. This is another potentially lengthy GPU stage.

Each full sketch is padded to multiples of 64 with no resizing. The full canvas is generated in one pass; only padding is cropped away. Raw outputs are saved before exact red/white projection. Designs above the configured pixel budget stop instead of being resized; larger native canvases may need a larger GPU even with model offload. No separate-tile stitching is performed.

Scores report pixel agreement, red IoU and red Dice. They are not staircase/gap pass rates. Review complete contours and spacing with the designer before interpreting a 70–80% similarity score as usable production quality.
'''),code('''from diffusers import ControlNetModel,StableDiffusionControlNetPipeline,UniPCMultistepScheduler
from textile.paired_training import native_inference_canvas,evaluate_split
import gc
MAX_NATIVE_CANVAS_PIXELS=1024*1024  # Memory guard, not a promise of GPU fit. Never resizes.
INFERENCE_STEPS=30
GUIDANCE_SCALE=5.0
CONTROL_SCALE=1.0
if not complete.exists():raise ValueError('Complete training before the held-out comparison.')
for row in rows:
    if row['split']=='test':
        with Image.open(row['source']) as im: padded,_=native_inference_canvas(im)
        if padded.width*padded.height>MAX_NATIVE_CANVAS_PIXELS:
            raise ValueError(f"{row['key']} exceeds the native inference budget. Use sufficient GPU memory and raise the budget explicitly; no resizing was applied.")
results={}
for method,path in [('pretrained',PRETRAINED_CONTROLNET),('finetuned',str(TRAIN_OUTPUT))]:
    controlnet=ControlNetModel.from_pretrained(path,torch_dtype=torch.float16)
    pipe=StableDiffusionControlNetPipeline.from_pretrained(BASE_MODEL,revision=BASE_REVISION,
        controlnet=controlnet,torch_dtype=torch.float16)
    pipe.scheduler=UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    pipe.enable_vae_slicing()
    def predict(sketch,seed):
        canvas,box=native_inference_canvas(sketch)
        generated=pipe(CAPTION,image=canvas,height=canvas.height,width=canvas.width,
            num_inference_steps=INFERENCE_STEPS,guidance_scale=GUIDANCE_SCALE,
            controlnet_conditioning_scale=CONTROL_SCALE,
            generator=torch.Generator(device='cpu').manual_seed(seed)).images[0]
        if generated.size!=canvas.size:raise ValueError('Model output size changed; no resize permitted.')
        return generated.crop(box)
    evaluation_settings=dict(training_config=training_config,model_artifacts=json.loads(complete.read_text())['artifacts'],inference_steps=INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,control_scale=CONTROL_SCALE,max_native_canvas_pixels=MAX_NATIVE_CANVAS_PIXELS)
    results[method]=evaluate_split(PREPARED,JOB/'test_results',predict,method,evaluation_settings,progress=print)
    del pipe,controlnet
    gc.collect();torch.cuda.empty_cache()
print(json.dumps(results,indent=2))
print('Change in percentage points:',{m:results['finetuned']['mean_percentages'][m]-results['pretrained']['mean_percentages'][m]
    for m in ('pixel_agreement','red_iou','red_dice')})
'''),md('''## 8. Inspect outputs and use the trained model for a future sketch
Comparison panels, coloured differences, PNG/BMP and scores stay in `test_results/pretrained` and `test_results/finetuned`. Every output retains its sketch's native size. Save the entire trained_controlnet folder; it contains the learned adapter, not a standalone replacement for the base model.
For future sketches use the final adapter and the same native padding/prompt/inference settings, without a designer target. The next cell provides this path.
'''),code('''for folder in sorted((JOB/'test_results/finetuned').glob('design_*'))[:5]:
    display(Image.open(folder/'comparison.png'))
print('Persistent trained weights:',TRAIN_OUTPUT)
print('Persistent scores and images:',JOB/'test_results')
'''),md('''## Optional: predict a new Step_2 (no designer target required)
Set NEW_SKETCH to a Drive path, then run this cell. It saves a raw draft plus exact-palette native PNG/BMP. No production approval is inferred.
'''),code('''NEW_SKETCH=''
if NEW_SKETCH:
    # This optional cell also works when the lengthy held-out comparison is skipped.
    from diffusers import ControlNetModel,StableDiffusionControlNetPipeline,UniPCMultistepScheduler
    from textile.paired_training import read,restore_native,native_inference_canvas
    import gc
    MAX_NATIVE_CANVAS_PIXELS=globals().get('MAX_NATIVE_CANVAS_PIXELS',1024*1024)
    INFERENCE_STEPS=globals().get('INFERENCE_STEPS',30)
    GUIDANCE_SCALE=globals().get('GUIDANCE_SCALE',5.0)
    CONTROL_SCALE=globals().get('CONTROL_SCALE',1.0)
    if not (JOB/'training_complete.json').exists():raise ValueError('Complete training first.')
    sketch=read(NEW_SKETCH);canvas,box=native_inference_canvas(sketch)
    if canvas.width*canvas.height>MAX_NATIVE_CANVAS_PIXELS:raise ValueError('Native canvas exceeds the configured GPU budget.')
    controlnet=ControlNetModel.from_pretrained(TRAIN_OUTPUT,torch_dtype=torch.float16)
    pipe=StableDiffusionControlNetPipeline.from_pretrained(BASE_MODEL,revision=BASE_REVISION,
        controlnet=controlnet,torch_dtype=torch.float16)
    pipe.scheduler=UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload();pipe.enable_vae_slicing()
    raw=pipe(CAPTION,image=canvas,height=canvas.height,width=canvas.width,
        num_inference_steps=INFERENCE_STEPS,guidance_scale=GUIDANCE_SCALE,
        controlnet_conditioning_scale=CONTROL_SCALE,generator=torch.Generator(device='cpu').manual_seed(42)).images[0]
    native=restore_native(raw,canvas.size,box)
    from datetime import datetime,timezone
    out=JOB/'future_predictions'/datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True,exist_ok=False)
    raw.save(out/'raw_padded.png');native.save(out/'step_3_review.png');native.save(out/'step_3_review.bmp')
    print('Native review output:',out);display(native)
    del pipe,controlnet
    gc.collect();torch.cuda.empty_cache()
else:
    print('Optional future-sketch cell: set NEW_SKETCH when ready.')
''')]
nb=dict(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
    'language_info':{'name':'python'},'colab':{'name':'Paired_ControlNet_Training.ipynb','accelerator':'GPU'},'accelerator':'GPU'},nbformat=4,nbformat_minor=4)
(root/'notebooks/Paired_ControlNet_Training.ipynb').write_text(json.dumps(nb,indent=1))
print('Built paired ControlNet training notebook.')
