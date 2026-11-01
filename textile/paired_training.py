"""Native-pixel paired ControlNet data preparation and held-out evaluation.

Training is performed by the pinned upstream Diffusers trainer in vendor/.
This module does not import GPU libraries so data checks can run independently.
"""
import csv
import hashlib
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from .core import rgb, exact_palette, validate
from .comparison import compare

BASE_MODEL = 'stable-diffusion-v1-5/stable-diffusion-v1-5'
CONTROLNET_MODEL = 'lllyasviel/control_v11p_sd15_lineart'
CAPTION = 'Flat dense textile design, solid pure red shapes on pure white, crisp pixel edges, separate leaves and petals, clean white channels, stepped tapering contours, preserve the sketch composition.'


def file_hash(path):
    checksum=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):
            checksum.update(block)
    return checksum.hexdigest()


def read(path):
    with Image.open(path) as im: return rgb(im)


def pixel_hash(path):
    im=read(path)
    return hashlib.sha256(im.tobytes()+str(im.size).encode()).hexdigest()


def verify_row(row):
    for field,hashfield in [('source','source_hash'),('target','target_hash')]:
        if pixel_hash(row[field])!=row[hashfield]:
            raise ValueError(f'{row["key"]}: original {field} changed; re-audit in a new experiment.')
    if not validate(read(row['target']),read(row['source']).size)['automated_checks_passed']:
        raise ValueError(f'{row["key"]}: target must match sketch dimensions and exact red/white palette.')


def split_80_20(rows,seed=42):
    if not rows or len({r['key'] for r in rows})!=len(rows):
        raise ValueError('Audited pairs must have unique keys.')
    groups={}
    for row in rows: groups.setdefault(row['group'],[]).append(row)
    if len(groups)<5: raise ValueError('Need at least five independent families for an 80:20 experiment.')
    keys=sorted(groups);random.Random(seed).shuffle(keys)
    test_groups=set(keys[:max(1,round(len(keys)*.2))])
    result=[dict(r,split='test' if r['group'] in test_groups else 'train') for r in rows]
    seen={}
    for row in result:
        for field in ('source_hash','target_hash','group'):
            token=(field,row[field])
            if token in seen and seen[token]!=row['split']:
                raise ValueError('Duplicate input/target or design family crosses train/test splits. Re-audit grouping.')
            seen[token]=row['split']
    return result


def patch_origins(length,size):
    if length<=size:return [0]
    stride=size-size//4
    return sorted(set(list(range(0,length-size+1,stride))+[length-size]))


def native_patches(source,target,size=512):
    if source.size!=target.size or size<64 or size%8:
        raise ValueError('Aligned pairs and a patch size divisible by 8 (at least 64) are required.')
    for y in patch_origins(source.height,size):
        for x in patch_origins(source.width,size):
            w,h=min(size,source.width),min(size,source.height)
            left,top=(size-w)//2,(size-h)//2
            box=(x,y,x+w,y+h)
            a,b=Image.new('RGB',(size,size),'white'),Image.new('RGB',(size,size),'white')
            a.paste(source.crop(box),(left,top));b.paste(target.crop(box),(left,top))
            yield a,b,dict(source_box=list(box),padding_left=left,padding_top=top)


def prepare_pairs(audit_manifest,output,size=512,seed=42,groups_reviewed=False,progress=print):
    rows=json.loads(Path(audit_manifest).read_text())
    split=split_80_20(rows,seed)
    # Validate ALL originals before writing a successful prepared-data manifest.
    for row in split: verify_row(row)
    config=dict(version=1,size=size,seed=seed,groups_reviewed=bool(groups_reviewed),caption=CAPTION,rows=split)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    marker=output/'preparation.json'
    if marker.exists() and json.loads(marker.read_text())!=config:
        raise ValueError('Training preparation changed. Use a new experiment folder.')
    marker.write_text(json.dumps(config,indent=2))
    (output/'split.json').write_text(json.dumps(split,indent=2))
    patches=[]
    for i,row in enumerate(r for r in split if r['split']=='train'):
        if progress:progress(f'Preparing native training patches: {row["key"]}')
        source,target=read(row['source']),read(row['target'])
        for j,(s,t,geometry) in enumerate(native_patches(source,target,size)):
            folder=output/'patches'/f'{i:05d}_{j:04d}'
            folder.mkdir(parents=True,exist_ok=True)
            for im,name in [(s,'sketch.png'),(t,'designer.png')]:
                path=folder/name
                if path.exists() and read(path).tobytes()!=im.tobytes():
                    raise ValueError('Prepared patch was modified; use a new experiment folder.')
                if not path.exists(): im.save(path)
            patches.append(dict(image=str((folder/'designer.png').resolve()),
                conditioning_image=str((folder/'sketch.png').resolve()),text=CAPTION,
                pair_key=row['key'],group=row['group'],geometry=geometry))
    with (output/'train_patches.jsonl').open('w') as handle:
        for record in patches:handle.write(json.dumps(record)+'\n')
    summary=dict(training_pairs=sum(r['split']=='train' for r in split),test_pairs=sum(r['split']=='test' for r in split),
        training_families=len({r['group'] for r in split if r['split']=='train'}),
        test_families=len({r['group'] for r in split if r['split']=='test'}),training_patches=len(patches),
        split='80:20 by family, rounded; pair counts can differ',groups_reviewed=bool(groups_reviewed),
        native_geometry='Training crops and padding only. No scaling, rotation or interpolation.',
        test_targets_in_training=False)
    (output/'summary.json').write_text(json.dumps(summary,indent=2))
    return summary


def native_inference_canvas(source,multiple=64):
    source=rgb(source)
    w=((source.width+multiple-1)//multiple)*multiple
    h=((source.height+multiple-1)//multiple)*multiple
    x,y=(w-source.width)//2,(h-source.height)//2
    canvas=Image.new('RGB',(w,h),'white');canvas.paste(source,(x,y))
    return canvas,(x,y,x+source.width,y+source.height)


def restore_native(raw,canvas_size,box):
    raw=rgb(raw)
    if raw.size!=tuple(canvas_size):raise ValueError('Model output dimensions differ; refusing automatic resizing.')
    return exact_palette(raw.crop(box))


def evaluate_split(prepared,output,predict,method,settings,progress=print):
    """predict receives sketch ONLY plus a fixed seed, and returns native raw output.

    Target images are used solely after prediction for native-coordinate scores.
    The predictor must not resize the sketch. Inference implementation is in Colab.
    """
    if method not in ('pretrained','finetuned'):raise ValueError('Unknown comparison method.')
    prepared=Path(prepared);output=Path(output)/method
    split=json.loads((prepared/'split.json').read_text())
    if split!=json.loads((prepared/'preparation.json').read_text())['rows']:
        raise ValueError('Prepared split changed; do not mix training and held-out evaluation.')
    rows=[r for r in split if r['split']=='test']
    for row in rows:verify_row(row)
    config=dict(method=method,settings=settings,test_rows=rows,caption=CAPTION)
    output.mkdir(parents=True,exist_ok=True)
    marker=output/'evaluation.json'
    if marker.exists() and json.loads(marker.read_text())!=config:raise ValueError('Evaluation settings changed; use a new output folder.')
    marker.write_text(json.dumps(config,indent=2))
    summaries=[]
    for i,row in enumerate(rows):
        folder=output/f'design_{i+1:04d}';folder.mkdir(exist_ok=True)
        raw_path=folder/'raw_native.png';seed=42+i
        if progress:progress(f'{method}: test {i+1}/{len(rows)} — {row["key"]}')
        source=read(row['source'])
        if not raw_path.exists():
            raw=predict(source,seed)
            if raw.size!=source.size:raise ValueError('Predictor changed native dimensions.')
            raw.save(raw_path)
        raw=read(raw_path)
        if raw.size!=source.size:raise ValueError('Saved output native dimensions changed.')
        candidate=exact_palette(raw)
        truth=read(row['target'])
        report,difference=compare(truth,candidate)
        report.update(key=row['key'],group=row['group'],method=method,seed=seed,
                      production_ready=False,semantic_rules_verified=False)
        candidate.save(folder/'step_3.png');candidate.save(folder/'step_3.bmp')
        difference.save(folder/'difference.png')
        (folder/'metrics.json').write_text(json.dumps(report,indent=2))
        panel=Image.new('RGB',(source.width*4,source.height+25),'white');draw=ImageDraw.Draw(panel)
        for k,(label,im) in enumerate([('Sketch',source),('Designer',truth),(method,candidate),('Difference',difference)]):
            draw.text((k*source.width+3,5),label,fill='black');panel.paste(im,(k*source.width,25))
        panel.save(folder/'comparison.png')
        summaries.append(dict(key=row['key'],method=method,**{k:report[k]*100 for k in ('pixel_agreement','red_iou','red_dice')}))
        with (output/'scores.csv').open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(summaries[0]));writer.writeheader();writer.writerows(summaries)
    summary=dict(method=method,evaluated_test_designs=len(summaries),planned_test_designs=len(rows),
        mean_percentages={metric:float(np.mean([r[metric] for r in summaries])) for metric in ('pixel_agreement','red_iou','red_dice')},
        semantic_rules_verified=False,production_ready=False,
        grouping_reviewed=json.loads((prepared/'preparation.json').read_text())['groups_reviewed'])
    (output/'summary.json').write_text(json.dumps(summary,indent=2))
    return summary
