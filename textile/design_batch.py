"""Full-design image API generation and held-out target comparisons.

Inputs are verified before any paid request. Query designer images are never
generation inputs. Format checks do not establish semantic rule compliance.
"""
import csv
import hashlib
import json
import re
import shutil
import zipfile
from pathlib import Path
from PIL import Image, ImageDraw
from .core import validate
from .genai import generate, POLICY
from .comparison import compare


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _image(path):
    with Image.open(path) as image:return image.convert('RGB')


def run_batch(inputs, output, client, model='gpt-image-2.5-sunburst', references=(), progress=print):
    if client is None:raise ValueError('Connect an authenticated image API client first.')
    inputs=Path(inputs);output=Path(output)
    records=json.loads((inputs/'manifest.json').read_text());prepared=[];ids=set()
    for record in records:
        key=record['design_id']
        if not re.fullmatch(r'GNR_[0-9]+',key) or key in ids:raise ValueError('Invalid/duplicate design ID.')
        ids.add(key);folder=inputs/key;source=folder/'step_2.bmp';target=folder/'designer_step_3.bmp'
        if _sha(source)!=record['source_sha256'] or _sha(target)!=record['designer_sha256']:
            raise ValueError(f'{key}: original input changed.')
        sketch=_image(source);designer=_image(target)
        if sketch.size!=designer.size or not validate(designer,sketch.size)['automated_checks_passed']:
            raise ValueError(f'{key}: designer palette/dimensions invalid; do not silently modify evaluation truth.')
        prepared.append((record,source,target,sketch,designer))
    if not prepared:raise ValueError('No design pairs supplied.')
    # None of the evaluation queries/targets can double as a prompting reference.
    evaluation_pixels={(_image(p).size,_image(p).tobytes()) for _,s,t,_,_ in prepared for p in (s,t)}
    for pair in references:
        for path in pair:
            image=_image(path)
            if (image.size,image.tobytes()) in evaluation_pixels:
                raise ValueError('Evaluation design/target cannot be a generation reference.')
    config=dict(version=1,model=model,inputs=[dict(design_id=r['design_id'],source_sha256=r['source_sha256'],designer_sha256=r['designer_sha256']) for r,_,_,_,_ in prepared],
                reference_hashes=[[_sha(p) for p in pair] for pair in references],policy_sha256=hashlib.sha256(POLICY.encode()).hexdigest(),
                request_contract='native-grid-v2',generation_size='1024x1024',quality='high')
    output.mkdir(parents=True,exist_ok=True);job=output/'job.json'
    if job.exists() and json.loads(job.read_text())!=config:
        raise ValueError('Run configuration changed; use a new output folder rather than mixing results.')
    job.write_text(json.dumps(config,indent=2));summaries=[]
    for index,(record,source,target,sketch,designer) in enumerate(prepared,1):
        key=record['design_id'];folder=output/key;folder.mkdir(exist_ok=True)
        for src,name in [(source,'step_2.bmp'),(target,'designer_step_3.bmp')]:
            dest=folder/name
            if not dest.exists():shutil.copyfile(src,dest)
            elif _sha(dest)!=_sha(src):raise ValueError(f'{key}: saved original changed.')
        generation=folder/'generation';draft=generation/'draft.png'
        if progress:progress(f'{index}/{len(prepared)}: {key} '+('reusing saved generation' if draft.exists() else 'generating full image'))
        if not draft.exists():
            if generation.exists():
                # Partial raw responses cannot be assumed equivalent to a finished generation.
                raise ValueError(f'{key}: partial generation exists; inspect it before retrying with a new run folder.')
            try:generate(source,generation,client=client,references=references,model=model)
            except Exception as exc:
                failure=dict(status='generation_failed',design_id=key,error_type=type(exc).__name__,http_status=getattr(exc,'status_code',None),generated=False)
                (folder/'status.json').write_text(json.dumps(failure,indent=2))
                raise RuntimeError(f'{key}: image request failed ({failure["error_type"]}, HTTP {failure["http_status"]}). Earlier completed designs are retained. Check API access/billing and rerun the same folder if no partial generation exists.') from None
        candidate=_image(draft);checks=validate(candidate,sketch.size)
        if not checks['automated_checks_passed']:raise ValueError(f'{key}: generated draft failed native raster checks.')
        report,difference=compare(designer,candidate)
        report.update(design_id=key,requested_model=model,designer_target_sent_to_generation=False,
                      semantic_business_rules_verified=False,production_ready=False,
                      source_sha256=record['source_sha256'],designer_sha256=record['designer_sha256'],
                      generated_png_sha256=_sha(draft),raster_validation=checks,
                      interpretation='Native-coordinate similarity to held-out designer target, not a business-rule pass rate. Density/gaps/staircases were prompted; only size/palette were deterministically enforced.')
        candidate.save(folder/'ai_step_3.png');candidate.save(folder/'ai_step_3.bmp');difference.save(folder/'difference_native.png')
        (folder/'comparison.json').write_text(json.dumps(report,indent=2))
        (folder/'status.json').write_text(json.dumps(dict(status='generated_compared_unreviewed',generated=True,design_id=key,production_ready=False),indent=2))
        w,h=sketch.size;scale=2;frames=[('Step 2',sketch),('Designer Step 3',designer),('Sunburst AI draft',candidate),('Difference: blue missing; orange extra',difference)]
        sheet=Image.new('RGB',(w*scale*4,h*scale+30),'white');draw=ImageDraw.Draw(sheet)
        for n,(title,image) in enumerate(frames):
            draw.text((n*w*scale+6,8),title,fill='black');sheet.paste(image.resize((w*scale,h*scale),Image.Resampling.NEAREST),(n*w*scale,30))
        sheet.save(folder/'comparison.png')
        row=dict(design_id=key,width=w,height=h,pixel_agreement_pct=100*report['pixel_agreement'],red_iou_pct=100*report['red_iou'],red_dice_pct=100*report['red_dice'],status='generated_compared_unreviewed')
        summaries.append(row)
        (output/'comparison_summary.json').write_text(json.dumps(summaries,indent=2))
        with (output/'comparison_summary.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(row));writer.writeheader();writer.writerows(summaries)
        if progress:progress(f'{key}: pixel agreement {row["pixel_agreement_pct"]:.2f}%; red IoU {row["red_iou_pct"]:.2f}%')
    with zipfile.ZipFile(output/'all_designs_review.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.rglob('*')):
            if path.is_file() and path.suffix!='.zip':archive.write(path,str(path.relative_to(output)))
    return summaries
