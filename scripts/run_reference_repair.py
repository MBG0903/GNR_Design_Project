"""Reproduce the TARGET-ASSISTED repair experiment; never an unseen-image test."""
from pathlib import Path
import json,gzip
import numpy as np
from PIL import Image,ImageDraw
from textile.reference_repair import fit_reference
from textile.comparison import compare
from textile.core import render


def save(source,result,repair,root):
    out=root/'examples/reference_assisted_repair';out.mkdir(exist_ok=True)
    before=render(source>0);after=render(result>0)
    with Image.open(root/'examples/step_3_reference.bmp') as opened:designer=opened.convert('RGB')
    initial,_=compare(designer,before);final,diff=compare(designer,after)
    assert final['mismatched_pixels']==repair['result_mismatches']
    assert initial['mismatched_pixels']-final['mismatched_pixels']==sum(o['reference_mismatch_reduction'] for o in repair['operations'])
    after.save(out/'step_3_reference_repaired.png');after.save(out/'step_3_reference_repaired.bmp')
    diff.save(out/'difference_native.png');np.save(out/'piece_labels.npy',result)
    report={k:v for k,v in repair.items() if k!='operations'}
    report.update(before_comparison=initial,after_comparison=final,
                  input='examples/whole_image_hierarchy_fallback_pass/piece_labels.npy',
                  designer_target='examples/step_3_reference.bmp')
    (out/'audit.json').write_text(json.dumps(report,indent=2))
    (out/'operations.json.gz').write_bytes(gzip.compress(json.dumps(repair['operations'],indent=2).encode(),mtime=0))
    # Native raster is separate from enlarged diagnostics.
    s=3;w,h=after.size
    sheet=Image.new('RGB',(w*s*3,h*s+45),'white');draw=ImageDraw.Draw(sheet)
    for i,(name,im) in enumerate([('Before: 87.39% agreement',before),('After reference-assisted fitting',after),('Designer reference',designer)]):
        sheet.paste(im.resize((w*s,h*s),Image.Resampling.NEAREST),(i*w*s,28))
        draw.text((i*w*s+5,8),name,fill='black')
    sheet.save(out/'before_after_designer.png')
    # The first diagnosed target: lower tail/teardrop, at a readable pixel scale.
    box=(244,244,332,332);s=6;cw,ch=box[2]-box[0],box[3]-box[1]
    detail=Image.new('RGB',(cw*s*3,ch*s+40),'white');draw=ImageDraw.Draw(detail)
    for i,(name,im) in enumerate([('Before',before),('After',after),('Designer',designer)]):
        detail.paste(im.crop(box).resize((cw*s,ch*s),Image.Resampling.NEAREST),(i*cw*s,25))
        draw.text((i*cw*s+5,7),name,fill='black')
    detail.save(out/'tail_detail_before_after.png')
    (out/'summary.md').write_text(f'''# Reference-assisted whole-image repair\n\nThis experiment uses the approved designer target to fit local boundary strips. It is not a learned conversion or an independent accuracy test. No GenAI model was retrained or evaluated on unseen sketches.\n\n| Measure | Before | After |\n|---|---:|---:|\n| Pixel agreement | {initial['pixel_agreement']:.2%} | {final['pixel_agreement']:.2%} |\n| Red IoU | {initial['red_iou']:.2%} | {final['red_iou']:.2%} |\n| Red Dice | {initial['red_dice']:.2%} | {final['red_dice']:.2%} |\n| Mismatched pixels | {initial['mismatched_pixels']:,} | {final['mismatched_pixels']:,} |\n\n{repair['accepted_strips']} accepted one-pixel boundary-strip corrections across {repair['regions_edited']} regions changed {repair['changed_pixels']:,} final pixels. The 401 × 340 canvas, pure palette, labelled two-pixel clearance, existing verified hierarchy traces, red connectivity and internal white-region counts were checked. Boundary movement was limited to two pixels from each original component edge. Compact shapes were left unchanged.\n\nExisting semantic joins, failed anchor comparisons and infeasible hierarchy arcs remain unresolved. Geometry safeguards do not establish semantic production readiness. The original output is retained.\n''')
    print(json.dumps({k:v for k,v in report.items() if k not in ['before_comparison','after_comparison']},indent=2))
    print('BEFORE',initial['pixel_agreement'],initial['red_iou'],'AFTER',final['pixel_agreement'],final['red_iou'])
    return report

if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]
    source=np.load(root/'examples/whole_image_hierarchy_fallback_pass/piece_labels.npy')
    with Image.open(root/'examples/step_3_reference.bmp') as im:target=np.all(np.asarray(im)==[255,0,0],axis=2)
    result,repair=fit_reference(source,target)
    save(source,result,repair,root)
