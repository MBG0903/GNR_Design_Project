"""Rank exact-coordinate discrepancies for SEMANTIC review, never auto-fill them.

This is a same-design diagnostic using an approved target; production inference
must operate without that target and be evaluated on independent designs.
"""
import csv,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from textile.core import rgb
from textile.comparison import compare


def build(root):
    with Image.open(root/'examples/step_2.bmp') as im:sketch=rgb(im)
    with Image.open(root/'examples/step_3_reference.bmp') as im:designer=rgb(im)
    with Image.open(root/'examples/generated_dense_draft/step_3_dense_draft.png') as im:base=rgb(im)
    with Image.open(root/'examples/whole_image_hierarchy_fallback_pass/step_3_whole_image_hierarchy.png') as im:current=rgb(im)
    assert len({im.size for im in [sketch,designer,base,current]})==1
    metrics,difference=compare(designer,current)
    red=lambda im:np.all(np.asarray(im)==[255,0,0],axis=2)
    truth,pred,initial=red(designer),red(current),red(base)
    missing=truth&~pred;extra=~truth&pred
    original_error=truth!=initial;final_error=truth!=pred
    introduced=final_error&~original_error
    resolved=original_error&~final_error
    persistent=original_error&final_error
    rows=[];w,h=current.size
    for y in range(0,h,64):
        for x in range(0,w,64):
            x1,y1=min(x+64,w),min(y+64,h);box=np.s_[y:y1,x:x1]
            m,e=int(missing[box].sum()),int(extra[box].sum())
            if not m+e:continue
            kind='mostly_missing_red' if m>3*e else 'mostly_excess_red' if e>3*m else 'mixed_boundary_or_structure'
            rows.append(dict(box=[x,y,x1,y1],missing_red=m,excess_red=e,mismatches=m+e,
                             persistent_from_generation=int(persistent[box].sum()),
                             introduced_during_rule_passes=int(introduced[box].sum()),
                             geometric_error_type=kind,semantic_decision='pending_designer_or_vision_review'))
    rows.sort(key=lambda r:-r['mismatches'])
    for i,r in enumerate(rows,1):r['priority']=i
    assert sum(r['mismatches'] for r in rows)==metrics['mismatched_pixels']
    assert int(persistent.sum()+introduced.sum())==metrics['mismatched_pixels']
    out=root/'examples/repair_queue';out.mkdir(exist_ok=True)
    summary=dict(status='diagnostic_queue_only_no_pixels_edited',
                 total_mismatches=metrics['mismatched_pixels'],
                 generation_stage_mismatches=int(original_error.sum()),
                 persistent_mismatches_from_generation=int(persistent.sum()),
                 mismatches_introduced_during_rule_passes=int(introduced.sum()),
                 mismatches_resolved_during_rule_passes=int(resolved.sum()),
                 core_tile_size=64,tile_count=len(rows),
                 next_actions=['Audit whether the designer reference satisfies the same requested pixel rules.',
                               'Review each crop with sketch context; identify underfill, intentional white and displaced edges.',
                               'Propose local edits with piece identity, affected contour and confidence; retain uncertain cases.',
                               'Check exact palette/size, labelled two-pixel separation, protected contours, internal white and motif fidelity before accepting.',
                               'Measure reference similarity separately from rule compliance; do not accept target-copy pixels as model improvement.'],
                 limitation='The reference is available for diagnosis of this example only. This queue is not an automatic semantic edit plan or unseen-image evaluation.',
                 queue=rows)
    (out/'repair_queue.json').write_text(json.dumps(summary,indent=2))
    with (out/'repair_queue.csv').open('w',newline='') as handle:
        fields=['priority','box','mismatches','missing_red','excess_red','persistent_from_generation','introduced_during_rule_passes','geometric_error_type','semantic_decision']
        writer=csv.DictWriter(handle,fieldnames=fields,lineterminator='\n');writer.writeheader();writer.writerows(rows)
    # Context padding is visual only; counts use disjoint 64px core tiles.
    for row in rows[:6]:
        x,y,x1,y1=row['box'];context=(max(0,x-12),max(0,y-12),min(w,x1+12),min(h,y1+12))
        scale=5;cw,ch=context[2]-context[0],context[3]-context[1]
        sheet=Image.new('RGB',(cw*scale*4,ch*scale+55),'white');draw=ImageDraw.Draw(sheet)
        for j,(title,im) in enumerate([('Sketch',sketch),('Designer',designer),('Current',current),('Difference',difference)]):
            sheet.paste(im.crop(context).resize((cw*scale,ch*scale),Image.Resampling.NEAREST),(j*cw*scale,25))
            draw.text((j*cw*scale+5,6),title,fill='black')
        draw.text((5,ch*scale+34),f"Priority {row['priority']} | core box {row['box']} | blue: missing red | orange: extra red",fill='black')
        sheet.save(out/f"priority_{row['priority']:02d}.png")
    (out/'summary.md').write_text(f'''# Next repair priorities\n\nNo pixels were edited by this diagnostic.\n\n- Current mismatches: {metrics['mismatched_pixels']:,}.\n- Persistent from the dense GenAI draft: {int(persistent.sum()):,}.\n- Introduced during spacing/hierarchy passes: {int(introduced.sum()):,}.\n- Fixed during those passes: {int(resolved.sum()):,}.\n\nEach priority sheet shows the sketch, designer target, current candidate and colour difference at the same coordinates. Blue indicates missing designer red; orange indicates excess red. Metrics use disjoint 64px core tiles; preview context overlaps.\n\nThe target image is for diagnosis, not a production input. Every crop still needs semantic review before changes. Reference similarity and pixel-rule compliance are separate acceptance criteria.\n''')
    return summary

if __name__=='__main__':
    summary=build(Path(__file__).resolve().parents[1])
    print(json.dumps({k:v for k,v in summary.items() if k!='queue'},indent=2))
    print('Top priority:',summary['queue'][0])
