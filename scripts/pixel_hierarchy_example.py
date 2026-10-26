"""Reproduce one audited contour correction, not image-wide certification."""
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from textile.pixel_hierarchy import replace_trace,check_runs,trace_on_boundary
from textile.core import render,validate
root=Path(__file__).resolve().parents[1]
labels=np.load(root/'examples/separation_2px_pass/provisional_piece_labels.npy')
old=[[204,139],[204,148],[203,148],[203,150],[202,150],[202,153],[201,153],[201,154],
     [200,154],[200,156],[199,156],[199,157],[198,157],[198,158]]
new=[[204,139],[204,148],[203,148],[203,151],[202,151],[202,153],[201,153],[201,155],
     [200,155],[200,156],[199,156],[199,157],[198,157],[198,158]]
assert trace_on_boundary(labels==40,[[212,140],[212,156]])
result,report=replace_trace(labels,40,old,new,'y')
out=root/'examples/pixel_hierarchy_pass';out.mkdir(exist_ok=True)
image=render(result>0);image.save(out/'step_3_hierarchy_draft.png');image.save(out/'step_3_hierarchy_draft.bmp')
np.save(out/'piece_labels.npy',result)
report.update(piece_id=40,front_anchor_length=9,back_anchor_length=16,
              front_shorter_than_back=True,front_trace_before=old,front_trace_after=new,
              corrected_edge_scope='Lower inner/concave contour only, until next octant transition.',
              other_edges_verified=False,image_wide_approval=False,
              unresolved=['Other contours, including this piece\'s opposite edge, are not yet corrected.',
                          'Feather region 78: longest inner run 38 pixels exceeds outer run 12; needs reshaping.',
                          'Short octants may not fit all 3/2/1 levels and strict counts without moving endpoints.',
                          'Semantic joins flagged by the previous separation pass remain unresolved.'],
              raster_checks=validate(image,(401,340)))
(out/'audit.json').write_text(json.dumps(report,indent=2))
# Diagnostic enlargement with actual pixel grid; never used as production raster.
zoom=12;box=(194,136,216,162)
panel=Image.new('RGB',((box[2]-box[0])*zoom*2, (box[3]-box[1])*zoom+30),'white')
d=ImageDraw.Draw(panel)
for i,(title,frame) in enumerate([('Before',render(labels>0)),('After',image)]):
    crop=frame.crop(box).resize(((box[2]-box[0])*zoom,(box[3]-box[1])*zoom),Image.Resampling.NEAREST)
    x=i*crop.width;panel.paste(crop,(x,30));d.text((x+4,8),title,fill='black')
    for j in range(box[2]-box[0]+1):d.line((x+j*zoom,30,x+j*zoom,panel.height),fill='#bbbbbb')
    for j in range(box[3]-box[1]+1):d.line((x,30+j*zoom,x+crop.width,30+j*zoom),fill='#bbbbbb')
panel.save(out/'selected_contour_before_after.png')
print(json.dumps(report,indent=2))
