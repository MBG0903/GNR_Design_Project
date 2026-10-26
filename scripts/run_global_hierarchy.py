"""Run the whole-image experiment and produce native rasters + review artifacts."""
from pathlib import Path
import csv,gzip,json
import numpy as np
from PIL import Image,ImageDraw
from textile.global_hierarchy import run_global
from textile.core import render,validate


def save_artifacts(source_labels,result,report,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    image=render(result>0)
    checks=validate(image,(source_labels.shape[1],source_labels.shape[0]))
    if not checks['automated_checks_passed']:raise RuntimeError('Raster validation failed.')
    image.save(output/'step_3_whole_image_hierarchy.png')
    image.save(output/'step_3_whole_image_hierarchy.bmp')
    np.save(output/'piece_labels.npy',result)
    summary={k:v for k,v in report.items() if k!='arcs'}
    summary['raster_checks']=checks
    summary['source']='examples/pixel_hierarchy_pass/piece_labels.npy'
    summary['previous_semantic_separation_issues_resolved']=False
    (output/'audit.json').write_text(json.dumps(summary,indent=2))
    (output/'contour_audit.json.gz').write_bytes(gzip.compress(json.dumps(report,indent=2).encode(),mtime=0))
    with (output/'contour_outcomes.csv').open('w',newline='') as handle:
        writer=csv.writer(handle);writer.writerow(['index','piece','axis','status','mode','preferred_321_outcome','start_x','start_y','before_runs','after_runs'])
        for i,arc in enumerate(report['arcs']):
            writer.writerow([i,arc['piece'],arc['axis'],arc['status'],arc.get('mode'),arc.get('preferred_321_outcome'),*arc['points'][0],
                             json.dumps(arc['before_runs']),json.dumps(arc.get('after_runs'))])
    before=render(source_labels>0);scale=3
    panel=Image.new('RGB',(image.width*scale*2,image.height*scale+26),'white')
    draw=ImageDraw.Draw(panel)
    for i,(title,frame) in enumerate([('Before whole-image pass',before),('After whole-image pass',image)]):
        x=i*image.width*scale
        panel.paste(frame.resize((image.width*scale,image.height*scale),Image.Resampling.NEAREST),(x,26))
        draw.text((x+6,8),title,fill='black')
    panel.save(output/'whole_image_before_after.png')
    change=np.array(image).copy();change[source_labels!=result]=[0,80,255]
    Image.fromarray(change).resize((image.width*scale,image.height*scale),Image.Resampling.NEAREST).save(output/'changed_pixels_blue.png')
    # Magnify local clusters of changed pixels so individual stair steps are visible.
    from scipy import ndimage
    changes=source_labels!=result
    clusters,n=ndimage.label(ndimage.binary_dilation(changes,iterations=2))
    boxes=[]
    for k,sl in enumerate(ndimage.find_objects(clusters),1):
        if sl is None:continue
        y,x=sl
        if x.stop-x.start<=70 and y.stop-y.start<=70:
            boxes.append((int((changes&(clusters==k)).sum()),(max(0,x.start-2),max(0,y.start-2),min(image.width,x.stop+2),min(image.height,y.stop+2))))
    boxes=sorted(boxes,reverse=True)[:4]
    details=Image.new('RGB',(960,max(1,len(boxes))*400),'#eeeeee');draw=ImageDraw.Draw(details)
    for row,(_,box) in enumerate(boxes):
        for col,(title,frame) in enumerate([('Before',before),('After',image)]):
            crop=frame.crop(box);s=max(1,min(470//crop.width,365//crop.height))
            crop=crop.resize((crop.width*s,crop.height*s),Image.Resampling.NEAREST)
            details.paste(crop,(col*480,row*400+28))
            draw.text((col*480+5,row*400+8),f'{title} native box {box}',fill='black')
    details.save(output/'detail_before_after.png')
    return summary


if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]
    source=np.load(root/'examples/pixel_hierarchy_pass/piece_labels.npy')
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--strict-321',action='store_true',help='Disable the designer-approved 2/1 fallback.')
    args=parser.parse_args()
    result,report=run_global(source,max_displacement=2,allow_21=not args.strict_321)
    folder='whole_image_hierarchy_strict_pass' if args.strict_321 else 'whole_image_hierarchy_fallback_pass'
    summary=save_artifacts(source,result,report,root/'examples'/folder)
    print({k:v for k,v in summary.items() if k!='pieces'})
