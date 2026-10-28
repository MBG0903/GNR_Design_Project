"""Reassemble recorded chat-tool generations; never invokes a generation API.

Code only restores native size/palette and merges cores. No staircase repairs.
"""
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from textile.tiles import read, _load, _digest
from textile.core import exact_palette, validate
from textile.tile_batch import assemble_preview


def screen_contours(image):
    from collections import Counter
    from textile.global_hierarchy import boundary_loops,traces
    from textile.pixel_hierarchy import contour_runs,check_runs
    counts=Counter();samples=[]
    mask=np.asarray(image)[:,:,1]==0
    for loop in boundary_loops(mask):
        for points,axis in traces(loop):
            runs=contour_runs(points,axis)
            if len(runs)<4:continue
            verdict=check_runs(runs,allow_21=True);counts[verdict['mode'] or 'unresolved']+=1
            if not verdict['passed'] and len(samples)<12:
                samples.append(dict(points=points,axis=axis,runs=runs,semantic_review_required=True))
    return dict(candidate_trace_counts=dict(counts),unresolved_trace_samples=samples,
        scope='Geometry-only screening, including overlapping/reversed/sub-arcs and possibly exempt shapes. These are not unique motifs or semantic pass rates; front/back identity and two-pixel semantic gaps are not certified.')


def run(root):
    root=Path(root);manifest,base=read(root)
    generation=json.loads((root/'generation_manifest.json').read_text())
    records=json.loads((root/'tool_generation_records.json').read_text())
    if {r['tile_id'] for r in records}!={t['id'] for t in manifest['tiles']} or len(records)!=len(manifest['tiles']):
        raise ValueError('Need one genuine generation record per tile.')
    lookup={t['id']:t for t in generation['tiles']};details=[]
    for record in records:
        key=record['tile_id'];tile=lookup[key];transform=tile['generation_transform']
        path=root/record['raw_file'];raw=_load(path)
        if raw.width!=raw.height:raise ValueError(f'{key}: generated square registration lost; review before assembly.')
        side=transform['side'];left=transform['left'];top=transform['top'];w,h=transform['native_size']
        restored=raw.resize((side,side),Image.Resampling.NEAREST).crop((left,top,left+w,top+h))
        projected=exact_palette(restored);original=base.crop(tile['context'])
        candidate=original.copy();x0,y0,_,_=tile['core_local']
        candidate.paste(projected.crop(tile['core_local']),(x0,y0))
        folder=root/'ai_proposals'/key/'0001'
        if folder.exists():
            old=json.loads((folder/'proposal.json').read_text())
            if old['candidate_rgb_sha256']!=_digest(candidate):raise ValueError('Existing proposal differs; preserve prior run.')
        else:
            folder.mkdir(parents=True);candidate.save(folder/'candidate.png');original.save(folder/'input.png')
            metadata=dict(status='unreviewed_ai_tile_proposal',tile_id=key,revision=1,
                requested_model=record['model'],model_identity_verified=False,generator=record['generator'],
                full_draft_rgb_sha256=manifest['draft_rgb_sha256'],context=tile['context'],core=tile['core_local'],
                native_size=list(candidate.size),candidate_rgb_sha256=_digest(candidate),input_rgb_sha256=_digest(original),
                prompt=record['prompt'],raw_file=record['raw_file'],raw_size=list(raw.size),
                semantic_hierarchy_verified=False,automatic_merge_allowed=False,
                processing='Square restored with nearest-neighbour; white padding cropped; palette projected to red/white; halo restored; owned core only.')
            (folder/'proposal.json').write_text(json.dumps(metadata,indent=2))
        details.append(dict(tile_id=key,raw_size=list(raw.size),native_size=list(candidate.size),
            native_core=tile['core'],changed_core_pixels=int(np.any(np.asarray(candidate)!=np.asarray(original),axis=2).sum())))
    output,report=assemble_preview(root)
    combined=_load(output/'combined_unreviewed.png');delta=np.any(np.asarray(combined)!=np.asarray(base),axis=2)
    seams=np.zeros(delta.shape,bool)
    for x in sorted({t['core'][0] for t in manifest['tiles']} - {0}):seams[:,max(0,x-2):x+2]=True
    for y in sorted({t['core'][1] for t in manifest['tiles']} - {0}):seams[max(0,y-2):y+2,:]=True
    report.update(generator='chat_image_generation_tool',model_identity_verified=False,
        requested_sunburst=False,tile_details=details,
        changed_pixels_near_seams=int((delta&seams).sum()),
        seam_metric_note='Changed pixels within two pixels of a tile boundary; diagnostic only, not a seam violation count.',
        raw_generations_retained=True,processing_applied='Native square restoration, exact palette projection, original halo restoration, disjoint core assembly.',
        hierarchy_code_repairs_applied=False,reference_fitting_applied=False,
        prior_reference_guidance='The initial dense draft used the same-design designer reference; no new reference was supplied to tile generation.')
    report['contour_screening_before']=screen_contours(base)
    report['contour_screening_after']=screen_contours(combined)
    (output/'report.json').write_text(json.dumps(report,indent=2))
    combined.save(root/'step_3_combined_review.png');combined.save(root/'step_3_combined_review.bmp')
    scale=4;panel=Image.new('RGB',(base.width*scale*2,base.height*scale+32),'white');draw=ImageDraw.Draw(panel)
    for i,(title,im) in enumerate([('Before: dense AI draft',base),('After: 12 regenerated tile cores',combined)]):
        draw.text((i*base.width*scale+8,8),title,fill='black')
        panel.paste(im.resize((base.width*scale,base.height*scale),Image.Resampling.NEAREST),(i*base.width*scale,32))
    panel.save(root/'before_after_review.png')
    cellw=324;cellh=368;sheet=Image.new('RGB',(cellw*4,cellh*3),'#eeeeee');draw=ImageDraw.Draw(sheet)
    for t in manifest['tiles']:
        im=_load(output/'tiles'/f"{t['id']}_core.png");x=t['column']*cellw+8;y=t['row']*cellh+24
        sheet.paste(im.resize((im.width*3,im.height*3),Image.Resampling.NEAREST),(x,y))
        draw.text((x,y-18),t['id']+' - native core',fill='black')
    sheet.save(root/'all_12_tiles_review.png')
    (root/'audit.json').write_text(json.dumps(report,indent=2))
    import zipfile
    with zipfile.ZipFile(root/'all_tiles_and_combined_review.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for name in ['step_3_combined_review.png','step_3_combined_review.bmp','before_after_review.png','all_12_tiles_review.png','audit.json','manifest.json']:
            archive.write(root/name,name)
        for path in sorted((output/'tiles').iterdir()):archive.write(path,'tiles/'+path.name)
        archive.write(output/'tile_seam_guide_diagnostic.png','tile_seam_guide_diagnostic.png')
    print(json.dumps(dict(output=str(output),changed_pixels=report['changed_pixels'],
        near_seams=report['changed_pixels_near_seams'],checks=validate(combined,base.size)),indent=2))
    return output,report


if __name__=='__main__':
    run(Path(__file__).resolve().parents[1]/'examples/chat_genai_tile_pass')
