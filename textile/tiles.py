"""Native-pixel split/review/merge experiment; no automatic contour repair.

Disjoint cores own output pixels. Overlapping halos are read-only context.
Merging does not certify staircase or semantic rules, even after tile sign-off.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from .core import rgb,validate


def _load(value):
    if isinstance(value,(str,Path)):
        with Image.open(value) as im:return rgb(im)
    return rgb(value)


def _digest(im):
    return hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest()


def split(sketch,draft,directory,rows=3,columns=4,overlap=32):
    source,base=_load(sketch),_load(draft);w,h=base.size
    if source.size!=base.size:raise ValueError('Sketch and draft dimensions must match; no resizing.')
    if not validate(base,source.size)['palette_ok']:raise ValueError('Draft must be exact red/white.')
    if any(type(v) is not int for v in [rows,columns,overlap]) or not(1<=rows<=h and 1<=columns<=w and overlap>=0):
        raise ValueError('Grid must use positive integer counts within the canvas and nonnegative overlap.')
    directory=Path(directory)
    if directory.exists():raise ValueError('Use a new experiment directory; existing files are preserved.')
    tiles=[]
    for r in range(rows):
        for c in range(columns):
            x0,x1=c*w//columns,(c+1)*w//columns;y0,y1=r*h//rows,(r+1)*h//rows
            context=[max(0,x0-overlap),max(0,y0-overlap),min(w,x1+overlap),min(h,y1+overlap)]
            tiles.append(dict(id=f'r{r+1}_c{c+1}',row=r,column=c,core=[x0,y0,x1,y1],context=context,
                              core_local=[x0-context[0],y0-context[1],x1-context[0],y1-context[1]]))
    manifest=dict(version=1,width=w,height=h,rows=rows,columns=columns,overlap=overlap,
                  draft_rgb_sha256=_digest(base),sketch_rgb_sha256=_digest(source),tiles=tiles)
    directory.mkdir(parents=True);base.save(directory/'original.png');source.save(directory/'sketch.png')
    (directory/'tiles').mkdir()
    for tile in tiles:
        base.crop(tile['context']).save(directory/'tiles'/f"{tile['id']}_draft.png")
        source.crop(tile['context']).save(directory/'tiles'/f"{tile['id']}_sketch.png")
    (directory/'manifest.json').write_text(json.dumps(manifest,indent=2))
    # The blue ownership grid is a diagnostic, never part of the textile raster.
    guide=base.resize((w*3,h*3),Image.Resampling.NEAREST);draw=ImageDraw.Draw(guide)
    for tile in tiles:
        x0,y0,x1,y1=tile['core'];draw.rectangle((x0*3,y0*3,x1*3-1,y1*3-1),outline='#0066ff',width=2)
        draw.text((x0*3+5,y0*3+5),tile['id'],fill='black',stroke_width=2,stroke_fill='white')
    guide.save(directory/'grid_preview.png')
    return manifest


def read(directory):
    directory=Path(directory);m=json.loads((directory/'manifest.json').read_text());base=_load(directory/'original.png')
    if m.get('version')!=1 or base.size!=(m['width'],m['height']) or _digest(base)!=m['draft_rgb_sha256']:
        raise ValueError('Manifest does not match the unedited source raster.')
    if m.get('sketch_rgb_sha256') and _digest(_load(directory/'sketch.png'))!=m['sketch_rgb_sha256']:
        raise ValueError('Original sketch changed; use a new review run.')
    h,w=m['height'],m['width'];coverage=np.zeros((h,w),np.uint16);ids=set()
    if len(m['tiles'])!=m['rows']*m['columns']:raise ValueError('Grid membership is incomplete.')
    for tile in m['tiles']:
        expected=f"r{tile['row']+1}_c{tile['column']+1}"
        if tile['id']!=expected or expected in ids:raise ValueError('Invalid or duplicate tile ID.')
        ids.add(expected);core=tile['core'];context=tile['context']
        if any(type(v) is not int for v in core+context):raise ValueError('Pixel coordinates must be integers.')
        x0,y0,x1,y1=core;cx0,cy0,cx1,cy1=context
        if not(0<=cx0<=x0<x1<=cx1<=w and 0<=cy0<=y0<y1<=cy1<=h):raise ValueError('Invalid tile bounds.')
        if tile['core_local']!=[x0-cx0,y0-cy0,x1-cx0,y1-cy0]:raise ValueError('Invalid local coordinates.')
        coverage[y0:y1,x0:x1]+=1
    if not np.all(coverage==1):raise ValueError('Cores must partition every source pixel exactly once.')
    return m,base


def merge(directory,proposals=None,reviews=None):
    m,base=read(directory);proposals=proposals or {};reviews=reviews or {}
    known={t['id'] for t in m['tiles']}
    if set(proposals)-known:raise ValueError('Unknown proposal tile.')
    output=base.copy();changed=[]
    for tile in m['tiles']:
        key=tile['id']
        if key not in proposals:continue
        proposal=_load(proposals[key]);original=base.crop(tile['context'])
        if proposal.size!=original.size:raise ValueError(f'{key}: context dimensions changed.')
        if not validate(proposal,original.size)['palette_ok']:raise ValueError(f'{key}: only red/white allowed.')
        delta=np.any(np.asarray(proposal)!=np.asarray(original),axis=2)
        if not delta.any():continue
        x0,y0,x1,y1=tile['core_local'];inside=np.zeros(delta.shape,bool);inside[y0:y1,x0:x1]=True
        if np.any(delta&~inside):raise ValueError(f'{key}: halo edits rejected; expand/reassign the affected curve first.')
        review=reviews.get(key,{})
        required=['complete_front_back_and_tips_reviewed','hierarchy_reviewed','two_pixel_gaps_reviewed']
        if not all(review.get(k) is True for k in required) or not isinstance(review.get('reviewer'),str) or not review['reviewer'].strip() or review.get('unresolved')!=[]:
            raise ValueError(f'{key}: complete-contour/gap/hierarchy review and reviewer required; unresolved issues block merge.')
        core=proposal.crop(tile['core_local']);output.paste(core,tuple(tile['core'][:2]));changed.append(key)
    checks=validate(output,base.size)
    if not checks['palette_ok'] or not checks['size_ok']:raise RuntimeError('Merge failed native raster checks.')
    return output,dict(status='merged_review_draft' if changed else 'unedited_roundtrip',changed_tiles=changed,
                       changed_pixels=int(np.any(np.asarray(output)!=np.asarray(base),axis=2).sum()),
                       raster_checks=checks,source_sha256=m['draft_rgb_sha256'],
                       whole_image_validation_required=True,automatic_staircase_compliance=False,
                       note='Tile attestations are review records, not machine proof. Inspect full contours and seams globally before approval.')


def save_proposal(directory,key,proposal,review):
    # Validate the complete candidate, including halo ownership, before saving.
    merge(directory,{key:proposal},{key:review})
    folder=Path(directory)/'proposals'/key;folder.mkdir(parents=True,exist_ok=True)
    versions=[int(p.stem) for p in folder.glob('*.json') if p.stem.isdigit()]
    version=max(versions,default=0)+1;stem=f'{version:04d}'
    _load(proposal).save(folder/f'{stem}.png')
    (folder/f'{stem}.json').write_text(json.dumps(review,indent=2))
    return str(folder/f'{stem}.png')


def merge_saved(directory):
    m,_=read(directory);proposals={};reviews={}
    for tile in m['tiles']:
        folder=Path(directory)/'proposals'/tile['id']
        records=sorted((p for p in folder.glob('*.json') if p.stem.isdigit() and p.with_suffix('.png').is_file()),key=lambda p:int(p.stem))
        if records:
            latest=records[-1];proposals[tile['id']]=latest.with_suffix('.png');reviews[tile['id']]=json.loads(latest.read_text())
    return merge(directory,proposals,reviews)


def paint_pixel(image,tile,preview_xy,zoom,colour):
    """One click on an integer-zoom preview changes exactly one owned pixel."""
    if type(zoom) is not int or zoom<1 or colour not in ['red','white']:
        raise ValueError('Choose an integer zoom and red or white.')
    if len(preview_xy)!=2 or any(type(v) is not int or v<0 for v in preview_xy):
        raise ValueError('Expected nonnegative integer preview coordinates.')
    im=_load(image);x,y=preview_xy[0]//zoom,preview_xy[1]//zoom
    x0,y0,x1,y1=tile['core_local']
    if not(x0<=x<x1 and y0<=y<y1 and x<im.width and y<im.height):
        raise ValueError('Click inside the editable core; context pixels are read-only.')
    im.putpixel((x,y),(255,0,0) if colour=='red' else (255,255,255))
    return im
