"""Optional paid Sunburst tile proposals, hard-clipped to owned native cores.

This does not enforce semantic staircase rules and never auto-approves a tile.
"""
from pathlib import Path
from io import BytesIO
import base64,json
from PIL import Image,ImageDraw
from .tiles import read,_load,_digest
from .core import validate,exact_palette
from .genai import POLICY


def _png(im):
    stream=BytesIO();im.save(stream,format='PNG');return stream.getvalue()


def propose(directory,key,current,client,model='gpt-image-2.5-sunburst'):
    if client is None:raise ValueError('Configure an API client in Colab; a subscription alone is not API authentication.')
    directory=Path(directory);manifest,base=read(directory);tiles={t['id']:t for t in manifest['tiles']}
    if key not in tiles:raise ValueError('Unknown tile.')
    tile=tiles[key];original=base.crop(tile['context']);current=_load(current)
    if current.size!=original.size or not validate(current,original.size)['palette_ok']:
        raise ValueError('Normalize the tile palette and preserve native dimensions before generation.')
    # Never accept pre-existing context edits as a generation input.
    import numpy as np
    outside=np.ones((current.height,current.width),bool)
    x0,y0,x1,y1=tile['core_local'];outside[y0:y1,x0:x1]=False
    if np.any(np.any(np.asarray(current)!=np.asarray(original),axis=2)&outside):
        raise ValueError('Context border is read-only; undo halo edits before proposing.')
    sketch=_load(directory/'sketch.png');local_sketch=sketch.crop(tile['context'])
    side=max(current.size);left=(side-current.width)//2;top=(side-current.height)//2
    def square(im):
        canvas=Image.new('RGB',(side,side),'white');canvas.paste(im,(left,top))
        return canvas.resize((1024,1024),Image.Resampling.NEAREST)
    mask=Image.new('RGBA',(side,side),(255,255,255,255))
    ImageDraw.Draw(mask).rectangle((left+x0,top+y0,left+x1-1,top+y1-1),fill=(255,255,255,0))
    mask=mask.resize((1024,1024),Image.Resampling.NEAREST)
    prompt=POLICY+f'''
This task is a LOCAL CORRECTION of an existing dense Step_3, not a new design.
Image 1: current Step_3 tile on a registered square canvas. Edit only the transparent mask area.
Image 2: same tile's Step_2 sketch on the identical square canvas.
Image 3: full original sketch for motif identity. Image 4: full AI design for composition context only.
Native tile dimensions: {current.width}x{current.height}. Editable local core: {tile['core_local']}.
Tile global context bounds: {tile['context']}; full dimensions {manifest['width']}x{manifest['height']}.
Preserve canvas registration, surrounding halo, motif identity, positions and established dense fill.
Focus on complete paired front/back staircase contours and tips: front shorter than back;
prefer 3/2/1 with n3<n2<n1; use 2/1 with n2<n1 if size/shape prevents 3/2/1.
Counts and two-pixel gaps refer to NATIVE tile pixels, not the 1024px canvas.
Do not redraw neighbouring motifs, add density for its own sake, move ornaments or invent missing geometry.
If a complete affected curve cannot be inferred safely from the local and full context, leave it unchanged for designer review.
Only pure red/white. The result will be reviewed; do not claim correctness or add labels.
'''
    result=client.images.edit(model=model,image=[('current_tile.png',_png(square(current)),'image/png'),
        ('sketch_tile.png',_png(square(local_sketch)),'image/png'),('full_sketch.png',_png(sketch),'image/png'),
        ('full_design.png',_png(base),'image/png')],mask=('editable_core.png',_png(mask),'image/png'),
        prompt=prompt,size='1024x1024',quality='high')
    if not result.data or not result.data[0].b64_json:raise RuntimeError('Image API returned no base64 image.')
    raw=base64.b64decode(result.data[0].b64_json,validate=True)
    with Image.open(BytesIO(raw)) as im:
        if im.size!=(1024,1024):raise ValueError('Unexpected generated canvas size; refusing unregistered merge.')
        restored=im.convert('RGB').resize((side,side),Image.Resampling.NEAREST).crop((left,top,left+current.width,top+current.height))
    restored=exact_palette(restored)
    # Soft API masks are NOT trusted as a hard ownership constraint.
    candidate=original.copy();candidate.paste(restored.crop(tile['core_local']),(x0,y0))
    folder=directory/'ai_proposals'/key;folder.mkdir(parents=True,exist_ok=True)
    revisions=[int(p.name) for p in folder.iterdir() if p.is_dir() and p.name.isdigit()]
    revision=max(revisions,default=0)+1;out=folder/f'{revision:04d}';out.mkdir()
    candidate.save(out/'candidate.png');current.save(out/'input.png');(out/'raw_generated.png').write_bytes(raw)
    metadata=dict(status='unreviewed_ai_tile_proposal',tile_id=key,revision=revision,requested_model=model,
                  full_draft_rgb_sha256=manifest['draft_rgb_sha256'],context=tile['context'],core=tile['core_local'],
                  native_size=list(candidate.size),candidate_rgb_sha256=_digest(candidate),input_rgb_sha256=_digest(current),prompt=prompt,
                  semantic_hierarchy_verified=False,automatic_merge_allowed=False,
                  note='Halo restored exactly by code. Size/palette checks do not validate hierarchy, composition, or seams.')
    (out/'proposal.json').write_text(json.dumps(metadata,indent=2))
    return candidate,metadata


def proposal_input(directory,key,revision,manifest):
    """Resolve a saved AI candidate as the before image of a human correction."""
    if type(revision) is not int or revision<1:raise ValueError('Invalid AI proposal revision.')
    folder=Path(directory)/'ai_proposals'/key/f'{revision:04d}'
    metadata=json.loads((folder/'proposal.json').read_text())
    if metadata['tile_id']!=key or metadata['full_draft_rgb_sha256']!=manifest['draft_rgb_sha256']:
        raise ValueError('AI proposal does not belong to this source/tile.')
    tile=next(t for t in manifest['tiles'] if t['id']==key)
    if metadata['context']!=tile['context'] or metadata['core']!=tile['core_local']:
        raise ValueError('AI proposal belongs to different tile bounds.')
    candidate=_load(folder/'candidate.png')
    if _digest(candidate)!=metadata['candidate_rgb_sha256']:
        raise ValueError('AI proposal changed after generation; do not overwrite proposal inputs.')
    return candidate,metadata
