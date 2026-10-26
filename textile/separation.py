"""Two-native-pixel minimum clearance between explicitly labelled pieces.

Labels are semantic annotations, not a claim that connected components are pieces.
White = 0; each continuous semantic piece has its own positive integer label.
Clearance uses the 8-neighbour/chessboard metric: red pixel centres must be at
least 3 grid steps apart to leave two white pixels at the closest approach.
Wider background is retained. No filling, resizing or taper correction is done.
"""
import numpy as np
from scipy import ndimage


def conflicts(labels, gap=2):
    h,w=labels.shape
    pairs=[]
    for dy in range(gap+1):
        for dx in range(-gap,gap+1):
            if dy==0 and dx<=0:continue
            y0=max(0,-dy); y1=min(h,h-dy)
            x0=max(0,-dx); x1=min(w,w-dx)
            a=labels[y0:y1,x0:x1]; b=labels[y0+dy:y1+dy,x0+dx:x1+dx]
            yy,xx=np.nonzero((a>0)&(b>0)&(a!=b))
            pairs.extend(((int(y+y0),int(x+x0)),(int(y+y0+dy),int(x+x0+dx))) for y,x in zip(yy,xx))
    return pairs


def _can_remove(labels, point, anchors):
    y,x=point; value=labels[y,x]
    if not value or (y,x) in anchors:return False
    # A local simple-point check preserves 8-connected red continuity.
    patch=labels[max(0,y-1):y+2,max(0,x-1):x+2]==value
    patch[y-max(0,y-1),x-max(0,x-1)]=False
    return ndimage.label(patch,np.ones((3,3)))[1]==1


def separate(labels):
    labels=np.asarray(labels)
    if labels.ndim!=2 or labels.dtype.kind not in 'iu' or np.any(labels<0):
        raise ValueError('Expected a nonnegative integer piece-label raster.')
    result=labels.copy(); anchors=set(); original_ids=set(np.unique(labels))-{0}
    for value in sorted(original_ids):
        mask=labels==value
        if ndimage.label(mask,np.ones((3,3)))[1]!=1:
            raise ValueError(f'Piece {value} is disconnected; review the annotation.')
        d=ndimage.distance_transform_edt(np.pad(mask,1))[1:-1,1:-1]
        anchors.add(tuple(np.unravel_index(d.argmax(),d.shape)))
    removed=0
    while True:
        pairs=conflicts(result)
        changed=False
        for a,b in pairs:
            if not result[a] or not result[b] or result[a]==result[b]:continue
            # Use the larger instance ID first for deterministic one-sided edits.
            candidates=sorted([a,b],key=lambda p:(int(result[p]),p),reverse=True)
            for point in candidates:
                if _can_remove(result,point,anchors):
                    result[point]=0;removed+=1;changed=True;break
        if not changed:break
    remaining=conflicts(result)
    for value in original_ids:
        if ndimage.label(result==value,np.ones((3,3)))[1]!=1:
            raise RuntimeError('Unexpected piece deletion/split; refuse result.')
    return result,dict(required_white_gap=2,metric='chessboard / 8-neighbour',
                       interpretation='Minimum 2 white pixels between labelled neighbours; wider background retained.',
                       labelled_pieces=len(original_ids),removed_red_pixels=removed,
                       unresolved_pixel_pairs=len(remaining),clearance_passed=not remaining,
                       semantic_approval=False,
                       limitation='Only labelled-piece clearance is verified. Touching semantic pieces sharing a label still need annotation review.')
