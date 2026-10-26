"""Validate explicitly selected native-pixel contour staircases.

This is a contour run-length rule, not a stroke-width rule. A trace follows
pixel-grid edges from its long anchor toward one tip/turn. Each trace covers
one monotone staircase; split at direction reversals. Front/back assignment
is supplied by the designer, never inferred from component counts or lengths.
"""
import numpy as np


def contour_runs(points, axis):
    if axis not in ('x','y'):
        raise ValueError('axis must be x or y (the long run direction).')
    if not isinstance(points,list) or len(points)<2:
        raise ValueError('At least two grid-edge vertices are required.')
    if any(not isinstance(p,list) or len(p)!=2 or any(type(v) is not int for v in p) for p in points):
        raise ValueError('Vertices must be native integer [x,y] coordinates.')
    segments=[]
    for a,b in zip(points,points[1:]):
        dx,dy=b[0]-a[0],b[1]-a[1]
        if (dx==0)==(dy==0):
            raise ValueError('Trace segments must be nonzero and axis-aligned.')
        direction=('x',1 if dx>0 else -1) if dx else ('y',1 if dy>0 else -1)
        length=abs(dx)+abs(dy)
        if segments and segments[-1][:2]==direction:
            segments[-1]=(direction[0],direction[1],segments[-1][2]+length)
        else:segments.append((*direction,length))
    if segments[0][0]!=axis or segments[-1][0]!=axis:
        raise ValueError('Start with the anchor and end with a final run on the same axis.')
    signs={}
    for direction,sign,length in segments:
        if direction in signs and signs[direction]!=sign:
            raise ValueError('Direction reversal: split the contour at its turn.')
        signs[direction]=sign
        if direction!=axis and length!=1:
            raise ValueError('Staircase connectors must each advance one pixel.')
    return [length for direction,_,length in segments if direction==axis]


def trace_on_boundary(mask, points):
    """Check grid edges against actual red/white pixel boundaries, not a preview."""
    a=np.asarray(mask,dtype=bool)
    if a.ndim!=2:raise ValueError('Expected a two-dimensional red mask.')
    h,w=a.shape
    if any(not (0<=x<=w and 0<=y<=h) for x,y in points):return False
    def red(x,y):
        return bool(a[y,x]) if 0<=x<w and 0<=y<h else False
    for (x0,y0),(x1,y1) in zip(points,points[1:]):
        if y0==y1:
            for x in range(min(x0,x1),max(x0,x1)):
                if red(x,y0-1)==red(x,y0):return False
        elif x0==x1:
            for y in range(min(y0,y1),max(y0,y1)):
                if red(x0-1,y)==red(x0,y):return False
        else:return False
    return True


def check_runs(runs, allow_21=False):
    if not isinstance(runs,list) or len(runs)<2 or any(type(n) is not int or n<1 for n in runs):
        raise ValueError('Supply an anchor followed by positive integer run lengths.')
    anchor,tail=runs[0],runs[1:]
    counts={str(n):tail.count(n) for n in (3,2,1)}
    order=all(n in (3,2,1) for n in tail) and tail==sorted(tail,reverse=True)
    all_levels=all(counts[str(n)]>0 for n in (3,2,1))
    dominance=counts['3']<counts['2']<counts['1']
    strict_pass=order and all_levels and dominance
    fallback_pass=(allow_21 and all(n in (2,1) for n in tail)
                   and tail==sorted(tail,reverse=True) and 0<counts['2']<counts['1'])
    return dict(anchor_length=anchor,tail=tail,counts=counts,
                order_3_then_2_then_1=order,all_levels_present=all_levels,
                increasing_step_counts=dominance,
                mode='3-2-1' if strict_pass else '2-1' if fallback_pass else None,
                passed=bool(strict_pass or fallback_pass))



def check_pair(front_points, back_points, axis, mask=None, allow_21=False):
    front=check_runs(contour_runs(front_points,axis),allow_21=allow_21)
    back=check_runs(contour_runs(back_points,axis),allow_21=allow_21)
    comparison=front['anchor_length']<back['anchor_length']
    on_boundary=None if mask is None else all(trace_on_boundary(mask,p) for p in [front_points,back_points])
    return dict(front=front,back=back,front_anchor_shorter_than_back=comparison,
                traces_match_raster=on_boundary,
                sequence_checks_passed=bool(front['passed'] and back['passed'] and comparison),
                annotated_raster_checks_passed=bool(front['passed'] and back['passed'] and comparison and on_boundary is True),
                image_wide_approval=False,
                scope='Only the two supplied traces. Semantic front/back assignment and all other curves need review.')


def plan_counts(span, steps):
    """Find 3/2/1 counts preserving a selected tail's axial span and step count.

    Return no solution rather than move endpoints, skip a level or violate the
    strict count hierarchy. This proposes counts only; it does not edit pixels.
    """
    if type(span) is not int or type(steps) is not int or span<1 or steps<1:
        raise ValueError('span and steps must be positive integers.')
    plans=[]
    for n3 in range(1,steps):
        for n2 in range(n3+1,steps-n3):
            n1=steps-n3-n2
            if n1>n2 and 3*n3+2*n2+n1==span:
                plans.append(dict(n3=n3,n2=n2,n1=n1,runs=[3]*n3+[2]*n2+[1]*n1))
    return plans


def plan_counts_21(span, steps):
    """Fallback two-level counts, with strictly more 1-pixel than 2-pixel runs."""
    if type(span) is not int or type(steps) is not int or span<1 or steps<1:
        raise ValueError('span and steps must be positive integers.')
    n2=span-steps; n1=steps-n2
    return [dict(n3=0,n2=n2,n1=n1,runs=[2]*n2+[1]*n1)] if 0<n2<n1 else []


def replace_trace(labels, piece_id, old_points, new_points, axis, preserve_anchor=True, allow_21=False):
    """Replace one annotated contour, preserving endpoints, labels and clearance.

    Both traces are grid-edge vertices. Fill their symmetric-difference polygon
    by pixel-centre sampling. Refuse edits touching another piece or breaking
    connectivity. This validates one selected edge, never all image curves.
    """
    from scipy import ndimage
    from .separation import conflicts
    a=np.asarray(labels)
    if a.ndim!=2 or a.dtype.kind not in 'iu':raise ValueError('Expected integer labels.')
    if old_points[0]!=new_points[0] or old_points[-1]!=new_points[-1]:
        raise ValueError('Correction must preserve both contour endpoints.')
    old_runs=contour_runs(old_points,axis);new_runs=contour_runs(new_points,axis)
    if preserve_anchor and (old_runs[0]!=new_runs[0] or old_points[1]!=new_points[1]):
        raise ValueError('This correction must preserve the long anchor.')
    hierarchy=check_runs(new_runs,allow_21=allow_21)
    if not hierarchy['passed']:raise ValueError('Proposed contour fails the allowed hierarchy.')
    piece=a==piece_id
    if not piece.any() or not trace_on_boundary(piece,old_points):
        raise ValueError('Original trace does not match the selected piece.')
    if conflicts(a):raise ValueError('Resolve existing labelled-piece gap conflicts first.')
    h,w=a.shape
    if any(not(0<=x<=w and 0<=y<=h) for x,y in new_points):
        raise ValueError('New contour outside native canvas.')
    polygon=old_points+new_points[::-1]
    yy,xx=np.mgrid[:h,:w];xs=xx+0.5;ys=yy+0.5
    toggle=np.zeros((h,w),bool)
    for p,q in zip(polygon,polygon[1:]+polygon[:1]):
        if p[1]==q[1]:continue
        cross=((p[1]>ys)!=(q[1]>ys)) & (xs<(q[0]-p[0])*(ys-p[1])/(q[1]-p[1])+p[0])
        toggle^=cross
    if np.any(toggle & (a!=0) & (a!=piece_id)):
        raise ValueError('Correction would alter a neighbouring piece.')
    updated=piece^toggle
    if ndimage.label(updated,np.ones((3,3)))[1]!=1:
        raise ValueError('Correction would split or delete the piece.')
    result=a.copy();result[piece]=0;result[updated]=piece_id
    if conflicts(result):raise ValueError('Correction would violate two-pixel separation.')
    if not trace_on_boundary(updated,new_points):
        raise ValueError('Rasterized correction does not match the proposed contour.')
    return result,dict(before_runs=old_runs,after_runs=new_runs,hierarchy=hierarchy,
                       changed_pixels=int(toggle.sum()),endpoints_preserved=True,
                       anchor_before=old_runs[0],anchor_after=new_runs[0],
                       two_pixel_clearance_preserved=True,image_wide_approval=False)
