"""Whole-raster hierarchy experiment: discover, correct feasible arcs, audit others.

Automatic geometry cannot establish semantic exemptions or front/back identities
for every motif. This reports scan coverage separately from rule compliance.
"""
from collections import defaultdict
import numpy as np
from scipy import ndimage
from .pixel_hierarchy import contour_runs,check_runs,plan_counts,plan_counts_21,replace_trace,trace_on_boundary
from .separation import conflicts


def boundary_loops(mask):
    """Clockwise pixel-grid boundary loops, including internal white holes."""
    h,w=mask.shape;edges=set()
    for y,x in zip(*np.nonzero(mask)):
        x=int(x);y=int(y)
        if y==0 or not mask[y-1,x]:edges.add(((x,y),(x+1,y)))
        if x==w-1 or not mask[y,x+1]:edges.add(((x+1,y),(x+1,y+1)))
        if y==h-1 or not mask[y+1,x]:edges.add(((x+1,y+1),(x,y+1)))
        if x==0 or not mask[y,x-1]:edges.add(((x,y+1),(x,y)))
    outgoing=defaultdict(list)
    for a,b in edges:outgoing[a].append(b)
    def direction(a,b):
        dx,dy=b[0]-a[0],b[1]-a[1]
        return 0 if dx>0 else 1 if dy>0 else 2 if dx<0 else 3
    remaining=set(edges);loops=[]
    while remaining:
        first=min(remaining);a,b=first;points=[a];previous=direction(a,b)
        while True:
            remaining.remove((a,b));points.append(b)
            if b==points[0]:break
            options=[c for c in outgoing[b] if (b,c) in remaining]
            if not options:raise ValueError('Boundary graph cannot be closed.')
            # At a diagonal contact turn right, retaining pixel-side ownership.
            priority={1:0,0:1,3:2,2:3}
            c=min(options,key=lambda c:(priority[(direction(b,c)-previous)%4],c))
            a,b=b,c;previous=direction(a,b)
        loops.append(points)
    return loops


def compress(points):
    segments=[]
    for a,b in zip(points,points[1:]):
        d=(int(np.sign(b[0]-a[0])),int(np.sign(b[1]-a[1])))
        if segments and segments[-1][2]==d:segments[-1]=(segments[-1][0],b,d)
        else:segments.append((a,b,d))
    if len(segments)>1 and segments[0][2]==segments[-1][2]:
        segments[0]=(segments[-1][0],segments[0][1],segments[0][2]);segments.pop()
    return segments


def traces(loop):
    for oriented in [loop,list(reversed(loop))]:
        segments=compress(oriented);n=len(segments)
        for i,(a,b,direction) in enumerate(segments):
            anchor=abs(b[0]-a[0])+abs(b[1]-a[1])
            if anchor<3:continue
            points=[list(a),list(b)];connector=None
            for k in range(1,n-1,2):
                cross=segments[(i+k)%n];run=segments[(i+k+1)%n]
                length=abs(cross[1][0]-cross[0][0])+abs(cross[1][1]-cross[0][1])
                if length!=1 or run[2]!=direction or cross[2]==direction:break
                if connector is not None and cross[2]!=connector:break
                connector=cross[2]
                points.extend([list(cross[1]),list(run[1])])
            if len(points)>2:
                yield points,'x' if direction[0] else 'y'


def pair_geometry(mask):
    """Heuristic crescent profile pairing; ambiguous shapes remain unassigned."""
    ys,xs=np.nonzero(mask)
    if not len(ys):return None
    axis='y' if np.ptp(ys)>=np.ptp(xs) else 'x'
    a=mask if axis=='y' else mask.T
    rows=np.where(a.any(1))[0]
    if len(rows)<12:return None
    left=[];right=[]
    for row in rows:
        cols=np.where(a[row])[0]
        if np.any(np.diff(cols)!=1):return None
        left.append(int(cols.min()));right.append(int(cols.max()+1))
    mid=len(rows)//2
    def bend(values):
        chord=np.linspace(values[0],values[-1],len(values))
        residual=np.array(values)-chord
        return float(np.median(residual[len(rows)//3:2*len(rows)//3]))
    bl,br=bend(left),bend(right)
    if bl>2 and br>2:front,back=left,right
    elif bl<-2 and br<-2:front,back=right,left
    else:return None
    def longest(values):
        best=length=1
        for previous,current in zip(values,values[1:]):
            length=length+1 if previous==current else 1;best=max(best,length)
        return best
    f,b=longest(front),longest(back)
    return dict(axis=axis,front_anchor=f,back_anchor=b,passed=f<b,
                assignment='Geometric concavity heuristic; requires semantic confirmation.')


def _proposal(points,axis,runs):
    primary=0 if axis=='x' else 1;secondary=1-primary
    sign=1 if points[1][primary]>points[0][primary] else -1
    cross=points[2][secondary]-points[1][secondary]
    p=points[0].copy();result=[p.copy()]
    for i,length in enumerate(runs):
        if i:p[secondary]+=cross;result.append(p.copy())
        p[primary]+=sign*length;result.append(p.copy())
    return result


def _guard_mask(points,shape):
    protected=np.zeros(shape,bool);h,w=shape
    for (x0,y0),(x1,y1) in zip(points,points[1:]):
        if x0==x1:
            for y in range(min(y0,y1),max(y0,y1)):
                for x in [x0-1,x0]:
                    if 0<=y<h and 0<=x<w:protected[y,x]=True
        else:
            for x in range(min(x0,x1),max(x0,x1)):
                for y in [y0-1,y0]:
                    if 0<=y<h and 0<=x<w:protected[y,x]=True
    return protected


def run_global(labels,max_displacement=2,allow_21=True):
    if type(max_displacement) is not int or not 0<=max_displacement<=4:
        raise ValueError('Displacement must be 0–4 native pixels.')
    labels=np.asarray(labels)
    if labels.ndim!=2 or labels.dtype.kind not in 'iu' or np.any(labels<0):
        raise ValueError('Expected nonnegative integer semantic/provisional labels.')
    if conflicts(labels):raise ValueError('Input violates labelled two-pixel clearance.')
    result=labels.copy();ids=sorted(set(np.unique(labels))-{0})
    candidates=[];records=[];pieces=[]
    for value in ids:
        value=int(value);mask=labels==value
        if ndimage.label(mask,np.ones((3,3)))[1]!=1:
            raise ValueError(f'Piece {value} is disconnected.')
        loops=boundary_loops(mask);ys,xs=np.nonzero(mask)
        width=int(np.ptp(xs)+1);height=int(np.ptp(ys)+1)
        compact=(min(width,height)/max(width,height)>0.65 and mask.sum()/(width*height)>0.65)
        pieces.append(dict(piece=value,boundary_loops=len(loops),
                           treatment='compact_shape_review_no_auto_edit' if compact else 'staircase_scan',
                           pairing_before=pair_geometry(mask)))
        seen=set()
        for loop in loops:
            for points,axis in traces(loop):
                key=tuple(map(tuple,points))
                if key in seen:continue
                seen.add(key)
                runs=contour_runs(points,axis)
                entry=dict(piece=value,axis=axis,points=points,before_runs=runs)
                # All discovered arcs are inventoried, including short ones.
                if compact:entry['status']='compact_shape_requires_semantic_exemption_review'
                elif len(runs)-1<(3 if allow_21 else 6):entry['status']='too_short_for_allowed_counts'
                elif check_runs(runs)['passed']:entry.update(status='already_ordered',mode='3-2-1')
                elif any(x in [0,labels.shape[1]] or y in [0,labels.shape[0]] for x,y in points):
                    entry['status']='frame_boundary_review'
                else:
                    entry['status']='pending';candidates.append(entry)
                records.append(entry)
    candidates.sort(key=lambda c:(-sum(c['before_runs']),c['piece'],c['points']))
    protected=np.zeros(labels.shape,bool);accepted=[];retained=[]
    for entry in records:
        if entry['status']=='already_ordered':protected|=_guard_mask(entry['points'],labels.shape)
    # Prefer 3/2/1 across the WHOLE image before attempting any 2/1 fallback.
    work=[(0,e) for e in candidates]+([(1,e) for e in candidates] if allow_21 else [])
    for stage,entry in work:
        if entry['status'] in ('corrected','already_ordered_fallback_21','overlap_with_previous_edit'):continue
        if stage==1:
            entry['preferred_321_outcome']=entry['status']
            entry['preferred_321_reasons']=entry.get('reasons',[])
        planner=plan_counts if stage==0 else plan_counts_21
        piece=entry['piece'];points=entry['points'];axis=entry['axis'];runs=entry['before_runs']
        if not trace_on_boundary(result==piece,points):
            entry['status']='overlap_with_previous_edit';continue
        options=[]
        for shift in range(-max_displacement,max_displacement+1):
            anchor=runs[0]+shift
            if anchor<3:continue
            for plan in planner(sum(runs)-anchor,len(runs)-1):
                proposed=[anchor]+plan['runs']
                deviation=np.abs(np.cumsum(runs)-np.cumsum(proposed))
                if deviation.max()<=max_displacement:
                    options.append((int(deviation.sum()),abs(shift),proposed))
        options.sort()
        if not options:entry['status']='no_endpoint_preserving_plan_within_envelope';continue
        reasons=[];before_pair=pair_geometry(result==piece)
        old_holes=ndimage.label(~np.pad(result==piece,1),np.ones((3,3)))[1]
        for _,_,proposed in options:
            new_points=_proposal(points,axis,proposed)
            try:
                changed,report=replace_trace(result,piece,points,new_points,axis,preserve_anchor=False,allow_21=bool(stage))
                delta=changed!=result
                if np.any(delta&protected):raise ValueError('Would invalidate a previously verified trace.')
                if ndimage.label(~np.pad(changed==piece,1),np.ones((3,3)))[1]!=old_holes:
                    raise ValueError('Would change internal white-channel topology.')
                after_pair=pair_geometry(changed==piece)
                if before_pair and before_pair['passed'] and (not after_pair or not after_pair['passed']):
                    raise ValueError('Would lose an existing front/back anchor relationship.')
            except ValueError as exc:
                reasons.append(str(exc));continue
            result=changed;protected|=_guard_mask(new_points,labels.shape)
            entry.update(status='corrected' if report['changed_pixels'] else 'already_ordered_fallback_21',after_runs=proposed,after_points=new_points,
                         mode='3-2-1' if stage==0 else '2-1',changed_pixels=report['changed_pixels'])
            (accepted if report['changed_pixels'] else retained).append(entry);break
        else:
            entry.update(status='geometry_or_clearance_conflict',reasons=sorted(set(reasons)))
    for entry in accepted+retained:
        if not check_runs(entry['after_runs'],allow_21=allow_21)['passed']:
            raise RuntimeError('Verified contour no longer passes hierarchy.')
        if not trace_on_boundary(result==entry['piece'],entry['after_points']):
            raise RuntimeError('A later correction invalidated a verified trace.')
    for item in pieces:item['pairing_after']=pair_geometry(result==item['piece'])
    statuses={name:sum(r['status']==name for r in records) for name in sorted({r['status'] for r in records})}
    final_violations=0;final_complete_arcs=0;final_modes={'3-2-1':0,'2-1':0}
    for value in ids:
        for loop in boundary_loops(result==value):
            for points,axis in traces(loop):
                runs=contour_runs(points,axis)
                if len(runs)-1>=(3 if allow_21 else 6):
                    final_complete_arcs+=1
                    verdict=check_runs(runs,allow_21=allow_21)
                    final_violations+=not verdict['passed']
                    if verdict['passed']:final_modes[verdict['mode']]+=1
    report=dict(status='whole_image_best_effort_experiment',width=labels.shape[1],height=labels.shape[0],
                scanned_regions=len(ids),boundary_loops=sum(p['boundary_loops'] for p in pieces),
                discovered_arcs=len(records),arc_outcomes=statuses,corrected_arcs=len(accepted),
                fallback_21_enabled=allow_21,mode_priority='3-2-1 whole-image phase, then 2-1 fallback phase',
                corrected_arcs_by_mode={m:sum(e['mode']==m for e in accepted) for m in ['3-2-1','2-1']},
                retained_fallback_21_arcs=len(retained),final_geometric_passing_modes=final_modes,
                regions_edited=len({e['piece'] for e in accepted}),
                changed_pixels=int(np.count_nonzero(result!=labels)),
                max_boundary_displacement=max_displacement,
                final_geometric_complete_arcs=final_complete_arcs,final_geometric_hierarchy_violations=final_violations,
                two_pixel_clearance_passed=not conflicts(result),
                all_corrected_traces_reverified=True,image_wide_rule_complete=False,
                limitations=['A whole-image scan is not whole-image compliance.',
                            'Compact shapes, semantic exemptions, joined semantic pieces and front/back assignments need review.',
                            'Short/infeasible arcs were retained rather than distorting motifs or closing gaps.',
                            'Geometric pairing is a heuristic; unresolved anchor comparisons are not silently passed.'],
                pieces=pieces,arcs=records)
    return result,report
