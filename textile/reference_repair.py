"""Reference-assisted local contour fitting for a paired-example experiment.

This uses the designer TARGET as an optimization objective. It is not an
inference model and cannot run on unseen sketches without an approved target.
Candidates move short straight boundary runs by one native pixel, not arbitrary
copies of reference pixels. Approved hierarchy traces and clearance are guarded.
"""
import numpy as np
from scipy import ndimage
from .global_hierarchy import boundary_loops,compress,traces,pair_geometry,_guard_mask
from .pixel_hierarchy import contour_runs,check_runs,trace_on_boundary
from .separation import conflicts


def fit_reference(labels,target,max_sweeps=3):
    a=np.asarray(labels)
    target=np.asarray(target)
    if a.ndim!=2 or a.dtype.kind not in 'iu' or target.shape!=a.shape or target.dtype!=bool:
        raise ValueError('Require integer piece labels and a same-size boolean target.')
    if conflicts(a):raise ValueError('Input needs labelled two-pixel clearance.')
    if type(max_sweeps) is not int or not 1<=max_sweeps<=5:
        raise ValueError('max_sweeps must be 1–5.')
    result=a.copy();h,w=a.shape;protected=np.zeros(a.shape,bool)
    ids=sorted(int(k) for k in np.unique(a) if k)
    verified=[];pieces={};skipped_compact=[]
    for k in ids:
        mask=a==k;ys,xs=np.nonzero(mask)
        if ndimage.label(mask,np.ones((3,3)))[1]!=1:raise ValueError('Disconnected label.')
        width=int(np.ptp(xs)+1);height=int(np.ptp(ys)+1)
        compact=min(width,height)/max(width,height)>0.65 and mask.sum()/(width*height)>0.65
        if compact:skipped_compact.append(k)
        # Every currently passing trace remains at the same pixel-grid location.
        for loop in boundary_loops(mask):
            for points,axis in traces(loop):
                if check_runs(contour_runs(points,axis),allow_21=True)['passed']:
                    protected|=_guard_mask(points,a.shape)
                    verified.append((k,points))
        edge=mask&~ndimage.binary_erosion(mask,np.ones((3,3)),border_value=0)
        pieces[k]=dict(envelope=ndimage.distance_transform_cdt(~edge,metric='chessboard')<=2,
                       holes=ndimage.label(~np.pad(mask,1),np.ones((3,3)))[1],
                       compact=compact)
    operations=[];outcomes={};history=[]
    def reject(reason):outcomes[reason]=outcomes.get(reason,0)+1
    for sweep in range(max_sweeps):
        candidates=[]
        for k in ids:
            if pieces[k]['compact']:continue
            for loop in boundary_loops(result==k):
                for start,end,direction in compress(loop):
                    length=abs(end[0]-start[0])+abs(end[1]-start[1])
                    if not 2<=length<=32:continue
                    for grow in [True,False]:
                        coords=[]
                        if start[1]==end[1]:
                            y=start[1]+(-1 if direction[0]>0 else 0) if grow else start[1]+(0 if direction[0]>0 else -1)
                            coords=[(y,x) for x in range(min(start[0],end[0]),max(start[0],end[0]))]
                        else:
                            x=start[0]+(0 if direction[1]>0 else -1) if grow else start[0]+(-1 if direction[1]>0 else 0)
                            coords=[(y,x) for y in range(min(start[1],end[1]),max(start[1],end[1]))]
                        if any(not(0<=y<h and 0<=x<w) for y,x in coords):continue
                        desired=sum(bool(target[y,x])==grow for y,x in coords)
                        gain=2*desired-len(coords)
                        # A broad supporting target band, not isolated target-pixel copying.
                        if desired/len(coords)<0.8 or gain<=0:continue
                        candidates.append((-gain,k,start,end,grow,coords))
        candidates.sort();accepted=0
        for neg_gain,k,start,end,grow,coords in candidates:
            ys=np.array([p[0] for p in coords]);xs=np.array([p[1] for p in coords])
            if np.any(protected[ys,xs]):reject('protected_hierarchy');continue
            if not np.all(pieces[k]['envelope'][ys,xs]):reject('outside_two_pixel_original_envelope');continue
            if grow and np.any(result[ys,xs]!=0):reject('stale_or_occupied');continue
            if not grow and np.any(result[ys,xs]!=k):reject('stale_or_occupied');continue
            if not trace_on_boundary(result==k,[list(start),list(end)]):reject('stale_boundary');continue
            y0=max(0,int(ys.min())-2);y1=min(h,int(ys.max())+3)
            x0=max(0,int(xs.min())-2);x1=min(w,int(xs.max())+3)
            if grow:
                neighbourhood=result[y0:y1,x0:x1]
                other=(neighbourhood!=0)&(neighbourhood!=k)
                blocked=ndimage.maximum_filter(other,size=5,mode='constant',cval=0)
                if np.any(blocked[ys-y0,xs-x0]):reject('two_pixel_neighbour_clearance');continue
            old=result==k;new=old.copy();new[ys,xs]=grow
            if ndimage.label(new,np.ones((3,3)))[1]!=1:reject('piece_connectivity');continue
            if ndimage.label(~np.pad(new,1),np.ones((3,3)))[1]!=pieces[k]['holes']:
                reject('internal_white_topology');continue
            before_pair=pair_geometry(old);after_pair=pair_geometry(new)
            if before_pair and before_pair['passed'] and (not after_pair or not after_pair['passed']):
                reject('front_back_anchor_regression');continue
            # Baseline verified traces are protected; newly passing traces are
            # added after each sweep and checked again on the final raster.
            result[ys,xs]=k if grow else 0
            accepted+=1
            operations.append(dict(sweep=sweep+1,piece=k,operation='grow_boundary_strip' if grow else 'trim_boundary_strip',
                                   edge_start=list(start),edge_end=list(end),pixels=len(coords),
                                   reference_mismatch_reduction=-neg_gain))
        # Preserve additional verified sequences created by this sweep.
        for k in ids:
            for loop in boundary_loops(result==k):
                for points,axis in traces(loop):
                    if check_runs(contour_runs(points,axis),allow_21=True)['passed']:
                        protected|=_guard_mask(points,a.shape);verified.append((k,points))
        history.append(dict(sweep=sweep+1,accepted_strips=accepted,mismatches=int(((result>0)!=target).sum())))
        if not accepted:break
    if conflicts(result):raise RuntimeError('Final clearance check failed.')
    for k,points in verified:
        if not trace_on_boundary(result==k,points):raise RuntimeError('A protected hierarchy trace was lost.')
    for k in ids:
        if ndimage.label(result==k,np.ones((3,3)))[1]!=1:raise RuntimeError('Final connectivity check failed.')
    before_error=int(((a>0)!=target).sum());after_error=int(((result>0)!=target).sum())
    if after_error>before_error:raise RuntimeError('Reference fitting regressed similarity.')
    return result,dict(status='reference_assisted_contour_fitting_experiment',
                       uses_designer_target=True,unseen_inference_accuracy_claim=False,
                       source_mismatches=before_error,result_mismatches=after_error,
                       mismatch_reduction=before_error-after_error,
                       changed_pixels=int((a!=result).sum()),regions_edited=len({o['piece'] for o in operations}),
                       accepted_strips=len(operations),max_boundary_displacement=2,
                       sweeps=history,rejections=outcomes,
                       compact_regions_not_edited=skipped_compact,
                       verified_trace_instances=len(verified),verified_traces_preserved=True,
                       labelled_two_pixel_clearance_passed=True,
                       full_semantic_rule_compliance=False,operations=operations,
                       limitations=['Designer target was required: this is supervised example repair, not a learned model.',
                                    'Existing unresolved joins, anchor failures and hierarchy issues remain for review.',
                                    'Geometry checks preserve red connectivity and internal white-region counts, not semantic proof.'])
