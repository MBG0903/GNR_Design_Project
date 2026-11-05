"""A/M/B native-pixel corner grammar, bounded constructor and review gates.

This follows the user's latest diagram: shrink run LENGTHS from either flat
anchor toward a shared one-pixel transition. There is no step-count dominance.
Uncertain segmentation is reported, not passed. Semantic applicability still
requires designer review. No model, notebook cell or API is executed here.
"""
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi
from PIL import Image

VERSION = '5.0.0-amb'
N8 = np.ones((3,3),bool)
N4 = np.array([[0,1,0],[1,1,1],[0,1,0]],bool)


def groups(vals):
    vals=list(vals)
    if not vals:return []
    out=[];s=0
    for i in range(1,len(vals)+1):
        if i==len(vals) or vals[i]!=vals[s]:out.append((int(vals[s]),s,i-s));s=i
    return out


def extreme_lines(vals):
    n=len(vals);g=groups(vals)
    if not n:return []
    if len(g)>1 and g[0][0]==g[-1][0]:
        v,s,l=g[-1];g=[(v,s,l+g[0][2])]+g[1:-1]
    if len(g)<3:return []
    return [(s%n,l) for i,(v,s,l) in enumerate(g)
            if (g[i-1][0]>v and g[(i+1)%len(g)][0]>v)
            or (g[i-1][0]<v and g[(i+1)%len(g)][0]<v)]


def _half_ok(runs):
    return (bool(runs) and runs[0] in (2,3,4) and runs[-1]==2
            and all(2<=v<=4 for v in runs)
            and all(a-b in (0,1) for a,b in zip(runs,runs[1:])))


def corner_check(arc,start_h,end_h):
    """Check one complete orthogonal-flat corner; no colour/image resampling."""
    result={'status':'unresolved','reason':'','A':None,'M':None,'B':None}
    p=np.asarray(arc)
    if p.ndim!=2 or p.shape[1]!=2 or len(p)<5 or not np.isfinite(p).all() or not np.equal(p,np.round(p)).all():
        return dict(result,reason='not a complete integer contour')
    p=p.astype(int)
    if start_h==end_h:return dict(result,reason='same-axis anchors / S-curve need designer subdivision')
    if not start_h:p=p[::-1]
    d=np.diff(p,axis=0)
    if (np.abs(d)>1).any() or (np.abs(d).sum(axis=1)==0).any():
        return dict(result,reason='contour is not an ordered 8-connected pixel path')
    sx=int(np.sign(p[-1,0]-p[0,0]));sy=int(np.sign(p[-1,1]-p[0,1]))
    if not sx or not sy:return dict(result,reason='missing orthogonal anchor extent')
    q=(p-p[0])*[sx,sy]
    if (np.diff(q,axis=0)<0).any():return dict(result,reason='non-monotone corner / junction')
    gy,gx=groups(q[:,1]),groups(q[:,0])
    lh,lv=gy[0][2],gx[-1][2]
    if lh<2 or lv<2:return dict(result,reason='flat anchor too short to locate reliably')
    ry=np.empty(len(q),int);rx=np.empty(len(q),int)
    for _,s,n in gy:ry[s:s+n]=n
    for _,s,n in gx:rx[s:s+n]=n
    mid=np.flatnonzero((ry==1)&(rx==1))
    mid=mid[(mid>=lh)&(mid<len(q)-lv)]
    result.update(A=p[lh-1].tolist(),B=p[len(p)-lv].tolist(),flat_lengths=[lh,lv])
    if len(mid)!=1:return dict(result,status='failed',reason='expected one shared 1-pixel transition M')
    m=int(mid[0]);result['M']=p[m].tolist()
    h=[v[2] for v in groups(q[lh:m,1])]
    v=[r[2] for r in groups(q[m+1:len(q)-lv,0])][::-1]
    result.update(horizontal_toward_M=h+[1],vertical_toward_M=v+[1])
    if not _half_ok(h) or not _half_ok(v):
        return dict(result,status='failed',reason='run lengths must hold or decrease one level toward M: 4/3/2 → … → 2 → 1')
    return dict(result,status='passed',reason='both flat-to-M halves satisfy the diagram')


def curve_ok(arc,start_h,end_h):
    return corner_check(arc,start_h,end_h)['status']=='passed'


def _arcs(points):
    n=len(points)
    ext=sorted(set([(s,l,True) for s,l in extreme_lines(points[:,1])]
                   +[(s,l,False) for s,l in extreme_lines(points[:,0])]))
    if len(ext)<2:return []
    out=[]
    for (a,la,ha),(b,lb,hb) in zip(ext,ext[1:]+[(ext[0][0]+n,ext[0][1],ext[0][2])]):
        count=b+lb-a
        if count>=n or count<5:
            out.append({'start':a,'length':count,'check':{'status':'unresolved','reason':'insufficient complete-corner anchors'},'points':None})
            continue
        arc=points[(np.arange(a,a+count)%n)]
        out.append(dict(start=a,length=count,start_h=ha,end_h=hb,points=arc,check=corner_check(arc,ha,hb)))
    return out


def curve_audit(red):
    """All detected contours are accounted for, including tiny/unsupported ones."""
    red=np.asarray(red,dtype=bool)
    contours,hierarchy=cv2.findContours(red.astype(np.uint8),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    records=[]
    for ci,c in enumerate(contours):
        pts=c[:,0,:];arcs=_arcs(pts)
        if not arcs:
            records.append(dict(contour=ci,status='unresolved',reason='no reliable orthogonal flat anchors; applicability needs review',bbox=list(cv2.boundingRect(c))))
        for i,a in enumerate(arcs):
            records.append(dict(contour=ci,corner=i,**a['check'],bbox=list(cv2.boundingRect(c))))
    if not records:
        records.append({'status':'unresolved','reason':'no red contour to assess'})
    counts=Counter(r['status'] for r in records)
    return dict(version=VERSION,passed=counts['passed'],failed=counts['failed'],unresolved=counts['unresolved'],
        detected_corners=len(records),pass_percent=round(100*counts['passed']/len(records),2) if records else None,
        records=records,semantic_applicability_verified=False,
        interpretation='Native-grid corner screening, not semantic or whole-design production certification.')


def curve_rule_pct(red,min_piece=0,min_arc=0):
    report=curve_audit(red)
    return report['pass_percent'] if report['pass_percent'] is not None else 0.0


def build_corner(horizontal,vertical,flat_h=3,flat_v=3):
    """Construct a canonical corner from two half-run lists EXCLUDING shared M.

    vertical is given from its flat toward M, as in the diagram; output is an
    ordered path beginning horizontally and ending vertically.
    """
    h=list(horizontal);v=list(vertical)
    if not _half_ok(h) or not _half_ok(v) or flat_h<2 or flat_v<2:
        raise ValueError('Invalid diagram half-runs or flat anchors.')
    pts=[[x,0] for x in range(int(flat_h))];x,y=int(flat_h)-1,0
    for length in h:
        y+=1
        for _ in range(length):x+=1;pts.append([x,y])
    x+=1;y+=1;pts.append([x,y]) # M belongs to both halves.
    for length in v[::-1]:
        x+=1
        for _ in range(length):y+=1;pts.append([x,y])
    x+=1
    for _ in range(int(flat_v)):y+=1;pts.append([x,y])
    return np.asarray(pts,int)


@lru_cache(maxsize=8)
def _sequences(max_repeat):
    out=[]
    for n4 in range(max_repeat+1):
        for n3 in range(1 if n4 else 0,max_repeat+1):
            for n2 in range(1,max_repeat+1):out.append((4,)*n4+(3,)*n3+(2,)*n2)
    return tuple(out)


def construct_corner(arc,start_h,end_h,max_shift=2,max_repeat=8):
    """Enumerate feasible A/M/B paths with fixed endpoints and bounded displacement.

    Returns the best independently checked complete path or None. This is a
    bounded local search, not a guarantee that every source curve is feasible.
    """
    original=np.asarray(arc,int);p=original if start_h else original[::-1]
    if start_h==end_h or len(p)<7:return None
    sx=int(np.sign(p[-1,0]-p[0,0]));sy=int(np.sign(p[-1,1]-p[0,1]))
    if not sx or not sy:return None
    q=(p-p[0])*[sx,sy]
    if (np.diff(q,axis=0)<0).any():return None
    if (np.abs(np.diff(q,axis=0))>1).any():return None
    lh=groups(q[:,1])[0][2];lv=groups(q[:,0])[-1][2]
    if lh<2 or lv<2:return None
    width,height=map(int,q[-1])
    if width>256 or height>256:return None # explicit bounded-search scope
    by_extent=defaultdict(list)
    seqs=_sequences(int(max_repeat))
    for v in seqs:by_extent[(len(v),sum(v))].append(v)
    # Distance field of the original boundary, for native Chebyshev displacement.
    field=np.ones((height+1,width+1),bool);field[q[:,1],q[:,0]]=False
    old_dist=ndi.distance_transform_cdt(field,metric='chessboard')
    best=None
    for h in seqs:
        sh,nh=sum(h),len(h)
        if sh>width:continue
        for fh in range(max(2,lh-max_shift),lh+max_shift+1):
            nv=width-fh-sh-1
            if nv<1:continue
            for fv in range(max(2,lv-max_shift),lv+max_shift+1):
                sv=height-nh-fv-1
                for v in by_extent.get((nv,sv),[]):
                    candidate=build_corner(h,v,fh,fv)
                    if tuple(candidate[-1])!=(width,height):continue
                    if old_dist[candidate[:,1],candidate[:,0]].max()>max_shift:continue
                    cfield=np.ones_like(field);cfield[candidate[:,1],candidate[:,0]]=False
                    d=ndi.distance_transform_cdt(cfield,metric='chessboard')
                    if d[q[:,1],q[:,0]].max()>max_shift:continue
                    score=float(old_dist[candidate[:,1],candidate[:,0]].sum()+d[q[:,1],q[:,0]].sum())
                    score+=.01*(abs(fh-lh)+abs(fv-lv))
                    if best is None or score<best[0]:
                        mapped=candidate*[sx,sy]+p[0]
                        mapped=mapped if start_h else mapped[::-1]
                        if corner_check(mapped,start_h,end_h)['status']=='passed':best=(score,mapped)
    return None if best is None else best[1]


def _on_contour(path,contour):
    p=contour[:,0,:];n=len(p)
    if len(path)>n:return False
    for candidate in (path,path[::-1]):
        for i in np.flatnonzero((p==candidate[0]).all(axis=1)):
            if np.array_equal(p[(int(i)+np.arange(len(path)))%n],candidate):return True
    return False


def _old_passes_preserved(before,after,max_shift=2):
    valid=[r for r in after['records'] if r['status']=='passed']
    for old in before['records']:
        if old['status']!='passed':continue
        if not any(all(np.max(np.abs(np.asarray(old[k])-np.asarray(new[k])))<=max_shift
                       for k in ('A','M','B')) for new in valid):return False
    return True


def transition_guard(before,after,protected_white=None):
    """For arbitrary cleanup/LLM changes, require affected full components to pass.

    Existing unrelated issues may remain, but are carried into the export report.
    Semantic identities are not inferred from connected-component counts.
    """
    a=np.asarray(before,bool);b=np.asarray(after,bool)
    if a.shape!=b.shape:return False,{'reason':'native dimensions changed'}
    change=a^b
    if not change.any():return True,{'reason':'unchanged'}
    if protected_white is not None and (b&np.asarray(protected_white,bool)).any():
        return False,{'reason':'protected white filled'}
    la,na=ndi.label(a,N8);lb,nb=ndi.label(b,N8)
    near=ndi.binary_dilation(change,N8)
    old_ids=np.unique(la[near]);old_ids=old_ids[old_ids>0]
    new_ids=np.unique(lb[near]);new_ids=new_ids[new_ids>0]
    # Require a one-to-one mapping for affected raster regions, not just equal counts.
    mapping=[]
    for oid in old_ids:
        ids=np.unique(lb[a&(la==oid)]);ids=ids[ids>0]
        if len(ids)!=1:return False,{'reason':'an affected region disappeared or split'}
        mapping.append(int(ids[0]))
    if len(set(mapping))!=len(mapping) or set(mapping)!=set(map(int,new_ids)):
        return False,{'reason':'an affected region merged or an untracked piece appeared'}
    affected=np.isin(lb,new_ids)&b
    audit=curve_audit(affected)
    if audit['failed'] or audit['unresolved']:
        return False,{'reason':'affected complete contours need staircase repair/review','audit':audit}
    for nid in new_ids:
        own=(lb==nid);other=b&~own
        if other.any():
            d=ndi.distance_transform_cdt(~other,metric='chessboard')
            if (d[own]<3).any():return False,{'reason':'fewer than two white pixels to another raster region'}
    return True,{'reason':'affected contours and raster-region clearance passed','audit':audit}


def repair_curves(red,max_shift=2,max_repeat=8,max_edits=250,protected_white=None):
    out=np.asarray(red,bool).copy();fixed=0;skips=Counter()
    # Re-extract contours after every accepted edit; never reuse stale contour paths.
    attempted=set()
    while fixed<max_edits:
        contours,hierarchy=cv2.findContours(out.astype(np.uint8),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        changed=False
        before_audit=curve_audit(out)
        for ci,c in enumerate(contours):
            p=c[:,0,:];n=len(p)
            for segment in _arcs(p):
                if segment['check']['status']=='passed':continue
                if segment['points'] is None:
                    skips['incomplete anchors']+=1;continue
                old=segment['points']
                key=(hashlib.sha256(old.tobytes()).hexdigest(),segment['start_h'],segment['end_h'])
                if key in attempted:continue
                attempted.add(key)
                candidate=construct_corner(old,segment['start_h'],segment['end_h'],max_shift,max_repeat)
                if candidate is None:
                    skips['no feasible bounded A/M/B corner']+=1;continue
                rotated=np.roll(p,-segment['start'],axis=0)
                replaced=np.concatenate([candidate,rotated[segment['length']:]],axis=0)
                proposed=[v.copy() for v in contours];proposed[ci]=replaced[:,None,:].astype(np.int32)
                trial=np.zeros(out.shape,np.uint8)
                cv2.drawContours(trial,proposed,-1,1,cv2.FILLED,hierarchy=hierarchy)
                trial=trial.astype(bool)
                if np.array_equal(trial,out):continue
                if protected_white is not None and (trial&np.asarray(protected_white,bool)).any():
                    skips['protected white']+=1;continue
                lab,_=ndi.label(out,N8)
                touched=np.unique(lab[out&~trial]);touched=touched[touched>0]
                # All additions must remain >=3 native red-centre steps from unrelated regions.
                area=ndi.binary_dilation(out^trial,N8)
                ids=np.unique(lab[area]);ids=ids[ids>0]
                if len(ids)!=1:
                    skips['ambiguous neighbouring region ownership']+=1;continue
                other=out&(lab!=ids[0])
                if other.any() and (ndi.distance_transform_cdt(~other,metric='chessboard')[trial&~out]<3).any():
                    skips['neighbour clearance']+=1;continue
                # Rendering must preserve the ACTUAL proposed pixel path.
                actual,_=cv2.findContours(trial.astype(np.uint8),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
                if not any(_on_contour(candidate,cnt) for cnt in actual):
                    skips['rendered contour differs from constructed corner']+=1;continue
                after_audit=curve_audit(trial)
                if not _old_passes_preserved(before_audit,after_audit,max_shift):
                    skips['would invalidate a previously passing corner']+=1;continue
                # Preserve raster-region/hole identities as an additional geometric guard.
                if ndi.label(out,N8)[1]!=ndi.label(trial,N8)[1] or ndi.label(~out,N4)[1]!=ndi.label(~trial,N4)[1]:
                    skips['raster topology changed']+=1;continue
                out=trial;fixed+=1;changed=True;break
            if changed:break
        if not changed:break
    return out,fixed,{'skipped':dict(skips),'limit_reached':fixed>=max_edits,'audit':curve_audit(out)}


def gap_audit(red):
    red=np.asarray(red,bool);lab,n=ndi.label(red,N8);violations=[]
    for pid,sl in enumerate(ndi.find_objects(lab),1):
        if sl is None:continue
        y0,y1=max(0,sl[0].start-3),min(red.shape[0],sl[0].stop+3)
        x0,x1=max(0,sl[1].start-3),min(red.shape[1],sl[1].stop+3)
        local=lab[y0:y1,x0:x1];other=(local>0)&(local!=pid)
        if other.any():
            d=ndi.distance_transform_cdt(~other,metric='chessboard')
            minimum=int(d[local==pid].min())-1
            if minimum<2:violations.append({'region':pid,'minimum_white_gap':minimum})
    return {'metric':'8-neighbour/Chebyshev red-centre distance minus one','narrow_region_count':len(violations),
            'violations':violations,'semantic_separation_verified':False}


def export_review(red,pair_id,namespace):
    red=np.asarray(red,bool)
    s2=namespace.get('S2',{})
    if pair_id not in s2:raise ValueError('Original Step_2 path is required to validate native dimensions.')
    with Image.open(s2[pair_id]) as src:expected=src.size
    if red.ndim!=2 or (red.shape[1],red.shape[0])!=expected:
        raise ValueError(f'Native size mismatch: expected {expected}; refusing export.')
    folder=Path(namespace['PIPE'])/'02_Review_Output_v5';folder.mkdir(parents=True,exist_ok=True)
    base=folder/f'{pair_id} GNR 3_v5_review.bmp';v=2
    while base.exists():base=folder/f'{pair_id} GNR 3_v5_review_{v}.bmp';v+=1
    a=np.full((*red.shape,3),255,np.uint8);a[red]=[255,0,0]
    im=Image.fromarray(a);im.save(base);im.save(base.with_suffix('.png'))
    curves=curve_audit(red);gaps=gap_audit(red)
    report={'version':VERSION,'source_step2':str(s2[pair_id]),'native_size':list(expected),
        'curves':curves,'gaps':gaps,'production_ready':False,
        'status':'unresolved_geometry_review' if curves['failed'] or curves['unresolved'] or gaps['narrow_region_count'] else 'geometry_screened_semantic_review_required'}
    base.with_suffix('.json').write_text(json.dumps(report,indent=2))
    namespace['V5_LAST_EXPORT_REPORT']=report
    with Image.open(base) as saved:
        check=np.asarray(saved.convert('RGB'))
        ok=saved.size==expected and np.array_equal(check,a)
    return base,base.with_suffix('.png'),bool(ok)


def install_bindings(namespace):
    """Override known legacy entry points, including the notebook's batch library."""
    namespace.update(curve_ok=curve_ok,curve_rule_pct=curve_rule_pct,curve_audit=curve_audit,extreme_lines=extreme_lines,groups=groups)
    def fix_curves(red,min_piece=0,min_arc=0):
        result,count,info=repair_curves(red,max_shift=namespace.get('MAX_SHIFT',2),max_repeat=namespace.get('MAX_REPEAT',8),
            max_edits=namespace.get('V5_MAX_CORNER_EDITS',250),protected_white=namespace.get('V5_PROTECTED_WHITE'))
        namespace['SKIPPED']=info['skipped'];namespace['V5_LAST_CURVE_AUDIT']=info['audit']
        return result,count
    def save_production(red,pair_id):return export_review(red,pair_id,namespace)
    namespace.update(fix_curves=fix_curves,save_production=save_production,V5_GEOMETRY_VERSION=VERSION)
    return VERSION


def self_test():
    good=build_corner([3,2,2],[3,2,2],11,7)
    assert curve_ok(good,True,False)
    for k in range(4):
        for flip in (False,True):
            p=good.copy()
            if flip:p[:,0]*=-1
            for _ in range(k):p=np.column_stack([-p[:,1],p[:,0]])
            assert curve_ok(p,k%2==0,k%2!=0)
    for start in (2,3,4):
        half=list(range(start,1,-1))
        assert curve_ok(build_corner(half,half),True,False)
    bad=good.copy();bad[12,1]+=1
    assert not curve_ok(bad,True,False)
    repairable=good.copy()
    repairable[(repairable[:,0]==16)&(repairable[:,1]==3),1]=2
    assert not curve_ok(repairable,True,False)
    repaired=construct_corner(repairable,True,False,max_repeat=3)
    assert repaired is not None and curve_ok(repaired,True,False)
    assert np.array_equal(repaired[[0,-1]],repairable[[0,-1]])
    return {'version':VERSION,'checks':'diagram, 8 orientations, 2/3/4 starts, invalid-path rejection and bounded construction passed',
            'note':'Synthetic geometry tests only; no training, API calls or semantic production certification.'}
