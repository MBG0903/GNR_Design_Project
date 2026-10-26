"""Native-coordinate binary-raster comparison; no alignment or resizing."""
import numpy as np
from PIL import Image
from scipy import ndimage
from .core import rgb,validate


def compare(designer, candidate):
    a=rgb(designer);b=rgb(candidate)
    if a.size!=b.size:raise ValueError('Dimensions differ; comparison refuses resizing.')
    for name,im in [('designer',a),('candidate',b)]:
        if not validate(im,a.size)['palette_ok']:
            raise ValueError(f'{name} must use only pure red/white.')
    truth=np.all(np.asarray(a)==[255,0,0],axis=2)
    pred=np.all(np.asarray(b)==[255,0,0],axis=2)
    tp=int((truth&pred).sum());tn=int((~truth&~pred).sum())
    fp=int((~truth&pred).sum());fn=int((truth&~pred).sum())
    def ratio(n,d):return float(n/d) if d else 1.0
    ta=truth&~ndimage.binary_erosion(truth,np.ones((3,3)),border_value=0)
    pb=pred&~ndimage.binary_erosion(pred,np.ones((3,3)),border_value=0)
    boundary=None
    if ta.any() and pb.any():
        candidate_to_designer=ndimage.distance_transform_edt(~ta)[pb]
        designer_to_candidate=ndimage.distance_transform_edt(~pb)[ta]
        combined=np.concatenate([candidate_to_designer,designer_to_candidate])
        boundary=dict(symmetric_mean_distance_pixels=float(combined.mean()),
                      symmetric_95th_percentile_pixels=float(np.percentile(combined,95)),
                      maximum_distance_pixels=float(combined.max()),
                      symmetric_fraction_within_1_pixel=float((combined<=1).mean()),
                      symmetric_fraction_within_2_pixels=float((combined<=2).mean()),
                      definition='Red-side 8-neighbour boundaries; symmetric Euclidean nearest-edge distances, no alignment.')
    report=dict(width=a.width,height=a.height,total_pixels=truth.size,
                matched_red_pixels=tp,matched_white_pixels=tn,
                excess_red_pixels=fp,missing_red_pixels=fn,mismatched_pixels=fp+fn,
                pixel_agreement=ratio(tp+tn,truth.size),
                red_iou=ratio(tp,tp+fp+fn),red_dice=ratio(2*tp,2*tp+fp+fn),
                white_iou=ratio(tn,tn+fp+fn),
                red_precision=ratio(tp,tp+fp),red_recall=ratio(tp,tp+fn),
                designer_red_fraction=float(truth.mean()),candidate_red_fraction=float(pred.mean()),
                boundary=boundary,
                registration='Exact native coordinates. No alignment, crop, warp, or resize.',
                interpretation='Pixel similarity does not establish semantic hierarchy, separation or production readiness.')
    difference=np.full((*truth.shape,3),255,np.uint8)
    difference[truth&pred]=[255,0,0]
    difference[truth&~pred]=[0,100,255]   # Blue: missing designer red.
    difference[~truth&pred]=[255,160,0]   # Orange: excess candidate red.
    return report,Image.fromarray(difference)
