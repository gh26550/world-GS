"""Fixed, class-independent reference validation. No detector-label exceptions."""
import numpy as np
from scipy.ndimage import binary_dilation,binary_erosion

def edges(depth):
    import cv2
    d=np.asarray(depth,float);valid=np.isfinite(d)&(d>.03)
    if not valid.any():raise ValueError('Depth has no finite positive pixels')
    lo,hi=np.percentile(d[valid],[1,99]);x=np.zeros(d.shape,np.uint8)
    if hi-lo>1e-6:x[valid]=np.uint8(np.clip((d[valid]-lo)/(hi-lo),0,1)*255)
    return (cv2.Canny(x,25,76)>0)&valid

def assess_reference(mesh_depth,proxy_depth,photo_depth,rgb,face_ids,owners,size,components=None):
    import cv2
    shape=mesh_depth.shape
    if any(np.shape(d)!=shape for d in [proxy_depth,photo_depth,face_ids]) or rgb.shape!=shape+(3,):raise ValueError('Reference evidence has incompatible dimensions')
    if not np.isfinite(photo_depth).all() or np.any(photo_depth<=0):raise ValueError('Invalid predicted depth')
    tolerance=max(1,round(10*size/512));expected=edges(mesh_depth);proxy_edges=binary_dilation(edges(proxy_depth),iterations=tolerance);photo_edges=binary_dilation(edges(photo_depth),iterations=tolerance);rgb_edges=binary_dilation(cv2.Canny(cv2.cvtColor(np.asarray(rgb,np.uint8),cv2.COLOR_RGB2GRAY),25,76)>0,iterations=tolerance)
    valid=face_ids>=0;ids=np.full(shape,-1,int);ids[valid]=owners[face_ids[valid]]
    def measure(mask,required_edges=expected):
        observed=mask&required_edges&proxy_edges;uncertain=mask&required_edges&~proxy_edges
        depth_recall=float(photo_edges[observed].mean()) if observed.any() else None
        rgb_recall=float(rgb_edges[uncertain].mean()) if uncertain.any() else None
        return dict(passed=(depth_recall is None or depth_recall>=.55) and (rgb_recall is None or rgb_recall>=.6),observable_depth_edges=int(observed.sum()),unobservable_depth_edges=int(uncertain.sum()),depth_recall=depth_recall,rgb_recall=rgb_recall)
    global_check=measure(valid);group_results=[]
    groups={}
    for owner in np.unique(ids[valid]):
        name=components[int(owner)].get('group',str(owner)) if components is not None else str(owner)
        groups.setdefault(name,[]).append(int(owner))
    for name,group_owners in groups.items():
        mask=valid&np.isin(ids,group_owners)
        if mask.sum()<max(64,size*size*.002):continue
        band=binary_dilation(mask,iterations=2)&~binary_erosion(mask,iterations=2);silhouette=(mask&~binary_erosion(mask))&binary_erosion(np.ones(shape,bool),iterations=2)
        row=measure(band,expected|silhouette if any(owner>=6 for owner in group_owners) else expected)
        components_row=dict(group=name,owners=group_owners,pixels=int(mask.sum()),**row)
        group_results.append(components_row)
    passed=global_check['passed'] and all(r['passed'] for r in group_results)
    values=[r[k] for r in [global_check]+group_results for k in ['depth_recall','rgb_recall'] if r[k] is not None]
    return dict(passed=passed,global_check=global_check,components=group_results,score=float(np.mean(values)) if values else 0.,policy=dict(granularity='physical_group_silhouette',depth_recall_min=.55,rgb_recall_min=.6,tolerance_pixels=tolerance),limitations=['Geometry-edge evidence only; does not certify object identity or detect every additional object','Unobservable depth boundaries require RGB evidence; texture edges can cause false positives'])
