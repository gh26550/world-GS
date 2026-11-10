"""Depth-edge verification independent of monocular absolute depth scale."""
import numpy as np
from scipy.ndimage import binary_dilation

def depth_edges(depth):
    import cv2
    d=np.asarray(depth,float);valid=np.isfinite(d)&(d>.03)
    if not valid.any():raise ValueError('No valid depth')
    lo,hi=np.percentile(d[valid],[1,99]);normalized=np.zeros(d.shape,np.uint8)
    if hi-lo>1e-6:normalized[valid]=np.uint8(np.clip((d[valid]-lo)/(hi-lo),0,1)*255)
    return (cv2.Canny(normalized,25,76,apertureSize=3)>0)&valid

def verify_depth(mesh_depth,predicted_depth,tolerance=10,min_recall=.55,proxy_depth=None,rgb=None):
    if np.shape(mesh_depth)!=np.shape(predicted_depth):raise ValueError('Depth/image shape differs from camera calibration')
    expected=depth_edges(mesh_depth);observed=depth_edges(predicted_depth);observable=expected.copy()
    if proxy_depth is not None:observable &= binary_dilation(depth_edges(proxy_depth),iterations=tolerance)
    unobservable=expected&~observable;n=int(expected.sum());n_observable=int(observable.sum());recall=float(binary_dilation(observed,iterations=tolerance)[observable].mean()) if n_observable else 1.
    rgb_recall=None
    if unobservable.any() and rgb is not None:
        import cv2
        edges=cv2.Canny(cv2.cvtColor(np.asarray(rgb,np.uint8),cv2.COLOR_RGB2GRAY),25,76)>0
        rgb_recall=float(binary_dilation(edges,iterations=tolerance)[unobservable].mean())
    fallback_pass=not unobservable.any() or (rgb_recall is not None and rgb_recall>=.6)
    return dict(passed=recall>=min_recall and fallback_pass,depth_edge_recall=recall,expected_edge_pixels=n,observable_edge_pixels=n_observable,unobservable_edge_pixels=int(unobservable.sum()),rgb_boundary_recall_unobservable=rgb_recall,min_recall=min_recall,tolerance_pixels=tolerance,informative=bool(n_observable),limitation='Proxy-estimator observability + RGB edge fallback; extra objects and exact geometry require separate layout checking')


def verify_layout(objects,detections,rgb,hit=None,owners=None,predicted_depth=None):
    from .layout_consistency import layout_check
    h,w=rgb.shape[:2];filtered=[];unlocalized=[];architecture_ambiguities=[]
    for d in detections:
        b=np.asarray(d['box']);area=float(np.maximum(b[2:]-b[:2],0).prod())
        # DINO emits whole-image classes on plain walls; no localized evidence.
        if not objects and area>.95*h*w and float(np.asarray(rgb,float).std(axis=(0,1)).mean())<12:
            unlocalized.append(d);continue
        if hit is not None and predicted_depth is not None:
            x0,y0,x1,y1=np.rint(b).astype(int);x0,x1=np.clip([x0,x1],0,w);y0,y1=np.clip([y0,y1],0,h)
            face=hit['faces'][y0:y1,x0:x1];valid=face>=0;ids=owners[np.maximum(face,0)];patch=rgb[y0:y1,x0:x1]
            if valid.any() and float((valid&(ids<6)).mean())>.95 and patch.size and float(patch.astype(float).std(axis=(0,1)).mean())<12:
                values=1/np.maximum(predicted_depth[y0:y1,x0:x1],.03);yy,xx=np.indices(values.shape);A=np.column_stack([xx[valid]/w,yy[valid]/h,np.ones(valid.sum())]);v=values[valid];fitted=A@np.linalg.lstsq(A,v,rcond=None)[0];scale=np.ptp(1/np.maximum(predicted_depth,.03));error=float(np.quantile(np.abs(v-fitted),.9)/max(scale,1e-6))
                if error<.03:
                    architecture_ambiguities.append(dict(**d,inverse_depth_plane_error=error,reason='Low-texture structural surface with no localized depth relief'));continue
        filtered.append(d)
    check=layout_check(objects,filtered,(h,w));check['unlocalized_flat_wall_detections']=unlocalized;check['planar_architecture_ambiguities']=architecture_ambiguities
    return check

def camera_context(hit,owners,components,labels):
    from collections import Counter
    from .layout_consistency import projected_objects
    parts=[dict(c,material='generated_asset') if c.get('group')=='rug' else c for c in components];labels=dict(labels)
    for c in parts:
        if c.get('group')=='rug':labels[c['name']]='rug'
    objects=projected_objects(hit,owners,parts,labels,min_pixels=max(16,hit['depth'].size//4096));counts=Counter(o['label'] for o in objects)
    valid=hit['faces']>=0;ids=owners[np.maximum(hit['faces'],0)];window=any(np.any(valid&(ids==i)) for i,c in enumerate(components) if c.get('group')=='window')
    if window:
        mask=valid&np.isin(ids,[i for i,c in enumerate(components) if c.get('group')=='window']);y,x=np.where(mask)
        objects.append(dict(id='scene_window',label='window',box=[int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)],pixels=int(mask.sum()),mask=mask,requires_detection=bool(mask.sum()>=256)));counts['window']=1
    visible=', '.join(str(n)+' '+name for name,n in sorted(counts.items())) or 'no furniture'
    absent=[name for name in ['sofa','bookshelf','chair','table','bed','cabinet','plant'] if name not in counts]
    text='This camera sees ONLY '+visible+'. '+('The guide-visible window must stay at its exact position. ' if window else 'NO windows are visible from this camera. ')
    text+='Do NOT show '+', '.join(absent)+'. Furniture elsewhere in the room is behind this camera or occluded: do not rotate toward it, invent another copy, or move it into view. Match the guide including partial clipping.'
    return text,objects

def instance_boundaries(hit,owners):
    ids=np.full(hit['faces'].shape,-1,int);valid=hit['faces']>=0;ids[valid]=owners[hit['faces'][valid]];edge=np.zeros(ids.shape,bool)
    edge[1:]|=ids[1:]!=ids[:-1];edge[:,1:]|=ids[:,1:]!=ids[:,:-1]
    return np.uint8(edge&valid)*255
