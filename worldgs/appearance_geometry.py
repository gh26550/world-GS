"""Geometry-conditioned appearance transfer and conservative image checks."""
import numpy as np
from scipy.ndimage import binary_erosion, gaussian_filter, sobel, distance_transform_edt, map_coordinates, binary_dilation


def depth_condition(depth):
    valid=np.isfinite(depth)&(depth>.03)
    inverse=np.zeros_like(depth,dtype=float);inverse[valid]=1/depth[valid]
    lo,hi=np.percentile(inverse[valid],[1,99]) if valid.any() else (0,1)
    return np.uint8(np.clip((inverse-lo)/max(hi-lo,1e-6),0,1)*255)


def pose_distance(a,b):
    a,b=np.asarray(a),np.asarray(b)
    return float(1-np.clip(a[:3,2]@b[:3,2],-1,1)+.08*np.linalg.norm(a[:3,3]-b[:3,3]))


def transfer_appearance(hit,frame,cache,max_sources=6):
    """Inverse-project target surface samples; reject occluded source pixels."""
    shape=hit['depth'].shape
    points=hit['xyz'].reshape(-1,3);normals=hit['normals'].reshape(-1,3)
    output=np.zeros((len(points),3),np.uint8);score=np.zeros(len(points));confidence=np.zeros(len(points))
    pose=np.asarray(frame['transform_matrix'])
    chosen=sorted(cache,key=lambda e:pose_distance(pose,e['frame']['transform_matrix']))[:max_sources]
    anchors=[e for e in cache if e.get('source_anchor')]
    chosen=anchors+[e for e in chosen if not e.get('source_anchor')]
    for entry in chosen:
        source_pose=np.asarray(entry['frame']['transform_matrix'])
        camera=(points-source_pose[:3,3])@source_pose[:3,:3]
        uvh=camera@np.asarray(entry['frame']['K']).T
        uv=uvh[:,:2]/np.maximum(uvh[:,2:],1e-8)
        h,w=entry['depth'].shape
        valid=(camera[:,2]>.03)&(uv[:,0]>=1)&(uv[:,0]<w-2)&(uv[:,1]>=1)&(uv[:,1]<h-2)&(hit['faces'].ravel()>=0)
        ids=np.flatnonzero(valid)
        if not len(ids):continue
        x,y=uv[ids].T;ix,iy=np.rint(x).astype(int),np.rint(y).astype(int)
        d=entry['depth'][iy,ix]
        visible=entry['valid'][iy,ix]&(np.abs(d-camera[ids,2])<.02+.01*d)
        ids=ids[visible];x,y=x[visible],y[visible];ix,iy=ix[visible],iy[visible]
        if not len(ids):continue
        ray=points[ids]-source_pose[:3,3];distance=np.linalg.norm(ray,axis=1)
        incidence=np.abs((normals[ids]*ray).sum(1)/np.maximum(distance,1e-8))
        conf=entry['confidence'][iy,ix]
        weight=conf*incidence**2/np.sqrt(np.maximum(distance,.3))
        better=weight>np.maximum(score[ids],.04)
        ids,x,y,conf,weight=ids[better],x[better],y[better],conf[better],weight[better]
        for channel in range(3):
            output[ids,channel]=np.clip(map_coordinates(entry['rgb'][...,channel].astype(float),[y,x],order=1),0,255)
        score[ids]=weight;confidence[ids]=conf
    known=(score>0).reshape(shape)
    # Let inpainting resolve jagged projection boundaries rather than freeze seams.
    known=binary_erosion(known,iterations=2,border_value=0)
    return output.reshape(*shape,3),known,confidence.reshape(shape)


def geometry_check(depth,predicted_inverse,faces,owners,tolerance=10,min_recall=.45,generated_mask=None):
    valid=faces>=0
    expected=depth_condition(depth).astype(float)/255
    prediction=np.asarray(predicted_inverse,float)
    lo,hi=np.percentile(prediction[valid],[2,98]) if valid.any() else (0,1)
    prediction=np.clip((prediction-lo)/max(hi-lo,1e-6),0,1)
    def gradient(x):
        x=gaussian_filter(x,1.)
        return np.hypot(sobel(x,0),sobel(x,1))/8
    expected_gradient=gradient(expected)
    owner=np.full(faces.shape,-1,int);owner[valid]=owners[faces[valid]]
    boundaries=np.zeros(faces.shape,bool)
    boundaries[1:]|=owner[1:]!=owner[:-1];boundaries[:,1:]|=owner[:,1:]!=owner[:,:-1]
    expected_edges=((expected_gradient>.025)|boundaries)&binary_erosion(valid,iterations=3)
    all_expected_edges=expected_edges.copy()
    if generated_mask is not None:expected_edges &= binary_dilation(generated_mask,iterations=3)
    predicted_edges=gradient(prediction)>.012
    distance=distance_transform_edt(~predicted_edges)
    count=int(expected_edges.sum())
    recall=float((distance[expected_edges]<=tolerance).mean()) if count else 1.
    return dict(passed=recall>=min_recall,depth_edge_recall=recall,expected_edge_pixels=count,
                min_recall=min_recall,tolerance_pixels=tolerance,
                all_region_recall=float((distance[all_expected_edges]<=tolerance).mean()) if all_expected_edges.any() else 1.,
                checked_region='generated pixels and 3px boundary' if generated_mask is not None else 'all pixels',
                note='Heuristic depth-edge recall; does not certify semantic identity or exact geometry')


def structural_planarity(predicted_inverse,faces,owners,generated,min_pixels=256,max_error=.16):
    valid=faces>=0
    prediction=np.asarray(predicted_inverse,float)
    lo,hi=np.percentile(prediction[valid],[2,98])
    prediction=(prediction-lo)/max(hi-lo,1e-6)
    y,x=np.indices(faces.shape);ids=owners[np.maximum(faces,0)]
    rows=[]
    for component in range(6):
        mask=binary_erosion(valid&(ids==component),iterations=4)&generated
        if mask.sum()<min_pixels:continue
        coordinates=np.column_stack([x[mask]/faces.shape[1],y[mask]/faces.shape[0],np.ones(mask.sum())])
        values=prediction[mask];keep=np.ones(len(values),bool)
        for _ in range(3):
            fitted=coordinates@np.linalg.lstsq(coordinates[keep],values[keep],rcond=None)[0]
            error=np.abs(fitted-values);keep=error<=np.quantile(error,.8)
        p90=float(np.quantile(error,.9));rows.append(dict(component=component,p90_inverse_depth_plane_error=p90))
    return dict(passed=all(r['p90_inverse_depth_plane_error']<=max_error for r in rows),planes=rows,max_error=max_error,
                note='Generated structural pixels only; inherited photographic details are excluded')


def semantic_support(detections,faces,owners,components,rgb=None,generated_mask=None,baseline_detections=None):
    labels={}
    for index,component in enumerate(components):
        name=component['name'].lower()
        for label in ('sofa','table','bookshelf','chair'):
            if label in name:labels.setdefault(label,[]).append(index)
    ids=owners[np.maximum(faces,0)];violations=[];ignored=[]
    for detection in detections:
        label=next((k for k in ('sofa','table','bookshelf','chair') if k in detection['label'].lower()),None)
        if label is None:continue
        x0,y0,x1,y1=np.rint(detection['box']).astype(int)
        x0,x1=np.clip([x0,x1],0,faces.shape[1]);y0,y1=np.clip([y0,y1],0,faces.shape[0])
        compatible=labels.get(label,[])
        if label in ('chair','sofa'):compatible=labels.get('chair',[])+labels.get('sofa',[])
        region=(faces[y0:y1,x0:x1]>=0)&np.isin(ids[y0:y1,x0:x1],compatible)
        support=float(region.mean()) if region.size else 0.
        if support<.05 and baseline_detections:
            same_guide=False
            for baseline in baseline_detections:
                other=baseline['label'].lower()
                compatible_label=label in other or (label in ('chair','sofa') and any(k in other for k in ('chair','sofa')))
                a=np.array([x0,y0,x1,y1],float);b=np.array(baseline['box'],float)
                intersection=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod()
                union=np.maximum(a[2:]-a[:2],0).prod()+np.maximum(b[2:]-b[:2],0).prod()-intersection
                if compatible_label and intersection/max(union,1)>.5:same_guide=True
            if same_guide:
                ignored.append(dict(label=label,reason='same detector response already present in the mesh guide'))
                continue
        if support<.05 and generated_mask is not None:
            changed=generated_mask[y0:y1,x0:x1]
            if changed.size and changed.mean()<.2:
                ignored.append(dict(label=label,reason='mostly inherited appearance; no newly generated object'))
                continue
        full_frame=(x1-x0)*(y1-y0)>.9*faces.size
        smooth=False
        if rgb is not None and full_frame:
            span=np.percentile(rgb,95,axis=(0,1))-np.percentile(rgb,5,axis=(0,1))
            smooth=float(span.max())<255*.15
        if support<.05 and full_frame and smooth:
            ignored.append(dict(label=label,reason='unlocalized full-frame detection on a low-contrast surface'))
            continue
        if support<.05:violations.append(dict(label=label,box=detection['box'],mesh_support_fraction=support))
    return dict(passed=not violations,unexpected_furniture=violations,ignored_unlocalized_detections=ignored,detection_count=len(detections),
                note='Detector-based rejection of unsupported furniture; false negatives remain possible')
