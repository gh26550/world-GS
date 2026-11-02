"""Shared layout contract, region-conditioned synthesis and coarse layout checks.

No texture matching or pixel-perfect silhouette requirement. Detectors remain
fallible; a pass is evidence of coarse arrangement, not a correctness proof.
"""
import numpy as np
from scipy.ndimage import binary_dilation
from scipy.optimize import linear_sum_assignment


def category(text):
    text=text.lower()
    aliases={'bookshelf':('bookshelf','bookshelve','bookcase'), 'sofa':('sofa','couch'),
             'chair':('chair',), 'table':('table',), 'bed':('bed',),
             'cabinet':('cabinet','dresser'), 'rug':('rug','carpet'),
             'lamp':('lamp',), 'plant':('plant',), 'vase':('vase',)}
    return next((key for key,words in aliases.items() if any(w in text for w in words)),text.strip())


def projected_objects(hit,owners,components,labels,min_pixels=64):
    valid=hit['faces']>=0;ids=owners[np.maximum(hit['faces'],0)];rows=[]
    for index,comp in enumerate(components):
        if comp.get('material')!='generated_asset':continue
        mask=valid & (ids==index)
        if mask.sum()<min_pixels:continue
        y,x=np.where(mask)
        rows.append(dict(id=comp['name'],label=category(labels.get(comp['name'],comp['name'])),
                         box=[int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)],
                         pixels=int(mask.sum()),mask=mask))
    return rows


def synthesis_regions(objects,shape):
    """Background first; each object gets a loose envelope, excluding other owners.

    Bounding envelopes permit local shape variation; all other objects remain
    protected during that object's synthesis pass, preserving coarse occlusion.
    """
    h,w=shape;foreground=np.zeros(shape,bool)
    for obj in objects:foreground |= obj['mask']
    regions=[dict(id='background',label='room architecture',mask=~binary_dilation(foreground,iterations=2))]
    for obj in objects:
        x0,y0,x1,y1=np.asarray(obj['box']);pad=max(3,int(.10*max(x1-x0,y1-y0)))
        mask=np.zeros(shape,bool);mask[max(0,y0-pad):min(h,y1+pad),max(0,x0-pad):min(w,x1+pad)]=True
        others=foreground & ~obj['mask']
        mask &= ~binary_dilation(others,iterations=2)
        regions.append(dict(id=obj['id'],label=obj['label'],description=obj.get('description',obj['label']),mask=mask))
    return regions


def layout_check(objects,detections,shape):
    """One-to-one class/position/scale matching, permitting approximate geometry.

    Expected visible objects are mandatory, unexpected instances are rejected.
    Low-confidence / tiny detections must be handled by the detector, not waived
    here. No seating-family exception and no guide-detection override.
    """
    h,w=shape;diagonal=np.hypot(h,w)
    detected=[]
    # Suppress duplicate detector boxes within one category before counting.
    for d in sorted(detections,key=lambda d:-d.get('score',1)):
        b=np.asarray(d['box'],float);label=category(d['label'])
        if np.any(~np.isfinite(b)) or np.any(b[2:]<=b[:2]):continue
        duplicate=False
        for other in detected:
            if other['label']!=label:continue
            a=np.asarray(other['box']);inter=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod()
            union=np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-inter
            if inter/max(union,1)>.65 or inter/max(min(np.prod(a[2:]-a[:2]),np.prod(b[2:]-b[:2])),1)>.85:duplicate=True
        if not duplicate:detected.append(dict(label=label,box=b.tolist()))
    cost=np.full((len(objects),len(detected)),1e4)
    metrics={}
    for i,obj in enumerate(objects):
        a=np.asarray(obj['box'],float);extent=a[2:]-a[:2]
        for j,d in enumerate(detected):
            if obj['label']!=d['label']:continue
            b=np.asarray(d['box']);shift=np.linalg.norm((a[:2]+a[2:]-b[:2]-b[2:])/2)
            scale=(b[2:]-b[:2])/np.maximum(extent,1)
            allowed=max(.02*diagonal,.12*np.linalg.norm(extent))
            ok=shift<=allowed and np.all((scale>=.6)&(scale<=1.7))
            cost[i,j]=shift/diagonal+float(np.abs(np.log(scale)).mean()) if ok else 1e4
            metrics[i,j]=dict(center_shift_pixels=float(shift),center_tolerance_pixels=float(allowed),scale_ratio=scale.tolist())
    paired=[];matched_o=set();matched_d=set()
    if cost.size:
        for i,j in zip(*linear_sum_assignment(cost)):
            if cost[i,j]>=1e4:continue
            paired.append(dict(id=objects[i]['id'],label=objects[i]['label'],**metrics[i,j]));matched_o.add(i);matched_d.add(j)
    missing=[o['id'] for i,o in enumerate(objects) if i not in matched_o]
    unexpected=[d for j,d in enumerate(detected) if j not in matched_d]
    return dict(passed=not missing and not unexpected,matched=paired,missing_or_displaced=missing,
                unexpected_or_displaced=unexpected,expected_count=len(objects),detected_count=len(detected),
                policy='class + one-to-one assignment + coarse center/scale; no texture or exact silhouette matching',
                limitation='Detector mistakes can reject valid images or accept incorrect ones')


def layout_contract(geometry,components,plan,mesh_sha,source_sha):
    rows=[]
    for i,c in enumerate(components):
        if c.get('material')!='generated_asset':continue
        faces=geometry['faces'][geometry['owners']==i]
        points=geometry['vertices'][np.unique(faces)]
        obj=next(o for o in plan['objects'] if o['id']==c['name'])
        rows.append(dict(id=c['name'],label=category(obj['label']),description=obj['description'],
                         bounds=[points.min(0).tolist(),points.max(0).tolist()],yaw=obj.get('yaw')))
    return dict(version=17,mesh_sha256=mesh_sha,source_sha256=source_sha,objects=rows,
                scope='Shared coarse world placement and object identity; mesh surface detail is not ground truth',
                prompt=plan.get('prompt'),texture_matching_required=False)


def layout_displacement_loss(delta,component_ids,tolerance=.04):
    """Penalize coherent object translation, permitting zero-mean local detail.

    Component assignments follow the trainer's seed anchors; this is a soft
    stabilization term, not a rigid-object or semantic-identity guarantee.
    """
    import torch
    ids=torch.as_tensor(component_ids,device=delta.device)
    terms=[]
    for component in torch.unique(ids):
        if int(component)<6:continue
        displacement=delta[ids==component].mean(0).norm()
        terms.append(torch.relu(displacement-tolerance))
    return torch.stack(terms).mean() if terms else delta.sum()*0
