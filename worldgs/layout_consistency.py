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
                         pixels=int(mask.sum()),mask=mask,requires_detection=bool(mask.sum()>=max(256,valid.size*.005))))
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
        if not obj.get('requires_detection',True):continue
        x0,y0,x1,y1=np.asarray(obj['box']);pad=max(3,int(.10*max(x1-x0,y1-y0)))
        mask=np.zeros(shape,bool);mask[max(0,y0-pad):min(h,y1+pad),max(0,x0-pad):min(w,x1+pad)]=True
        others=foreground & ~obj['mask']
        mask &= ~binary_dilation(others,iterations=2)
        regions.append(dict(id=obj['id'],label=obj['label'],description=obj.get('description',obj['label']),mask=mask))
    return regions


def layout_check(objects,detections,shape,source_parts=None):
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
    retained=[]
    for i,obj in enumerate(objects):
        if not obj.get('requires_detection',True) and i not in matched_o:
            matched_o.add(i);retained.append(obj['id'])
    missing=[o['id'] for i,o in enumerate(objects) if i not in matched_o]
    unexpected=[];parts=[];ambiguous=[]
    for j,d in enumerate(detected):
        if j in matched_d:continue
        b=np.asarray(d['box']);area=np.prod(b[2:]-b[:2]);owner=None
        for i in matched_o:
            obj=objects[i]
            if d['label'] not in (source_parts or {}).get(obj['id'],[]):continue
            a=np.asarray(obj['box']);inter=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod()
            if inter/max(area,1)>=.9:owner=obj['id'];break
        if owner is None:
            # A detector box spanning already verified furniture is not evidence
            # of an additional physical object. Keep the ambiguity explicit.
            if objects and all('mask' in obj for obj in objects):
                x0,y0,x1,y1=np.rint(b).astype(int);x0,x1=np.clip([x0,x1],0,w);y0,y1=np.clip([y0,y1],0,h)
                occupied=np.zeros((h,w),bool)
                for i in matched_o:occupied |= objects[i]['mask']
                patch=occupied[y0:y1,x0:x1]
                support=float(patch.mean()) if patch.size else 0.
            else:support=0.
            if support>=.65:
                ambiguous.append(dict(**d,verified_instance_coverage=support,reason='Alternative/aggregate recognition on existing verified furniture'))
            else:unexpected.append(d)
        else:parts.append(dict(parent_id=owner,**d,reason='Source-mask-confirmed nested part; parent instance separately matched'))
    return dict(passed=not missing and not unexpected,matched=paired,missing_or_displaced=missing,
                unexpected_or_displaced=unexpected,source_confirmed_parts=parts,recognition_ambiguities=ambiguous,retained_small_instances=retained,expected_count=len(objects),detected_count=len(detected),
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


def calibrate_source_parts(objects,detections,shape,source_masks):
    """Resolve source-image part detections using independent instance masks.

    A parent must first match as its own class. The extra box must lie inside
    that parent and be predominantly covered by that parent's source mask.
    This cannot explain away a missing parent or a new object elsewhere.
    """
    baseline=layout_check(objects,detections,shape);matched={r['id'] for r in baseline['matched']};parts={};evidence=[]
    h,w=shape
    for det in baseline['unexpected_or_displaced']:
        b=np.asarray(det['box']);x0,y0,x1,y1=np.rint(b).astype(int);x0,x1=np.clip([x0,x1],0,w);y0,y1=np.clip([y0,y1],0,h)
        for obj in objects:
            if obj['id'] not in matched or obj['id'] not in source_masks:continue
            a=np.asarray(obj['box']);inter=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod()
            patch=source_masks[obj['id']][y0:y1,x0:x1]
            support=float(patch.mean()) if patch.size else 0.
            if inter/max(np.prod(b[2:]-b[:2]),1)>=.9 and support>=.5:
                parts.setdefault(obj['id'],[]).append(det['label']);evidence.append(dict(id=obj['id'],part=det['label'],source_mask_support=support,box=det['box']));break
    return dict(parts=parts,evidence=evidence,policy='Source-mask ownership and separately matched parent required')


def structure_palette_guide(rgb,hit,owners):
    """Remove object-shaped detail baked into architectural material swatches.

    Geometry and material palette stay intact. Photographic appearance is then
    synthesized in the structural region, rather than reusing false furniture.
    """
    result=rgb.copy();valid=hit['faces']>=0;ids=owners[np.maximum(hit['faces'],0)]
    for i in range(6):
        mask=valid & (ids==i)
        if mask.any():result[mask]=np.median(rgb[mask],axis=0).astype(np.uint8)
    return result


def surface_regions(objects,shape):
    """Image edits are attached to projected visible world instances only.

    A small contour band permits local shape refinement. Architecture is not
    independently regenerated, and another instance's pixels stay protected.
    """
    foreground=np.zeros(shape,bool)
    for obj in objects:foreground |= obj['mask']
    rows=[];slack=max(2,round(min(shape)*.012))
    for obj in objects:
        if not obj.get('requires_detection',True):continue
        mask=binary_dilation(obj['mask'],iterations=slack)
        mask &= ~binary_dilation(foreground & ~obj['mask'],iterations=1)
        rows.append(dict(id=obj['id'],label=obj['label'],description=obj.get('description',obj['label']),mask=mask))
    return rows


def ownership_check(rgb,guide,regions):
    editable=np.zeros(rgb.shape[:2],bool)
    for r in regions:editable |= r['mask']
    diff=np.abs(rgb.astype(float)-guide.astype(float)).mean(-1)
    outside=float(diff[~editable].max()) if (~editable).any() else 0.
    return dict(passed=outside==0.,outside_instance_max_change=outside,
                editable_fraction=float(editable.mean()),policy='Shared-world instance surface edits; immutable architecture and other-instance pixels')


def source_exclusion_mask(shape,objects,detections,source_size):
    """Exclude unmatched furniture detections from architecture projection.

    Matched instances already have pixel-accurate SAM ownership. Excluding their
    entire rectangles would also discard the valid floor between table legs.
    """
    h,w=shape;scale=np.array([w/source_size,h/source_size,w/source_size,h/source_size]);mask=np.zeros(shape,bool)
    for det in detections:
        b=np.asarray(det['box'],float)*scale;matched=False
        for obj in objects:
            if obj.get('evidence')!='front' or category(obj['label'])!=category(det['label']):continue
            a=np.asarray(obj['source_box'],float)*np.array([w,h,w,h]);inter=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod()
            union=np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-inter
            if inter/max(union,1)>.2:matched=True;break
        if not matched:
            x0,y0,x1,y1=np.rint(b).astype(int);mask[max(0,y0):min(h,y1),max(0,x0):min(w,x1)]=True
    return mask
