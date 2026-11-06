"""Whole-room source anchoring and translated-view supervision policy."""
import numpy as np


def whole_room_prompt(prompt):
    return (prompt + ', architectural interior photograph taken from the back corner of the room, '
            'wide establishing view showing nearly the entire room except behind the camera, '
            'floor and ceiling and three walls visible, all main furniture fully in frame, '
            'realistic perspective, straight vertical lines, natural materials, detailed textures, '
            'no close-up, no fisheye, no panorama')


def protection_mask(hit, frame, known, source_pose, owners):
    """Keep source view exact; allow translated furniture surfaces to be resynthesized.

    Projected colour remains a diffusion guide, not immutable evidence of shape.
    Structural planes are constrained by depth, not frozen low-detail pixels.
    """
    pose=np.asarray(frame['transform_matrix']);source_pose=np.asarray(source_pose)
    if np.allclose(pose,source_pose,atol=1e-6):return known.copy()
    return np.zeros_like(known,dtype=bool)


def parallax_audit(renderer, components, owners, frames, size=64, min_pixels=16,
                   min_baseline=.25, min_angle=12.):
    """Count geometric diversity, not repeated image files or camera rotations.

    Baseline and object-centred ray angle must both pass for each generated asset.
    This is a necessary supervision check, not an image consistency certificate.
    """
    observations={i:[] for i,c in enumerate(components) if c.get('material')=='generated_asset'}
    for frame in frames:
        K=np.asarray(frame['K'],float).copy()
        original_size=2*K[0,2]+1
        scale=size/original_size
        K[:2,:2]*=scale;K[:2,2]=(K[:2,2]+.5)*scale-.5
        pose=np.asarray(frame['transform_matrix'])
        hit=renderer.render(K,pose,size);valid=hit['faces']>=0
        ids=owners[np.maximum(hit['faces'],0)]
        for component,items in observations.items():
            mask=valid & (ids==component)
            if mask.sum()>=min_pixels:
                items.append((frame['file_path'],pose[:3,3],hit['xyz'][mask].mean(0)))
    rows=[]
    for component,items in observations.items():
        best=None
        if items:
            center=np.mean([r[2] for r in items],axis=0)
            for i,a in enumerate(items):
                for b in items[i+1:]:
                    baseline=float(np.linalg.norm(a[1]-b[1]))
                    x,y=a[1]-center,b[1]-center
                    angle=float(np.degrees(np.arccos(np.clip(x@y/max(np.linalg.norm(x)*np.linalg.norm(y),1e-9),-1,1))))
                    if baseline>=min_baseline and angle>=min_angle:
                        if best is None or angle>best['angle_degrees']:
                            best=dict(frames=[a[0],b[0]],baseline=baseline,angle_degrees=angle)
        rows.append(dict(object=components[component]['name'],visible_views=len(items),passed=best is not None,pair=best))
    return dict(passed=bool(rows) and all(r['passed'] for r in rows),objects=rows,
                min_baseline=min_baseline,min_angle_degrees=min_angle,
                note='Geometric diversity only; generated images are not independent observations')
