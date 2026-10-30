"""Fit actual reconstructed meshes to their source image before GS teaching.

Research-inspired reprojection/pointmap alignment, not a reproduction of the
WorldMesh or SAM3D test-time optimizer. Geometry is not deformed to fit boxes.
"""
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.optimize import least_squares
from .geometry import intrinsics
from .mesh_scene import look_at


def crop_region(box, size):
    size=np.array(size)
    lo=np.floor(np.asarray(box[:2])*size).astype(int)
    hi=np.ceil(np.asarray(box[2:])*size).astype(int)
    margin=np.maximum(2,np.rint((hi-lo)*.06).astype(int))
    return np.maximum(lo-margin,0),np.minimum(hi+margin,size)


def source_mask(folder, obj, size):
    """Undo the exact padded crop, not the detector's rectangular mask."""
    mask=Image.open(folder/'assets'/obj['id']/'mask.png').convert('L')
    lo,hi=crop_region(obj['source_box'],size)
    mask=mask.crop((32,32,mask.width-32,mask.height-32)).resize(tuple(hi-lo),Image.Resampling.NEAREST)
    result=np.zeros((size[1],size[0]),bool)
    result[lo[1]:hi[1],lo[0]:hi[0]]=np.asarray(mask)>127
    return result


def projection(vertices, K, pose, size):
    camera=(np.asarray(vertices)-pose[:3,3])@pose[:3,:3]
    uvh=camera@K.T
    uv=uvh[:,:2]/np.maximum(camera[:,2:],.03)
    return np.r_[uv.min(0),uv.max(0)]/np.tile(size,2),camera[:,2]


def box_iou(a,b):
    a,b=np.clip(a,0,1),np.clip(b,0,1)
    inter=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod()
    return float(inter/max(1e-9,np.maximum(a[2:]-a[:2],0).prod()+np.maximum(b[2:]-b[:2],0).prod()-inter))


def estimate_floor(points, valid, excluded):
    """Robust horizontal-plane prior from unmasked lower-image points."""
    h,w=valid.shape
    y,x=np.indices((h,w))
    select=valid&~excluded&(y>h*.68)&np.isfinite(points).all(-1)
    cloud=points[select]
    if len(cloud)<100:
        return None
    rng=np.random.default_rng(0)
    cloud=cloud[rng.choice(len(cloud),min(3000,len(cloud)),replace=False)]
    best=None
    tolerance=max(.006,float(np.median(cloud[:,2]))*.015)
    for _ in range(160):
        tri=cloud[rng.choice(len(cloud),3,replace=False)]
        n=np.cross(tri[1]-tri[0],tri[2]-tri[0])
        n/=max(np.linalg.norm(n),1e-9)
        if n[1]<0:n=-n
        if n[1]<.9:continue
        d=-float(tri[0]@n)
        inlier=np.abs(cloud@n+d)<tolerance
        if best is None or inlier.sum()>best[0].sum():best=(inlier,n,d)
    if best is None or best[0].mean()<.25:return None
    subset=cloud[best[0]]
    _,_,vh=np.linalg.svd(subset-subset.mean(0),full_matrices=False)
    n=vh[-1]
    if n[1]<0:n=-n
    d=-float(subset.mean(0)@n)
    if abs(d)<.02:return None
    return dict(normal=n.tolist(),distance=abs(d),inlier_fraction=float(best[0].mean()),
                pitch=float(-np.arctan2(n[2],n[1])))


def fit_front(plan, meshes, folder, *, max_nfev=180):
    """Uniform scales, x/z, yaw and camera intrinsics/extrinsics jointly fit."""
    folder=Path(folder)
    with Image.open(folder/'source_front.png') as opened:
        image=opened.convert('RGB')
    size=np.array(image.size,float)
    w,h,l=plan['room']['size']
    original_objects=copy.deepcopy(plan['objects'])
    observed=[(i,o) for i,o in enumerate(plan['objects']) if o.get('evidence')=='front' and o.get('support','floor')=='floor']
    if not observed:raise ValueError('Front fit requires image-grounded meshes')
    targets=[];masks=[];samples=[]
    for i,obj in observed:
        mask=source_mask(folder,obj,image.size)
        yy,xx=np.where(mask)
        if len(xx)<30:raise ValueError('Insufficient source segmentation: '+obj['id'])
        targets.append(np.array([xx.min(),yy.min(),xx.max()+1,yy.max()+1])/np.tile(size,2))
        masks.append(mask)
        vertices=meshes[i].vertices
        ids=np.linspace(0,len(vertices)-1,min(1800,len(vertices))).astype(int)
        samples.append(vertices[ids])
    geometry=np.load(folder/'front_pointmap.npz') if (folder/'front_pointmap.npz').exists() else None
    fov=75.;floor=None;depths=[None]*len(observed)
    if geometry is not None:
        normalized=geometry['intrinsics']
        fov=float(np.rad2deg(2*np.arctan(.5/normalized[0,0])))
        valid=geometry['mask']>.5
        floor=estimate_floor(geometry['points'],valid,np.logical_or.reduce(masks))
        if floor:
            for j,mask in enumerate(masks):
                d=geometry['depth'][mask&valid]
                if len(d)>30:depths[j]=float(np.median(d)/floor['distance']*1.3)
    fov=float(np.clip(fov,38,100))
    pitch=float(np.clip(floor['pitch'] if floor else 0.,-.25,.25))
    initial=[fov,h/2-1.3,-l/2+.35,pitch]
    lower=[max(35,fov-12),-h/2+.35,-l/2+.15,-.35]
    upper=[min(110,fov+12),h/2-.65,-.15,.35]
    for i,obj in observed:
        initial.extend([*obj['position'],0.,0.])
        lower.extend([-w/2+.1,-l/2+.1,np.log(.3),-np.pi/3])
        upper.extend([w/2-.1,l/2-.1,np.log(1.8),np.pi/3])
        if obj.get('size_locked'):
            lower[-2],upper[-2]=-1e-7,1e-7
    margin=np.minimum(1e-5,(np.asarray(upper)-lower)*.1)
    initial=np.clip(initial,np.asarray(lower)+margin,np.asarray(upper)-margin)
    def camera(values):
        f,cy,cz,angle=values[:4]
        eye=[0,cy,cz]
        pose=look_at(eye,[0,cy-np.tan(angle),cz+1])
        return intrinsics(image.width,float(f)),pose
    def points(j,values):
        i,obj=observed[j]
        x,z,scale,yaw=values[4+j*4:8+j*4]
        v=samples[j]*np.exp(scale)
        a=np.deg2rad(obj['yaw'])+yaw
        R=np.array([[np.cos(a),0,np.sin(a)],[0,1,0],[-np.sin(a),0,np.cos(a)]])
        v=v@R.T
        # Ground contact is exact, mesh shape/aspect ratio stays unchanged.
        top=meshes[i].bounds[1,1]*np.exp(scale)
        return v+np.array([x,h/2-top,z])
    def residual(values):
        K,pose=camera(values)
        residuals=[(values[0]-fov)/30,(values[1]-(h/2-1.3))/.6,(values[3]-pitch)/.3]
        for j,(i,obj) in enumerate(observed):
            v=points(j,values)
            box,z=projection(v,K,pose,size)
            target=targets[j]
            error=box-target
            # A truncated observed silhouette gives a one-sided constraint.
            for edge in (0,1):
                if target[edge]<.012:error[edge]=max(0,error[edge])
            for edge in (2,3):
                if target[edge]>.988:error[edge]=min(0,error[edge])
            residuals.extend(error*12)
            residuals.append(max(0,.15-float(z.min()))*12)
            excess=np.maximum(np.abs(v[:,[0,2]])-np.array([w,l])/2+.04,0).max(0)
            residuals.extend(excess*5)
            residuals.append(values[6+j*4]*.12)
            residuals.append(values[7+j*4]*.15)
            if depths[j] is not None:
                residuals.append((np.quantile(z,.25)-depths[j])*.25)
        return np.asarray(residuals)
    before=np.linalg.norm(residual(initial))
    solved=least_squares(residual,initial,bounds=(lower,upper),loss='soft_l1',f_scale=.3,max_nfev=max_nfev)
    values=solved.x
    K,pose=camera(values)
    after_objective=float(np.linalg.norm(residual(values)))
    fitted_boxes=[projection(points(j,values),K,pose,size)[0] for j in range(len(observed))]
    rows=[]
    oldK=intrinsics(image.width)
    oldpose=np.eye(4)
    for j,(i,obj) in enumerate(observed):
        old=copy.deepcopy(meshes[i])
        old.apply_transform(__import__('trimesh').transformations.rotation_matrix(np.deg2rad(obj['yaw']),[0,1,0]))
        old.apply_translation([obj['position'][0],h/2-old.bounds[1,1],obj['position'][1]])
        oldbox,_=projection(old.vertices,oldK,oldpose,size)
        x,z,scale,yaw=values[4+j*4:8+j*4]
        meshes[i].apply_scale(np.exp(scale))
        obj['dimensions']=meshes[i].extents.tolist()
        obj['position']=[float(x),float(z)]
        obj['yaw']=float(obj['yaw']+np.rad2deg(yaw))
        obj['front_fit_uniform_scale']=float(np.exp(scale))
        box=fitted_boxes[j]
        rows.append(dict(id=obj['id'],source_box=targets[j].tolist(),before_box=oldbox.tolist(),after_box=box.tolist(),
                         before_iou=box_iou(oldbox,targets[j]),after_iou=box_iou(box,targets[j])))
    report=dict(method='robust front reprojection with optional MoGe camera/depth/floor priors',
        reference='source segmentation, not independent 3D ground truth',floor=floor,
        objective_before=float(before),objective_after=after_objective,
        optimized_fov=float(values[0]),camera_to_world=pose.tolist(),objects=rows,
        mean_box_iou_before=float(np.mean([r['before_iou'] for r in rows])),
        mean_box_iou_after=float(np.mean([r['after_iou'] for r in rows])),
        source_sha256=hashlib.sha256((folder/'source_front.png').read_bytes()).hexdigest(),
        limitations=['box fit is not full silhouette or texture matching','occluded geometry is inferred','metric scale depends on camera-height prior'])
    plan['reference_camera']=dict(eye=pose[:3,3].tolist(),target=(pose[:3,3]+pose[:3,2]).tolist())
    plan['reference_intrinsics']=K.tolist()
    # Completion views have known scaffold camera directions. Place inferred
    # furniture along those walls, instead of silently returning it to +z.
    placement=[]
    for i,obj in enumerate(plan['objects']):
        if obj.get('evidence')!='inferred' or obj.get('support','floor')!='floor':continue
        provenance=json.loads((folder/'assets'/obj['id']/'source.json').read_text())
        name=Path(provenance['source_image']).stem
        if name not in ('yaw_090','yaw_180','yaw_270'):continue
        yaw=int(name.split('_')[1]);angle=np.deg2rad(yaw)
        forward=np.array([np.sin(angle),np.cos(angle)])
        right=np.array([np.cos(angle),-np.sin(angle)])
        distance=(w if yaw in (90,270) else l)/2-meshes[i].extents[2]/2-.08
        box=provenance['source_box']
        offset=((box[0]+box[2])/2-.5)*2*np.tan(np.deg2rad(75)/2)*distance
        center=forward*distance+right*offset
        extent=np.array([meshes[i].extents[2],meshes[i].extents[0]]) if yaw in (90,270) else meshes[i].extents[[0,2]]
        bound=np.array([w,l])/2-extent/2-.04
        center=np.clip(center,-bound,bound)
        placement.append(dict(id=obj['id'],before=obj['position'],after=center.tolist(),source_yaw=yaw))
        obj['position']=center.tolist();obj['yaw']=yaw
    report['inferred_view_placement']=placement
    original={o['id']:o for o in original_objects}
    from .semantic_plan import resolve_supports
    by_name={o['id']:(i,o) for i,o in enumerate(plan['objects'])}
    for obj in resolve_supports(plan):
        parent=obj.get('support','floor')
        if parent=='floor':continue
        _,base=by_name[parent];old=original[parent]
        ratio=float(np.median(np.asarray(base['dimensions'])/old['dimensions']))
        angle=np.deg2rad(base['yaw']-old['yaw'])
        R=np.array([[np.cos(angle),np.sin(angle)],[-np.sin(angle),np.cos(angle)]])
        offset=np.asarray(original[obj['id']]['position'])-old['position']
        obj['position']=(np.asarray(base['position'])+R@offset*ratio).tolist()
        obj['yaw']+=base['yaw']-old['yaw']
        if not obj.get('size_locked'):
            index,_=by_name[obj['id']]
            meshes[index].apply_scale(ratio)
            obj['dimensions']=meshes[index].extents.tolist()
    report['camera_bound_warning']=[name for j,name in enumerate(('fov','eye_y','eye_z','pitch'))
        if min(values[j]-lower[j],upper[j]-values[j])<.01]
    plan['front_alignment']=report
    (folder/'front_alignment.json').write_text(json.dumps(report,indent=2))
    return plan


def audit_front(folder):
    """Actual z-buffered silhouette overlap, distinct from fitted box overlap."""
    import trimesh
    from .mesh_scene import MeshRenderer
    folder=Path(folder)
    meta=json.loads((folder/'scene.json').read_text())
    layout=json.loads((folder/'layout.json').read_text())
    geometry=np.load(folder/'mesh_geometry.npz')
    mesh=trimesh.Trimesh(geometry['vertices'],geometry['faces'],process=False)
    owners=geometry['owners']
    pose=np.asarray(meta['frames'][0]['transform_matrix'])
    hit=MeshRenderer(mesh,layout['components'],owners).render(np.asarray(meta['K']),pose,meta['size'])
    visible=hit['faces']>=0
    owner_image=np.full(visible.shape,-1,int)
    owner_image[visible]=owners[hit['faces'][visible]]
    plan=json.loads((folder/'planning/image_room_plan.json').read_text())
    names={c['name']:i for i,c in enumerate(layout['components'])}
    rows=[]
    for obj in plan['objects']:
        if obj.get('evidence')!='front':continue
        target=source_mask(folder,obj,(meta['size'],meta['size']))
        predicted=owner_image==names[obj['id']]
        union=(target|predicted).sum();intersection=(target&predicted).sum()
        rows.append(dict(id=obj['id'],silhouette_iou=float(intersection/max(1,union)),
                         target_recall=float(intersection/max(1,target.sum())),
                         predicted_precision=float(intersection/max(1,predicted.sum()))))
    report=dict(reference='source-image visible segmentation; no independent geometry ground truth',objects=rows,
                mean_silhouette_iou=float(np.mean([r['silhouette_iou'] for r in rows])),
                image_size=meta['size'],source_sha256=hashlib.sha256((folder/'source_front.png').read_bytes()).hexdigest())
    report['criterion']='mean silhouette IoU >=0.4 and each observed object >=0.2; initial engineering gate, not a research benchmark'
    report['passed']=bool(rows) and report['mean_silhouette_iou']>=.4 and all(r['silhouette_iou']>=.2 for r in rows)
    report['projection_fingerprint']=projection_fingerprint(meta)
    (folder/'front_silhouette_audit.json').write_text(json.dumps(report,indent=2))
    return report


def projection_fingerprint(meta):
    record=dict(K=meta['K'],camera=meta['frames'][0]['transform_matrix'],mesh_digest=meta['mesh_digest'])
    return hashlib.sha256(json.dumps(record,sort_keys=True).encode()).hexdigest()


def validate_front_audit(folder, meta):
    if not meta.get('front_alignment'):return
    folder=Path(folder)
    audit=json.loads((folder/'front_silhouette_audit.json').read_text())
    if not audit.get('passed'):
        raise ValueError('Front alignment quality gate failed; do not train GS on this teacher')
    if audit.get('projection_fingerprint')!=projection_fingerprint(meta):
        raise ValueError('Front alignment audit is stale for this camera/mesh')
    if audit.get('source_sha256')!=hashlib.sha256((folder/'source_front.png').read_bytes()).hexdigest():
        raise ValueError('Front source image changed after alignment audit')


def ensure_front_geometry(folder):
    from .learned_scene import backend_paths, backend_environment, run_worker
    folder=Path(folder)
    output=folder/'front_pointmap.npz'
    source_hash=hashlib.sha256((folder/'source_front.png').read_bytes()).hexdigest()
    manifest=folder/'front_pointmap_source.json'
    if output.exists() and manifest.exists() and json.loads(manifest.read_text()).get('source_sha256')==source_hash:return
    _,_,python,_=backend_paths()
    run_worker([python,Path(__file__).with_name('front_geometry_worker.py'),folder/'source_front.png',output],
               folder/'logs/front_geometry.log',backend_environment(python))
    manifest.write_text(json.dumps(dict(source_sha256=source_hash,model='Ruicheng/moge-vitl'),indent=2))


def refit_scene(path, output):
    """Clone source inputs only; never overwrite previous GS/reference results."""
    import shutil
    from .learned_scene import generate_learned_scene
    source=Path(path).resolve();target=Path(output).resolve()
    if source==target or source in target.parents or target in source.parents:
        raise ValueError('Refit output must be a separate sibling run')
    if target.exists() and any(target.iterdir()):raise FileExistsError('Use an empty output directory')
    config=json.loads((source/'request.json').read_text())
    if config.get('mode')!='image-room':raise ValueError('refit-front needs an image-room source run')
    target.mkdir(parents=True,exist_ok=True)
    for name in ('assets','planning','materials','room_images'):
        shutil.copytree(source/name,target/name)
    for name in ('source_front.png','image_room_provenance.json'):
        shutil.copy2(source/name,target/name)
    config['geometry_pipeline_version']=13
    if config.get('front_image_sha256'):
        # The supplied original may have been a JPEG or a non-square image.
        # This replay uses the saved canonical PNG, not the now-unneeded file.
        config['front_image_sha256']=hashlib.sha256((target/'source_front.png').read_bytes()).hexdigest()
    (target/'request.json').write_text(json.dumps(config,indent=2))
    options={key:config[key] for key in ('prompt','size','views','seed','room_size','planner_model')}
    if config.get('front_image_sha256'):
        options['front_image']=str(target/'source_front.png')
    return generate_learned_scene(output=target,resume=True,image_room=True,**options)
