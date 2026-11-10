"""Occlusion-aware source-image fitting of rigid, uniformly scaled meshes."""
import numpy as np
from PIL import Image
from scipy.optimize import minimize


def refine(values,lower,upper,meshes,observed,masks,camera,residual,source_size,resolution=96):
    import trimesh
    from .mesh_scene import MeshRenderer
    from .reference_coverage import scaled_intrinsics
    targets=np.stack([np.asarray(Image.fromarray(m).resize((resolution,resolution),Image.Resampling.NEAREST),bool) for m in masks])
    weights=targets.sum((1,2))/(targets.sum((1,2))+12.)
    vertices=[meshes[i].vertices for i,_ in observed]
    faces=[];owners=[];offset=0
    for j,(i,_) in enumerate(observed):
        faces.append(meshes[i].faces+offset);owners.append(np.full(len(meshes[i].faces),j,np.int32));offset+=len(vertices[j])
    faces=np.concatenate(faces);owners=np.concatenate(owners);calls=0
    def evaluate(v,details=False):
        nonlocal calls
        calls+=1;world=[]
        for j,(i,obj) in enumerate(observed):
            x,z,logscale,yaw=v[4+j*4:8+j*4];scale=np.exp(logscale);a=np.deg2rad(obj['yaw'])+yaw
            R=np.array([[np.cos(a),0,np.sin(a)],[0,1,0],[-np.sin(a),0,np.cos(a)]])
            world.append(vertices[j]*scale@R.T+[x,obj['bottom_y']-meshes[i].bounds[1,1]*scale,z])
        mesh=trimesh.Trimesh(np.concatenate(world),faces,process=False)
        comps=[dict(name=o['id'],group=o['id'],material='generated_asset') for _,o in observed]
        K,pose=camera(v);hit=MeshRenderer(mesh,comps,owners).render(scaled_intrinsics(K,source_size,resolution),pose,resolution)
        ids=np.where(hit['faces']>=0,owners[np.maximum(hit['faces'],0)],-1)
        ious=np.array([np.logical_and(ids==j,t).sum()/max(1,np.logical_or(ids==j,t).sum()) for j,t in enumerate(targets)])
        # Box/depth/room priors remain, but cannot hide incorrect occlusion.
        score=float(np.sum(weights*(1-ious))+.025*np.sum(np.minimum(np.asarray(residual(v))**2,4)))
        return (score,ious.tolist()) if details else score
    best=np.array(values);before=evaluate(best,True);trials=[]
    for _ in range(2):
        for j,(_,obj) in enumerate(observed):
            start=4+4*j;indices=np.arange(start,start+4);lo=np.asarray(lower)[indices].copy();hi=np.asarray(upper)[indices].copy()
            lo[-1],hi[-1]=-np.pi,np.pi
            def objective(part):
                v=best.copy();v[indices]=part;return evaluate(v)
            part=best[indices].copy();cost=objective(part)
            for angle in [-np.pi,-np.pi/2,0,np.pi/2]:
                trial=part.copy();trial[-1]=angle;c=objective(trial)
                if c<cost:part,cost=trial,c
            result=minimize(objective,part,method='Powell',bounds=list(zip(lo,hi)),options=dict(maxfev=110,xtol=.015,ftol=.002))
            if result.fun<evaluate(best):best[indices]=result.x
            trials.append(dict(object=obj['id'],score=float(evaluate(best))))
    after=evaluate(best,True)
    return best,dict(method='rigid uniform-scale source silhouette and occlusion refinement',resolution=resolution,before_score=before[0],after_score=after[0],before_iou=before[1],after_iou=after[1],evaluations=calls,trials=trials,limitations=['Visible segmentation can be wrong','Hidden geometry remains inferred','No anisotropic mesh deformation'])
