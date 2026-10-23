"""Attach physically rendered references and initial colours to the same mesh."""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from .mesh_scene import MeshRenderer
from .teacher_geometry import sample_teacher_surface,structure_report,mesh_digest
from .mesh_pipeline import make_contact


def save_references(folder,prompt,seed,size,mesh,components,owners,job):
    folder=Path(folder)
    K=np.asarray(job['K'])
    renderer=MeshRenderer(mesh,components,owners)
    visible=np.zeros(len(mesh.faces),bool)
    frames,heldout=[],[]
    for source in job['frames']:
        frame=source.copy()
        split=frame.pop('split')
        stem=Path(frame['file_path']).stem
        hit=renderer.render(K,np.asarray(frame['transform_matrix']),size)
        valid=hit['faces']>=0
        for field,data in [('depth',hit['depth']),('mask',valid),('normals',hit['normals'].astype(np.float32)),('faces',hit['faces'])]:
            filename=f'{stem}_{field}.npy'
            np.save(folder/filename,data)
            frame[{'normals':'normal','faces':'face'}.get(field,field)+'_file']=filename
        frame['reference_source']='Cycles RGB and matching exact mesh depth/normal/ID'
        if split=='view':
            frames.append(frame)
            visible[hit['faces'][valid]]=True
        else:
            heldout.append(frame)
    points=sample_teacher_surface(mesh,renderer,owners,visible,seed=seed)
    scores=np.zeros(len(points['xyz']))
    for frame in frames:
        pose=np.asarray(frame['transform_matrix'])
        camera=(points['xyz']-pose[:3,3])@pose[:3,:3]
        uvh=camera@K.T
        uv=np.rint(uvh[:,:2]/np.maximum(uvh[:,2:],1e-8)).astype(int)
        valid=(camera[:,2]>.03)&(uv[:,0]>=0)&(uv[:,0]<size)&(uv[:,1]>=0)&(uv[:,1]<size)
        ids=np.flatnonzero(valid)
        x,y=uv[ids].T
        depth=np.load(folder/frame['depth_file'])[y,x]
        faces=np.load(folder/frame['face_file'])[y,x]
        visible=(faces>=0)&(owners[np.maximum(faces,0)]==points['component_ids'][ids])&(np.abs(depth-camera[ids,2])<.02+.01*depth)
        ids=ids[visible]
        if not len(ids):
            continue
        ray=points['xyz'][ids]-pose[:3,3]
        distance=np.linalg.norm(ray,axis=1)
        incidence=np.abs((points['normals'][ids]*ray).sum(1)/distance)
        weight=incidence/np.maximum(distance,.3)
        better=weight>scores[ids]
        ids=ids[better]
        x,y=uv[ids].T
        points['rgb'][ids]=np.asarray(Image.open(folder/frame['file_path']).convert('RGB'))[y,x]
        scores[ids]=weight[better]
    keep=scores>0
    points={name:value[keep] for name,value in points.items()}
    if len(points['xyz'])<1000:
        raise ValueError('Insufficient physically observed surface samples')
    np.savez_compressed(folder/'points.npz',**points)
    quality=structure_report(mesh,components,points)
    (folder/'structure_quality.json').write_text(json.dumps(quality,indent=2))
    if not quality['passed']:
        raise ValueError('Assembled room/seed geometry failed checks')
    meta=dict(mode='learned_mesh_v10',prompt=prompt,seed=seed,demo=False,size=size,K=K.tolist(),frames=frames,heldout_frames=heldout,
        geometry_constraint='soft_mesh_structure_prior',convention='OpenCV camera-to-world, x right y down z forward',
        mesh_digest=mesh_digest(mesh.vertices,mesh.faces),mesh_file_sha256=hashlib.sha256((folder/'scene_mesh.glb').read_bytes()).hexdigest(),
        point_count=len(points['xyz']),mesh_triangles=len(mesh.faces),mesh_components=len(components),
        room_size_prior=job['plan']['room']['size'],planner=job['plan']['planner'],
        reference_source='UV textured SAM3D meshes with physically rendered room materials and lighting',
        depth_source='ray triangle intersection; no per-view depth prediction',
        limitations=['LLM completeness audit is not a proof of arbitrary prompt satisfaction',
          'Single-image object backsides are inferred, and planned dimensions are priors',
          'Learned textures may already contain illumination; not intrinsic albedo recovery',
          'GS bakes illumination; Unity dynamic relighting requires a separate renderer',
          'Heldout mesh renders are not independent real-world ground truth'])
    (folder/'scene.json').write_text(json.dumps(meta,indent=2))
    Image.open(folder/frames[0]['file_path']).save(folder/'front.png')
    make_contact(folder,frames,'references.jpg')
    make_contact(folder,heldout,'heldout_references.jpg')
    return meta
