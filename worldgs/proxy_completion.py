from pathlib import Path
import json,shutil,numpy as np,trimesh,subprocess
from PIL import Image
from worldgs.pipeline import Models
from worldgs.observed_architecture import observed_window,observed_rug
from worldgs.window_geometry import add_window
from worldgs.learned_scene import backend_paths,run_worker
from worldgs.learned_references import save_references
from worldgs.front_alignment import audit_front


def complete_proxy(path,output):
    src=Path(path).resolve();p=Path(output).resolve()
    if src==p or src in p.parents or p in src.parents:raise ValueError("Use a separate sibling output")
    if p.exists() and any(p.iterdir()):raise FileExistsError('Use an empty proxy output')
    p.mkdir(exist_ok=True)
    for name in ['assets','planning','materials','room_images']:shutil.copytree(src/name,p/name,dirs_exist_ok=True)
    for name in ['source_front.png','front_alignment.json','request.json','image_room_provenance.json']:shutil.copy2(src/name,p/name)
    m=json.loads((src/'scene.json').read_text());job=json.loads((src/'render_job.json').read_text());layout=json.loads((src/'layout.json').read_text());plan=job['plan'];front=job['frames'][0];K=np.array(front['K']);pose=np.array(front['transform_matrix']);size=m['size']
    detections=Models(84).detect(Image.open(p/'source_front.png').convert('RGB'),'',labels=['window','rug'],confidence=.25,text_confidence=.2,max_detections=8)
    (p/'planning/architecture_detections.json').write_text(json.dumps(detections,indent=2))
    parts=[trimesh.load(src/'meshes'/(c['name']+'.glb'),force='mesh',process=False) for c in layout['components']];components=layout['components'];evidence=[]
    windows=[d for d in detections if 'window' in d['label']]
    if windows:
     d=max(windows,key=lambda d:d['score']);plan['room']['window']=observed_window(d['box'],K,pose,plan['room']['size']);add_window(plan['room'],parts,components);evidence.append(dict(kind='window',detection=d,geometry=plan['room']['window']))
    rugs=[d for d in detections if 'rug' in d['label']]
    if rugs:
     d=max(rugs,key=lambda d:d['score']);lo,hi=observed_rug(d['box'],K,pose,plan['room']['size']);rug=trimesh.creation.box(extents=hi-lo);rug.apply_translation((hi+lo)/2);rug.visual.face_colors=[161,147,119,255];parts.append(rug);components.append(dict(name='observed_rug',group='rug',material='fabric',faces=len(rug.faces),bounds=rug.bounds.tolist()));evidence.append(dict(kind='rug',detection=d,bounds=rug.bounds.tolist()))
    plan['room']['floor']='wood';Image.new('RGB',(64,64),(139,100,65)).save(p/'materials/floor.png')
    scene=trimesh.Scene();vertices=[];faces=[];owners=[];offset=0
    (p/'meshes').mkdir()
    for i,(part,c) in enumerate(zip(parts,components)):
     scene.add_geometry(part,geom_name=c['name'],node_name=c['name']);part.export(p/'meshes'/(c['name']+'.glb'));vertices.append(part.vertices);faces.append(part.faces+offset);owners.append(np.full(len(part.faces),i,np.int32));offset+=len(part.vertices)
    mesh=trimesh.Trimesh(np.concatenate(vertices),np.concatenate(faces),process=False);owners=np.concatenate(owners)
    scene.export(p/'scene_mesh.glb');np.savez_compressed(p/'mesh_geometry.npz',vertices=mesh.vertices.astype(np.float32),faces=mesh.faces.astype(np.int32),owners=owners)
    (p/'layout.json').write_text(json.dumps(layout,indent=2));(p/'planning/architecture_evidence.json').write_text(json.dumps(evidence,indent=2));job['folder']=str(p);job['plan']=plan
    (p/'render_job.json').write_text(json.dumps(job,indent=2));(p/'planning/image_room_plan.json').write_text(json.dumps(plan,indent=2))
    _,_,_,blender=backend_paths();(p/'logs').mkdir();run_worker([blender,'-b','--python',Path(__file__).with_name('physical_render.py'),'--',p/'render_job.json'],p/'logs/render.log')
    meta=save_references(p,m['prompt'],m['seed'],size,mesh,components,owners,job)
    for key in ['mode','generation_source','input_image','front_alignment']:meta[key]=m[key]
    (p/'scene.json').write_text(json.dumps(meta,indent=2));audit_front(p)
    print(json.dumps(evidence,indent=2));print('COMPLETE',p)
    
    return meta
