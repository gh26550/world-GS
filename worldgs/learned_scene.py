"""Resumable prompt -> LLM graph -> learned meshes -> physical RGB-D -> GS."""
import gc
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime,timezone
import numpy as np
from PIL import Image
import trimesh
from .semantic_plan import plan_with_llm,validate_plan,repair_layout,resolve_supports,normalize_size_locks


def backend_paths():
    base=Path.home()/'worldmesh-main-optimized'
    root=Path(os.environ.get('WORLDGS_SAM3D_ROOT',base/'sam-3d-objects'))
    envs=Path.home()/'miniforge/envs'
    segment=Path(os.environ.get('WORLDGS_SAM3_PYTHON',envs/'worldmesh-sam3/bin/python'))
    reconstruct=Path(os.environ.get('WORLDGS_SAM3D_PYTHON',envs/'worldmesh-sam3d-objects/bin/python'))
    candidates=sorted((Path.home()/'software').glob('blender-*/blender'))
    blender=Path(os.environ.get('WORLDGS_BLENDER',shutil.which('blender') or (str(candidates[-1]) if candidates else '')))
    required=[root/'checkpoints/hf/pipeline.yaml',segment,reconstruct,blender]
    if any(not p.is_file() for p in required):
        raise FileNotFoundError('Missing generation backends: '+', '.join(str(p) for p in required if not p.is_file()))
    return root,segment,reconstruct,blender


def run_worker(command,log,environment=None,cwd=None):
    print('Running backend; log: '+str(log),flush=True)
    if log.exists():
        index=1
        while log.with_name(f'{log.stem}.attempt_{index:02d}.log').exists():
            index+=1
        shutil.copyfile(log,log.with_name(f'{log.stem}.attempt_{index:02d}.log'))
    with open(log,'w') as stream:
        with subprocess.Popen([str(x) for x in command],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                              text=True,bufsize=1,env=environment,cwd=cwd) as process:
            for line in process.stdout:
                stream.write(line)
                if any(marker in line for marker in ('Segmented ','Reconstructed ','Physical reference ',
                    'Sampling sparse structure:','Sampling sparse latent:','Postprocessing mesh with option')):
                    print(line.strip(),flush=True)
                    stream.flush()
            code=process.wait()
    if code:
        raise RuntimeError(f'Backend failed (exit {code}); inspect {log}. No placeholder mesh was substituted.')


def backend_environment(python,base=None):
    environment=(base or os.environ).copy()
    prefix=Path(python).parent.parent
    cu13=sorted(prefix.glob('lib/python*/site-packages/nvidia/cu13/lib'))
    others=sorted(prefix.glob('lib/python*/site-packages/nvidia/*/lib'))
    paths=cu13+[p for p in others if p not in cu13]+[prefix/'lib']
    environment['LD_LIBRARY_PATH']=':'.join(map(str,paths))+(':'+environment['LD_LIBRARY_PATH'] if environment.get('LD_LIBRARY_PATH') else '')
    environment['CONDA_PREFIX']=str(prefix)
    return environment


def usable_image(path):
    return path.exists() and np.asarray(Image.open(path)).mean()>1.


def configured_front_camera(layout,spec):
    from .mesh_scene import look_at
    eye,target=np.asarray(spec['eye'],float),np.asarray(spec['target'],float)
    if eye.shape!=(3,) or target.shape!=(3,) or not np.isfinite([eye,target]).all() or np.linalg.norm(eye-target)<.1:
        raise ValueError('Invalid reference camera')
    lo,hi=np.asarray(layout[0]['min']),np.asarray(layout[0]['max'])
    if not ((eye>lo+.08)&(eye<hi-.08)).all() or any(
        ((eye>np.asarray(s['min'])-.05)&(eye<np.asarray(s['max'])+.05)).all() for s in layout[1:]):
        raise ValueError('Reference camera is outside room or inside furniture')
    return look_at(eye,target)


def learned_cameras(layout,views):
    from .mesh_scene import teacher_camera_set,look_at
    from .room import pose_at
    cameras=teacher_camera_set(layout,views)
    lo,hi=np.asarray(layout[0]['min']),np.asarray(layout[0]['max'])
    center=(lo+hi)/2
    size=hi-lo
    def safe(eye):
        return ((eye>lo+.08)&(eye<hi-.08)).all() and not any(
            ((eye>np.asarray(s['min'])-.08)&(eye<np.asarray(s['max'])+.08)).all() for s in layout[1:])
    # Close translated views keep small accessories resolved in the references.
    for shape in layout[1:]:
        a,b=np.asarray(shape['min']),np.asarray(shape['max'])
        target=(a+b)/2
        radius=max(.55,float(np.linalg.norm(b-a))*.9)
        for angle in range(0,360,60):
            radians=np.deg2rad(angle)
            eye=target+radius*np.array([np.sin(radians),-.45,np.cos(radians)])
            if safe(eye):
                cameras.append((f'object_detail_{shape["name"]}_{angle}',look_at(eye,target)))
    # Diagonally grazing room edges were missing from the colour-initialized
    # seed cloud. Explicit corner views and vertical translated views observe
    # those surfaces directly instead of hallucinating/filling their geometry.
    for x in (-1,1):
        for y in (-1,1):
            for z in (-1,1):
                target=center+np.array([x,y,z])*(size/2-.05)
                cameras.append((f'structure_corner_{x}_{y}_{z}',look_at(center,target)))
    for ix,x in enumerate((-.3,0,.3)):
        for iz,z in enumerate((-.3,0,.3)):
            for pitch,height in [(90,.22),(-90,-.22)]:
                eye=center+size*np.array([x,height,z])
                if safe(eye):
                    cameras.append((f'structure_grid_{ix}_{iz}_{pitch}',pose_at(0,pitch,eye)))
    return cameras


def generate_images(plan,folder,seed,retry_ids=None):
    from .pipeline import Models
    import torch
    from diffusers import StableDiffusionPipeline
    objects=plan['objects']
    missing=[o for o in objects if o['id'] in retry_ids] if retry_ids is not None else [o for o in objects if not (folder/'assets'/o['id']/'candidate_03.png').exists() and not (folder/'assets'/o['id']/'reference_asset.png').exists()]
    materials=[('floor',plan['room']['floor']),('walls',plan['room']['walls']),('ceiling',plan['room']['ceiling'])]
    missing_materials=[(n,d) for n,d in materials if not usable_image(folder/'materials'/f'{n}.png')]
    if not missing and not missing_materials:
        return
    models=Models(seed)
    pipe=StableDiffusionPipeline.from_pretrained('stable-diffusion-v1-5/stable-diffusion-v1-5',torch_dtype=models.dtype).to(models.device)
    pipe.enable_attention_slicing()
    try:
        for index,obj in enumerate(objects):
            if retry_ids is not None and obj['id'] not in retry_ids:
                continue
            directory=folder/'assets'/obj['id']
            directory.mkdir(parents=True,exist_ok=True)
            if (directory/'reference_asset.png').exists():
                continue
            existing=sorted(directory.glob('candidate_*.png'))
            start=max([int(p.stem.split('_')[-1]) for p in existing],default=-1)+1 if retry_ids is not None else 0
            for candidate in range(start,start+4):
                filename=directory/f'candidate_{candidate:02d}.png'
                if filename.exists():
                    continue
                def limited(text,count):
                    return pipe.tokenizer.decode(pipe.tokenizer.encode(text,add_special_tokens=False)[:count])
                prompt=('Wide distant product photograph of one small '+obj['label']+
                        ', centered with ample white space, entire object fully visible. '+limited(obj['description'],23)+
                        '. '+limited(plan['style'],10)+'. White studio background.')
                tall=bool(re.search(r'bookcase|bookshelf',obj['label'],re.I))
                if tall:
                    prompt=('Full-length photograph of ONE tall narrow freestanding bookcase, entire frame and feet visible with empty margins. '+
                            limited(obj['description'],25)+'. '+limited(plan['style'],10)+'. White studio background.')
                image=pipe(prompt,negative_prompt='close-up, cropped, cut off, multiple objects, room, interior, walls, people, text, watermark, collage, duplicate, floor platform',
                    width=512 if tall else (768 if candidate>=2 else 512),height=768 if tall else 512,num_inference_steps=35,guidance_scale=7.,
                    generator=torch.Generator(models.device).manual_seed(seed+index*19+candidate)).images[0]
                image.save(filename)
                filename.with_suffix('.json').write_text(json.dumps(dict(model='stable-diffusion-v1-5/stable-diffusion-v1-5',
                    prompt=prompt,seed=seed+index*19+candidate,steps=35,size=list(image.size)),indent=2))
                (directory/'image_prompt.txt').write_text(prompt,encoding='utf-8')
        (folder/'materials').mkdir(exist_ok=True)
        for index,(name,description) in enumerate(materials):
            filename=folder/'materials'/f'{name}.png'
            if usable_image(filename):
                continue
            prompt=f'Seamless flat material texture of {description}, orthographic close up surface, uniform illumination, albedo texture, physically realistic, no objects, no perspective'
            for attempt in range(3):
                image=pipe(prompt,negative_prompt='room, furniture, windows, doors, perspective, objects, text, shadows',width=512,height=512,num_inference_steps=30,
                           generator=torch.Generator(models.device).manual_seed(seed+100+index+attempt*1000)).images[0]
                if np.asarray(image).mean()>1.:
                    image.save(filename)
                    break
            else:
                raise RuntimeError('Material generation returned black/rejected images: '+name)
    finally:
        del pipe,models
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def assemble(plan,folder):
    from .mesh_scene import build_components
    w,h,l=plan['room']['size']
    room=dict(name='room',kind='room',min=[-w/2,-h/2,-l/2],max=[w/2,h/2,l/2])
    _,parts,components,_=build_components([room],Image.new('RGB',(64,64),'ivory'))
    from .window_geometry import add_window
    add_window(plan['room'],parts,components)
    geometries=list(parts)
    scene=trimesh.Scene()
    for i,part in enumerate(parts):
        scene.add_geometry(part,geom_name=components[i]['name'],node_name=components[i]['name'])
    layout=[room]
    normalized=[]
    for obj in plan['objects']:
        source=folder/'assets'/obj['id']/'generated.glb'
        mesh=trimesh.load(source,force='mesh',process=False)
        if len(mesh.faces)<100 or not np.isfinite(mesh.vertices).all():
            raise ValueError('Invalid generated mesh: '+obj['id'])
        mesh.update_faces(mesh.nondegenerate_faces())
        # SAM3D GLB is y-up, front +z. Rotate x by pi into camera-style y-down,
        # front -z. Fit uniformly inside the planned envelope (no anisotropic warp).
        mesh.apply_transform(np.diag([1.,-1.,-1.,1.]))
        extent=mesh.extents
        if (extent<1e-4).any():
            raise ValueError('Generated object is degenerate: '+obj['id'])
        # Dimensions from language are uncertain priors. A robust uniform scale
        # preserves generated aspect ratio; placement is solved again afterward.
        scale=float(min(np.median(np.asarray(obj['dimensions'])/extent),(h-.15)/extent[1]))
        mesh.apply_scale(scale)
        if re.search(r'\b(rug|carpet|mat)\b',obj['label'],re.I):
            mesh.apply_scale([1,.012/mesh.extents[1],1])
            obj['floor_covering_thickness']=.012
        if obj.get('size_locked') and (np.abs(mesh.extents/np.asarray(obj['dimensions'])-1)>.15).any():
            raise ValueError('Generated shape cannot meet explicit dimensions without distortion: '+obj['id'])
        center=(mesh.bounds[0]+mesh.bounds[1])/2
        mesh.apply_translation(-center)
        obj['planned_dimensions']=obj['dimensions']
        obj['dimensions']=mesh.extents.tolist()
        obj['uniform_scale']=scale
        normalized.append(mesh)
    plan=repair_layout(plan)
    validate_plan(plan)
    normalized={o['id']:m for o,m in zip(plan['objects'],normalized)}
    placed={}
    contacts=[]
    for obj in resolve_supports(plan):
        mesh=normalized[obj['id']]
        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.deg2rad(obj['yaw']),[0,1,0]))
        x,z=obj['position']
        mesh.apply_translation([x,obj['bottom_y']-mesh.bounds[1,1],z])
        parent=obj.get('support','floor')
        if parent!='floor':
            bottom=contact_height(placed[parent],mesh)
            mesh.apply_translation([0,bottom-mesh.bounds[1,1],0])
            contacts.append(dict(object=obj['id'],support=parent,bottom_y=bottom))
        placed[obj['id']]=mesh
        mesh.export(folder/'assets'/obj['id']/'placed.glb')
        actual=mesh.bounds.tolist()
        components.append(dict(name=obj['id'],group=obj['id'],material='generated_asset',faces=len(mesh.faces),bounds=actual))
        geometries.append(mesh)
        layout.append(dict(name=obj['id'],label=obj['label'],kind='generated_asset',min=actual[0],max=actual[1]))
        scene.add_geometry(mesh,geom_name=obj['id'],node_name=obj['id'])
        obj['realized_dimensions']=mesh.extents.tolist()
    (folder/'meshes').mkdir(exist_ok=True)
    for mesh,part in zip(geometries,components):
        mesh.export(folder/'meshes'/f'{part["name"]}.glb')
    scene.export(folder/'scene_mesh.glb')
    # Concatenation intentionally drops UV materials here: this array is only
    # for intersections, IDs and seed attachment. GLB retains per-object UVs.
    vertices,faces,owners=[],[],[]
    offset=0
    for i,mesh in enumerate(geometries):
        vertices.append(mesh.vertices)
        faces.append(mesh.faces+offset)
        owners.append(np.full(len(mesh.faces),i,np.int32))
        offset+=len(mesh.vertices)
    combined=trimesh.Trimesh(np.concatenate(vertices),np.concatenate(faces),process=False)
    owners=np.concatenate(owners)
    np.savez_compressed(folder/'mesh_geometry.npz',vertices=combined.vertices.astype(np.float32),faces=combined.faces.astype(np.int32),owners=owners)
    (folder/'layout.json').write_text(json.dumps(dict(shapes=layout,components=components),indent=2))
    (folder/'assembly.json').write_text(json.dumps(dict(objects=plan['objects'],source='SAM3D generated UV-textured meshes',
        placeholder_fallback=False,support_contacts=contacts),indent=2))
    return combined,components,owners,layout


def contact_height(parent,child):
    """Check actual support triangles at a 3x3 footprint grid, not just its box."""
    from trimesh.ray.ray_pyembree import RayMeshIntersector
    lo,hi=child.bounds
    origins=np.array([[x,parent.bounds[0,1]-.1,z]
        for x in np.linspace(lo[0],hi[0],3) for z in np.linspace(lo[2],hi[2],3)])
    directions=np.tile([0.,1.,0.],(len(origins),1))
    hits,ids,_=RayMeshIntersector(parent).intersects_location(origins,directions,multiple_hits=False)
    if len(np.unique(ids))!=len(origins):
        raise ValueError('Accessory footprint is not fully supported by the actual mesh')
    if np.ptp(hits[:,1])>.035:
        raise ValueError('Support surface is not sufficiently flat for this accessory')
    return float(hits[:,1].min())


def generate_learned_scene(prompt,output,size,views,seed,room_size=None,resume=False,planner_model=None):
    if size<64 or size%8 or views<8:
        raise ValueError('size >=64 divisible by8; views>=8')
    root,segment,reconstruct,blender=backend_paths()
    folder=Path(output).resolve()
    config=dict(prompt=prompt,size=size,views=views,seed=seed,room_size=list(room_size) if room_size else None,planner_model=planner_model)
    folder.mkdir(parents=True,exist_ok=True)
    manifest=folder/'request.json'
    if any(folder.iterdir()):
        if not resume or not manifest.exists() or json.loads(manifest.read_text())!=config:
            raise FileExistsError('Use a new output directory, or --resume with the identical request')
    manifest.write_text(json.dumps(config,indent=2))
    (folder/'logs').mkdir(exist_ok=True)
    def status(stage,**details):
        record=dict(stage=stage,success=None,updated_at=datetime.now(timezone.utc).isoformat(),**details)
        (folder/'status.json').write_text(json.dumps(record,indent=2))
        print('WorldGS stage: '+stage,flush=True)
    try:
        status('planning')
        plan_file=folder/'planning/scene_plan.json'
        if plan_file.exists():
            plan=normalize_size_locks(json.loads(plan_file.read_text()),prompt)
            plan_file.write_text(json.dumps(plan,indent=2,ensure_ascii=False),encoding='utf-8')
            validate_plan(plan)
        else:
            options=dict(model_id=planner_model) if planner_model else {}
            plan=plan_with_llm(prompt,folder/'planning',room_size,**options)
        status('object_images',objects=len(plan['objects']))
        generate_images(plan,folder,seed)
        job=dict(seed=seed,objects=[dict(id=o['id'],label=o['label'],folder=str(folder/'assets'/o['id']),instance_of=o.get('asset_instance_of')) for o in plan['objects']])
        (folder/'asset_job.json').write_text(json.dumps(job,indent=2))
        worker=Path(__file__).with_name('asset_worker.py')
        environment=os.environ.copy()
        environment.update(ATTN_BACKEND='sdpa',SPARSE_ATTN_BACKEND='sdpa',WORLDMESH_LOW_VRAM='1',XFORMERS_DISABLED='1')
        if any(not (folder/'assets'/o['id']/'segmentation.json').exists() for o in plan['objects']):
            status('segmentation')
            for attempt in range(7):
                try:
                    run_worker([segment,worker,'segment',folder/'asset_job.json','--root',root],folder/'logs/segment.log',backend_environment(segment,environment))
                    break
                except RuntimeError:
                    match=re.search(r'Could not isolate ([a-z][a-z0-9_]*):',(folder/'logs/segment.log').read_text())
                    if match is None or attempt==6:
                        raise
                    status('retry_object_images',object=match.group(1),attempt=attempt+1)
                    generate_images(plan,folder,seed,retry_ids={match.group(1)})
        if any(not (folder/'assets'/o['id']/'asset.json').exists() for o in plan['objects']):
            status('mesh_generation')
            run_worker([reconstruct,worker,'reconstruct',folder/'asset_job.json','--root',root],folder/'logs/reconstruct.log',backend_environment(reconstruct,environment),cwd=root)
        status('assembly')
        mesh,components,owners,layout=assemble(plan,folder)
        from .geometry import intrinsics
        cameras=learned_cameras(layout,views)
        if plan.get('reference_camera'):
            cameras[0]=('front',configured_front_camera(layout,plan['reference_camera']))
        frames=[]
        for i,(target,pose) in enumerate(cameras):
            prefix='heldout' if i%5==2 and not target.startswith('structure_') else 'view'
            index=sum(f['split']==prefix for f in frames)
            frames.append(dict(file_path=f'{prefix}_{index:03d}.png',split=prefix,target=target,transform_matrix=pose.tolist()))
        render_job=dict(folder=str(folder),size=size,K=intrinsics(size).tolist(),frames=frames,plan=plan)
        (folder/'render_job.json').write_text(json.dumps(render_job,indent=2))
        status('physical_references',cameras=len(frames))
        run_worker([blender,'-b','--python',Path(__file__).with_name('physical_render.py'),'--',folder/'render_job.json'],folder/'logs/render.log')
        from .learned_references import save_references
        status('reference_validation')
        result=save_references(folder,prompt,seed,size,mesh,components,owners,render_job)
        (folder/'status.json').write_text(json.dumps(dict(stage='references_ready',success=True),indent=2))
        return result
    except Exception as error:
        (folder/'status.json').write_text(json.dumps(dict(stage='failed',success=False,error=str(error)),indent=2))
        raise
