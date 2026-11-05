"""Local camera-centred room reference generation with depth verification."""
import gc,hashlib,json,shutil
from pathlib import Path
import numpy as np
from PIL import Image
import trimesh
from scipy.ndimage import binary_erosion
from .mesh_scene import MeshRenderer
from .room_cameras import room_camera_plan,augment_coverage,rotational_order
from .room_verification import verify_depth,verify_layout,camera_context,instance_boundaries
from .appearance_geometry import depth_condition,transfer_appearance
from .local_runtime import LocalRuntime

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2))

def load_proxy(src):
    g=np.load(src/'mesh_geometry.npz');layout=json.loads((src/'layout.json').read_text());mesh=trimesh.Trimesh(g['vertices'],g['faces'],process=False);colors=[]
    for c in layout['components']:
        part=trimesh.load(src/'meshes'/(c['name']+'.glb'),force='mesh',process=False)
        if part.visual.kind=='texture':part.visual=part.visual.to_color()
        colors.append(part.visual.face_colors)
    mesh.visual.face_colors=np.concatenate(colors);return mesh,layout,g['owners']

def validate_local(folder,meta):
    if not meta.get('local_room_audit'):return
    from .appearance import projection_fingerprint
    folder=Path(folder);a=json.loads((folder/meta['local_room_audit']).read_text())
    if not a.get('complete') or not a.get('passed'):raise ValueError('Local room references incomplete or rejected')
    if a['projection_sha256']!=projection_fingerprint(meta) or a['mesh_sha256']!=sha(folder/'scene_mesh.glb') or a['source_sha256']!=sha(folder/'source_front.png'):raise ValueError('Local room geometry/camera/source changed')
    if {r['file_path'] for r in a['views']}!={f['file_path'] for f in meta['frames']}:raise ValueError('Local audit missing training camera')
    for row in a['views']:
        if not row['passed'] or sha(folder/row['file_path'])!=row['teacher_sha256']:raise ValueError('Local reference RGB changed')
        if row.get('record_file'):
            if sha(folder/row['record_file'])!=row['record_sha256']:raise ValueError('Local acceptance record changed')
            record=json.loads((folder/row['record_file']).read_text())
            if record.get('appearance_reference') and sha(folder/record['appearance_reference'])!=record['appearance_reference_sha256']:raise ValueError('Local appearance reference changed')
            for field in ['raw','guide','mesh_depth','predicted_depth']+(['boundaries'] if 'boundaries' in record else [])+(['proxy_depth'] if 'proxy_depth' in record else []):
                if sha(folder/record[field])!=record[field+'_sha256']:raise ValueError('Local reference evidence changed: '+field)

def generate_room_views(path,output,perimeter=16,overhead=8,adaptive=8,fov=75,pilot_views=None,prepare_only=False,resume=False,retries=3,backend_root=None,port=8197,train_steps=0,image_model="controlnet",conditioning="structure"):
    src=Path(path).resolve();folder=Path(output).resolve()
    if src==folder or src in folder.parents or folder in src.parents:raise ValueError('Use a separate sibling output')
    if retries<1 or retries>5 or adaptive<0 or adaptive>32:raise ValueError('Invalid retry/adaptive count')
    original=json.loads((src/'scene.json').read_text());size=original['size'];mesh,layout,owners=load_proxy(src);renderer=MeshRenderer(mesh,layout['components'],owners)
    frames=room_camera_plan(layout['shapes'],size,perimeter,overhead,fov);frames,coverage=augment_coverage(renderer,frames,layout['shapes'],size,adaptive);frames=rotational_order(frames)
    runtime=LocalRuntime(folder/'logs',backend_root,port,image_model);models=runtime.preflight()
    request=dict(version=33,conditioning=conditioning,layout_model="IDEA-Research/grounding-dino-tiny",source=str(src),mesh_sha256=sha(src/'scene_mesh.glb'),source_sha256=sha(src/'source_front.png'),seed=original['seed'],size=size,perimeter=perimeter,overhead=overhead,adaptive=adaptive,fov=fov,retries=retries,models=models,cameras=frames)
    manifest=folder/'local_request.json'
    if folder.exists() and any(folder.iterdir()):
        if not resume or not manifest.exists() or json.loads(manifest.read_text())!=request:raise FileExistsError('Use empty output or identical --resume request')
    else:
        folder.mkdir(parents=True,exist_ok=True)
        for name in ['assets','planning','materials','room_images','meshes']:
            if (src/name).exists():shutil.copytree(src/name,folder/name)
        for name in ['scene_mesh.glb','mesh_geometry.npz','layout.json','source_front.png','front_alignment.json','image_room_provenance.json','request.json']:
            if (src/name).exists():shutil.copy2(src/name,folder/name)
        write(manifest,request)
    for name in ['guides','raw','records','depth','logs']:(folder/name).mkdir(exist_ok=True)
    if conditioning not in ['structure','atlas']:raise ValueError('Use structure or atlas conditioning')
    from .appearance_reference import appearance_atlas
    appearance_atlas(Image.open(folder/'source_front.png'),original['seed']).save(folder/'source_appearance_atlas.png')
    write(folder/'camera_plan.json',dict(cameras=frames,coverage=coverage,convention='OpenCV x right y down z forward'))
    def status(stage,**kw):write(folder/'status.json',dict(stage=stage,**kw))
    # Render all exact-camera evidence before any AI call. No source-photo projection.
    for f in frames:
        stem=Path(f['file_path']).stem;hit=renderer.render(np.array(f['K']),np.array(f['transform_matrix']),size)
        Image.fromarray(hit['rgb']).save(folder/'guides'/(stem+'_proxy.png'));Image.fromarray(depth_condition(hit['depth'])).save(folder/'guides'/(stem+'_depth.png'))
        np.save(folder/'depth'/(stem+'_mesh.npy'),hit['depth']);Image.fromarray(instance_boundaries(hit,owners)).save(folder/'guides'/(stem+'_boundaries.png'))
    if prepare_only:status('local_guides_ready',cameras=len(frames),success=True);return request
    accepted=[];cache=[]
    try:
        with runtime:
            for f in frames:
                if pilot_views is not None and len(accepted)>=pilot_views:status('local_pilot_ready',completed=len(accepted),total=len(frames),trainable=False);return dict(completed=len(accepted),total=len(frames),output=str(folder))
                stem=Path(f['file_path']).stem;record_file=folder/'records'/(stem+'.json');hit=renderer.render(np.array(f['K']),np.array(f['transform_matrix']),size);valid=hit['faces']>=0;ids=owners[np.maximum(hit['faces'],0)];architecture=valid&(ids<6)
                if record_file.exists():
                    record=json.loads(record_file.read_text())
                    if not record['passed'] or record['teacher_sha256']!=sha(folder/f['file_path']):raise ValueError('Stale cached photograph')
                    rgb=np.asarray(Image.open(folder/f['file_path']).convert('RGB'))
                else:
                    guide=hit['rgb'].copy();style=folder/'source_appearance_atlas.png'
                    if cache:
                        transferred,known,_=transfer_appearance(hit,f,cache);guide[known&architecture]=transferred[known&architecture]
                    guide_file=folder/'guides'/(stem+'_condition.png');Image.fromarray(guide).save(guide_file)
                    from .layout_consistency import layout_contract
                    plan=json.loads((folder/'planning/image_room_plan.json').read_text());contract=layout_contract(dict(vertices=mesh.vertices,faces=mesh.faces,owners=owners),layout['components'],plan,sha(folder/'scene_mesh.glb'),sha(folder/'source_front.png'));labels={o['id']:o['label'] for o in contract['objects']}
                    context,projected=camera_context(hit,owners,layout['components'],labels)
                    prompt=('Edit IMAGE 1 ONLY: turn this room proxy into a photorealistic photograph, preserving its exact camera, perspective, furniture positions, sizes, orientations, visible silhouettes and occlusions. '+context+' Image 2 is an UNORDERED appearance-patch atlas from the initial room photo: use its materials and colors, never a layout. Image 3 is the exact mesh DEPTH MAP for image 1. Image 4, when supplied, is an UNORDERED appearance atlas from a nearby accepted camera for consistent materials only. Do not change camera or zoom. Fill guide-visible bookcase compartments with books; side panels stay wood. Preserve the guide rug shape. One square opaque photograph, no text, borders, collage or panels.')
                    if conditioning=='structure':
                        descriptions=[o.get('description',o['label']) for o in plan['objects'] if o['id'] in {row['id'] for row in projected}]
                        prompt=('Edit IMAGE 1 only into one photorealistic interior photograph. Keep EXACTLY its camera, framing, perspective, object positions, sizes, orientations, silhouettes and occlusions. Image 2 is its exact depth map. '+context+' Visible-item appearance: '+', '.join(descriptions)+'. Room materials: '+str(plan.get('style',''))+'. Realistic warm daylight and material detail. Fill guide-visible bookshelves with books. No camera zoom or rotation, added furniture or openings. One square image, no text or borders.')
                    previous=None
                    if cache and conditioning=='atlas':
                        rotation=np.array(f['transform_matrix'])[:3,:3];previous=max(cache,key=lambda e:np.trace(rotation.T@np.array(e['frame']['transform_matrix'])[:3,:3]))['image_path']
                    if previous is not None:
                        previous_image=previous;previous=folder/'guides'/(stem+'_previous_appearance.png');appearance_atlas(Image.open(previous_image),original['seed']).save(previous)
                    proxy_depth_file=folder/'depth'/(stem+'_proxy_predicted.npy')
                    if not proxy_depth_file.exists():runtime.depth(folder/'guides'/(stem+'_proxy.png'),f['K'][0][0],proxy_depth_file)
                    attempts=[]
                    for attempt in range(retries):
                        status('local_generation',camera=f['target'],completed=len(accepted),total=len(frames),attempt=attempt+1,success=None)
                        raw=folder/'raw'/f'{stem}_{attempt:02d}.png';runtime.generate(guide_file,style,prompt,original['seed']+len(accepted)*101+attempt,raw,size,kind='structure' if conditioning=='structure' else 'edit',previous=previous,depth_guide=folder/'guides'/(stem+'_depth.png'),visible_labels=[o['label'] for o in projected],appearance=plan.get('style','')+'; '+('; '.join(o.get('description',o['label']) for o in plan['objects'] if o['id'] in {r['id'] for r in projected})),edge_guide=folder/'guides'/(stem+'_boundaries.png'))
                        image=Image.open(raw).convert('RGB')
                        if abs(image.width/image.height-1)>.002:raise ValueError('Generated aspect ratio differs; cropping forbidden')
                        image=image.resize((size,size),Image.Resampling.BOX);rgb=np.asarray(image);teacher=folder/f['file_path'];image.save(teacher)
                        predicted_file=folder/'depth'/f'{stem}_predicted_{attempt:02d}.npy';runtime.depth(teacher,f['K'][0][0],predicted_file);predicted=np.load(predicted_file);check=verify_depth(hit['depth'],predicted,tolerance=round(10*size/512),proxy_depth=np.load(proxy_depth_file),rgb=rgb)
                        layout_file=folder/'records'/f'{stem}_layout_{attempt:02d}.json';runtime.layout(teacher,projected,layout_file);detections=json.loads(layout_file.read_text());check['layout']=verify_layout(projected,detections['detections'],rgb,hit=hit,owners=owners,predicted_depth=predicted);check['depth_passed']=check['passed'];check['passed']=check['passed'] and check['layout']['passed'];attempts.append(check)
                        write(folder/'records'/(stem+'_attempts.json'),attempts)
                        if check['passed']:break
                    if not check['passed']:raise ValueError('Depth/layout verification rejected camera '+f['target']+'; no GS training started')
                    record=dict(file_path=f['file_path'],target=f['target'],passed=True,depth_check=check,prompt=prompt,previous_style=str(previous) if previous else None,teacher_sha256=sha(teacher),provider=models,conditioning=conditioning,appearance_reference='source_appearance_atlas.png',appearance_reference_sha256=sha(style),preprocessing=dict(resampling='Pillow BOX',size=[size,size],crop=None,color_normalization=False),inherited_fraction=0.,layout_limitation='Depth boundaries and common mesh do not certify exact semantic identity or furniture count')
                    for field,value in [('raw',raw),('guide',guide_file),('mesh_depth',folder/'depth'/(stem+'_mesh.npy')),('predicted_depth',predicted_file),('boundaries',folder/'guides'/(stem+'_boundaries.png')),('proxy_depth',proxy_depth_file)]:record[field]=str(value.relative_to(folder));record[field+'_sha256']=sha(value)
                    write(record_file,record)
                accepted.append(f)
                cache.append(dict(frame=f,depth=hit['depth'],valid=binary_erosion(architecture,iterations=4),rgb=rgb,confidence=np.ones((size,size)),image_path=folder/f['file_path']))
                print('Accepted room camera '+f['target'],flush=True)
        status('local_packaging',completed=len(accepted),total=len(frames),success=None)
        from .learned_references import save_references
        from .front_alignment import audit_front
        from .appearance import projection_fingerprint
        source=dict(original['frames'][0],file_path='source_anchor.png',split='view');shutil.copy2(folder/'source_front.png',folder/source['file_path']);job=json.loads((src/'render_job.json').read_text());job.update(folder=str(folder),frames=[source]+accepted)
        # Held-out mesh probes carry no independent photographic RGB ground truth.
        for i,f in enumerate(frames[:4]):
            probe=dict(f,file_path=f'geometry_probe_{i:02d}.png',split='heldout',photographic_ground_truth=False);hit=renderer.render(np.array(f['K']),np.array(f['transform_matrix']),size);Image.fromarray(hit['rgb']).save(folder/probe['file_path']);job['frames'].append(probe)
        write(folder/'render_job.json',job);meta=save_references(folder,original['prompt'],original['seed'],size,mesh,layout['components'],owners,job)
        meta.update(mode='image_room_mesh_v12',front_alignment='front_alignment.json',local_room_audit='local_room_audit.json',training_geometry_policy='shared_layout_local_detail_v17',reference_source='Local FLUX.2 Klein per-camera photographs; Depth Pro verification; mesh depth for GS',generation_method='whole-room cameras + mesh depth -> local multi-reference generation -> estimated-depth edge verification -> GS',rgb_evaluation_is_independent=False)
        rows=[dict(file_path=source['file_path'],passed=True,teacher_sha256=sha(folder/source['file_path']))]
        rows += [dict(file_path=f['file_path'],passed=True,teacher_sha256=sha(folder/f['file_path']),record_file='records/'+Path(f['file_path']).stem+'.json',record_sha256=sha(folder/'records'/(Path(f['file_path']).stem+'.json'))) for f in accepted]
        write(folder/'local_room_audit.json',dict(complete=True,passed=True,mesh_sha256=sha(folder/'scene_mesh.glb'),source_sha256=sha(folder/'source_front.png'),projection_sha256=projection_fingerprint(meta),views=rows,models=models,coverage=coverage));write(folder/'scene.json',meta);audit_front(folder);validate_local(folder,meta);status('local_references_ready',completed=len(accepted),success=True)
        if train_steps:
            from .train import train
            return train(folder,steps=train_steps,max_points=180000)
        return meta
    except BaseException as error:status('local_failed',success=False,error=str(error),completed=len(accepted));raise


def create_local_proxy(prompt,output,size=512,views=16,seed=0,front_image=None,room_size=None,resume=False,planner_model=None):
    from .learned_scene import generate_learned_scene
    from .proxy_completion import complete_proxy
    folder=Path(output).resolve();front_folder=folder.with_name(folder.name+'_front');scaffold=folder.with_name(folder.name+'_scaffold');completed=folder.with_name(folder.name+'_proxy')
    if not front_image:
        front_folder.mkdir(parents=True,exist_ok=True);front_image=front_folder/'source.png';front_request=dict(prompt=prompt,seed=seed,size=size,model='black-forest-labs/FLUX.2-klein-base-9b-fp8')
        if front_image.exists():
            if not resume or json.loads((front_folder/'request.json').read_text())!=front_request:raise FileExistsError('Existing initial image requires identical --resume')
        else:
            Image.new('RGB',(size,size),'white').save(front_folder/'size_canvas.png')
            with LocalRuntime(front_folder/'logs') as runtime:
                runtime.generate(front_folder/'size_canvas.png',front_folder/'size_canvas.png','One photorealistic wide-angle room photograph, almost the entire room visible except behind the camera, all main furniture fully visible, natural perspective. '+prompt,seed,front_image,size,kind='front')
            write(front_folder/'request.json',front_request)
    generate_learned_scene(prompt,scaffold,size,views,seed,room_size,resume,planner_model,image_room=True,front_image=front_image)
    if not (completed/'scene.json').exists():complete_proxy(scaffold,completed)
    return completed


def generate_local_scene(prompt,output,size=512,views=16,seed=0,front_image=None,room_size=None,resume=False,planner_model=None):
    from .reference_pipeline import run_reference_experiment
    proxy=create_local_proxy(prompt,output,size,views,seed,front_image,room_size,resume,planner_model)
    return run_reference_experiment(proxy,output,resume=resume)
