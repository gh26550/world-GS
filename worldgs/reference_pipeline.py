"""Small, camera-first reference experiment with separate collection/assessment/export."""
import hashlib,json,shutil
from pathlib import Path
import numpy as np
from PIL import Image
from .reference_cameras import worldmesh_cameras,closest_accepted,dense_camera_plan
from .reference_geometry import assess_reference
from .local_runtime import LocalRuntime
from .local_room import load_proxy
from .mesh_scene import MeshRenderer
from .appearance_geometry import depth_condition

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2))

def boundaries(hit,owners):
    ids=np.full(hit['faces'].shape,-1,int);valid=hit['faces']>=0;ids[valid]=owners[hit['faces'][valid]];edge=np.zeros(ids.shape,bool);edge[1:]|=ids[1:]!=ids[:-1];edge[:,1:]|=ids[:,1:]!=ids[:,:-1];return np.uint8(edge&valid)*255

def validate_reference_dataset(folder,meta):
    if not meta.get('reference_v2_audit'):return
    from .appearance import projection_fingerprint
    folder=Path(folder);audit=json.loads((folder/meta['reference_v2_audit']).read_text())
    if not audit.get('complete') or not audit.get('passed'):raise ValueError('Reference benchmark has unresolved cameras')
    if audit['mesh_sha256']!=sha(folder/'scene_mesh.glb') or audit['source_sha256']!=sha(folder/'source_front.png') or audit['projection_sha256']!=projection_fingerprint(meta):raise ValueError('Reference geometry/source/cameras changed')
    if {r['file_path'] for r in audit['views']}!={f['file_path'] for f in meta['frames']}:raise ValueError('Reference audit missing a training image')
    for row in audit['views']:
        if sha(folder/row['file_path'])!=row['sha256']:raise ValueError('Reference training image changed')
    if 'rgb_weights' in audit and audit['rgb_weights']!={f['file_path']:f.get('rgb_weight',1.) for f in meta['frames']}:raise ValueError('RGB supervision weights changed')
    for row in audit.get('geometry_only',[]):
        if audit['rgb_weights'].get(row['file_path'])!=0.:raise ValueError('Rejected photograph must have RGB disabled')
        for name,digest in row['evidence'].items():
            if sha(folder/name)!=digest:raise ValueError('Rejected evidence changed')
    for row in audit['selected']:
        if not row['assessment']['passed']:raise ValueError('Rejected reference cannot be trained')
        for name,digest in row['evidence'].items():
            if sha(folder/name)!=digest:raise ValueError('Reference evidence changed: '+name)

def prepare_reference_experiment(src,folder,attempts,model,fov,rgb_target=None,camera_count=None):
    from .appearance_reference import appearance_atlas
    original=json.loads((src/'scene.json').read_text());mesh,layout,owners=load_proxy(src);frames=worldmesh_cameras(layout['shapes'],original['size'],fov);runtime=LocalRuntime(folder/'logs',image_model=model);models=runtime.preflight()
    if rgb_target is not None:
        if rgb_target<26:raise ValueError('RGB target includes source image and must be at least 26')
    if camera_count is not None and rgb_target is not None and camera_count<rgb_target-1:raise ValueError('Camera count cannot supply the requested RGB teacher target')
    if rgb_target is not None or camera_count is not None:
        frames=dense_camera_plan(layout['shapes'],original['size'],camera_count if camera_count is not None else int(np.ceil((rgb_target-1)*1.6)),fov)
    recipe=dict(version=4,camera_policy='worldmesh_perimeter16_overhead8',model_profile=model,source=str(src),mesh_sha256=sha(src/'scene_mesh.glb'),source_sha256=sha(src/'source_front.png'),size=original['size'],seed=original['seed'],attempts=attempts,models=models,cameras=frames,validation='reference_geometry_v3_physical_groups')
    if rgb_target is not None:recipe.update(version=5,camera_policy='worldmesh_plus_safe_multiheight_multidistance',rgb_target=rgb_target,target_includes_source=True)
    if camera_count is not None:recipe.update(version=6,camera_policy='worldmesh_plus_safe_multiheight_multidistance',camera_count=camera_count,collect_all_planned=True)
    return original,mesh,layout,owners,frames,runtime,recipe

def initialize_experiment(src,folder,recipe,resume):
    if folder.exists() and any(folder.iterdir()):
        if not resume or not (folder/'experiment.json').exists() or json.loads((folder/'experiment.json').read_text())!=recipe:raise FileExistsError('Use an empty output or identical --resume experiment')
        return
    folder.mkdir(parents=True,exist_ok=True)
    for name in ['assets','planning','materials','room_images','meshes']:
        if (src/name).exists():shutil.copytree(src/name,folder/name)
    for name in ['scene_mesh.glb','mesh_geometry.npz','layout.json','source_front.png','front_alignment.json','image_room_provenance.json','request.json']: 
        if (src/name).exists():shutil.copy2(src/name,folder/name)
    write(folder/'experiment.json',recipe)

def render_benchmark(folder,renderer,frames,size):
    for f in frames:
        d=folder/'references'/f['id'];d.mkdir(parents=True,exist_ok=True);hit=renderer.render(np.array(f['K']),np.array(f['transform_matrix']),size)
        Image.fromarray(hit['rgb']).save(d/'proxy.png');Image.fromarray(depth_condition(hit['depth'])).save(d/'depth.png');Image.fromarray(boundaries(hit,renderer.owners)).save(d/'boundaries.png');np.save(d/'mesh_depth.npy',hit['depth'])

def collect_candidates(folder,renderer,frames,runtime,recipe):
    from .appearance_reference import appearance_atlas
    from .layout_consistency import layout_contract
    from .room_verification import camera_context
    plan=json.loads((folder/'planning/image_room_plan.json').read_text());contract=layout_contract(dict(vertices=renderer.mesh.vertices,faces=renderer.mesh.faces,owners=renderer.owners),renderer.components,plan,recipe['mesh_sha256'],recipe['source_sha256']);labels={o['id']:o['label'] for o in contract['objects']};size=recipe['size'];appearance_atlas(Image.open(folder/'source_front.png'),recipe['seed']).save(folder/'appearance.png');results=[];accepted=[]
    with runtime:
        for index,f in enumerate(frames):
            d=folder/'references'/f['id'];hit=renderer.render(np.array(f['K']),np.array(f['transform_matrix']),size)
            if not (d/'proxy_depth.npy').exists():runtime.depth(d/'proxy.png',f['K'][0][0],d/'proxy_depth.npy')
            nearest=closest_accepted(f,accepted);style=folder/'appearance.png'
            if nearest is not None:
                style=d/'nearest_appearance.png';appearance_atlas(Image.open(folder/nearest['row']['image']),recipe['seed']).save(style)
            context,objects=camera_context(hit,renderer.owners,renderer.components,labels);prompt='One photorealistic image from the EXACT proxy camera. '+context;rows=[]
            for attempt in range(recipe['attempts']):
                name=f'candidate_{attempt:02d}.png';photo=d/name;pred=d/f'predicted_{attempt:02d}.npy';assessment_file=d/f'assessment_{attempt:02d}.json'
                if assessment_file.exists():
                    row=json.loads(assessment_file.read_text())
                    if all(sha(folder/path)==digest for path,digest in row['evidence'].items()):rows.append(row)
                    else:raise ValueError('Cached candidate evidence changed')
                else:
                    write(folder/'status.json',dict(stage='collecting',camera=f['id'],attempt=attempt+1,completed=len(results),total=len(frames)))
                    runtime.generate(d/'proxy.png',style,prompt,recipe['seed']+index*101+attempt,photo,size,kind='structure',depth_guide=d/'depth.png',edge_guide=d/'boundaries.png',visible_labels=[o['label'] for o in objects],appearance=plan.get('style',''))
                    image=Image.open(photo).convert('RGB')
                    if abs(image.width/image.height-1)>.002:raise ValueError('Generated camera aspect changed; cropping is forbidden')
                    image.resize((size,size),Image.Resampling.BOX).save(photo)
                    runtime.depth(photo,f['K'][0][0],pred)
                    result=assess_reference(hit['depth'],np.load(d/'proxy_depth.npy'),np.load(pred),np.asarray(Image.open(photo)),hit['faces'],renderer.owners,size,components=renderer.components)
                    evidence=[d/'proxy.png',d/'depth.png',d/'boundaries.png',d/'mesh_depth.npy',d/'proxy_depth.npy',photo,pred,style]
                    row=dict(camera=f['id'],reference_camera=nearest['frame']['id'] if nearest else None,image=str(photo.relative_to(folder)),assessment=result,evidence={str(p.relative_to(folder)):sha(p) for p in evidence});write(assessment_file,row);rows.append(row)
                if rows[-1]['assessment']['passed']:break
            good=[r for r in rows if r['assessment']['passed']];selected=max(good or rows,key=lambda r:r['assessment']['score']);results.append(selected)
            if recipe.get('collect_all_planned'):
                selected['rgb_teacher']=selected['assessment']['passed'] and (recipe.get('rgb_target') is None or sum(uses_rgb(r) for r in results[:-1])+1<recipe['rgb_target'])
            if selected['assessment']['passed']:accepted.append(dict(frame=f,row=selected))
            rgb_count=sum(uses_rgb(r) for r in results)+1;reached=recipe.get('rgb_target') is not None and rgb_count>=recipe['rgb_target']
            complete=(reached and not recipe.get('collect_all_planned')) or len(results)==len(frames)
            write(folder/'assessment.json',dict(complete=complete,ready=complete and reached if recipe.get('rgb_target') else complete and all(r['assessment']['passed'] for r in results),rgb_teacher_count=rgb_count,rgb_target=recipe.get('rgb_target'),planned_cameras=len(frames),attempted_cameras=len(results),views=results));print(f"Camera {f['id']}: geometry_passed={selected['assessment']['passed']} RGB teachers={rgb_count}",flush=True)
            if reached and not recipe.get('collect_all_planned'):break
    return results

def write_review(folder,frames,results):
    from .reference_review import write_review as explain_review
    explain_review(folder,frames,results)

def uses_rgb(row):return row['assessment']['passed'] and row.get('rgb_teacher',True)


def training_view_source(folder,frame,row,allow_geometry_only=False):
    frame=dict(frame)
    if uses_rgb(row):return frame,folder/row['image']
    if not allow_geometry_only:raise ValueError('Rejected photograph cannot be an RGB teacher')
    frame.update(rgb_weight=0.,reference_source='Exact mesh depth/opacity only; photograph not selected for RGB supervision')
    return frame,folder/'references'/frame['id']/'proxy.png'


def package_dataset(src,folder,original,mesh,layout,owners,frames,results,allow_geometry_only=False):
    from .learned_references import save_references
    from .front_alignment import audit_front
    from .appearance import projection_fingerprint
    source={k:v for k,v in original['frames'][0].items() if k in ('K','transform_matrix','target')};source.update(file_path='source_anchor.png',split='view');shutil.copy2(folder/'source_front.png',folder/source['file_path']);training=[source]
    for f,row in zip(frames,results):
        target,image=training_view_source(folder,f,row,allow_geometry_only);shutil.copy2(image,folder/target['file_path']);training.append(target)
    job=json.loads((src/'render_job.json').read_text());job.update(folder=str(folder),frames=list(training))
    for i,f in enumerate(frames):
        probe=dict(f,file_path=f'probe_{i:02d}.png',split='heldout',photographic_ground_truth=False);shutil.copy2(folder/'references'/f['id']/'proxy.png',folder/probe['file_path']);job['frames'].append(probe)
    write(folder/'render_job.json',job);meta=save_references(folder,original['prompt'],original['seed'],original['size'],mesh,layout['components'],owners,job);meta.update(mode='image_room_mesh_v12',front_alignment='front_alignment.json',reference_v2_audit='reference_audit.json',training_geometry_policy='shared_layout_local_detail_v17',reference_source='WorldMesh-inspired multi-camera local mesh/depth/boundary conditioned photographs',rgb_evaluation_is_independent=False)
    audit=dict(complete=True,passed=True,mesh_sha256=sha(folder/'scene_mesh.glb'),source_sha256=sha(folder/'source_front.png'),projection_sha256=projection_fingerprint(meta),views=[dict(file_path=f['file_path'],sha256=sha(folder/f['file_path'])) for f in training],selected=[r for r in results if uses_rgb(r)],geometry_only=[dict(camera=f['id'],file_path=f['file_path'],assessment=r['assessment'],evidence=r['evidence']) for f,r in zip(frames,results) if not uses_rgb(r)],policy='selected_photos_with_mesh_only_other_views' if allow_geometry_only else 'all_photographs_passed',rgb_weights={f['file_path']:f.get('rgb_weight',1.) for f in training},all_photographs_passed=all(r['assessment']['passed'] for r in results))
    meta['all_photographs_passed']=audit['all_photographs_passed']
    write(folder/'reference_audit.json',audit);write(folder/'scene.json',meta);audit_front(folder);validate_reference_dataset(folder,meta);return meta

def reuse_reference_evidence(old,folder,recipe):
    old=Path(old);previous=json.loads((old/'experiment.json').read_text())
    for key in ['mesh_sha256','source_sha256','seed','size','model_profile']:
        if previous[key]!=recipe[key]:raise ValueError('Cannot reuse different reference evidence: '+key)
    for frame in previous['cameras']:
        match=next((f for f in recipe['cameras'] if f['id']==frame['id']),None)
        if match!=frame:raise ValueError('Cannot reuse changed camera '+frame['id'])
        source=old/'references'/frame['id'];target=folder/'references'/frame['id']
        if target.exists():continue
        if source.exists():shutil.copytree(source,target)
    write(folder/'reused_evidence.json',dict(source=str(old),policy='Exact same source, mesh, cameras and local model; cached evidence hashes verified during collection'))


def run_reference_experiment(path,output,resume=False,prepare_only=False,attempts=2,model='controlnet',fov=60,train_steps=0,max_points=100000,rgb_target=None,reuse_references=None,camera_count=None):
    src=Path(path).resolve();folder=Path(output).resolve()
    if src==folder or src in folder.parents or folder in src.parents:raise ValueError('Use a separate sibling output')
    if not 1<=attempts<=3:raise ValueError('Use 1..3 attempts per camera')
    original,mesh,layout,owners,frames,runtime,recipe=prepare_reference_experiment(src,folder,attempts,model,fov,rgb_target,camera_count);initialize_experiment(src,folder,recipe,resume)
    if reuse_references:reuse_reference_evidence(reuse_references,folder,recipe)
    renderer=MeshRenderer(mesh,layout['components'],owners);render_benchmark(folder,renderer,frames,original['size'])
    if prepare_only:write(folder/'status.json',dict(stage='prepared',cameras=len(frames)));return recipe
    try:
        results=collect_candidates(folder,renderer,frames,runtime,recipe);used_frames=frames[:len(results)];write_review(folder,used_frames,results)
        if rgb_target is not None:
            write(folder/'status.json',dict(stage='rgb_target_ready' if sum(uses_rgb(r) for r in results)+1>=rgb_target else 'rgb_target_unmet',rgb_teachers=sum(uses_rgb(r) for r in results)+1,rgb_target=rgb_target,attempted_cameras=len(results),planned_cameras=len(frames)))
            return results
        if not all(r['assessment']['passed'] for r in results):write(folder/'status.json',dict(stage='review_required',trainable=False,passed=sum(r['assessment']['passed'] for r in results),total=len(frames)));return results
        meta=package_dataset(src,folder,original,mesh,layout,owners,frames,results);write(folder/'status.json',dict(stage='references_ready',trainable=True,cameras=len(frames)))
        if train_steps:
            from .train import train
            return train(folder,train_steps,max_points)
        return meta
    except Exception as error:write(folder/'status.json',dict(stage='failed',error=str(error),trainable=False));raise


def evaluate_reference_experiment(path,output,train_steps=0,max_points=100000):
    old=Path(path).resolve();folder=Path(output).resolve();old_recipe=json.loads((old/'experiment.json').read_text());src=Path(old_recipe['source'])
    if old==folder or old in folder.parents or folder in old.parents:raise ValueError('Use a separate sibling output')
    model=old_recipe.get('model_profile') or ('controlnet' if 'depth_control_model' in old_recipe['models'] else 'base' if 'base' in old_recipe['models']['image_model'] else 'distilled')
    original,mesh,layout,owners,frames,runtime,recipe=prepare_reference_experiment(src,folder,old_recipe['attempts'],model,70)
    # Evaluation uses precisely the recorded camera poses, not a newly inferred layout.
    frames=old_recipe['cameras'];recipe['cameras']=frames
    if recipe['mesh_sha256']!=old_recipe['mesh_sha256'] or recipe['source_sha256']!=old_recipe['source_sha256']:raise ValueError('Original proxy/source changed')
    initialize_experiment(src,folder,recipe,False);shutil.copytree(old/'references',folder/'references');shutil.copy2(old/'appearance.png',folder/'appearance.png');write(folder/'origin.json',dict(experiment=str(old),recipe=old_recipe));renderer=MeshRenderer(mesh,layout['components'],owners);results=[]
    for f in frames:
        d=folder/'references'/f['id'];hit=renderer.render(np.array(f['K']),np.array(f['transform_matrix']),recipe['size']);rows=[]
        for assessment_file in sorted(p for p in d.glob('assessment_*.json') if p.stem.removeprefix('assessment_').isdigit()):
            old_row=json.loads(assessment_file.read_text())
            if not all(sha(folder/name)==digest for name,digest in old_row['evidence'].items()):raise ValueError('Old candidate evidence changed')
            index=assessment_file.stem.split('_')[-1];image=folder/old_row['image'];prediction=d/('predicted_'+index+'.npy');check=assess_reference(hit['depth'],np.load(d/'proxy_depth.npy'),np.load(prediction),np.asarray(Image.open(image)),hit['faces'],owners,recipe['size'],components=layout['components']);row=dict(old_row,assessment=check,previous_assessment=old_row['assessment'],evaluation_version=3);write(assessment_file.with_suffix('.previous.json'),old_row);write(assessment_file,row);rows.append(row)
        good=[r for r in rows if r['assessment']['passed']];results.append(max(good or rows,key=lambda r:r['assessment']['score']))
    ready=all(r['assessment']['passed'] for r in results);write(folder/'assessment.json',dict(complete=True,ready=ready,views=results));write_review(folder,frames,results)
    if not ready:write(folder/'status.json',dict(stage='review_required',passed=sum(r['assessment']['passed'] for r in results),total=len(frames),trainable=False));return results
    meta=package_dataset(src,folder,original,mesh,layout,owners,frames,results);write(folder/'status.json',dict(stage='references_ready',trainable=True,cameras=len(frames)))
    if train_steps:
        from .train import train
        return train(folder,train_steps,max_points)
    return meta


def export_accepted_references(path,output,train_steps=3000,max_points=180000):
    """Explicit production export: rejected RGB never supervises GS appearance."""
    old=Path(path).resolve();folder=Path(output).resolve();recipe=json.loads((old/'experiment.json').read_text());assessment=json.loads((old/'assessment.json').read_text())
    if old==folder or old in folder.parents or folder in old.parents:raise ValueError('Use separate sibling output')
    if folder.exists() and any(folder.iterdir()):raise FileExistsError('Use empty production output')
    if not assessment['complete']:raise ValueError('Collect every planned view before production export')
    results=assessment['views'];passed=sum(uses_rgb(r) for r in results)
    if recipe.get('rgb_target') and passed+1!=recipe['rgb_target']:raise ValueError('Requested RGB teacher target has not been reached exactly')
    if passed<8:raise ValueError('Production export needs at least eight accepted translated photographs')
    src=Path(recipe['source']);original=json.loads((src/'scene.json').read_text());mesh,layout,owners=load_proxy(src)
    if sha(src/'scene_mesh.glb')!=recipe['mesh_sha256'] or sha(src/'source_front.png')!=recipe['source_sha256']:raise ValueError('Source changed')
    initialize_experiment(src,folder,recipe,False);shutil.copytree(old/'references',folder/'references');shutil.copy2(old/'appearance.png',folder/'appearance.png');write(folder/'assessment.json',assessment);write(folder/'origin.json',dict(experiment=str(old),accepted=passed,total=len(results),policy='Rejected RGB excluded; planned mesh geometry retained'))
    frames=recipe['cameras'][:len(results)];renderer=MeshRenderer(mesh,layout['components'],owners)
    if [f['id'] for f in frames]!=[row['camera'] for row in results]:raise ValueError('Collected camera order differs from plan')
    from .reference_coverage import scaled_intrinsics
    observations={i:[] for i,c in enumerate(layout['components']) if c.get('material')=='generated_asset'}
    for f,row in zip(frames,results):
        if not uses_rgb(row):continue
        hit=renderer.render(scaled_intrinsics(f['K'],recipe['size'],96),np.asarray(f['transform_matrix']),96);valid=hit['faces']>=0;visible_owners=owners[hit['faces'][valid]]
        for owner in observations:
            if np.count_nonzero(visible_owners==owner)>=4:observations[owner].append(f)
    coverage={layout['components'][owner]['name']:dict(views=len(items),translated_pair=any(np.linalg.norm(np.asarray(a['transform_matrix'])[:3,3]-np.asarray(b['transform_matrix'])[:3,3])>.25 for a in items for b in items)) for owner,items in observations.items()}
    write(folder/'photographic_object_coverage.json',coverage)
    if not all(row['translated_pair'] for row in coverage.values()):raise ValueError('An object lacks two translated accepted photographic views')
    write_review(folder,frames,results);meta=package_dataset(src,folder,original,mesh,layout,owners,frames,results,allow_geometry_only=True)
    write(folder/'status.json',dict(stage='references_ready',accepted_photos=passed,total=len(results),all_photographs_passed=passed==len(results)))
    if train_steps:
        from .train import train
        train(folder,train_steps,max_points)
    return meta
