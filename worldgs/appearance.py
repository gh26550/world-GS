"""Mesh -> mutually conditioned photographic references -> GS-ready dataset."""
import gc
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
import trimesh
from scipy.ndimage import binary_erosion
from .mesh_scene import MeshRenderer
from .reference_coverage import scaled_intrinsics
from .appearance_geometry import transfer_appearance,pose_distance,geometry_check,structural_planarity,semantic_support


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def projection_fingerprint(meta):
    rows=[dict(file_path=f['file_path'],K=f.get('K',meta.get('K')),pose=f.get('transform_matrix')) for f in meta['frames']+meta['heldout_frames']]
    return hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()


def synthesize_appearance(path,output,size=512,pilot_views=None,resume=False,_completion_round=0):
    source=Path(path).resolve();folder=Path(output).resolve()
    if source==folder or source in folder.parents or folder in source.parents:
        raise ValueError('Appearance output must be a separate sibling run')
    if size<256 or size%8:raise ValueError('Appearance size must be >=256 and divisible by 8')
    original=json.loads((source/'scene.json').read_text())
    if original.get('appearance_audit'):raise ValueError('Use an original mesh-reference run, not an already stylized result')
    request=dict(version=15,source=str(source),mesh_sha256=sha(source/'scene_mesh.glb'),
                 front_sha256=sha(source/'source_front.png'),camera_sha256=sha(source/'scene.json'),
                 size=size,seed=original['seed'],controlnet='lllyasviel/control_v11f1p_sd15_depth',
                 base='stable-diffusion-v1-5/stable-diffusion-inpainting',ip_adapter='h94/IP-Adapter:ip-adapter_sd15.bin',
                 ip_scale=.2,min_depth_edge_recall=.55,source_mask_policy='bookshelf_composite_hull_v1',view_order='bootstrap_quadrants_v1',
                 diffusion_strength=[.6,.5,.4],validation='generated_edges_planes_seating_family_v2')
    manifest=folder/'appearance_request.json'
    if folder.exists() and any(folder.iterdir()):
        if not resume or not manifest.exists() or json.loads(manifest.read_text())!=request:
            raise FileExistsError('Use an empty output, or --resume with the identical appearance request')
    else:
        folder.mkdir(parents=True,exist_ok=True)
        for name in ('assets','planning','materials','room_images'):
            if (source/name).exists():shutil.copytree(source/name,folder/name)
        for name in ('scene_mesh.glb','mesh_geometry.npz','layout.json','source_front.png','image_room_provenance.json','front_alignment.json','request.json','reference_coverage.json'):
            if (source/name).exists():shutil.copy2(source/name,folder/name)
        manifest.write_text(json.dumps(request,indent=2))
    (folder/'appearance_records').mkdir(exist_ok=True);(folder/'appearance_candidates').mkdir(exist_ok=True)
    (folder/'logs').mkdir(exist_ok=True)
    def status(stage,**extra):
        record=dict(stage=stage,success=None);record.update(extra)
        (folder/'status.json').write_text(json.dumps(record,indent=2))
    status('appearance_preparation')
    geometry=np.load(folder/'mesh_geometry.npz');owners=geometry['owners']
    mesh=trimesh.Trimesh(geometry['vertices'],geometry['faces'],process=False)
    layout=json.loads((folder/'layout.json').read_text());components=layout['components']
    renderer=MeshRenderer(mesh,components,owners)
    job=json.loads((source/'render_job.json').read_text());job.update(folder=str(folder),size=size)
    job['K']=scaled_intrinsics(original['K'],original['size'],size).tolist()
    frames=[]
    for split,items in [('view',original['frames']),('heldout',original['heldout_frames'])]:
        for old in items:
            f={k:v for k,v in old.items() if k in ('target','file_path','transform_matrix')}
            f.update(split=split,K=scaled_intrinsics(old.get('K',original['K']),original['size'],size).tolist(),
                     confidence_file=Path(f['file_path']).stem+'_confidence.npy')
            frames.append(f)
    job['frames']=frames
    (folder/'render_job.json').write_text(json.dumps(job,indent=2))
    style=Image.open(folder/'source_front.png').convert('RGB').resize((size,size),Image.Resampling.LANCZOS)
    front=frames[0];front_hit=renderer.render(np.asarray(front['K']),np.asarray(front['transform_matrix']),size)
    original_owner=np.full((size,size),-1,int)
    from .front_alignment import source_mask
    plan=json.loads((folder/'planning/image_room_plan.json').read_text())
    names={c['name']:i for i,c in enumerate(components)}
    for obj in plan['objects']:
        if obj.get('evidence')=='front':
            mask=source_mask(folder,obj,(size,size))
            if 'bookshelf' in obj['id'] and mask.any():
                from scipy.spatial import ConvexHull
                from PIL import ImageDraw
                y,x=np.where(mask);points=np.column_stack([x,y])
                hull=ConvexHull(points);canvas=Image.new('L',(size,size))
                ImageDraw.Draw(canvas).polygon([tuple(p) for p in points[hull.vertices]],fill=255)
                mask=np.asarray(canvas)>0
            original_owner[mask]=names[obj['id']]
    hit_owner=owners[np.maximum(front_hit['faces'],0)]
    source_valid=(front_hit['faces']>=0)&np.where(original_owner>=0,hit_owner==original_owner,hit_owner<6)
    source_valid=binary_erosion(source_valid,iterations=1)
    cache=[dict(frame=front,depth=front_hit['depth'],valid=source_valid,rgb=np.asarray(style),
                confidence=np.ones((size,size)),source_anchor=True)]
    remaining=[i for i,f in enumerate(frames) if f['split']=='view'];order=[];last=front
    while remaining:
        index=min(remaining,key=lambda i:pose_distance(last['transform_matrix'],frames[i]['transform_matrix']))
        remaining.remove(index);order.append(index);last=frames[index]
    priority=[]
    for target in ('front','room_yaw_0_pitch_0','room_yaw_180_pitch_0','room_yaw_90_pitch_0','room_yaw_270_pitch_0'):
        priority += [i for i in order if frames[i]['target']==target]
    order=priority+[i for i in order if i not in priority]
    order += [i for i,f in enumerate(frames) if f['split']=='heldout']
    # A later completion round can use every already validated TRAINING image.
    # Heldout images remain excluded even when they are available on disk.
    for frame in frames:
        record_file=folder/'appearance_records'/(Path(frame['file_path']).stem+'.json')
        if frame['split']!='view' or not record_file.exists():continue
        old_record=json.loads(record_file.read_text())
        if not old_record.get('passed') or sha(folder/frame['file_path'])!=old_record['image_sha256']:continue
        hit=renderer.render(np.asarray(frame['K']),np.asarray(frame['transform_matrix']),size)
        cache.append(dict(frame=frame,depth=hit['depth'],valid=hit['faces']>=0,
                          rgb=np.asarray(Image.open(folder/frame['file_path']).convert('RGB')),
                          confidence=np.load(folder/frame['confidence_file'])))
    records=[];rejections=[];models=None
    try:
        for position,index in enumerate(order):
            if pilot_views is not None and position>=pilot_views:
                status('appearance_pilot_ready',completed=len(records),total=len(frames),trainable=False)
                return dict(pilot=True,completed=len(records),total=len(frames),output=str(folder))
            frame=frames[index];stem=Path(frame['file_path']).stem
            record_file=folder/'appearance_records'/f'{stem}.json'
            hit=front_hit if index==0 else renderer.render(np.asarray(frame['K']),np.asarray(frame['transform_matrix']),size)
            status('appearance_synthesis',completed=position,total=len(frames),frame=frame['file_path'])
            if record_file.exists():
                record=json.loads(record_file.read_text())
                if not record.get('passed') or sha(folder/frame['file_path'])!=record['image_sha256']:
                    raise ValueError('Cached appearance image failed its recorded fingerprint')
                rgb=np.asarray(Image.open(folder/frame['file_path']).convert('RGB'))
                confidence=np.load(folder/frame['confidence_file'])
            else:
                transferred,known,confidence=transfer_appearance(hit,frame,cache)
                guide=np.array(Image.open(source/frame['file_path']).convert('RGB').resize((size,size),Image.Resampling.LANCZOS))
                floor_ids=[i for i,c in enumerate(components) if c['name']=='room_1_1']
                floor=(hit['faces']>=0)&np.isin(owners[np.maximum(hit['faces'],0)],floor_ids)
                guide[floor]=[148,113,75]
                guide[known]=transferred[known]
                visible_ids=owners[hit['faces'][hit['faces']>=0]]
                descriptions=[]
                for obj in job['plan']['objects']:
                    if (visible_ids==names[obj['id']]).sum()>=size*size*.01:
                        descriptions.append('bookshelves filled with books' if 'bookshelf' in obj['id'] else obj['label'])
                room=job['plan']['room']
                material=', '.join(' '.join(str(room[name]).split()[:3])+' '+label for name,label in [('ceiling','ceiling'),('walls','walls'),('floor','floor')])
                style_words=' '.join(str(job['plan'].get('style','')).split()[:8])
                prompt='Photorealistic interior photograph of the same room. '+material+'. '+style_words+'. '
                prompt += ('Visible furniture: '+', '.join(descriptions)+'.') if descriptions else 'Plain room surfaces, no additional furniture.'
                attempts=[];guide_check=None;guide_detections=None
                for attempt in range(3):
                    if models is None:
                        from .appearance_models import AppearanceModels
                        models=AppearanceModels(original['seed'])
                    if guide_check is None:
                        guide_check=geometry_check(hit['depth'],models.inverse_depth(guide),hit['faces'],owners,tolerance=round(10*size/512),generated_mask=~known)
                        guide_detections=models.detections(guide)
                    try:
                        rgb=models.generate(prompt,guide,known,hit['depth'],style,index,attempt)
                    except RuntimeError as error:
                        if 'black/rejected image' not in str(error):raise
                        check=dict(passed=False,generation_rejected=True,attempt=attempt)
                        attempts.append(check)
                        continue
                    inverse=models.inverse_depth(rgb)
                    check=geometry_check(hit['depth'],inverse,hit['faces'],owners,tolerance=round(10*size/512),min_recall=min(.55,max(.05,.9*guide_check['depth_edge_recall'])),generated_mask=~known)
                    check['guide_edge_recall']=guide_check['depth_edge_recall']
                    check['edge_threshold_policy']='min(0.55, max(0.05, 0.9 * guide recall)); calibrated for depth-estimator observability'
                    check['planarity']=structural_planarity(inverse,hit['faces'],owners,~known)
                    check['semantics']=semantic_support(models.detections(rgb),hit['faces'],owners,components,rgb=rgb,generated_mask=~known,baseline_detections=guide_detections)
                    check['passed']=check['passed'] and check['planarity']['passed'] and check['semantics']['passed']
                    check['protected_pixel_l1']=float(np.abs(rgb[known].astype(float)-guide[known]).mean()/255) if known.any() else 0.
                    check['passed']=check['passed'] and check['protected_pixel_l1']<=1/255
                    attempts.append(check)
                    Image.fromarray(rgb).save(folder/'appearance_candidates'/f'{stem}_attempt_{attempt:02d}.png')
                    if check['passed']:break
                if not check['passed']:
                    (folder/'appearance_records'/f'{stem}_rejected.json').write_text(json.dumps(attempts,indent=2))
                    rejections.append(dict(frame=frame['file_path'],target=frame['target'],attempts=attempts))
                    print('Rejected appearance reference: '+frame['file_path'],flush=True)
                    if index==0:raise ValueError('The source-aligned front appearance failed validation')
                    continue
                confidence=np.where(known,np.maximum(confidence,.55),.55).astype(np.float32)
                Image.fromarray(rgb).save(folder/frame['file_path'])
                Image.fromarray(guide).save(folder/'appearance_candidates'/f'{stem}_guide.png')
                Image.fromarray(np.uint8(known)*255).save(folder/'appearance_candidates'/f'{stem}_known.png')
                np.save(folder/frame['confidence_file'],confidence)
                record=dict(frame=frame['file_path'],target=frame['target'],split=frame['split'],passed=True,
                            inherited_fraction=float(known.mean()),generated_fraction=float((~known).mean()),
                            image_sha256=sha(folder/frame['file_path']),attempts=attempts,prompt=prompt,
                            geometry='shared exact mesh; generated depth used for checking only')
                record_file.write_text(json.dumps(record,indent=2))
            record['confidence_file']=frame['confidence_file']
            record['confidence_sha256']=sha(folder/frame['confidence_file'])
            record_file.write_text(json.dumps(record,indent=2))
            records.append(record)
            # Heldout synthesis must never affect any training reference.
            if frame['split']=='view':
                cache=[e for e in cache if e.get('source_anchor') or e['frame']['file_path']!=frame['file_path']]
                cache.append(dict(frame=frame,depth=hit['depth'],valid=hit['faces']>=0,rgb=rgb,confidence=confidence))
            print(f'Appearance reference {position+1}/{len(frames)} {frame["file_path"]}: inherited={record["inherited_fraction"]:.1%}',flush=True)
        del models;models=None;gc.collect()
        import torch
        torch.cuda.empty_cache()
        status('appearance_reference_packaging')
        from .reference_coverage import audit_planned_coverage,protect_training_coverage
        accepted={r['frame'] for r in records}
        planned=job['frames']
        job['frames']=[f for f in planned if f['file_path'] in accepted]
        split_report=protect_training_coverage(mesh,components,owners,job['frames'],size,rename_files=False)
        (folder/'appearance_split.json').write_text(json.dumps(split_report,indent=2))
        for record in records:
            record['split']=next(f['split'] for f in job['frames'] if f['file_path']==record['frame'])
        training=[f for f in job['frames'] if f['split']=='view']
        coverage=audit_planned_coverage(mesh,components,owners,planned,training,size)
        (folder/'appearance_coverage.json').write_text(json.dumps(coverage,indent=2))
        if not coverage['passed']:
            if _completion_round<2:
                print('Completing remaining surfaces with the expanded validated training appearance cache',flush=True)
                return synthesize_appearance(source,folder,size=size,resume=True,_completion_round=_completion_round+1)
            raise ValueError('Accepted photographic views leave unseen surfaces; inspect appearance_coverage.json')
        if sum(f['split']=='heldout' for f in job['frames'])<4:raise ValueError('Insufficient accepted heldout photographic references')
        (folder/'render_job.json').write_text(json.dumps(job,indent=2))
        from .learned_references import save_references
        meta=save_references(folder,original['prompt'],original['seed'],size,mesh,components,owners,job)
        probes=[]
        for i,frame in enumerate(planned):
            hit=renderer.render(np.asarray(frame['K']),np.asarray(frame['transform_matrix']),size)
            filename=f'coverage_probe_{i:03d}_mask.npy'
            np.save(folder/filename,hit['faces']>=0)
            probes.append(dict(target=frame['target'],K=frame['K'],transform_matrix=frame['transform_matrix'],mask_file=filename))
        meta['coverage_probe_frames']=probes
        for key in ('mode','generation_source','input_image','front_alignment'):
            if key in original:meta[key]=original[key]
        meta.update(appearance_audit='appearance_audit.json',reference_source='Depth-ControlNet + source-photo IP-Adapter + occlusion-tested appearance propagation',
                    proposal_images_are_training_targets=False,generated_appearance_images_are_training_targets=True)
        for frame in meta['frames']+meta['heldout_frames']:
            frame['reference_source']='mesh-conditioned synthesized photograph; exact shared mesh depth/normal/ID'
        meta['limitations']=[line for line in meta['limitations'] if line!='Heldout mesh renders are not independent real-world ground truth']
        meta['limitations'] += ['Generated references are hypotheses, not captured or independent ground truth',
                               'Mesh topology and missing architectural objects are not repaired by appearance synthesis',
                               'Depth-edge recall is a heuristic, not a guarantee of multi-view or semantic consistency']
        audit=dict(version=15,complete=True,passed=all(r['passed'] for r in records),mesh_digest=meta['mesh_digest'],
                   request_sha256=sha(manifest),projection_sha256=projection_fingerprint(meta),views=records,
                   rejected_views=rejections,completion_round=_completion_round+1,split_selection=split_report,planned_views=len(planned),coverage=coverage,heldout_used_for_training_conditioning=False)
        (folder/'appearance_audit.json').write_text(json.dumps(audit,indent=2))
        (folder/'scene.json').write_text(json.dumps(meta,indent=2))
        from .front_alignment import audit_front
        if meta.get('front_alignment') and not audit_front(folder)['passed']:raise ValueError('Front projection gate failed')
        html_file=folder/'references.html'
        text=html_file.read_text().replace('360-degree mesh references','Mesh-guided photographic references').replace('Physical RGB references used for Gaussian training.','Generated photographic RGB references used for Gaussian training; not captured ground truth.')
        html_file.write_text(text)
        status('appearance_ready',completed=len(records),total=len(frames),success=True)
        return meta
    except Exception as error:
        (folder/'status.json').write_text(json.dumps(dict(stage='appearance_failed',success=False,error=str(error)),indent=2))
        raise


def validate_appearance(folder,meta):
    folder=Path(folder)
    if not meta.get('appearance_audit'):
        if (folder/'appearance_request.json').exists():raise ValueError('Appearance references are incomplete')
        return
    audit=json.loads((folder/meta['appearance_audit']).read_text())
    if not audit.get('complete') or not audit.get('passed') or audit.get('mesh_digest')!=meta['mesh_digest']:
        raise ValueError('Appearance references are incomplete, rejected or stale')
    if audit['request_sha256']!=sha(folder/'appearance_request.json'):raise ValueError('Appearance request changed after validation')
    if audit.get('projection_sha256')!=projection_fingerprint(meta):raise ValueError('Appearance cameras changed after validation')
    expected={f['file_path'] for f in meta['frames']+meta['heldout_frames']}
    if {r['frame'] for r in audit['views']}!=expected:raise ValueError('Appearance audit does not cover every frame')
    for record in audit['views']:
        if record.get('confidence_file') and sha(folder/record['confidence_file'])!=record['confidence_sha256']:
            raise ValueError('Appearance confidence changed after validation')
        if not record['passed'] or sha(folder/record['frame'])!=record['image_sha256']:
            raise ValueError('Appearance RGB changed after validation')
