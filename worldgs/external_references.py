"""Import two-image generated photographs without projecting source pixels.

Raw images, prompts, guide/camera hashes and resampling are retained. Layout
recognition is a heuristic acceptance gate, not independent geometry truth.
"""
import hashlib,json,shutil
from pathlib import Path
import numpy as np
from PIL import Image
import trimesh

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def validate_external(folder,meta):
    if not meta.get('external_reference_audit'):return
    folder=Path(folder);a=json.loads((folder/meta['external_reference_audit']).read_text())
    if not a.get('complete') or not a.get('passed'):raise ValueError('External references are incomplete or rejected')
    from .appearance import projection_fingerprint
    if a['projection_sha256']!=projection_fingerprint(meta) or a['mesh_sha256']!=digest(folder/'scene_mesh.glb'):
        raise ValueError('External reference geometry/cameras changed')
    if a['source_sha256']!=digest(folder/'source_front.png'):raise ValueError('External reference source changed')
    expected={f['file_path'] for f in meta['frames']}
    if {r['file_path'] for r in a['views']}!=expected:raise ValueError('External audit does not cover training images')
    if 'rgb_weights' in a and a['rgb_weights']!={f['file_path']:f.get('rgb_weight',1.) for f in meta['frames']}:raise ValueError('External RGB supervision weights changed')
    for row in a['views']:
        if not row['passed'] or digest(folder/row['file_path'])!=row['teacher_sha256']:raise ValueError('External teacher changed')
        if row.get('raw_file') and digest(folder/row['raw_file'])!=row['raw_sha256']:raise ValueError('External raw image changed')
        if row.get('guide_file') and digest(folder/row['guide_file'])!=row['guide_sha256']:raise ValueError('External proxy guide changed')

def import_reference_pack(path,manifest,output):
    from .appearance import projection_fingerprint
    from .appearance_models import AppearanceModels
    from .appearance_geometry import geometry_check,structural_planarity
    from .layout_consistency import layout_contract,projected_objects,layout_check,calibrate_source_parts
    from .mesh_scene import MeshRenderer
    from .learned_references import save_references
    from .front_alignment import audit_front,source_mask
    src=Path(path).resolve();folder=Path(output).resolve();manifest=Path(manifest).resolve()
    if src==folder or src in folder.parents or folder in src.parents:raise ValueError('Use a separate sibling output')
    if folder.exists() and any(folder.iterdir()):raise FileExistsError('Use an empty output')
    old=json.loads((src/'scene.json').read_text());rows=json.loads(manifest.read_text());size=old['size']
    for row in rows:
        for key in ['raw_path','guide_path']:
            if not Path(row[key]).is_file():raise FileNotFoundError(row[key])
    folder.mkdir(parents=True,exist_ok=True);(folder/'raw').mkdir();(folder/'guides').mkdir()
    for name in ['assets','planning','materials','room_images','meshes']:
        if (src/name).exists():shutil.copytree(src/name,folder/name)
    for name in ['scene_mesh.glb','mesh_geometry.npz','layout.json','source_front.png','front_alignment.json','image_room_provenance.json','request.json']:
        if (src/name).exists():shutil.copy2(src/name,folder/name)
    old=json.loads((src/'scene.json').read_text());rows=json.loads(manifest.read_text());size=old['size']
    geometry=np.load(folder/'mesh_geometry.npz');owners=geometry['owners'];mesh=trimesh.Trimesh(geometry['vertices'],geometry['faces'],process=False)
    components=json.loads((folder/'layout.json').read_text())['components'];plan=json.loads((folder/'planning/image_room_plan.json').read_text())
    renderer=MeshRenderer(mesh,components,owners);contract=layout_contract(geometry,components,plan,digest(folder/'scene_mesh.glb'),digest(folder/'source_front.png'))
    (folder/'layout_contract.json').write_text(json.dumps(contract,indent=2));labels={o['id']:o['label'] for o in contract['objects']};vocabulary=sorted(set(labels.values())|{'sofa','chair','table','bookshelf','bed','cabinet'})
    models=AppearanceModels(old['seed']);source_frame=dict(old['frames'][0],split='view',file_path='source_teacher.png');source_rgb=np.asarray(Image.open(folder/'source_front.png').convert('RGB'))
    hit=renderer.render(np.array(source_frame['K']),np.array(source_frame['transform_matrix']),size);objects=projected_objects(hit,owners,components,labels,min_pixels=max(16,size*size//4096))
    detections,_=models.layout_detections(source_rgb,objects,vocabulary)
    masks={o['id']:source_mask(folder,o,(size,size)) for o in plan['objects'] if o.get('evidence')=='front'}
    calibration=calibrate_source_parts(objects,detections,(size,size),masks);(folder/'source_instance_parts.json').write_text(json.dumps(calibration,indent=2))
    Image.fromarray(source_rgb).save(folder/source_frame['file_path']);frames=[source_frame];records=[dict(file_path=source_frame['file_path'],target='front',passed=True,teacher_sha256=digest(folder/source_frame['file_path']),policy='Exact initial reference; alignment checked separately')];rejected=[]
    for row in rows:
        f={k:row[k] for k in ['file_path','target','K','transform_matrix']};f['split']='view';name=Path(f['file_path']).stem
        raw=Path(row['raw_path']);guide=Path(row['guide_path']);raw_file='raw/'+name+'.png';guide_file='guides/'+name+'.png';shutil.copy2(raw,folder/raw_file);shutil.copy2(guide,folder/guide_file)
        image=Image.open(raw).convert('RGB');ratio=image.width/image.height
        if abs(ratio-1.)>.002:raise ValueError('Generated image aspect differs from proxy; cropping is forbidden')
        image=image.resize((size,size),Image.Resampling.BOX);rgb=np.asarray(image);image.save(folder/f['file_path'])
        hit=renderer.render(np.array(f['K']),np.array(f['transform_matrix']),size);objects=projected_objects(hit,owners,components,labels,min_pixels=max(16,size*size//4096))
        detections,conflicts=models.layout_detections(rgb,objects,vocabulary);layout=layout_check(objects,detections,(size,size),source_parts=calibration.get('parts',{}));inverse=models.inverse_depth(rgb)
        edges=geometry_check(hit['depth'],inverse,hit['faces'],owners,tolerance=10,min_recall=.45);planes=structural_planarity(inverse,hit['faces'],owners,np.ones((size,size),bool))
        review=row.get('layout_review')
        reviewed=bool(review and review.get('verdict')=='same_world_layout' and review.get('reason') and review.get('raw_sha256')==digest(raw) and review.get('guide_sha256')==digest(guide))
        passed=(layout['passed'] or reviewed) and edges['passed'] and planes['passed']
        record=dict(file_path=f['file_path'],target=f['target'],passed=passed,layout=layout,layout_review=review,automatic_layout_passed=layout['passed'],depth_edges=edges,planarity=planes,recognition_conflicts=conflicts,raw_file=raw_file,guide_file=guide_file,raw_sha256=digest(folder/raw_file),guide_sha256=digest(folder/guide_file),reference_sha256=digest(folder/'source_front.png'),teacher_sha256=digest(folder/f['file_path']),prompt=row['prompt'],provider=row['provider'],preprocessing=dict(operation='aspect_preserving_native_size_resample',output_size=[size,size],resampling='Pillow BOX',crop=None,color_normalization=False),inherited_fraction=0.)
        records.append(record)
        if passed:frames.append(f)
        else:rejected.append(f['file_path'])
        print(f"External reference {f['file_path']}: passed={passed}, missing={layout['missing_or_displaced']}, unexpected={layout['unexpected_or_displaced']}, edge={edges['depth_edge_recall']:.3f}",flush=True)
    del models
    import gc,torch
    gc.collect();torch.cuda.empty_cache()
    audit=dict(complete=not rejected,passed=not rejected,mesh_sha256=digest(folder/'scene_mesh.glb'),source_sha256=digest(folder/'source_front.png'),views=records,rejected=rejected)
    (folder/'external_reference_audit.json').write_text(json.dumps(audit,indent=2))
    if rejected:raise ValueError('Rejected generated references; inspect external_reference_audit.json: '+str(rejected))
    # Unphotographed surfaces contribute depth/opacity, never proxy RGB loss.
    selected_targets={f['target'] for f in frames}
    for i,f in enumerate(old['frames']):
        if f['target'] in selected_targets:continue
        name=f'geometry_only_{i:03d}.png';shutil.copy2(src/f['file_path'],folder/name)
        helper=dict(f,file_path=name,split='view',rgb_weight=0.,reference_source='Mesh depth/opacity only; proxy RGB is not a training target')
        frames.append(helper);records.append(dict(file_path=name,target=f['target'],passed=True,teacher_sha256=digest(folder/name),policy='Geometry only, RGB loss disabled',rgb_weight=0.))
    # Geometric probe views are not independent photographic evaluation targets.
    probes=[f for f in old['heldout_frames'] if not f['target'].startswith('room_')]
    probes=probes[:4] if len(probes)>=4 else old['heldout_frames'][:4]
    for i,f in enumerate(probes):
        f=dict(f,split='heldout',file_path=f'geometry_probe_{i:02d}.png',photographic_ground_truth=False);shutil.copy2(src/probes[i]['file_path'],folder/f['file_path']);frames.append(f)
    job=json.loads((src/'render_job.json').read_text());job.update(folder=str(folder),frames=frames);(folder/'render_job.json').write_text(json.dumps(job,indent=2))
    meta=save_references(folder,old['prompt'],old['seed'],size,mesh,components,owners,job)
    meta.update(mode='image_room_mesh_v12',front_alignment='front_alignment.json',external_reference_audit='external_reference_audit.json',layout_contract='layout_contract.json',training_geometry_policy='shared_layout_local_detail_v17',generation_method='initial photo -> image-grounded proxy incl window/rug -> per-camera proxy + source identity two-image generation -> strict layout/depth checks -> GS',reference_source='Two-image generated photographs; initial photo never projected onto novel-view guide',rgb_evaluation_is_independent=False)
    audit['rgb_weights']={f['file_path']:f.get('rgb_weight',1.) for f in meta['frames']};audit['projection_sha256']=projection_fingerprint(meta);(folder/'external_reference_audit.json').write_text(json.dumps(audit,indent=2));(folder/'scene.json').write_text(json.dumps(meta,indent=2));audit_front(folder)
    validate_external(folder,meta);return meta

def prepare_photograph_guides(path,output):
    """Write unprojected mesh guides, camera records and two-input prompts."""
    from .mesh_scene import MeshRenderer
    src=Path(path).resolve();out=Path(output).resolve()
    if src==out or src in out.parents or out in src.parents:raise ValueError('Use a separate sibling output')
    if out.exists() and any(out.iterdir()):raise FileExistsError('Use an empty guide output')
    meta=json.loads((src/'scene.json').read_text());layout=json.loads((src/'layout.json').read_text());g=np.load(src/'mesh_geometry.npz')
    mesh=trimesh.Trimesh(g['vertices'],g['faces'],process=False);colors=[]
    for c in layout['components']:
        part=trimesh.load(src/'meshes'/(c['name']+'.glb'),force='mesh',process=False)
        if part.visual.kind=='texture':part.visual=part.visual.to_color()
        colors.append(part.visual.face_colors)
    mesh.visual.face_colors=np.concatenate(colors);renderer=MeshRenderer(mesh,layout['components'],g['owners'])
    frames=meta['frames']+meta['heldout_frames'];groups={}
    for f in frames:
        if f['target'].startswith('object_detail_'):groups.setdefault(f['target'].rsplit('_',1)[0],[]).append(f)
    selected=[]
    for group in groups.values():
        if len(group)<2:raise ValueError('Need translated paired views for each object')
        a,b=max(((a,b) for i,a in enumerate(group) for b in group[i+1:]),key=lambda ab:np.linalg.norm(np.array(ab[0]['transform_matrix'])[:3,3]-np.array(ab[1]['transform_matrix'])[:3,3]));selected.extend([a,b])
    for yaw in [0,90,180,270]:
        choices=[f for f in frames if f['target']==f'room_yaw_{yaw}_pitch_0']
        if not choices:raise ValueError('Need cardinal yaw proxy views')
        selected.append(choices[0])
    out.mkdir(parents=True,exist_ok=True);shutil.copy2(src/'source_front.png',out/'source_front.png');rows=[]
    for i,f in enumerate(selected):
        guide=out/f'guide_{i:02d}.png';hit=renderer.render(np.array(f['K']),np.array(f['transform_matrix']),meta['size']);Image.fromarray(hit['rgb']).save(guide)
        prompt=('Use case: sketch-to-render. Create exactly one photorealistic photograph of the SAME room. Image 1 is authoritative for camera, perspective, object count, positions, orientations, major silhouettes and occlusions, including clipped furniture. Image 2 supplies furniture identity, colors, materials and lighting ONLY; never copy its viewpoint. Replace coarse surfaces with realistic materials. Preserve the guide-visible windows and rug. Do not add, remove or move furniture or openings. One opaque square 1:1 PNG, edge-to-edge. No text, borders or panels. Camera: '+f'dual_{i:02d}'+'. Original scene request: '+meta['prompt'])
        rows.append(dict(file_path=f'dual_{i:02d}.png',target=f['target'],K=f['K'],transform_matrix=f['transform_matrix'],guide_path=str(guide),guide_sha256=digest(guide),reference_path=str(out/'source_front.png'),reference_sha256=digest(out/'source_front.png'),raw_path=None,provider=None,prompt=prompt))
    (out/'manifest_template.json').write_text(json.dumps(rows,indent=2));return rows
