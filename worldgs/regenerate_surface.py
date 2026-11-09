"""Correct observed proxy errors and regenerate surface-targeted room teachers."""
import argparse,copy,hashlib,html,json,shutil,time
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion,binary_fill_holes,gaussian_filter
from .controlled_revision import dump,digest,assess_v4,camera_quality


def box(lo,hi,color):
 import trimesh
 lo,hi=np.array(lo,float),np.array(hi,float)
 if (hi-lo<=0).any():raise ValueError('Empty architectural box')
 m=trimesh.creation.box(extents=hi-lo);m.apply_translation((lo+hi)/2);m.visual.face_colors=color;return m


def correct_geometry(source,output,correction_plan=None):
 import trimesh
 from .observed_architecture import ray_plane
 src=Path(source).resolve();out=Path(output).resolve()
 if (out/'geometry_correction.json').exists():return out
 if correction_plan is None:raise ValueError('A source-hash-bound correction plan is required for a new mesh')
 correction=json.loads(Path(correction_plan).read_text(encoding='utf-8-sig'))
 if correction['source_mesh_sha256']!=digest(src/'scene_mesh.glb') or correction['source_image_sha256']!=digest(src/'source_front.png'):raise ValueError('Correction plan belongs to another scene')
 if out.exists() and any(out.iterdir()):raise FileExistsError('Use empty corrected proxy output')
 out.mkdir(parents=True,exist_ok=True)
 for n in ['assets','planning','materials','room_images','meshes']:shutil.copytree(src/n,out/n)
 for n in ['source_front.png','front_alignment.json','request.json','image_room_provenance.json','render_job.json','scene.json','layout.json']:
  if (src/n).exists():shutil.copy2(src/n,out/n)
 meta=json.loads((out/'scene.json').read_text());layout=json.loads((out/'layout.json').read_text());job=json.loads((out/'render_job.json').read_text());components=layout['components'];parts=[];new=[];changes=[]
 for c in components:
  if c['name'] in correction.get('remove_components',[]):changes.append(dict(component=c['name'],operation='Remove randomly fabricated outdoor branch geometry'));continue
  m=trimesh.load(src/'meshes'/(c['name']+'.glb'),force='mesh',process=False);c=copy.deepcopy(c)
  if c['name'] in correction.get('yaw_degrees',{}):
   if correction['yaw_degrees'][c['name']]!=180:raise ValueError('Only inspected 180-degree orientation probes are supported')
   center=m.bounds.mean(0);transform=np.eye(4);transform[:3,:3]=np.diag([-1,1,-1]);transform[:3,3]=center-transform[:3,:3]@center;m.apply_transform(transform);changes.append(dict(component=c['name'],operation='yaw_180_about_own_center',position_and_extent_preserved=True))
  if c['name']=='window_frame':
   from .window_geometry import window_bounds
   bounds=window_bounds(job['plan']['room'])
   if bounds is not None:
    a,b=bounds;pieces=[];z=float(a[2])
    for x in np.linspace(a[0],b[0],correction['window_panes'][0]+1):pieces.append(box([x-.035,a[1]-.04,z-.06],[x+.035,b[1]+.04,z+.06],[245,242,231,255]))
    for y in np.linspace(a[1],b[1],correction['window_panes'][1]+1):pieces.append(box([a[0]-.06,y-.035,z-.06],[b[0]+.06,y+.035,z+.06],[245,242,231,255]))
    m=trimesh.util.concatenate(pieces);changes.append(dict(component=c['name'],operation='Two columns and three rows of window panes, matching source image'))
  if c['name']=='window_outdoor':
   m.visual.face_colors=np.tile(np.array([223,231,229,255],np.uint8),(len(m.faces),1));changes.append(dict(component=c['name'],operation='Opaque distant daylight background; no invented branches'))
  c.update(faces=len(m.faces),bounds=m.bounds.tolist());parts.append(m);new.append(c)
 # Add the side opening visible at the left edge of the original photograph.
 detections=json.loads((src/'planning/architecture_detections.json').read_text());side=[d for d in detections if 'window' in d['label'] and d['box'][0]<meta['size']*.03 and d['box'][2]<meta['size']*.2]
 if side and correction.get('side_window_from_source'):
  d=max(side,key=lambda v:v['score']);K=np.array(meta['frames'][0]['K']);pose=np.array(meta['frames'][0]['transform_matrix']);lo=np.array(layout['shapes'][0]['min']);hi=np.array(layout['shapes'][0]['max']);x0,y0,x1,y1=d['box'];pts=ray_plane([[x0,y0],[x1,y0],[x0,y1],[x1,y1]],K,pose,0,lo[0]);a,b=pts.min(0),pts.max(0)
  if a[2]>=hi[2]-.12 and correction.get('infer_left_wall'):
   # The old width put the entire left wall outside the source frustum.
   # Infer an interior wall plane from the visible window, retaining furniture.
   corner=ray_plane([[x1,(y0+y1)/2]],K,pose,2,hi[2]-.18)[0]
   furniture_min=min(v['min'][0] for v in layout['shapes'][1:]);new_left=min(float(corner[0]),float(furniture_min)-.18)
   if lo[0]<new_left<hi[0]-3:
    previous=lo[0];lo[0]=new_left;layout['shapes'][0]['min'][0]=new_left
    for pi,c in enumerate(new):
     if c.get('group')!='room':continue
     if c['name']=='room_0_0':parts[pi]=box([new_left-.08,lo[1],lo[2]],[new_left,hi[1],hi[2]],[240,236,223,255])
     else:
      vertices=parts[pi].vertices.copy();vertices[:,0]=np.maximum(vertices[:,0],new_left);parts[pi].vertices=vertices
     c.update(bounds=parts[pi].bounds.tolist(),faces=len(parts[pi].faces))
    job['plan']['room']['size'][0]=float(hi[0]-new_left)
    changes.append(dict(component='left_wall_plane',operation='Infer visible side-wall plane from source window rays',old_x=float(previous),new_x=float(new_left),furniture_unchanged=True))
    pts=ray_plane([[x0,y0],[x1,y0],[x0,y1],[x1,y1]],K,pose,0,lo[0]);a,b=pts.min(0),pts.max(0)
  a[1:]=np.maximum(a[1:],lo[1:]+.12);b[1:]=np.minimum(b[1:],hi[1:]-.12)
  if (b[1:]-a[1:]>.15).all():
   idx=next(i for i,c in enumerate(new) if c['name']=='room_0_0');wall=[]
   for u,v in [([lo[0]-.08,lo[1],lo[2]],[lo[0],a[1],hi[2]]),([lo[0]-.08,b[1],lo[2]],[lo[0],hi[1],hi[2]]),([lo[0]-.08,a[1],lo[2]],[lo[0],b[1],a[2]]),([lo[0]-.08,a[1],b[2]],[lo[0],b[1],hi[2]])]:
    if (np.array(v)-u>1e-6).all():wall.append(box(u,v,[240,236,223,255]))
   parts[idx]=trimesh.util.concatenate(wall);new[idx].update(faces=len(parts[idx].faces),bounds=parts[idx].bounds.tolist(),opening=dict(min=a.tolist(),max=b.tolist()))
   frames=[]
   for z in [a[2],(a[2]+b[2])/2,b[2]]:frames.append(box([lo[0]-.04,a[1]-.035,z-.035],[lo[0]+.04,b[1]+.035,z+.035],[244,242,232,255]))
   for y in [a[1],(a[1]+b[1])/2,b[1]]:frames.append(box([lo[0]-.04,y-.035,a[2]-.035],[lo[0]+.04,y+.035,b[2]+.035],[244,242,232,255]))
   for name,m,mat in [('side_window_frame',trimesh.util.concatenate(frames),'window_frame'),('side_window_outdoor',box([lo[0]-.5,a[1]-.2,a[2]-.2],[lo[0]-.48,b[1]+.2,b[2]+.2],[223,231,229,255]),'window_sky')]:parts.append(m);new.append(dict(name=name,group='window',material=mat,faces=len(m.faces),bounds=m.bounds.tolist()))
   changes.append(dict(component='left_wall_window',operation='Source camera ray-plane intersection opening',detection=d,bounds=[a.tolist(),b.tolist()]))
 scene=trimesh.Scene();verts=[];faces=[];owners=[];offset=0
 for i,(m,c) in enumerate(zip(parts,new)):
  m.export(out/'meshes'/(c['name']+'.glb'));scene.add_geometry(m,geom_name=c['name'],node_name=c['name']);verts.append(m.vertices);faces.append(m.faces+offset);owners.append(np.full(len(m.faces),i,np.int32));offset+=len(m.vertices)
 scene.export(out/'scene_mesh.glb');np.savez_compressed(out/'mesh_geometry.npz',vertices=np.concatenate(verts).astype(np.float32),faces=np.concatenate(faces).astype(np.int32),owners=np.concatenate(owners));layout['components']=new;dump(out/'layout.json',layout)
 job.update(folder=str(out));dump(out/'render_job.json',job);dump(out/'planning/image_room_plan.json',job['plan'])
 from .local_room import load_proxy
 from .mesh_scene import MeshRenderer
 mesh,layout,owners=load_proxy(out);renderer=MeshRenderer(mesh,new,owners);f=meta['frames'][0];hit=renderer.render(np.array(f['K']),np.array(f['transform_matrix']),meta['size']);Image.fromarray(hit['rgb']).save(out/'corrected_front_mesh.png')
 from .teacher_geometry import mesh_digest
 meta.update(mesh_digest=mesh_digest(mesh.vertices,mesh.faces),mesh_file_sha256=digest(out/'scene_mesh.glb'));meta.pop('reference_v2_audit',None);dump(out/'scene.json',meta)
 from .front_alignment import audit_front
 audit_front(out)
 dump(out/'geometry_correction.json',dict(source=str(src),source_sha256=digest(src/'source_front.png'),old_mesh_sha256=digest(src/'scene_mesh.glb'),new_mesh_sha256=digest(out/'scene_mesh.glb'),changes=changes,limitations=['Window shape comes from one image and assumed room planes','Bookshelf contents and hidden surfaces are inferred, not recovered ground truth']))
 return out


def surface_cameras(layout,size,count=233):
 from .reference_cameras import dense_camera_plan
 from .room_cameras import safe_eye
 from .mesh_scene import look_at
 from .geometry import intrinsics
 frames=dense_camera_plan(layout['shapes'],size,count);room=layout['shapes'][0];lo,hi=np.array(room['min']),np.array(room['max']);targets=[]
 for shape in layout['shapes'][1:]:
  from .layout_consistency import category
  if category(shape.get('label','')) not in ['bookshelf','sofa','table']:continue
  center=(np.array(shape['min'])+np.array(shape['max']))/2
  for height in [1.05,1.85]:
   for i in range(16):
    angle=2*np.pi*(i+.19)/16;radius=1.9 if category(shape.get('label',''))=='table' else 1.6;eye=center+np.array([radius*np.cos(angle),0,radius*np.sin(angle)]);eye[1]=hi[1]-height
    if not safe_eye(eye,room,layout['shapes'][1:]):continue
    if any(np.linalg.norm(np.maximum(np.maximum(np.array(s['min'])-eye,eye-np.array(s['max'])),0))<.6 for s in layout['shapes'][1:]):continue
    targets.append(dict(id=f'surface_{len(targets):03d}',target='surface_'+shape['name'],focus_object=shape['name'],camera_kind='surface_targeted',K=intrinsics(size,65).tolist(),transform_matrix=look_at(eye,center).tolist(),split='view'))
 # Reserve eight complementary positions for each object when available.
 chosen=[]
 for name in dict.fromkeys(f['focus_object'] for f in targets):
  group=[f for f in targets if f['focus_object']==name];indices=np.linspace(0,len(group)-1,min(8,len(group))).astype(int);chosen.extend(group[i] for i in indices)
 for f in chosen:f['file_path']=f['id']+'.png'
 return frames[:count-len(chosen)]+chosen


def run(source,output,steps=12000,max_points=180000,prepare_only=False,correction_plan=None,previous=None):
 from .local_room import load_proxy
 from .mesh_scene import MeshRenderer
 from .reference_pipeline import initialize_experiment,package_dataset,boundaries,uses_rgb
 from .appearance_geometry import transfer_appearance,depth_condition
 from .appearance_reference import appearance_atlas
 from .local_runtime import LocalRuntime
 from .room_verification import camera_context
 src=Path(source).resolve();out=Path(output).resolve();proxy=correct_geometry(src,out.with_name(out.name+'_proxy'),correction_plan);original=json.loads((proxy/'scene.json').read_text());mesh,layout,owners=load_proxy(proxy);size=original['size'];frames=json.loads((out/'experiment.json').read_text())['cameras'] if (out/'experiment.json').exists() else surface_cameras(layout,size);renderer=MeshRenderer(mesh,layout['components'],owners);runtime=LocalRuntime(out/'logs',image_model='controlnet')
 recipe=dict(version=7,source=str(proxy),model_profile='controlnet',models=runtime.preflight(),size=size,seed=original['seed'],attempts=2,cameras=frames,mesh_sha256=digest(proxy/'scene_mesh.glb'),source_sha256=digest(proxy/'source_front.png'),rgb_target=200,camera_count=233,collect_all_planned=True,validation='physical_depth_v4_plus_visible_surface_preservation')
 initialize_experiment(proxy,out,recipe,(out/'experiment.json').exists())
 # Initial photograph is an observation of the corrected shared mesh, not a plane.
 initial=original['frames'][0];source_hit=renderer.render(np.array(initial['K']),np.array(initial['transform_matrix']),size);source_rgb=np.asarray(Image.open(out/'source_front.png').convert('RGB'));conf=np.full((size,size),.65,np.float32)
 conf[(source_hit['faces']>=0)&np.isin(owners[np.maximum(source_hit['faces'],0)],[0,1,2,4,5])]=0.
 source_owner_ids=np.where(source_hit['faces']>=0,owners[np.maximum(source_hit['faces'],0)],-1)
 from .front_alignment import source_mask
 plan=json.loads((out/'planning/image_room_plan.json').read_text());ids=owners[np.maximum(source_hit['faces'],0)]
 for obj in plan['objects']:
  if obj.get('evidence')!='front':continue
  own=next((i for i,c in enumerate(layout['components']) if c['name']==obj['id']),None)
  if own is None:continue
  mask=binary_fill_holes(source_mask(out,obj,(size,size)));on_object=(source_hit['faces']>=0)&(ids==own);conf[on_object]=np.where(mask[on_object],1.,0.)
 cache=[dict(frame=initial,rgb=source_rgb,depth=source_hit['depth'],valid=source_hit['faces']>=0,confidence=conf,source_anchor=True,owner_ids=source_owner_ids)]
 appearance_atlas(Image.fromarray(source_rgb),original['seed']).save(out/'appearance.png');results=[]
 if prepare_only:
  for f in frames:
   d=out/'references'/f['id'];d.mkdir(parents=True,exist_ok=True);hit=renderer.render(np.array(f['K']),np.array(f['transform_matrix']),size);Image.fromarray(hit['rgb']).save(d/'proxy.png')
  dump(out/'status.json',dict(stage='prepared',cameras=len(frames),surface_cameras=sum(f['camera_kind']=='surface_targeted' for f in frames)));return
 with runtime:
  for index,f in enumerate(frames):
   d=out/'references'/f['id'];d.mkdir(parents=True,exist_ok=True);hit=renderer.render(np.array(f['K']),np.array(f['transform_matrix']),size);hit['owner_ids']=np.where(hit['faces']>=0,owners[np.maximum(hit['faces'],0)],-1);photo=d/'photo.png';record=d/'decision.json'
   if record.exists():
    row=json.loads(record.read_text())
    if digest(photo)!=row['evidence'][str(photo.relative_to(out))]:raise ValueError('Cached photograph changed')
   else:
    transferred,known,confidence=transfer_appearance(hit,f,cache,max_sources=4)
    # Do not freeze low-confidence/grazing observations or a one-pixel seam.
    known &= confidence>=.6
    guide=hit['rgb'].copy();guide[known]=transferred[known]
    Image.fromarray(guide).save(d/'proxy.png');Image.fromarray(np.uint8(~known)*255).save(d/'edit_mask.png');Image.fromarray(depth_condition(hit['depth'])).save(d/'depth.png');Image.fromarray(boundaries(hit,owners)).save(d/'boundaries.png');np.save(d/'mesh_depth.npy',hit['depth']);runtime.depth(d/'proxy.png',f['K'][0][0],d/'proxy_depth.npy')
    labels={o['id']:o.get('label','') for o in plan['objects']};context,objects=camera_context(hit,owners,layout['components'],labels)
    checks=[]
    for attempt in range(2):
     trial=d/f'attempt_{attempt:02d}.png'
     try:
      runtime.generate(d/'proxy.png',out/'appearance.png',context,original['seed']+100003+index*103+attempt,trial,size,kind='structure',depth_guide=d/'depth.png',edge_guide=d/'boundaries.png',visible_labels=[o['label'] for o in objects],appearance=str(plan['room'].get('walls','warm beige plaster'))+', '+original['prompt'],generation_options=dict(strength=.6 if attempt==0 else .45,ip_scale=.2,depth_scale=1.5,boundary_scale=1.1,edit_mask=str(d/'edit_mask.png'),preserve_observed=True))
     except RuntimeError as error:
      if 'Local generator rejected the image' not in str(error):raise
      dump(d/f'rejected_attempt_{attempt:02d}.json',dict(reason=str(error),seed=original['seed']+100003+index*103+attempt));continue
     shutil.copy2(trial,photo)
     runtime.depth(photo,f['K'][0][0],d/'predicted.npy');check=assess_v4(hit,np.load(d/'proxy_depth.npy'),np.load(d/'predicted.npy'),np.asarray(Image.open(photo)),owners,layout['components']);checks.append(check)
     if check['passed']:break
    if not checks:
     shutil.copy2(d/'proxy.png',photo);check=dict(passed=False,score=0.,groups=[],reason='Image model rejected all attempts; proxy is a geometry-only placeholder')
    # Defer uncertainty; never treat the RGB target count as a reason to pass.
    row=dict(camera=f['id'],image=str(photo.relative_to(out)),assessment=check,rgb_teacher=check['passed'],preserved_fraction=float(known.mean()),attempts=len(checks),evidence={str(photo.relative_to(out)):digest(photo)})
    dump(record,row)
   from .reference_coverage import scaled_intrinsics
   coarse=renderer.render(scaled_intrinsics(f['K'],size,96),np.array(f['transform_matrix']),96)
   row['camera_quality']=camera_quality(coarse,owners,layout['components']);row['rgb_eligible']=row['camera_quality']['safe'];dump(record,row)
   results.append(row)
   if row['assessment']['passed'] and row['rgb_eligible']:
    furniture_owners=[j for j,c in enumerate(layout['components']) if c.get('material')=='generated_asset']
    observed_confidence=np.where(np.isin(hit['owner_ids'],furniture_owners),.75,0.).astype(np.float32)
    cache.append(dict(frame=f,rgb=np.asarray(Image.open(photo)),depth=hit['depth'],valid=hit['faces']>=0,confidence=observed_confidence,owner_ids=hit['owner_ids']))
   dump(out/'assessment.json',dict(complete=len(results)==len(frames),ready=False,views=results,rgb_teacher_count=1+sum(r['assessment']['passed'] for r in results),rgb_target=200));dump(out/'status.json',dict(stage='generating',completed=index+1,total=len(frames),accepted=1+sum(r['assessment']['passed'] for r in results)));print(f'{index+1}/233 {f["id"]}: {row["assessment"]["passed"]}, preserved={row["preserved_fraction"]:.3f}',flush=True)
 # Preserve targeted surface coverage before filling the remaining RGB quota.
 order=sorted(range(len(results)),key=lambda i:(frames[i]['camera_kind']!='surface_targeted',i));selected=[i for i in order if results[i]['assessment']['passed'] and results[i].get('rgb_eligible',True)][:199]
 for i,row in enumerate(results):row['rgb_teacher']=i in selected
 dump(out/'assessment.json',dict(complete=True,ready=len(selected)==199,views=results,rgb_teacher_count=len(selected)+1,rgb_target=200))
 if previous:
  from .surface_finish import finish
  return finish(out,previous,steps,max_points)
 if len(selected)<199:
  dump(out/'status.json',dict(stage='rgb_target_unmet',rgb_teacher_count=len(selected)+1,target=200));return
 # Exact geometry evidence is fresh for the corrected mesh and all new cameras.
 meta=package_dataset(proxy,out,original,mesh,layout,owners,frames,results,allow_geometry_only=True);meta['controlled_training']=dict(rgb_robust=True,rgb_scale=.12);dump(out/'scene.json',meta)
 # Same previously defined independent object-facing cameras, never training RGB.
 from .train import train
 train(out,steps,max_points)
 dump(out/'generation_result.json',dict(steps=steps,rgb_teachers=200,cameras=233,targeted_cameras=sum(f['camera_kind']=='surface_targeted' for f in frames),initial_source_sha256=digest(out/'source_front.png'),geometry_corrections=json.loads((proxy/'geometry_correction.json').read_text())))
 return meta

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--output',required=True);p.add_argument('--steps',type=int,default=12000);p.add_argument('--max-points',type=int,default=180000);p.add_argument('--prepare-only',action='store_true');p.add_argument('--correction-plan');p.add_argument('--previous');run(**vars(p.parse_args()))
