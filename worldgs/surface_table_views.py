"""Replace uninformative failed room views with missing table-facing views."""
import json,copy
from pathlib import Path
import numpy as np
from PIL import Image
from .controlled_revision import dump,assess_v4,camera_quality,digest

def add(path,mode="table",max_views=8):
 from .local_room import load_proxy
 from .mesh_scene import MeshRenderer
 from .regenerate_surface import surface_cameras
 from .layout_consistency import category
 from .appearance_geometry import transfer_appearance,depth_condition
 from .reference_pipeline import boundaries
 from .reference_coverage import scaled_intrinsics
 from .room_verification import camera_context
 from .local_runtime import LocalRuntime
 folder=Path(path).resolve();recipe=json.loads((folder/'experiment.json').read_text());assessment=json.loads((folder/'assessment.json').read_text());rows=assessment['views'];mesh,layout,owners=load_proxy(folder);renderer=MeshRenderer(mesh,layout['components'],owners);size=recipe['size'];plan=json.loads((folder/'planning/image_room_plan.json').read_text());source=Path(recipe['source']);original=json.loads((source/'scene.json').read_text());initial=original['frames'][0];h=renderer.render(np.array(initial['K']),np.array(initial['transform_matrix']),size);oi=np.where(h['faces']>=0,owners[np.maximum(h['faces'],0)],-1);conf=np.where(np.isin(oi,[0,1,2,4,5]),0.,.65).astype(np.float32);cache=[dict(frame=initial,rgb=np.array(Image.open(folder/'source_front.png')),depth=h['depth'],valid=h['faces']>=0,confidence=conf,owner_ids=oi,source_anchor=True)]
 existing={f['id'] for f in recipe['cameras']};planned=surface_cameras(layout,size);table_names={s['name'] for s in layout['shapes'][1:] if category(s.get('label',''))=='table'};new=[f for f in planned if f.get('focus_object') in table_names and f['id'] not in existing]
 if mode=='adaptive':
  from .room_cameras import safe_eye
  from .mesh_scene import look_at
  offset=1+max([int(n.rsplit('_',1)[-1]) for n in existing if n.startswith('adaptive_surface_')],default=-1)
  new=[];objects={o['name']:o for o in layout['shapes'][1:]};bases=[f for f,r in zip(recipe['cameras'],rows) if f.get('focus_object') in objects and (not r['assessment']['passed']) and r.get('rgb_eligible',True)]
  if not bases:bases=[f for f in recipe['cameras'] if f.get('focus_object') in objects]
  for base in bases:
   obj=objects[base['focus_object']];target=(np.array(obj['min'])+obj['max'])/2
   for delta in [[.22,0,0],[-.22,0,0],[0,0,.22],[0,0,-.22]]:
    eye=np.array(base['transform_matrix'])[:3,3]+delta
    if not safe_eye(eye,layout['shapes'][0],layout['shapes'][1:]):continue
    if any(np.linalg.norm(np.maximum(np.maximum(np.array(o['min'])-eye,eye-np.array(o['max'])),0))<.6 for o in layout['shapes'][1:]):continue
    f=copy.deepcopy(base);f.update(id='adaptive_surface_'+str(offset+len(new)).zfill(3),transform_matrix=look_at(eye,target).tolist());f['file_path']=f['id']+'.png'
    if f['id'] not in existing:new.append(f)
  new=new[:max_views]
 slots=[i for i,r in enumerate(rows) if not r['assessment']['passed'] and recipe['cameras'][i]['camera_kind']!='surface_targeted'];changes=[];runtime=LocalRuntime(folder/'logs',image_model='controlnet')
 with runtime:
  for slot,f in zip(slots,new):
   d=folder/'references'/f['id'];d.mkdir(exist_ok=True);hit=renderer.render(np.array(f['K']),np.array(f['transform_matrix']),size);hit['owner_ids']=np.where(hit['faces']>=0,owners[np.maximum(hit['faces'],0)],-1);transferred,known,confidence=transfer_appearance(hit,f,cache,max_sources=4);known&=confidence>=.6;guide=hit['rgb'].copy();guide[known]=transferred[known];Image.fromarray(guide).save(d/'proxy.png');Image.fromarray(np.uint8(~known)*255).save(d/'edit_mask.png');Image.fromarray(depth_condition(hit['depth'])).save(d/'depth.png');Image.fromarray(boundaries(hit,owners)).save(d/'boundaries.png');np.save(d/'mesh_depth.npy',hit['depth']);runtime.depth(d/'proxy.png',f['K'][0][0],d/'proxy_depth.npy');labels={o['id']:o['label'] for o in plan['objects']};context,objects=camera_context(hit,owners,layout['components'],labels);photo=d/'photo.png';runtime.generate(d/'proxy.png',folder/'appearance.png',context,recipe['seed']+910003+slot,photo,size,kind='structure',depth_guide=d/'depth.png',edge_guide=d/'boundaries.png',visible_labels=[o['label'] for o in objects],appearance=str(plan['room'].get('walls','warm beige plaster'))+', '+original['prompt'],generation_options=dict(strength=.55,ip_scale=.15,depth_scale=1.5,boundary_scale=1.2,edit_mask=str(d/'edit_mask.png'),preserve_observed=True));runtime.depth(photo,f['K'][0][0],d/'predicted.npy');check=assess_v4(hit,np.load(d/'proxy_depth.npy'),np.load(d/'predicted.npy'),np.array(Image.open(photo)),owners,layout['components']);q=camera_quality(renderer.render(scaled_intrinsics(f['K'],size,96),np.array(f['transform_matrix']),96),owners,layout['components']);row=dict(camera=f['id'],image=str(photo.relative_to(folder)),assessment=check,rgb_teacher=False,rgb_eligible=q['safe'],camera_quality=q,preserved_fraction=float(known.mean()),evidence={str(photo.relative_to(folder)):digest(photo)});dump(d/'decision.json',row);changes.append(dict(slot=slot,previous=recipe['cameras'][slot],new=f,reason='Adaptive surface coverage; replace failed ordinary view'));recipe['cameras'][slot]=f;rows[slot]=row;dump(folder/'experiment.json',recipe);dump(folder/'assessment.json',dict(complete=True,views=rows));dump(folder/(mode+'_camera_revision.json'),changes);print(f['id'],check['passed'],q['safe'],flush=True)
   if recipe.get('scene_specific_corrections') is False:
    from .room_verification import verify_layout
    runtime.layout(photo,objects,d/'layout_detection.json');detected=json.loads((d/'layout_detection.json').read_text());lc=verify_layout(objects,detected['detections'],np.array(Image.open(photo)),hit=hit,owners=owners,predicted_depth=np.load(d/'predicted.npy'));check.update(depth_passed=check['passed'],layout=lc,passed=check['passed'] and lc['passed']);row['assessment']=check;dump(d/'decision.json',row);dump(folder/'assessment.json',dict(complete=True,views=rows))
   if 1+sum(r['assessment']['passed'] and r.get('rgb_eligible',True) for r in rows)>=200:break
 from .preserved_observation_audit import review
 return review(folder)

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('path');p.add_argument('--mode',choices=['table','adaptive'],default='table');p.add_argument('--max-views',type=int,default=8);print(add(**vars(p.parse_args())))
