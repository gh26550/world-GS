"""Finalize teacher evidence, retry unresolved safe views, fit and evaluate GS."""
import json,shutil,hashlib,time
from pathlib import Path
import numpy as np
from PIL import Image
from .controlled_revision import dump,assess_v4,digest

def finish(path,previous,steps=12000,max_points=180000,allow_partial_rgb=False):
 from .preserved_observation_audit import review
 from .local_runtime import LocalRuntime
 from .local_room import load_proxy
 from .mesh_scene import MeshRenderer
 from .room_verification import camera_context
 from .reference_pipeline import package_dataset
 out=Path(path).resolve();recipe=json.loads((out/'experiment.json').read_text());src=Path(recipe['source']);original=json.loads((src/'scene.json').read_text());mesh,layout,owners=load_proxy(out);renderer=MeshRenderer(mesh,layout['components'],owners);count=review(out);attempted=json.loads((out/'repair_attempts.json').read_text()) if (out/'repair_attempts.json').exists() else []
 if count<200:
  rows=json.loads((out/'assessment.json').read_text())['views'];plan=json.loads((out/'planning/image_room_plan.json').read_text());runtime=LocalRuntime(out/'logs',image_model='controlnet')
  candidates=sorted([i for i,r in enumerate(rows) if not r['assessment']['passed'] and r.get('rgb_eligible',True)],key=lambda i:(recipe['cameras'][i]['camera_kind']!='surface_targeted',-rows[i]['assessment']['score']))
  with runtime:
   for i in candidates:
    f=recipe['cameras'][i];d=out/'references'/f['id'];hit=renderer.render(np.array(f['K']),np.array(f['transform_matrix']),recipe['size']);labels={o['id']:o.get('label','') for o in plan['objects']};context,objects=camera_context(hit,owners,layout['components'],labels)
    start=1+max([a['attempt'] for a in attempted if a['camera']==f['id']],default=1)
    for attempt,strength in enumerate([.35,.8],start):
     photo=d/f'repair_{attempt:02d}.png'
     try:runtime.generate(d/'proxy.png',out/'appearance.png',context,recipe['seed']+700001+i*107+attempt,photo,recipe['size'],kind='structure',depth_guide=d/'depth.png',edge_guide=d/'boundaries.png',visible_labels=[o['label'] for o in objects],appearance=str(plan['room'].get('walls','warm beige plaster'))+', '+original['prompt'],generation_options=dict(strength=strength,ip_scale=.15,depth_scale=1.7,boundary_scale=1.3,edit_mask=str(d/'edit_mask.png'),preserve_observed=True))
     except RuntimeError as error:
      if 'Local generator rejected the image' not in str(error):raise
      attempted.append(dict(camera=f['id'],attempt=attempt,rejected=str(error)));continue
     prediction=d/f'repair_depth_{attempt:02d}.npy';runtime.depth(photo,f['K'][0][0],prediction);check=assess_v4(hit,np.load(d/'proxy_depth.npy'),np.load(prediction),np.array(Image.open(photo)),owners,layout['components']);attempted.append(dict(camera=f['id'],attempt=attempt,passed=check['passed'],score=check['score']))
     if recipe.get('scene_specific_corrections') is False:
      from .room_verification import verify_layout
      runtime.layout(photo,objects,d/f'repair_layout_{attempt:02d}.json');detected=json.loads((d/f'repair_layout_{attempt:02d}.json').read_text());lc=verify_layout(objects,detected['detections'],np.asarray(Image.open(photo)),hit=hit,owners=owners,predicted_depth=np.load(prediction));check.update(depth_passed=check['passed'],layout=lc,passed=check['passed'] and lc['passed'])
     if check['passed']:
      old=d/'pre_repair_photo.png'
      if not old.exists():shutil.copy2(d/'photo.png',old)
      shutil.copy2(photo,d/'photo.png');shutil.copy2(prediction,d/'predicted.npy');rows[i].update(assessment=check,image=str((d/'photo.png').relative_to(out)),evidence={str((d/'photo.png').relative_to(out)):digest(d/'photo.png')});rows[i].pop('original_depth_assessment',None);dump(d/'decision.json',rows[i]);break
    dump(out/'assessment.json',dict(complete=True,views=rows));count=review(out);rows=json.loads((out/'assessment.json').read_text())['views'];dump(out/'status.json',dict(stage='repairing_teachers',rgb_teachers=count,target=200,attempts=len(attempted)));dump(out/'repair_attempts.json',attempted)
    if count>=200:break
 if count<200:
  from .surface_table_views import add
  for _ in range(3):
   count=add(out,mode='adaptive',max_views=8)
   if count>=200:break
 recipe=json.loads((out/'experiment.json').read_text())
 if count<200 and not allow_partial_rgb:raise RuntimeError(f'RGB target unmet after evidence review and reseeding: {count}/200')
 assessment=json.loads((out/'assessment.json').read_text());rows=assessment['views'];frames=recipe['cameras']
 for row in rows:
  d=out/'references'/row['camera']
  row['evidence'].update({str(p.relative_to(out)):digest(p) for p in [d/n for n in ['photo.png','proxy.png','edit_mask.png','depth.png','boundaries.png','mesh_depth.npy','proxy_depth.npy','predicted.npy']] if p.exists()})
 dump(out/'assessment.json',dict(assessment,views=rows))
 meta=package_dataset(src,out,original,mesh,layout,owners,frames,rows,allow_geometry_only=True);meta['controlled_training']=dict(rgb_robust=True,rgb_scale=.12);dump(out/'scene.json',meta);shutil.copy2(src/'corrected_front_mesh.png',out/'corrected_front_mesh.png')
 record=dict(mesh=digest(out/'scene_mesh.glb'),points=digest(out/'points.npz'),frames=[dict(file=f['file_path'],sha=digest(out/f['file_path']),weight=f.get('rgb_weight',1.),K=f['K'],pose=f['transform_matrix']) for f in meta['frames']],steps=steps,max_points=max_points);signature=hashlib.sha256(json.dumps(record,sort_keys=True).encode()).hexdigest();marker=out/'surface_training_signature.json';reuse=marker.exists() and json.loads(marker.read_text())['signature']==signature and (out/'training.json').exists()
 if not reuse:
  if (out/'training.json').exists():
   archive=out/('before_final_teacher_audit_'+str(int(time.time())));archive.mkdir()
   for p in list(out.iterdir()):
    if p.is_file() and (p.name in ['scene.ply','scene_unity.ply','gaussians.npz','training.json','evaluation.json','front_render.png','gs_coverage_quality.json'] or p.name.startswith(('heldout_render_','heldout_alpha_','heldout_depth_'))):shutil.move(str(p),archive/p.name)
  from .train import train
  train(out,steps,max_points);dump(marker,dict(signature=signature,record=record))
 from .surface_evaluation import evaluate
 metrics=evaluate(out,previous);dump(out/'generation_result.json',dict(complete=True,rgb_teachers=count,cameras=len(frames),targeted_cameras=sum(f['camera_kind']=='surface_targeted' for f in frames),steps=steps,initial_points=max_points,corrected_geometry=json.loads((src/'geometry_correction.json').read_text()),independent_evaluation=metrics,limitations=['Depth metrics use the inferred scaffold; not independent real-world truth','A different prompt is a generalization example, not a controlled improvement comparison']))
 return metrics

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('path');p.add_argument('--previous',required=True);p.add_argument('--steps',type=int,default=12000);p.add_argument('--max-points',type=int,default=180000);p.add_argument('--allow-partial-rgb',action='store_true');finish(**vars(p.parse_args()))
