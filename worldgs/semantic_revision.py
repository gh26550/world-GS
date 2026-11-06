"""Semantic confirmation for newly generated RGB teachers, before comparison."""
import json,shutil,html,os
from pathlib import Path
import numpy as np
from PIL import Image
from .controlled_revision import dump,digest

def audit(out):
 from .local_room import load_proxy
 from .mesh_scene import MeshRenderer
 from .room_verification import camera_context,verify_layout
 from .local_runtime import LocalRuntime
 out=Path(out).resolve();ref=out/'revised_references';src=Path(json.loads((out/'manifest.json').read_text())['source']);source_meta=json.loads((src/'scene.json').read_text());recipe=json.loads((ref/'experiment.json').read_text());frames=recipe['cameras'];rows=json.loads((ref/'assessment.json').read_text())['views'];orig_recipe=json.loads((src/'experiment.json').read_text());orig=json.loads((src/'assessment.json').read_text())['views'];diagnostics=json.loads((out/'camera_depth_audit.json').read_text());mesh,layout,owners=load_proxy(src);renderer=MeshRenderer(mesh,layout['components'],owners);runtime=LocalRuntime(ref/'logs',image_model='controlnet');checks=[]
 for i,row in enumerate(rows):
  if not diagnostics[i].get('regeneration',{}).get('used'):continue
  frame=frames[i];hit=renderer.render(np.array(frame['K']),np.array(frame['transform_matrix']),source_meta['size']);labels={o['name']:o.get('label',o['name']) for o in layout['shapes'][1:]};_,objects=camera_context(hit,owners,layout['components'],labels);p=ref/'references'/frame['id']/'semantic_check.json'
  if not p.exists():runtime.layout(ref/row['image'],objects,p)
  detected=json.loads(p.read_text());photo=np.asarray(Image.open(ref/row['image']));check=verify_layout(objects,detected['detections'],photo,hit,owners,np.load(ref/'references'/frame['id']/'revised_predicted.npy'))
  # Recorded full-frame visual inspection found a cushioned table in this image.
  # Never override this concrete counterexample merely because DINO agrees.
  review_file=out/'visual_review.json'
  reviews=json.loads(review_file.read_text()) if review_file.exists() else {}
  review=reviews.get(frame['id'],{})
  if review and review.get('sha256')!=digest(ref/row['image']):raise ValueError('Visual review image changed')
  manual_reject=review.get('reject',False)
  record=dict(camera=frame['id'],detector=check,visual_review_reject=manual_reject,visual_reason=review.get('reason'))
  good=(check['passed'] or review.get('accept',False)) and not manual_reject
  record['visual_review_accept']=review.get('accept',False);record['accepted']=good;checks.append(record)
  if not good:
   frames[i]=orig_recipe['cameras'][i];rows[i]=dict(orig[i],assessment=diagnostics[i]['new_check'],rgb_teacher=diagnostics[i]['new_check']['passed'] and diagnostics[i]['before']['safe'])
   d=ref/'references'/frame['id'];sd=src/'references'/frame['id']
   for name in ['proxy.png','depth.png','boundaries.png','mesh_depth.npy']:shutil.copy2(sd/name,d/name)
   diagnostics[i]['regeneration']['used']=False;diagnostics[i]['semantic_rejected']=record
 selected=[]
 for i,row in enumerate(rows):
  safe=diagnostics[i]['proposal']['safe'] if diagnostics[i].get('regeneration',{}).get('used') else diagnostics[i]['before']['safe']
  row['rgb_teacher']=row['assessment']['passed'] and safe and len(selected)<199
  if row['rgb_teacher']:selected.append(i)
 dump(out/'semantic_review.json',checks);dump(out/'camera_depth_audit.json',diagnostics);dump(ref/'experiment.json',dict(recipe,cameras=frames));dump(ref/'assessment.json',dict(complete=True,ready=len(selected)==199,rgb_teacher_count=len(selected)+1,rgb_target=200,views=rows))
 cards=[]
 for i,row in enumerate(rows):
  before=os.path.relpath(src/orig[i]['image'],out);after=os.path.relpath(ref/row['image'],out);use='RGB採用' if row['rgb_teacher'] else 'RGB除外・形状のみ'
  evidence=next((c for c in checks if c['camera']==row['camera']),None)
  cards.append('<section><h2>'+html.escape(row['camera'])+'：'+use+'</h2><div style="display:flex;gap:12px"><figure><img width="380" src="'+html.escape(before)+'"><figcaption>既存</figcaption></figure><figure><img width="380" src="'+html.escape(after)+'"><figcaption>実際に使用する教師</figcaption></figure></div><details><summary>検査と目視確認の記録</summary><pre>'+html.escape(json.dumps(dict(camera=diagnostics[i],semantic=evidence),ensure_ascii=False,indent=2))+'</pre></details></section>')
 (out/'review.html').write_text('<meta charset="utf-8"><h1>確定した教師画像と採否</h1><p>初期画像を含むRGB教師 '+str(len(selected)+1)+'枚。再生成画像を棄却した場合は右側にも既存画像を表示します。検出器の判断と目視確認を区別しています。</p>'+''.join(cards),encoding='utf-8')
 from .reference_pipeline import package_dataset
 package_dataset(src,ref,source_meta,mesh,layout,owners,frames,rows,allow_geometry_only=True)
 shutil.copy2(src/'points.npz',ref/'points.npz')
 # A remains valid. B/D must be repackaged to exactly match the revised evidence.
 for name in ['B_revised_legacy','D_revised_robust']:
  cell=out/name
  if cell.exists():
   archive=out/(name+'_before_semantic_review')
   if archive.exists():raise FileExistsError(archive)
   cell.rename(archive)
 return checks

if __name__=='__main__':
 import sys
 audit(sys.argv[1])

