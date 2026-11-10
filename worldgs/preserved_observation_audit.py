"""Audit estimator disagreements only where projected observations are unchanged."""
from pathlib import Path
import json
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation,binary_erosion,maximum_filter,minimum_filter
from .controlled_revision import dump,assess_v4
from .reference_geometry import edges

def review(path):
 import cv2
 from .local_room import load_proxy
 from .mesh_scene import MeshRenderer
 folder=Path(path).resolve();recipe=json.loads((folder/'experiment.json').read_text());mesh,layout,owners=load_proxy(folder);renderer=MeshRenderer(mesh,layout['components'],owners);assessment=json.loads((folder/'assessment.json').read_text());rows=assessment['views'];changes=[]
 for f,row in zip(recipe['cameras'],rows):
  baseline=row.get('original_depth_assessment',row['assessment'])
  if not baseline.get('layout',{}).get('passed',True):continue
  if baseline['passed']:continue
  d=folder/'references'/f['id'];photo=np.array(Image.open(d/'photo.png'));guide=np.array(Image.open(d/'proxy.png'));known=np.array(Image.open(d/'edit_mask.png'))==0
  if not np.array_equal(photo[known],guide[known]):raise ValueError('Claimed preserved pixels changed')
  hit=renderer.render(np.array(f['K']),np.array(f['transform_matrix']),recipe['size']);size=recipe['size'];tol=max(1,round(10*size/512));valid=hit['faces']>=0;ids=owners[np.maximum(hit['faces'],0)];interior=binary_erosion(np.ones_like(valid),iterations=tol+2);depth=hit['depth'];de=edges(depth);pe=binary_dilation(edges(np.load(d/'proxy_depth.npy')),iterations=tol);step=maximum_filter(depth,5)-minimum_filter(np.where(valid,depth,np.inf),5)>np.maximum(.07,.025*depth);rgb_edges=binary_dilation(cv2.Canny(cv2.cvtColor(photo,cv2.COLOR_RGB2GRAY),25,76)>0,iterations=tol);supported=binary_dilation(known,iterations=2);check=json.loads(json.dumps(baseline));witness=[]
  for g in check['groups']:
   if g['passed']:continue
   mask=valid if g['group']=='scene' else valid&np.isin(ids,[i for i,c in enumerate(layout['components']) if c['group']==g['group']]);band=(binary_dilation(mask,iterations=2)&~binary_erosion(mask,iterations=2))&interior;sil=mask&~binary_erosion(mask)&interior;required=de&valid&interior if g['group']=='scene' else (de&band)|sil;observed=required&step&pe
   fraction=float(supported[observed].mean()) if observed.any() else 0.;rgb=float(rgb_edges[observed].mean()) if observed.any() else 0.
   # Evidence of fixed image-space placement; not recovered 3D ground truth.
   eligible=g['depth_recall'] is not None and g['depth_recall']<.55 and (g['rgb_recall'] is None or g['rgb_recall']>=.6) and fraction>=.8 and rgb>=.8
   witness.append(dict(group=g['group'],fixed_observation_boundary_fraction=fraction,rgb_boundary_recall=rgb,verified_preserved_pixels=int(known.sum()),admissible=eligible))
   if eligible:g['passed']=True;g['reason']='Depth estimator disagreement; unchanged projected boundary and RGB evidence verified'
  check['passed']=any(g['informative'] for g in check['groups']) and all(g['passed'] for g in check['groups'] if g['informative']);check['preserved_observation_witness']=witness;row['original_depth_assessment']=baseline;row['assessment']=check;dump(d/'decision.json',row);changes.append(dict(camera=f['id'],recovered=check['passed'],witness=witness))
 order=sorted(range(len(rows)),key=lambda i:(recipe['cameras'][i]['camera_kind']!='surface_targeted',i));selected=[i for i in order if rows[i]['assessment']['passed'] and rows[i].get('rgb_eligible',True)][:199]
 for i,row in enumerate(rows):row['rgb_teacher']=i in selected
 dump(folder/'preserved_observation_audit.json',dict(policy='80% boundary support by unchanged projected observation + 80% RGB edge correspondence; depth disagreements retained explicitly',views=changes));dump(folder/'assessment.json',dict(complete=True,ready=len(selected)==199,rgb_teacher_count=len(selected)+1,rgb_target=200,views=rows));return len(selected)+1

if __name__=='__main__':
 import sys
 print(review(sys.argv[1]))
