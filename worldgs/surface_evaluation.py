"""Evaluate a regenerated checkpoint from the previous independent object cameras."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from .controlled_revision import dump

def evaluate(path,previous):
 import torch
 from gsplat import rasterization
 from .local_room import load_proxy
 from .mesh_scene import MeshRenderer
 folder=Path(path).resolve();previous=Path(previous).resolve() if previous else None;mesh,layout,owners=load_proxy(folder)
 if previous:frames=json.loads((previous/'object_evaluation_cameras.json').read_text())
 else:
  from .regenerate_surface import surface_cameras
  from .mesh_scene import look_at
  from .room_cameras import safe_eye
  frames=[];meta=json.loads((folder/'scene.json').read_text())
  for base in surface_cameras(layout,meta['size']):
   if base.get('camera_kind')!='surface_targeted':continue
   eye=np.array(base['transform_matrix'])[:3,3]+[.13,-.07,.11]
   if not safe_eye(eye,layout['shapes'][0],layout['shapes'][1:]):continue
   obj=next(o for o in layout['shapes'] if o['name']==base['focus_object']);f=dict(base,id=f"object_eval_{len(frames):02d}",target=obj['name'],transform_matrix=look_at(eye,(np.array(obj['min'])+obj['max'])/2).tolist());frames.append(f)
  frames=frames[::max(1,len(frames)//12)][:12];dump(folder/'object_evaluation_cameras.json',frames)
 renderer=MeshRenderer(mesh,layout['components'],owners);meta=json.loads((folder/'scene.json').read_text());size=meta['size'];g=np.load(folder/'gaussians.npz');p={k:torch.tensor(g[k],device='cuda') for k in g.files};degree=json.loads((folder/'training.json').read_text())['sh_degree'];rows=[];cards=[]
 for i,f in enumerate(frames):
  K=np.array(f['K']);pose=np.array(f['transform_matrix']);hit=renderer.render(K,pose,size)
  with torch.inference_mode():
   rgb,a,_=rasterization(means=p['means'],quats=p['quats'],scales=p['scales'].exp(),opacities=p['opacities'].sigmoid(),colors=torch.cat([p['sh0'],p['shN']],1),sh_degree=degree,viewmats=torch.linalg.inv(torch.tensor(pose,device='cuda',dtype=torch.float32))[None],Ks=torch.tensor(K,device='cuda',dtype=torch.float32)[None],width=size,height=size,packed=False,render_mode='RGB+ED')
  image=np.uint8(rgb[0,:,:,:3].clamp(0,1).cpu().numpy()*255);alpha=a[0,:,:,0].cpu().numpy();depth=rgb[0,:,:,3].cpu().numpy();valid=hit['faces']>=0;supported=valid&(alpha>.5);name=f'independent_object_{i:02d}.png';Image.fromarray(image).save(folder/name);np.save(folder/f'independent_alpha_{i:02d}.npy',alpha)
  ids=owners[np.maximum(hit['faces'],0)];groups=[]
  for group in dict.fromkeys(c['group'] for c in layout['components'] if c.get('material')=='generated_asset'):
   own=[j for j,c in enumerate(layout['components']) if c['group']==group];mask=valid&np.isin(ids,own)
   if mask.sum()>=64:groups.append(dict(object=group,pixels=int(mask.sum()),missing=float((mask&(alpha<.5)).sum()/mask.sum())))
  rows.append(dict(camera=f['id'],target=f['target'],missing=float((valid&(alpha<.5)).sum()/valid.sum()),depth_absrel=float((np.abs(depth[supported]-hit['depth'][supported])/hit['depth'][supported]).mean()),objects=groups))
  cards.append('<h2>'+f['id']+'：'+f['target']+'</h2><div style="display:flex;gap:16px"><figure><img width="450" src="previous_object_'+str(i).zfill(2)+'.png"><figcaption>前回D：3,000 steps / 100,000 initial points</figcaption></figure><figure><img width="450" src="'+name+'"><figcaption>今回：12,000 steps / 180,000 initial points</figcaption></figure></div>')
  import shutil
  if previous:shutil.copy2(previous/'D_revised_robust'/f'object_render_{i:02d}.png',folder/f'previous_object_{i:02d}.png')
  else:cards[-1]='<h2>'+f['id']+' : '+f['target']+'</h2><img width=450 src='+name+'>'
 summary=dict(views=rows,mean_missing=float(np.mean([r['missing'] for r in rows])),max_missing=float(np.max([r['missing'] for r in rows])),mean_depth_absrel=float(np.mean([r['depth_absrel'] for r in rows])),scope='Same previous cameras; changed geometry and budget' if previous else 'Novel translated object-facing cameras; depth reference is inferred mesh, not ground truth')
 dump(folder/'independent_evaluation.json',summary)
 intro='初期画像は同一。今回は概形と教師、学習予算を変更しています。画像は同じ8カメラで比較しますが、修正単独の効果を示す制御実験ではありません。' if previous else '新しいプロンプトから初期画像・概形・多視点教師を生成しました。シーン専用の補正値は使用していません。深度誤差は推定メッシュに対する値です。'
 (folder/'comparison.html').write_text('<meta charset="utf-8"><title>概形修正・不足面補完後の再生成</title><h1>再生成の比較</h1><p>'+intro+'</p><h2>正面</h2><div style="display:flex"><figure><img width="400" src="source_front.png"><figcaption>初期画像</figcaption></figure><figure><img width="400" src="corrected_front_mesh.png"><figcaption>修正概形</figcaption></figure><figure><img width="400" src="front_render.png"><figcaption>今回のGS</figcaption></figure></div>'+''.join(cards),encoding='utf-8')
 return summary

if __name__=='__main__':
 import sys
 evaluate(sys.argv[1],sys.argv[2])
