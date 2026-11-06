"""Independent object-facing views of frozen checkpoints; no further training."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from .controlled_revision import dump

def evaluate(output):
 import torch
 from gsplat import rasterization
 from .local_room import load_proxy
 from .mesh_scene import MeshRenderer,look_at
 from .room_cameras import safe_eye,nudge_eye
 from .geometry import intrinsics
 out=Path(output).resolve();spec=json.loads((out/'manifest.json').read_text());src=Path(spec['source']);mesh,layout,owners=load_proxy(src);renderer=MeshRenderer(mesh,layout['components'],owners);size=512;K=intrinsics(size,60);room=layout['shapes'][0];center=(np.array(room['min'])+np.array(room['max']))/2;objects=[s for s in layout['shapes'][1:] if any(v in s.get('label','') for v in ['sofa','table','bookshelf'])];frames=[]
 for i in range(8):
  a=2*np.pi*(i+.183)/8;eye=nudge_eye(center+np.array([2.1*np.cos(a),-.1 if i%2 else .35,2.1*np.sin(a)]),room,layout['shapes'][1:]);obj=objects[i%len(objects)];target=(np.array(obj['min'])+np.array(obj['max']))/2
  if not safe_eye(eye,room,layout['shapes'][1:]):continue
  frames.append(dict(id=f'object_eval_{i:02d}',target=obj['name'],K=K.tolist(),transform_matrix=look_at(eye,target).tolist()))
 dump(out/'object_evaluation_cameras.json',frames);hits=[renderer.render(K,np.array(f['transform_matrix']),size) for f in frames];summary={};pictures=[]
 for name in ['A_original_legacy','B_revised_legacy','C_original_robust','D_revised_robust']:
  cell=out/name;npz=np.load(cell/'gaussians.npz');p={k:torch.tensor(npz[k],device='cuda') for k in npz.files};stats=[]
  for i,(f,hit) in enumerate(zip(frames,hits)):
   with torch.inference_mode():
    image,alpha,_=rasterization(means=p['means'],quats=p['quats'],scales=p['scales'].exp(),opacities=p['opacities'].sigmoid(),colors=torch.cat([p['sh0'],p['shN']],1),sh_degree=3,viewmats=torch.linalg.inv(torch.tensor(f['transform_matrix'],device='cuda',dtype=torch.float32))[None],Ks=torch.tensor(K,device='cuda',dtype=torch.float32)[None],width=size,height=size,packed=False,render_mode='RGB+ED')
   rgb=image[0,:,:,:3].cpu().numpy();depth=image[0,:,:,3].cpu().numpy();a=alpha[0,:,:,0].cpu().numpy();valid=hit['faces']>=0;supported=valid&(a>.5);stats.append(dict(camera=f['id'],target=f['target'],depth_absrel=float((np.abs(depth[supported]-hit['depth'][supported])/hit['depth'][supported]).mean()),missing=float((valid&(a<.5)).sum()/max(valid.sum(),1))))
   Image.fromarray(np.uint8(np.clip(rgb,0,1)*255)).save(cell/f'object_render_{i:02d}.png')
  summary[name]=dict(views=stats,mean_depth_absrel=float(np.mean([s['depth_absrel'] for s in stats])),mean_missing=float(np.mean([s['missing'] for s in stats])))
  del p;torch.cuda.empty_cache()
 dump(out/'object_comparison.json',dict(scope='Eight additional shared object-facing novel cameras; same frozen checkpoints; depth against scaffold only',cells=summary))
 for i,f in enumerate(frames):pictures.append('<h2>家具を向いた未学習視点 '+f['id']+'：'+f['target']+'</h2><div style="display:flex;gap:12px">'+''.join('<figure style="width:24%;margin:0"><img style="width:100%" src="'+n+f'/object_render_{i:02d}.png"><figcaption>'+n+'</figcaption></figure>' for n in summary)+'</div>')
 page=(out/'comparison.html').read_text(encoding='utf-8');(out/'comparison.html').write_text(page+'<h1>追加：家具を向いた8視点</h1><p>部屋中心の12視点だけでは家具の確認が不足するため、同じチェックポイントを家具に向けた8つの共通カメラでも評価。再学習は行っていません。</p>'+''.join(pictures),encoding='utf-8');return summary

if __name__=='__main__':
 import sys
 evaluate(sys.argv[1])
