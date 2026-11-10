"""Common-view depth and visual report; no invented RGB ground truth."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from .controlled_revision import dump,project_overlap


def evaluate(path):
 out=Path(path);frames=json.loads((out/'evaluation_cameras.json').read_text());summary=json.loads((out/'comparison.json').read_text());cells=summary['cells']
 for name,row in cells.items():
  cell=out/name;errors=[];pairs=[];all_depth=[]
  for i,f in enumerate(frames):
   d=np.load(cell/f'heldout_depth_{i:03d}.npy');a=np.load(cell/f'heldout_alpha_{i:03d}.npy');target=np.load(cell/f['depth_file']);mask=np.load(cell/f['mask_file'])&(a>.5)&(d>.03)
   errors.extend((d[mask]-target[mask]).tolist());all_depth.append(d)
  for i,f in enumerate(frames):
   size=len(all_depth[i]);yy,xx=np.indices((size,size));rays=np.stack([xx,yy,np.ones_like(xx)],-1)@np.linalg.inv(f['K']).T;pose=np.array(f['transform_matrix']);mesh=np.load(cell/f['depth_file']);mesh_xyz=(rays*mesh[...,None])@pose[:3,:3].T+pose[:3,3];pred_xyz=(rays*all_depth[i][...,None])@pose[:3,:3].T+pose[:3,3]
   ids=np.load(cell/f['face_file']);hit=dict(xyz=mesh_xyz,faces=ids)
   for j in [k for k in range(len(frames)) if k!=i][:]:
    g=frames[j];other=np.load(cell/g['depth_file']);_,_,observable=project_overlap(hit,f,g,other,size)
    gp=np.asarray(g['transform_matrix']);camera=(pred_xyz-gp[:3,3])@gp[:3,:3];uv=camera@np.array(g['K']).T;uv=uv[...,:2]/np.maximum(uv[...,2:],1e-8);x,y=np.rint(uv).astype(int).transpose(2,0,1);valid=observable&(x>=2)&(y>=2)&(x<size-2)&(y<size-2)&(camera[...,2]>.03)&(np.load(cell/f'heldout_alpha_{i:03d}.npy')>.5)
    x,y=np.clip(x,0,size-1),np.clip(y,0,size-1);d=all_depth[j][y,x];valid&=(d>.03)&(np.load(cell/f'heldout_alpha_{j:03d}.npy')[y,x]>.5)
    if valid.sum()<64:continue
    err=np.abs(camera[...,2][valid]-d[valid]);pairs.append(dict(source=i,target=j,pixels=int(valid.sum()),absrel=float((err/d[valid]).mean()),mae_norm=float(err.mean()/np.median(d[valid]))))
  e=np.array(errors);row['mesh_mae']=float(np.abs(e).mean());row['mesh_rmse']=float(np.sqrt((e*e).mean()));row['cross_view_absrel']=float(np.mean([p['absrel'] for p in pairs]));row['cross_view_mae_norm']=float(np.mean([p['mae_norm'] for p in pairs]));row['reprojection_pairs']=len(pairs)
  dump(cell/'controlled_metrics.json',dict(**row,pairs=pairs,scope='Mesh-visible cross-view pairs, alpha > 0.5; no independent geometry truth; not an exact reproduction of WorldMesh implementation'))
 summary['metric_scope']='Mesh depth alignment + locally implemented scale-normalized cross-view reprojection. Masks/scale are explicitly recorded; do not compare raw values directly with WorldMesh tables.'
 dump(out/'comparison.json',summary)
 # Contact sheet only for navigation; full-size originals remain linked.
 thumbs=Image.new('RGB',(4*256,len(frames)*280),(245,245,245))
 from PIL import ImageDraw
 draw=ImageDraw.Draw(thumbs)
 for j,name in enumerate(cells):
  for i in range(len(frames)):
   thumbs.paste(Image.open(out/name/f'heldout_render_{i:03d}.png').resize((256,256)),(j*256,i*280+24));draw.text((j*256+5,i*280+5),name,fill=(30,30,30))
 thumbs.save(out/'comparison_contact.png')
 page=(out/'comparison.html').read_text(encoding='utf-8');table='<h2>追加の深度評価</h2><p>同じ未学習カメラ。メッシュ準拠と視点間の整合を分けて表示。WorldMeshと同一実装ではありません。</p><table><tr><th>条件</th><th>Mesh MAE ↓</th><th>Mesh RMSE ↓</th><th>再投影AbsRel ↓</th><th>再投影MAE-norm ↓</th></tr>'
 for name,r in cells.items():table+='<tr><td>'+name+'</td>'+''.join('<td>'+f'{r[k]:.4f}'+'</td>' for k in ['mesh_mae','mesh_rmse','cross_view_absrel','cross_view_mae_norm'])+'</tr>'
 (out/'comparison.html').write_text(page.replace('<h2>independent_',table+'</table><h2>independent_',1),encoding='utf-8')
 return summary

if __name__=='__main__':
 import sys
 evaluate(sys.argv[1])
