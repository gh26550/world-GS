"""Resume only the controlled training/evaluation phase after reference collection."""
import json,shutil,time
from pathlib import Path
import numpy as np
from .controlled_revision import clone_tree,dump,digest

def finish(output):
 out=Path(output).resolve();spec=json.loads((out/'manifest.json').read_text());src=Path(spec['source']);ref=out/'revised_references';held=json.loads((out/'evaluation_cameras.json').read_text());results={}
 if not held:raise ValueError('No independent evaluation cameras')
 cells=[('A_original_legacy',src,False),('B_revised_legacy',ref,False),('C_original_robust',src,True),('D_revised_robust',ref,True)]
 for name,dataset,robust in cells:
  cell=out/name
  if not cell.exists():
   clone_tree(dataset,cell)
   if (dataset/'references').exists():shutil.copytree(dataset/'references',cell/'references',copy_function=shutil.copy2)
  m=json.loads((cell/'scene.json').read_text());m['heldout_frames']=held;m['coverage_probe_frames']=[];m['controlled_training']=dict(rgb_robust=robust,rgb_scale=.12,geometry_confidence_separate=True)
  dump(cell/'scene.json',m)
  for f in held:
   for key in ['file_path','depth_file','mask_file','face_file']:shutil.copy2(out/f[key],cell/f[key])
  from .appearance import projection_fingerprint
  a=json.loads((cell/'reference_audit.json').read_text());a['projection_sha256']=projection_fingerprint(m);dump(cell/'reference_audit.json',a)
  from .front_alignment import audit_front
  audit_front(cell)
  if digest(cell/'points.npz')!=digest(src/'points.npz'):raise ValueError('Initial point clouds differ')
  if not (cell/'training.json').exists():
   dump(out/'status.json',dict(stage='training',cell=name,steps=spec['steps']));t=time.monotonic()
   from .train import train
   train(cell,spec['steps'],spec['max_points']);dump(cell/'controlled_timing.json',dict(training_and_export_seconds=time.monotonic()-t))
  ev=json.loads((cell/'evaluation.json').read_text())['frames'];train=json.loads((cell/'training.json').read_text())
  results[name]=dict(mean_depth_absrel=float(np.mean([r['depth_relative_error'] for r in ev if r['depth_relative_error'] is not None])),mean_missing=float(np.mean([r['alpha_below_half_on_target'] for r in ev])),max_missing=float(np.max([r['alpha_below_half_on_target'] for r in ev])),gaussians=train['final_points'],steps=train['steps'],initial_points=train['initial_points'],initial_seed_sha256=digest(cell/'points.npz'))
 summary=dict(design=spec,evaluation_cameras=len(held),cells=results,rgb_revised=json.loads((ref/'assessment.json').read_text())['rgb_teacher_count'],limits=['One room and one seed; not generalization evidence','Depth errors are against the same scaffold, not independent 3D truth','No unseen photographic ground truth; no PSNR/SSIM reported'])
 rows=[]
 for name,v in results.items():
  rows.append('<tr><td>'+name+'</td><td>'+str(v['steps'])+'</td><td>'+str(v['gaussians'])+'</td><td>'+f"{v['mean_depth_absrel']:.4f}"+'</td><td>'+f"{v['mean_missing']:.2%}"+'</td></tr>')
 pictures=[]
 for i in range(len(held)):
  pictures.append('<h2>'+held[i]['target']+'</h2><div style="display:flex;gap:12px">'+''.join('<figure style="margin:0;width:24%"><img style="width:100%" src="'+n+f'/heldout_render_{i:03d}.png"><figcaption>'+n+'</figcaption></figure>' for n,_,_ in cells)+'</div>')
 page='<meta charset="utf-8"><title>4条件GS比較</title><style>body{font:16px/1.6 sans-serif;margin:24px}td,th{padding:8px;border:1px solid #bbb}table{border-collapse:collapse}</style><h1>教師×GS学習方法の4条件比較</h1><p>A/C：既存教師、B/D：修正教師。A/B：従来L1、C/D：外れ値に頑健なRGB損失。初期点群・乱数・学習回数・評価カメラは共通。</p><p>深度は統合メッシュへの一致、欠損はα&lt;0.5の面積。実世界の正解や見た目の品質そのものではありません。</p><table><tr><th>条件</th><th>学習回数</th><th>ガウス数</th><th>深度AbsRel ↓</th><th>平均欠損 ↓</th></tr>'+''.join(rows)+'</table>'+''.join(pictures)
 (out/'comparison.html').write_text(page,encoding='utf-8');dump(out/'comparison.json',summary)
 from .controlled_metrics import evaluate
 summary=evaluate(out)
 from .controlled_object_eval import evaluate as evaluate_objects
 object_results=evaluate_objects(out)
 summary['object_evaluation_views']=len(next(iter(object_results.values()))['views'])
 dump(out/'comparison.json',summary)
 dump(out/'status.json',dict(stage='complete',**summary));return summary

if __name__=='__main__':
 import sys
 finish(sys.argv[1])
