"""Read-only source-photo / integrated-mesh correspondence inspection."""
import hashlib
import html
import json
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion,binary_dilation


def silhouette_metrics(target,predicted):
    target=np.asarray(target,bool);predicted=np.asarray(predicted,bool)
    if target.shape!=predicted.shape:raise ValueError('Masks must share the image coordinate system')
    intersection=int((target&predicted).sum());union=int((target|predicted).sum())
    def box(mask):
        yy,xx=np.nonzero(mask)
        return [int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)] if len(xx) else None
    def centroid(mask):
        yy,xx=np.nonzero(mask)
        return [float(xx.mean()),float(yy.mean())] if len(xx) else None
    a,b=centroid(target),centroid(predicted)
    return dict(iou=intersection/max(union,1),target_recall=intersection/max(int(target.sum()),1),predicted_precision=intersection/max(int(predicted.sum()),1),
                target_pixels=int(target.sum()),predicted_pixels=int(predicted.sum()),target_only_pixels=int((target&~predicted).sum()),predicted_only_pixels=int((predicted&~target).sum()),
                target_box=box(target),predicted_box=box(predicted),centroid_offset_pixels=float(np.linalg.norm(np.asarray(a)-b)) if a is not None and b is not None else None)


def inspect_front(path,output,orientation_probes=False):
    from .local_room import load_proxy
    from .mesh_scene import MeshRenderer
    from .front_alignment import source_mask,box_iou
    source=Path(path).resolve();folder=Path(output).resolve()
    if source==folder or source in folder.parents or folder in source.parents:raise ValueError('Use a separate inspection output')
    if folder.exists() and any(folder.iterdir()):raise FileExistsError('Use an empty inspection output; original evidence is preserved')
    def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
    before={name:sha(source/name) for name in ['source_front.png','scene_mesh.glb','mesh_geometry.npz','scene.json']}
    meta=json.loads((source/'scene.json').read_text());plan=json.loads((source/'planning/image_room_plan.json').read_text())
    mesh,layout,owners=load_proxy(source);frame=next((f for f in meta['frames'] if f.get('target')=='front' or f['file_path'] in ['source_front.png','source_anchor.png']),meta['frames'][0])
    K=np.asarray(frame.get('K',meta['K']));pose=np.asarray(frame['transform_matrix']);size=meta['size']
    image=Image.open(source/'source_front.png').convert('RGB')
    if image.size!=(size,size):raise ValueError('Source must match saved camera image dimensions; no implicit crop/resizing')
    renderer=MeshRenderer(mesh,layout['components'],owners);hit=renderer.render(K,pose,size);valid=hit['faces']>=0;ids=np.full(valid.shape,-1,int);ids[valid]=owners[hit['faces'][valid]]
    folder.mkdir(parents=True,exist_ok=True);image.save(folder/'source_front.png');Image.fromarray(hit['rgb']).save(folder/'mesh_same_camera.png')
    Image.fromarray(np.uint8(np.asarray(image,dtype=float)*.5+hit['rgb']*.5)).save(folder/'blend.png')
    np.savez_compressed(folder/'projection_evidence.npz',K=K,camera_to_world=pose,depth=hit['depth'],owner_ids=ids)
    names={c['name']:i for i,c in enumerate(layout['components'])};rows=[];cards=[];probes=[]
    labels={'sofa_00':'ソファ','bookshelf_left_00':'左本棚','bookshelf_right_00':'右本棚','coffee_table_00':'テーブル'}
    for obj in plan['objects']:
        if obj.get('evidence')!='front':continue
        target=source_mask(source,obj,image.size);predicted=ids==names[obj['id']];row=dict(id=obj['id'],label=labels.get(obj['id'],obj['id']),**silhouette_metrics(target,predicted));rows.append(row)
        name=obj['id'];Image.fromarray(np.uint8(target)*255).save(folder/(name+'_source_mask.png'));Image.fromarray(np.uint8(predicted)*255).save(folder/(name+'_mesh_mask.png'))
        a=binary_dilation(target&~binary_erosion(target),iterations=1);b=binary_dilation(predicted&~binary_erosion(predicted),iterations=1)
        overlay=np.uint8(np.asarray(image,dtype=float)*.65);overlay[a]=[0,230,255];overlay[b]=[255,40,190];overlay[a&b]=[255,255,255];Image.fromarray(overlay).save(folder/(name+'_contours.png'))
        error=np.uint8(np.asarray(image,dtype=float)*.3);error[target&predicted]=[220,220,220];error[target&~predicted]=[0,230,255];error[predicted&~target]=[255,40,190];Image.fromarray(error).save(folder/(name+'_difference.png'))
        cards.append(f'<section><h2>{html.escape(row["label"])}</h2><p>可視輪郭IoU {row["iou"]:.1%}／元マスクの被覆 {row["target_recall"]:.1%}／メッシュ輪郭の適合 {row["predicted_precision"]:.1%}</p><div class="images"><figure><img src="{name}_contours.png"><figcaption>輪郭：水色＝初期画像の分割、ピンク＝メッシュ、白＝重なる境界</figcaption></figure><figure><img src="{name}_difference.png"><figcaption>面の差分：灰＝一致、水色＝元マスクのみ、ピンク＝メッシュのみ。分割誤りも含み得る。</figcaption></figure><figure><img src="{name}_source_mask.png"><figcaption>初期画像から切り戻した分割マスク。これ自体が正しいかも確認する。</figcaption></figure></div></section>')
        if orientation_probes and obj.get('label')=='bookshelf':
            component=names[name];vertices=np.unique(mesh.faces[owners==component]);probe=mesh.copy();center=(probe.vertices[vertices].min(0)+probe.vertices[vertices].max(0))/2;v=probe.vertices.copy();v[vertices]=(v[vertices]-center)*[-1,1,-1]+center;probe.vertices=v
            result=MeshRenderer(probe,layout['components'],owners).render(K,pose,size);filename=name+'_yaw180_diagnostic.png';Image.fromarray(result['rgb']).save(folder/filename);probes.append(dict(id=name,yaw_delta_degrees=180,applied_to_source=False,file_path=filename))
    architecture=[];detection_file=source/'planning/architecture_detections.json'
    if detection_file.exists():
        for detection in json.loads(detection_file.read_text()):
            group=detection['label'];components=[i for i,c in enumerate(layout['components']) if c.get('group')==group]
            mask=valid&np.isin(ids,components);yy,xx=np.nonzero(mask)
            if not len(xx):continue
            observed=np.asarray(detection['box'])/size;predicted=np.array([xx.min(),yy.min(),xx.max()+1,yy.max()+1])/size
            architecture.append(dict(group=group,comparison='detector box only, not segmentation/3D truth',source_box=detection['box'],predicted_visible_box=(predicted*size).tolist(),box_iou=box_iou(observed,predicted)))
    fit=json.loads((source/'front_alignment.json').read_text()) if (source/'front_alignment.json').exists() else {}
    report=dict(source=str(source),source_fingerprints=before,camera_source=frame['file_path'],camera_K=K.tolist(),camera_to_world=pose.tolist(),horizontal_fov_degrees=float(np.rad2deg(2*np.arctan(size/(2*K[0,0])))),
                camera_fit_bound_warnings=fit.get('camera_bound_warning',[]),previous_box_fit_iou=fit.get('mean_box_iou_after'),objects=rows,mean_visible_silhouette_iou=float(np.mean([r['iou'] for r in rows])) if rows else None,
                architecture_box_checks=architecture,orientation_probes=probes,limitations=['Source masks and architectural detections are estimated, not ground truth','Silhouette overlap cannot certify front/back orientation, hidden geometry, room perspective or texture identity','No optimization or modification of source image, camera, mesh or Gaussian scene'])
    (folder/'report.json').write_text(json.dumps(report,indent=2))
    table=''.join(f'<tr><td>{html.escape(r["label"])}</td><td>{r["iou"]:.1%}</td><td>{r["target_recall"]:.1%}</td><td>{r["predicted_precision"]:.1%}</td></tr>' for r in rows)
    probe_cards=''.join(f'<figure><img src="{p["file_path"]}"><figcaption>{html.escape(labels.get(p["id"],p["id"]))}だけを180度回した診断用描画。元メッシュへは未適用。</figcaption></figure>' for p in probes)
    (folder/'comparison.html').write_text('<!doctype html><html lang="ja"><meta charset="utf-8"><title>初期画像と統合メッシュの照合</title><style>body{font:16px/1.7 sans-serif;margin:24px;color:#223}.images{display:flex;flex-wrap:wrap;gap:14px}figure{margin:0;width:min(31%,512px);min-width:240px}img{width:100%}table{border-collapse:collapse}td,th{padding:6px 14px;border:1px solid #bbb}section{border-top:1px solid #aaa;margin-top:30px}</style><h1>初期画像と統合メッシュの照合</h1><p>初期画像専用カメラの内部行列・姿勢を使用。ガウス描画ではなく、統合メッシュそのものを初期画像と照合する。カメラと形状の修正は行っていない。</p><div class="images"><figure><img src="source_front.png"><figcaption>初期参照画像</figcaption></figure><figure><img src="mesh_same_camera.png"><figcaption>同じカメラからの統合メッシュ。面色による確認用描画で、写真品質の比較ではない。</figcaption></figure><figure><img src="blend.png"><figcaption>50%重ね合わせ。配置・輪郭を確認。</figcaption></figure></div><h2>可視輪郭の対応</h2><table><tr><th>物体</th><th>IoU</th><th>元マスクの被覆</th><th>メッシュ輪郭の適合</th></tr>'+table+'</table><p>被覆が高く適合が低い場合は、メッシュの輪郭が元マスクより広い可能性がある。ただし初期分割マスクの取りこぼしでも同じ数値になる。IoUの高さだけでは前後の向きや配置の完全一致を保証しない。</p>'+''.join(cards)+'<h2>向きの診断用比較</h2><p>回転後に棚の開口が現れるかを確認するための仮の描画。良否の自動判定や修正適用は行っていない。</p><div class="images">'+probe_cards+'</div><p><a href="report.json">数値・カメラ・根拠ファイルの記録</a></p></html>',encoding='utf-8')
    after={name:sha(source/name) for name in before}
    if before!=after:raise ValueError('Source changed while inspection was running')
    return report
