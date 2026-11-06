"""Explain stored reference decisions without changing their thresholds or evidence."""
import html
import json
from pathlib import Path
import numpy as np
from PIL import Image


def name(group):
    return {'room':'部屋構造','window':'窓','rug':'ラグ','coffee_table_00':'テーブル',
            'bookshelf_left_00':'左の本棚','bookshelf_right_00':'右の本棚','sofa_00':'ソファ'}.get(group,group)


def missing_edges(mesh_depth,proxy_depth,photo_depth,face_ids,owners,group,size):
    """Reproduce the assessor's exact observable group boundary test."""
    from scipy.ndimage import binary_dilation,binary_erosion
    from .reference_geometry import edges
    tolerance=max(1,round(10*size/512));valid=face_ids>=0
    ids=np.full(face_ids.shape,-1,int);ids[valid]=owners[face_ids[valid]]
    mask=valid&np.isin(ids,group['owners'])
    band=binary_dilation(mask,iterations=2)&~binary_erosion(mask,iterations=2)
    silhouette=(mask&~binary_erosion(mask))&binary_erosion(np.ones(mask.shape,bool),iterations=2)
    expected=edges(mesh_depth)
    if any(owner>=6 for owner in group['owners']):expected|=silhouette
    observable=band&expected&binary_dilation(edges(proxy_depth),iterations=tolerance)
    matched=observable&binary_dilation(edges(photo_depth),iterations=tolerance)
    return observable,matched,observable&~matched


def diagnostic(folder,frame,row):
    """Produce a data overlay for a rejected view, retaining original images."""
    import trimesh
    from scipy.ndimage import binary_dilation
    from .mesh_scene import MeshRenderer
    geometry=np.load(folder/'mesh_geometry.npz');owners=geometry['owners']
    mesh=trimesh.Trimesh(geometry['vertices'],geometry['faces'],process=False)
    components=json.loads((folder/'layout.json').read_text())['components']
    renderer=MeshRenderer(mesh,components,owners);d=folder/'references'/frame['id']
    mesh_depth=np.load(d/'mesh_depth.npy');size=mesh_depth.shape[0]
    hit=renderer.render(np.array(frame['K']),np.array(frame['transform_matrix']),size)
    index=Path(row['image']).stem.removeprefix('candidate_')
    photo_depth=np.load(d/f'predicted_{index}.npy');proxy_depth=np.load(d/'proxy_depth.npy')
    red=np.zeros(mesh_depth.shape,bool);green=red.copy();counts={}
    for group in row['assessment']['components']:
        if group['passed']:continue
        observable,matched,missing=missing_edges(mesh_depth,proxy_depth,photo_depth,hit['faces'],owners,group,size)
        if int(observable.sum())!=group['observable_depth_edges']:raise ValueError('Diagnostic differs from recorded assessment')
        if observable.any() and abs(float(matched.sum()/observable.sum())-group['depth_recall'])>1e-10:raise ValueError('Diagnostic recall differs from recorded assessment')
        counts[group['group']]=dict(expected=int(observable.sum()),matched=int(matched.sum()),missing=int(missing.sum()))
        red|=missing;green|=matched
    overlay=np.uint8(np.asarray(Image.open(folder/row['image']).convert('RGB'))*.65)
    overlay[binary_dilation(green,iterations=1)]=[40,240,110]
    overlay[binary_dilation(red,iterations=1)]=[255,45,65]
    Image.fromarray(overlay).save(d/'rejection_overlay.png')
    return counts


def write_review(folder,frames,results):
    folder=Path(folder);cards=[];rejected=sum(not r['assessment']['passed'] for r in results)
    def pct(value):return '対象境界なし' if value is None else f'{value:.1%}'
    for frame,row in zip(frames,results):
        check=row['assessment'];policy=check['policy'];bad=[g for g in check['components'] if not g['passed']]
        counts=diagnostic(folder,frame,row) if not check['passed'] and (folder/'mesh_geometry.npz').exists() else {}
        label='形状チェック合格' if check['passed'] else '不合格：'+ '・'.join(name(g['group']) for g in bad)
        reasons=[]
        for group in bad:
            if group['depth_recall'] is not None and group['depth_recall']<policy['depth_recall_min']:
                count=counts.get(group['group']);detail=f"（期待した境界{count['expected']}画素中、対応あり{count['matched']}画素、対応なし{count['missing']}画素）" if count else ''
                reasons.append(f"{name(group['group'])}：深度境界の検出率 {pct(group['depth_recall'])} ＜ 基準 {pct(policy['depth_recall_min'])}{detail}")
            if group['rgb_recall'] is not None and group['rgb_recall']<policy['rgb_recall_min']:
                reasons.append(f"{name(group['group'])}：RGB境界の検出率 {pct(group['rgb_recall'])} ＜ 基準 {pct(policy['rgb_recall_min'])}")
        if not check['global_check']['passed']:reasons.append('画像全体の境界検査も基準未達です。')
        table=''.join(f"<tr><td>{html.escape(name(g['group']))}</td><td>{pct(g['depth_recall'])}</td><td>{pct(g['rgb_recall'])}</td><td>{'合格' if g['passed'] else '不合格'}</td></tr>" for g in check['components'])
        overlay=f'<figure><img src="references/{frame["id"]}/rejection_overlay.png"><figcaption>不合格領域の期待境界：緑＝対応する推定深度境界あり／赤＝対応なし。画像上の赤い面積が欠損率を示すわけではありません。</figcaption></figure>' if counts else ''
        usage='形状チェック合格。ただしRGB枚数上限により、写真の色は未使用。形状のみ使用。' if check['passed'] and not row.get('rgb_teacher',True) else '写真のRGBを学習に採用。' if check['passed'] else 'この写真のRGBは学習から除外。production出力では、このカメラの共通メッシュ深度だけを使用。'
        cards.append(f'<section id="{html.escape(frame["id"])}"><h2>{html.escape(frame["id"])}：{html.escape(label)}</h2><p>{usage}</p>'+''.join(f'<p class="reason">{html.escape(reason)}</p>' for reason in reasons)+f'<div class="images"><figure><img src="references/{frame["id"]}/proxy.png"><figcaption>基準メッシュ</figcaption></figure><figure><img src="{html.escape(row["image"],quote=True)}"><figcaption>生成した写真候補</figcaption></figure>{overlay}</div><details><summary>部位ごとの判定値</summary><table><tr><th>部位</th><th>深度境界</th><th>RGB境界</th><th>結果</th></tr>{table}</table><p>画像全体の深度境界：{pct(check["global_check"]["depth_recall"])}</p></details></section>')
    nav=' / '.join(f'<a href="#{html.escape(f["id"])}">{html.escape(f["id"])}</a>' for f,r in zip(frames,results) if not r['assessment']['passed'])
    (folder/'review.html').write_text('<!doctype html><html lang="ja"><meta charset="utf-8"><title>参照画像の採否と根拠</title><style>body{font:16px/1.6 sans-serif;margin:24px;color:#223}section{border-top:1px solid #aaa;padding:16px 0}.images{display:flex;flex-wrap:wrap;gap:12px}figure{margin:0;width:min(30%,512px);min-width:240px}img{width:100%}figcaption{font-size:14px}table{border-collapse:collapse}td,th{padding:6px 14px;border:1px solid #bbb}.reason{color:#a21;font-weight:bold}</style><h1>参照画像の採否と根拠</h1>'
        +f'<p>{len(results)}視点中、合格{len(results)-rejected}・不合格{rejected}。不合格へ移動：{nav}</p>'
        +'<p>深度境界の検出率は「メッシュで期待した奥行きの境目のうち、写真から推定した深度にも近くの境界がある割合」です。奥行き数値そのものの一致率や、画像全体の品質点ではありません。512画素画像では許容10画素、深度境界55%以上、推定器が概形でも検出できない境界ではRGB境界60%以上を要求します。全体検査と各部位の全条件を満たす必要があります。</p>'
        +'<p>不合格は、画像が実際に破綻していると断定するものではありません。生成画像の変化・深度推定の誤差・基準メッシュの誤りをこの検査だけで切り分けることはできません。特に窓や薄い物体には注意が必要です。合格も写真らしさや家具同一性の保証ではありません。</p>'+''.join(cards)+'</html>',encoding='utf-8')
