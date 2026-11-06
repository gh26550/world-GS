"""Visibility-aware reference diagnostics and controlled 2x2 GS experiments.

The shared mesh is immutable. No pixels are evaluated against invented RGB truth.
"""
import argparse, copy, hashlib, html, json, os, shutil, time
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter
from .reference_geometry import edges
from .reference_coverage import scaled_intrinsics


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def project_overlap(hit, frame, other, other_depth, size):
    """Target-to-reference mapping; reject hidden surfaces and image-border pixels."""
    xyz=hit['xyz'];pose=np.asarray(other['transform_matrix']);camera=(xyz-pose[:3,3])@pose[:3,:3]
    uv=camera@np.asarray(other['K']).T
    uv=uv[...,:2]/np.maximum(uv[...,2:],1e-8)
    x,y=np.rint(uv).astype(int).transpose(2,0,1)
    inside=(hit['faces']>=0)&(camera[...,2]>.03)&(x>=2)&(x<size-2)&(y>=2)&(y<size-2)
    xx,yy=np.clip(x,0,size-1),np.clip(y,0,size-1)
    z=other_depth[yy,xx]
    inside&=(z>.03)&(np.abs(z-camera[...,2])<.025+.012*z)
    return xx,yy,inside


def camera_quality(hit, owners, components):
    valid=hit['faces']>=0;ids=owners[np.maximum(hit['faces'],0)];size=valid.shape[0]
    border=np.zeros(valid.shape,bool);w=max(2,size//64);border[:w]=True;border[-w:]=True;border[:,:w]=True;border[:,-w:]=True
    rows=[]
    for group in sorted({c['group'] for c in components if c.get('material')=='generated_asset'}):
        own=[i for i,c in enumerate(components) if c.get('group')==group];m=valid&np.isin(ids,own)
        if m.sum()<size*size*.005:continue
        outline=m&~binary_erosion(m);crop=float((outline&border).sum()/max(outline.sum(),1));near=float(np.quantile(hit['depth'][m],.05))
        mask_area=float(m.mean())
        rows.append(dict(group=group,pixels=int(m.sum()),clipped_outline_fraction=crop,depth_p05=near,unsafe=bool(near<.55 or (crop>.5 and near<1.5 and mask_area>.08))))
    return dict(safe=not any(r['unsafe'] for r in rows),objects=rows,score=sum(r['clipped_outline_fraction']+max(0,.65-r['depth_p05']) for r in rows))


def assess_v4(hit, proxy_depth, photo_depth, rgb, owners, components):
    """Abstain on poorly supported/border edges; use RGB for coplanar boundaries.

    Physical depth contrast, not category labels, selects the cue. At least a
    minimum number of edge pixels is required; lack of evidence is not a pass.
    """
    import cv2
    size=len(photo_depth);tol=max(1,round(10*size/512));valid=hit['faces']>=0
    if photo_depth.shape!=valid.shape or not np.isfinite(photo_depth).all() or (photo_depth<=0).any():raise ValueError('Invalid photo depth')
    ids=owners[np.maximum(hit['faces'],0)];interior=binary_erosion(np.ones_like(valid),iterations=tol+2)
    de=edges(hit['depth']);pe=binary_dilation(edges(proxy_depth),iterations=tol);fe=binary_dilation(edges(photo_depth),iterations=tol)
    re=binary_dilation(cv2.Canny(cv2.cvtColor(np.uint8(rgb),cv2.COLOR_RGB2GRAY),25,76)>0,iterations=tol)
    rows=[];minimum=max(8,round(size/32))
    masks={'scene':valid}
    masks.update({g:valid&np.isin(ids,[i for i,c in enumerate(components) if c['group']==g]) for g in sorted({c['group'] for c in components})})
    for group,mask in masks.items():
        if mask.sum()<size*size*.002:continue
        band=(binary_dilation(mask,iterations=2)&~binary_erosion(mask,iterations=2))&interior
        silhouette=mask&~binary_erosion(mask)&interior
        required=(de&valid&interior) if group=='scene' else ((de&band)|silhouette)
        # A silhouette is not necessarily a real depth step (rug/window plane).
        from scipy.ndimage import maximum_filter,minimum_filter
        d=hit['depth'];depth_step=(maximum_filter(d,5)-minimum_filter(np.where(valid,d,np.inf),5))>np.maximum(.07,.025*d)
        observed=required&depth_step&pe
        uncertain=required&~observed
        nd,nr=int(observed.sum()),int(uncertain.sum())
        dr=float(fe[observed].mean()) if nd>=minimum else None
        rr=float(re[uncertain].mean()) if nr>=minimum else None
        # Reject a well-supported missing volumetric edge even if texture exists.
        passed=(dr is None or dr>=.55) and (rr is None or rr>=.6)
        informative=dr is not None or rr is not None
        rows.append(dict(group=group,depth_edges=nd,rgb_edges=nr,depth_recall=dr,rgb_recall=rr,informative=informative,passed=passed,reason='depth_and_rgb' if dr is not None and rr is not None else 'depth' if dr is not None else 'coplanar_or_estimator_unreliable_rgb' if rr is not None else 'insufficient_evidence'))
    checks=[r for r in rows if r['informative']]
    vals=[r[k] for r in checks for k in ['depth_recall','rgb_recall'] if r[k] is not None]
    return dict(passed=bool(checks) and all(r['passed'] for r in checks),score=float(np.mean(vals)) if vals else 0.,groups=rows,policy='v4_physical_depth_contrast_border_abstention',limitations=['RGB edges are not object identity evidence','No evidence is reported explicitly; global informative evidence still required'])


def clone_tree(src,dst):
    """Copy mutable root data; hardlink only read-only asset subdirectories."""
    dst.mkdir(parents=True,exist_ok=False)
    for p in src.iterdir():
        if p.is_file() and p.suffix not in ['.ply'] and not p.name.startswith(('heldout_','gaussians','front_render','training.json','evaluation.json','gs_coverage','status.json')):
            if p.suffix=='.json':shutil.copy2(p,dst/p.name)
            else:shutil.copy2(p,dst/p.name)
    for name in ['assets','meshes','planning','materials','room_images']:
        if (src/name).exists():shutil.copytree(src/name,dst/name,copy_function=os.link)


def run(path,output,steps=3000,max_points=100000,regenerate=16,resume=False):
    from .local_room import load_proxy
    from .mesh_scene import MeshRenderer,look_at
    from .reference_pipeline import boundaries,package_dataset
    from .appearance_geometry import depth_condition
    from .local_runtime import LocalRuntime
    src=Path(path).resolve();out=Path(output).resolve();out.mkdir(parents=True,exist_ok=True)
    if src==out or src in out.parents:raise ValueError('Use a sibling output')
    meta=json.loads((src/'scene.json').read_text());recipe=json.loads((src/'experiment.json').read_text());old=json.loads((src/'assessment.json').read_text())['views'];frames=recipe['cameras'][:len(old)]
    mesh,layout,owners=load_proxy(src);renderer=MeshRenderer(mesh,layout['components'],owners);size=meta['size'];started=time.monotonic()
    manifest=dict(source=str(src),mesh_sha256=digest(src/'scene_mesh.glb'),source_sha256=digest(src/'source_front.png'),seed=meta['seed'],steps=steps,max_points=max_points,cameras=len(frames),regenerate_limit=regenerate,policy='2x2 teacher(original/revised) x trainer(legacy/robust_rgb_only); fixed initial points, steps, seed and unseen evaluation cameras')
    if (out/'manifest.json').exists() and json.loads((out/'manifest.json').read_text())!=manifest:raise ValueError('Experiment settings changed')
    dump(out/'manifest.json',manifest)
    if (out/'evaluation_cameras.json').exists():
        from .controlled_finish import finish
        if not (out/'semantic_review.json').exists():
            from .semantic_revision import audit
            audit(out)
        return finish(out)
    ref=out/'revised_references'
    if not ref.exists():
        clone_tree(src,ref);shutil.copytree(src/'references',ref/'references',copy_function=shutil.copy2)
    results=[];hits={};newframes=[];audit=[]
    # CPU geometry pass at 96px; full-resolution evidence only when needed.
    for i,(f,row) in enumerate(zip(frames,old)):
        d=ref/'references'/f['id'];coarse=renderer.render(scaled_intrinsics(f['K'],size,96),np.array(f['transform_matrix']),96);quality=camera_quality(coarse,owners,layout['components']);nf=copy.deepcopy(f)
        if not quality['safe']:
            # Widen/re-aim toward the same room, without moving any geometry.
            eye=np.array(f['transform_matrix'])[:3,3];room=layout['shapes'][0];center=(np.array(room['min'])+np.array(room['max']))/2
            candidates=[(quality,nf)]
            for mult in [.9,.8]:
                K=np.array(f['K']);K[:2,:2]*=mult
                for target in [None,center+np.array([0,.25,0])]:
                    candidate=copy.deepcopy(f);candidate['K']=K.tolist()
                    if target is not None:candidate['transform_matrix']=look_at(eye,target).tolist()
                    h=renderer.render(scaled_intrinsics(K,size,96),np.array(candidate['transform_matrix']),96);q=camera_quality(h,owners,layout['components']);candidates.append((q,candidate))
            best,nf=min(candidates,key=lambda v:v[0]['score']);changed=nf!=f
            # Only change a camera if a photograph is actually regenerated.
        else:best=quality;changed=False
        pred=d/('predicted_'+Path(row['image']).stem.split('_')[-1]+'.npy')
        h=renderer.render(np.array(f['K']),np.array(f['transform_matrix']),size)
        check=assess_v4(h,np.load(src/'references'/f['id']/'proxy_depth.npy'),np.load(pred),np.asarray(Image.open(src/row['image']).convert('RGB')),owners,layout['components'])
        audit.append(dict(camera=f['id'],before=quality,proposal=best,camera_changed=changed,old_passed=row['assessment']['passed'],new_check=check));newframes.append(nf)
        results.append(dict(row,assessment=check,original_assessment=row['assessment']))
        if i%20==0:dump(out/'status.json',dict(stage='geometry_audit',completed=i,total=len(frames)))
    dump(out/'camera_depth_audit.json',audit)
    # Reproject neighbouring accepted photographs onto a target camera only when
    # their exact mesh depths agree. This also measures low-frequency appearance.
    rgb_ids=[i for i,r in enumerate(old) if r.get('rgb_teacher',r['assessment']['passed']) and r['assessment']['passed']]
    consistency=[];warps={}
    for i,(f,row) in enumerate(zip(frames,old)):
        h=renderer.render(scaled_intrinsics(f['K'],size,128),np.array(f['transform_matrix']),128);target=np.asarray(Image.open(src/row['image']).resize((128,128)),float)/255
        nearby=sorted([j for j in rgb_ids if j!=i],key=lambda j:np.linalg.norm(np.array(frames[j]['transform_matrix'])[:3,3]-np.array(f['transform_matrix'])[:3,3]))[:8]
        best=None
        for j in nearby:
            g=copy.deepcopy(frames[j]);g['K']=scaled_intrinsics(g['K'],size,128).tolist();depth=np.asarray(Image.fromarray(np.load(ref/'references'/g['id']/'mesh_depth.npy')).resize((128,128),Image.Resampling.NEAREST))
            xx,yy,m=project_overlap(h,f,g,depth,128);n=int(m.sum())
            if n<128:continue
            image=np.asarray(Image.open(src/old[j]['image']).resize((128,128)),float)/255;blur=gaussian_filter(image,(2,2,0));err=np.abs(gaussian_filter(target,(2,2,0))[m]-blur[yy[m],xx[m]]).mean()
            item=(n,dict(camera=f['id'],reference=g['id'],reference_index=j,overlap=float(m.mean()),low_frequency_l1=float(err)))
            if best is None or n>best[0]:best=item
        consistency.append(best[1] if best else dict(camera=f['id'],overlap=0.,low_frequency_l1=None))
    dump(out/'teacher_consistency_before.json',consistency)
    ranked=sorted(range(len(frames)),key=lambda i:(not audit[i]['before']['safe'],not results[i]['assessment']['passed'],consistency[i].get('low_frequency_l1') or 0),reverse=True)
    selected=set(ranked[:regenerate]);runtime=LocalRuntime(ref/'logs',image_model='controlnet');generated=0
    with runtime:
        for i in sorted(selected):
            f=newframes[i];d=ref/'references'/f['id'];photo=d/'revised.png';h=renderer.render(np.array(f['K']),np.array(f['transform_matrix']),size)
            guide=h['rgb'].copy();observed=np.zeros((size,size),bool);entry=consistency[i]
            if entry.get('reference_index') is not None:
                j=entry['reference_index'];other=frames[j];depth=np.load(ref/'references'/other['id']/'mesh_depth.npy');xx,yy,observed=project_overlap(h,f,other,depth,size)
                other_rgb=np.asarray(Image.open(src/old[j]['image']).convert('RGB'))
                # Propagate broad appearance only, preserving proxy silhouettes.
                smooth=gaussian_filter(other_rgb.astype(float),(2,2,0));guide[observed]=np.uint8(.65*smooth[yy[observed],xx[observed]]+.35*guide[observed])
            Image.fromarray(guide).save(d/'consistent_guide.png');Image.fromarray(depth_condition(h['depth'])).save(d/'revised_depth.png');Image.fromarray(boundaries(h,owners)).save(d/'revised_boundaries.png');np.save(d/'revised_mesh_depth.npy',h['depth'])
            if not (d/'revised_proxy_predicted.npy').exists():
                from .room_verification import camera_context
                labels={o['name']:o.get('label',o['name']) for o in layout['shapes'][1:]};context,objs=camera_context(h,owners,layout['components'],labels)
                labels=[next((s.get('label','') for s in layout['shapes'][1:] if s['name']==o['id']),o['label']) for o in objs]
                runtime.generate(d/'consistent_guide.png',src/'appearance.png',context,meta['seed']+i*101+900001,photo,size,kind='structure',depth_guide=d/'revised_depth.png',edge_guide=d/'revised_boundaries.png',visible_labels=labels,appearance='warm beige room, brown leather sofa, wooden table, bookcases',generation_options=dict(strength=.48,ip_scale=.15,depth_scale=1.5,boundary_scale=1.2))
                runtime.depth(photo,f['K'][0][0],d/'revised_predicted.npy');runtime.depth(d/'consistent_guide.png',f['K'][0][0],d/'revised_proxy_predicted.npy')
            check=assess_v4(h,np.load(d/'revised_proxy_predicted.npy'),np.load(d/'revised_predicted.npy'),np.asarray(Image.open(photo)),owners,layout['components'])
            q=camera_quality(renderer.render(scaled_intrinsics(f['K'],size,96),np.array(f['transform_matrix']),96),owners,layout['components'])
            # Keep original on an unchanged camera if generation worsens the check.
            use=check['passed'] and q['safe']
            if use:
                results[i]=dict(camera=f['id'],image=str(photo.relative_to(ref)),assessment=check,rgb_teacher=True,evidence={str(photo.relative_to(ref)):digest(photo)},reference_camera=entry.get('reference'))
                for name,source in [('proxy.png','consistent_guide.png'),('depth.png','revised_depth.png'),('boundaries.png','revised_boundaries.png'),('mesh_depth.npy','revised_mesh_depth.npy')]:shutil.copy2(d/source,d/name)
            else:newframes[i]=frames[i]
            audit[i]['regeneration']=dict(passed=check['passed'],safe=q['safe'],used=use,assessment=check);generated+=1
            dump(out/'camera_depth_audit.json',audit);dump(out/'status.json',dict(stage='regenerating',completed=generated,total=len(selected)))
    # Keep the same planned 233 cameras. Unsafe/unverified RGB is excluded rather
    # than silently accepted. A teacher shortfall is reported, never fabricated.
    for i,r in enumerate(results):
        q=audit[i]['proposal'] if audit[i].get('regeneration',{}).get('used') else audit[i]['before']
        r['rgb_teacher']=r['assessment']['passed'] and q['safe']
    accepted=[i for i,r in enumerate(results) if r['rgb_teacher']]
    accepted=accepted[:199]
    for i,r in enumerate(results):r['rgb_teacher']=i in accepted
    dump(ref/'assessment.json',dict(complete=True,views=results,rgb_teacher_count=len(accepted)+1,rgb_target=200,ready=len(accepted)==199))
    dump(ref/'experiment.json',dict(recipe,cameras=newframes,validation='v4',source=str(src)))
    # Reference review is separate from the old edge checker: no stale diagnostic.
    cards=[]
    for i,r in enumerate(results):
        reason='RGB採用' if r['rgb_teacher'] else 'RGB除外：カメラまたは形状の根拠不足'
        cards.append('<section><h3>'+html.escape(r['camera'])+'：'+reason+'</h3><img width="280" src="'+html.escape(str((src/old[i]['image']).as_uri()))+'"><img width="280" src="'+html.escape((ref/r['image']).as_uri())+'"><pre>'+html.escape(json.dumps(audit[i],ensure_ascii=False,indent=2))+'</pre></section>')
    (out/'review.html').write_text('<meta charset="utf-8"><h1>教師画像・カメラ・深度判定の比較</h1><p>左：既存、右：選択画像。根拠のない視点はRGB教師にしません。200枚未達は未達として記録。</p>'+''.join(cards),encoding='utf-8')
    # Package revised geometry while preserving the original shared seed cloud.
    package_dataset(src,ref,meta,mesh,layout,owners,newframes,results,allow_geometry_only=True)
    target=ref/'points.npz';target.unlink();shutil.copy2(src/'points.npz',target)
    # Fixed unseen camera set between all four cells; both data sets use it.
    held=[];room=layout['shapes'][0];center=(np.array(room['min'])+np.array(room['max']))/2
    for i in range(12):
        angle=2*np.pi*(i+.37)/12;eye=center+np.array([1.7*np.cos(angle),-.17 if i%2 else .37,1.7*np.sin(angle)])
        from .room_cameras import safe_eye
        if not safe_eye(eye,room,layout['shapes'][1:]):continue
        f=copy.deepcopy(frames[0]);f.update(target=f'independent_{i:02d}',file_path=f'eval_{i:02d}.png',transform_matrix=look_at(eye,center+np.array([0,.15,0])).tolist(),photographic_ground_truth=False)
        hit=renderer.render(np.array(f['K']),np.array(f['transform_matrix']),size);Image.fromarray(hit['rgb']).save(out/f['file_path'])
        for key,array in [('depth',hit['depth']),('mask',hit['faces']>=0),('faces',hit['faces'])]:
            name=f'eval_{i:02d}_{key}.npy';np.save(out/name,array);f[{'faces':'face'}.get(key,key)+'_file']=name
        held.append(f)
    dump(out/'evaluation_cameras.json',held)
    from .semantic_revision import audit
    audit(out)
    from .controlled_finish import finish
    return finish(out)


def main():
    p=argparse.ArgumentParser();p.add_argument('path');p.add_argument('--output',required=True);p.add_argument('--steps',type=int,default=3000);p.add_argument('--max-points',type=int,default=100000);p.add_argument('--regenerate',type=int,default=16);p.add_argument('--resume',action='store_true');a=p.parse_args();run(**vars(a))

if __name__=='__main__':main()
