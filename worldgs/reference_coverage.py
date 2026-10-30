"""Occlusion-aware split protection on a coarse mesh render (not image synthesis)."""
import numpy as np
from .mesh_scene import MeshRenderer


def scaled_intrinsics(K, original_size, size):
    K=np.asarray(K,float).copy()
    scale=size/original_size
    K[:2,:2]*=scale
    K[:2,2]=(K[:2,2]+.5)*scale-.5
    return K


def observed(points, frames, hits, original_size, size):
    covered=np.zeros(len(points),bool)
    for frame,hit in zip(frames,hits):
        pose=np.asarray(frame['transform_matrix'])
        camera=(points-pose[:3,3])@pose[:3,:3]
        K=scaled_intrinsics(frame['K'],original_size,size)
        uvh=camera@K.T
        uv=np.rint(uvh[:,:2]/np.maximum(uvh[:,2:],1e-8)).astype(int)
        good=(camera[:,2]>.03)&(uv[:,0]>=0)&(uv[:,0]<size)&(uv[:,1]>=0)&(uv[:,1]<size)
        ids=np.flatnonzero(good)
        x,y=uv[ids].T
        depth=hit['depth'][y,x]
        covered[ids]|=(hit['faces'][y,x]>=0)&(np.abs(depth-camera[ids,2])<.04+.02*depth)
    return covered


def protect_training_coverage(mesh,components,owners,frames,original_size,size=64,max_missing=.01):
    renderer=MeshRenderer(mesh,components,owners)
    hits=[renderer.render(scaled_intrinsics(f['K'],original_size,size),np.asarray(f['transform_matrix']),size) for f in frames]
    report={'resolution':size,'max_missing_fraction':max_missing,'promoted':[],
            'criterion':'Heldout surface samples must be observed in training with mesh depth occlusion testing',
            'limitations':'Coarse sampled visibility; not a proof of full continuous surface coverage'}
    for i,frame in enumerate(frames):
        if frame['split']!='heldout':continue
        hit=hits[i];valid=hit['faces']>=0
        y,x=np.where(valid)
        rays=np.column_stack([x,y,np.ones(len(x))])@np.linalg.inv(scaled_intrinsics(frame['K'],original_size,size)).T
        pose=np.asarray(frame['transform_matrix'])
        points=(rays*hit['depth'][y,x,None])@pose[:3,:3].T+pose[:3,3]
        train=[j for j,f in enumerate(frames) if f['split']=='view']
        missing=float((~observed(points,[frames[j] for j in train],[hits[j] for j in train],original_size,size)).mean()) if len(points) else 0.
        if missing>max_missing:
            report['promoted'].append({'target':frame['target'],'missing_fraction':missing})
            frame['split']='view'
    for split in ('view','heldout'):
        for index,frame in enumerate(f for f in frames if f['split']==split):frame['file_path']=f'{split}_{index:03d}.png'
    report['training_views']=sum(f['split']=='view' for f in frames)
    report['heldout_views']=sum(f['split']=='heldout' for f in frames)
    report['passed']=report['heldout_views']>0
    if not report['passed']:raise ValueError('No independent covered heldout views remain; add reference viewpoints')
    return report
