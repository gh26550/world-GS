"""Room-scale cameras; OpenCV y-down world, no per-object image generation."""
import numpy as np
from .mesh_scene import look_at
from .geometry import intrinsics

def safe_eye(eye,room,objects,margin=.12):
    eye=np.asarray(eye);lo,hi=np.asarray(room['min']),np.asarray(room['max'])
    return bool(np.all(eye>lo+margin) and np.all(eye<hi-margin) and not any(np.all(eye>np.asarray(o['min'])-margin) and np.all(eye<np.asarray(o['max'])+margin) for o in objects))

def nudge_eye(eye,room,objects):
    lo,hi=np.asarray(room['min']),np.asarray(room['max']);eye=np.clip(eye,lo+.18,hi-.18)
    if safe_eye(eye,room,objects):return eye
    candidates=[np.array([x,y,z]) for x in np.linspace(lo[0]+.2,hi[0]-.2,15) for z in np.linspace(lo[2]+.2,hi[2]-.2,15) for y in [eye[1],np.clip(eye[1]-.3,lo[1]+.2,hi[1]-.2)]]
    valid=[p for p in candidates if safe_eye(p,room,objects)]
    if not valid:raise ValueError('No collision-free camera position')
    return min(valid,key=lambda p:np.linalg.norm(p-eye))

def room_camera_plan(shapes,size=512,perimeter=16,overhead=8,fov=75):
    if perimeter<8 or overhead<4 or not 40<=fov<=100:raise ValueError('Use >=8 perimeter, >=4 overhead, FOV 40..100')
    room=shapes[0];objects=shapes[1:];lo,hi=np.asarray(room['min']),np.asarray(room['max']);center=(lo+hi)/2;span=hi-lo
    eye_y=float(np.clip(hi[1]-1.5,lo[1]+.25,hi[1]-.35));high_y=float(np.clip(hi[1]-2.4,lo[1]+.3,hi[1]-.5))
    K=intrinsics(size,fov).tolist();rows=[]
    def add(kind,eye,target):
        eye=nudge_eye(eye,room,objects);pose=look_at(eye,target)
        if any(np.linalg.norm(eye-np.array(r['transform_matrix'])[:3,3])<.12 and (np.array(r['transform_matrix'])[:3,2]@pose[:3,2])>.995 for r in rows):return
        i=len(rows);rows.append(dict(file_path=f'room_{i:03d}.png',target=f'{kind}_{i:03d}',camera_kind=kind,K=K,transform_matrix=pose.tolist(),split='view'))
    # Opposite shorter-wall midpoints anchor all four walls.
    axis=2 if span[0]<=span[2] else 0
    for sign in [-1,1]:
        eye=center.copy();eye[axis]+=sign*(span[axis]/2-.25);eye[1]=eye_y
        target=center.copy();target[1]=eye_y+.15;add('bootstrap',eye,target)
    length=2*(span[0]+span[2]);corners=[(lo[0]+.25,lo[2]+.25),(hi[0]-.25,lo[2]+.25),(hi[0]-.25,hi[2]-.25),(lo[0]+.25,hi[2]-.25)]
    edges=[np.linalg.norm(np.array(corners[(i+1)%4])-corners[i]) for i in range(4)];total=sum(edges)
    for count,kind,y in [(perimeter,'perimeter',eye_y),(overhead,'overhead',high_y)]:
        for j in range(count):
            d=(j+.25)*total/count;k=0
            while d>=edges[k]:d-=edges[k];k+=1
            point=np.array(corners[k])+(np.array(corners[(k+1)%4])-corners[k])*d/edges[k]
            target=center.copy();target[1]=min(hi[1]-.2,eye_y+(.65 if kind=='overhead' else .15))
            add(kind,[point[0],y,point[1]],target)
    return rows

def adaptive_candidates(shapes,size=512,fov=75):
    room=shapes[0];lo,hi=np.asarray(room['min']),np.asarray(room['max']);center=(lo+hi)/2;span=hi-lo;rows=[]
    for x in np.linspace(lo[0]+.6,hi[0]-.6,3):
        for z in np.linspace(lo[2]+.6,hi[2]-.6,3):
            for y in [hi[1]-1.2,max(lo[1]+.3,hi[1]-2.3)]:
                eye=np.array([x,y,z])
                if not safe_eye(eye,room,shapes[1:]):continue
                for target in [center+span*np.array([sx,.18,sz]) for sx,sz in [(-.3,-.3),(-.3,.3),(.3,-.3),(.3,.3)]]:
                    if np.linalg.norm(target-eye)<.4:continue
                    i=len(rows);rows.append(dict(file_path=f'adaptive_{i:03d}.png',target=f'adaptive_{i:03d}',camera_kind='adaptive',K=intrinsics(size,fov).tolist(),transform_matrix=look_at(eye,target).tolist(),split='view'))
    return rows

def visible_quality(renderer,frame,size):
    from .reference_coverage import scaled_intrinsics
    K=scaled_intrinsics(frame['K'],size,96);hit=renderer.render(K,np.array(frame['transform_matrix']),96);valid=hit['faces']>=0
    q=np.zeros(len(renderer.mesh.faces),np.float32)
    if valid.any():
        rays=hit['xyz'][valid]-np.array(frame['transform_matrix'])[:3,3];rays/=np.maximum(np.linalg.norm(rays,axis=1,keepdims=True),1e-6)
        incidence=np.abs((hit['normals'][valid]*rays).sum(1));np.maximum.at(q,hit['faces'][valid],incidence)
    return q

def coverage_weights(mesh,owners,components):
    areas=mesh.area_faces;weights=np.zeros_like(areas);eligible=np.ones(len(areas),bool)
    for owner,c in enumerate(components):
        mask=owners==owner
        if c.get('group')=='room':
            _,axis,side=c['name'].split('_');axis,side=int(axis),int(side);plane=np.array(c['bounds'])[1-side,axis]
            eligible[mask]=np.max(np.abs(mesh.triangles[mask,:,axis]-plane),axis=1)<1e-5
        selected=mask&eligible
        if selected.any():weights[selected]=areas[selected]/max(areas[selected].sum(),1e-9)
    return weights,eligible

def augment_coverage(renderer,frames,shapes,size,max_extra=8):
    quality=np.zeros(len(renderer.mesh.faces));areas=renderer.mesh.area_faces;weights,eligible=coverage_weights(renderer.mesh,renderer.owners,renderer.components)
    for f in frames:quality=np.maximum(quality,visible_quality(renderer,f,size))
    candidates=adaptive_candidates(shapes,size);scored=[(f,visible_quality(renderer,f,size)) for f in candidates];additions=[]
    for _ in range(max_extra):
        if not scored:break
        gains=[float((np.maximum(q-quality,0)*weights).sum()) for _,q in scored];i=int(np.argmax(gains))
        if gains[i]<.005:break
        f,q=scored.pop(i);f['coverage_gain']=gains[i];additions.append(f);quality=np.maximum(quality,q)
    return frames+additions,dict(adaptive_added=len(additions),face_quality_mean=float((quality*weights).sum()/max(weights.sum(),1e-9)),unseen_area_fraction=float(areas[(quality==0)&eligible].sum()/max(areas[eligible].sum(),1e-9)),limitation='Interior architectural faces and all furniture faces; ray visibility/incidence only; not texel-complete coverage or unknown-geometry recovery')

def rotational_order(frames):
    remaining=list(frames);ordered=[]
    while remaining:
        if not ordered or remaining[0]['camera_kind']=='bootstrap':i=0
        else:
            axes=[np.array(f['transform_matrix'])[:3,:3] for f in ordered]
            i=max(range(len(remaining)),key=lambda j:max((np.trace(a.T@np.array(remaining[j]['transform_matrix'])[:3,:3])-1)/2 for a in axes))
        ordered.append(remaining.pop(i))
    return ordered
