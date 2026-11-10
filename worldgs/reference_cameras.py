"""Four room-scale benchmark cameras; furniture is never moved."""
import numpy as np
from .geometry import intrinsics
from .mesh_scene import look_at
from .room_cameras import nudge_eye

def benchmark_cameras(shapes,size,fov=70):
    room=shapes[0];lo,hi=np.asarray(room['min'],float),np.asarray(room['max'],float);middle=(lo+hi)/2;span=hi-lo;y=float(np.clip(hi[1]-1.5,lo[1]+.25,hi[1]-.25))
    positions={'front':[middle[0],y,lo[2]+.4],'left':[lo[0]+.4,y,lo[2]+span[2]*.35],'right':[hi[0]-.4,y,lo[2]+span[2]*.35],'rear':[middle[0],y,hi[2]-.4]};rows=[]
    for name,position in positions.items():
        eye=nudge_eye(position,room,shapes[1:]);target=middle.copy();target[1]=min(hi[1]-.25,y+(.5 if name=='rear' else .35));rows.append(dict(id=name,file_path=name+'.png',target='benchmark_'+name,K=intrinsics(size,fov).tolist(),transform_matrix=look_at(eye,target).tolist(),split='view'))
    return rows


def worldmesh_cameras(shapes,size,fov=60,perimeter=16,overhead=8):
    """WorldMesh-inspired rectangular-room plan in our y-down coordinates."""
    from .room_cameras import safe_eye
    room=shapes[0];objects=shapes[1:];lo,hi=np.asarray(room['min'],float),np.asarray(room['max'],float);center=(lo+hi)/2;span=hi-lo
    eye_y=float(np.clip(hi[1]-1.6,lo[1]+.25,hi[1]-.25));high_y=float(np.clip(hi[1]-2.4,lo[1]+.25,eye_y-.1));rows=[]
    def add(name,kind,eye,target):
        eye=nudge_eye(eye,room,objects)
        if not safe_eye(eye,room,objects):raise ValueError('Camera remains inside geometry')
        rows.append(dict(id=name,file_path=name+'.png',target=name,camera_kind=kind,K=intrinsics(size,fov).tolist(),transform_matrix=look_at(eye,target).tolist(),split='view'))
    axis=2 if span[0]<=span[2] else 0
    for i,sign in enumerate([-1,1]):
        eye=center.copy();eye[axis]+=sign*(span[axis]/2-.2);eye[1]=eye_y;target=center.copy();target[1]=eye_y;add(f'bootstrap_{i:02d}','bootstrap',eye,target)
    corners=np.array([[lo[0],lo[2]],[hi[0],lo[2]],[hi[0],hi[2]],[lo[0],hi[2]]]);lengths=np.linalg.norm(np.roll(corners,-1,axis=0)-corners,axis=1);total=lengths.sum()
    for count,kind,height in [(perimeter,'perimeter',eye_y),(overhead,'overhead',high_y)]:
        for j in range(count):
            distance=(j+.5)*total/count;k=0
            while distance>=lengths[k]:distance-=lengths[k];k+=1
            a,b=corners[k],corners[(k+1)%4];point=a+(b-a)*distance/lengths[k];normal=np.array([-(b-a)[1],(b-a)[0]])/lengths[k];point+=normal*.4
            target=center.copy();target[1]=eye_y;eye=np.array([point[0],height,point[1]]);pose=look_at(nudge_eye(eye,room,objects),target)
            if kind=='perimeter' and any(np.array(r['transform_matrix'])[:3,2]@pose[:3,2]>np.cos(np.deg2rad(10)) for r in rows[:2]):continue
            add(f'{kind}_{j:02d}',kind,eye,target)
    # Keep overhead views last; greedily grow from both bootstrap orientations.
    ordered=rows[:2]
    for kind in ['perimeter','overhead']:
        remaining=[r for r in rows if r['camera_kind']==kind]
        while remaining:
            best=max(range(len(remaining)),key=lambda j:max(np.trace(np.array(r['transform_matrix'])[:3,:3].T@np.array(remaining[j]['transform_matrix'])[:3,:3]) for r in ordered))
            ordered.append(remaining.pop(best))
    return ordered


def closest_accepted(frame,accepted):
    if not accepted:return None
    rotation=np.array(frame['transform_matrix'])[:3,:3]
    return max(accepted,key=lambda entry:np.trace(rotation.T@np.array(entry['frame']['transform_matrix'])[:3,:3]))


def dense_camera_plan(shapes,size,count,fov=60,min_object_distance=.65):
    """Translated multi-height room views, retaining the original bootstrap plan."""
    from .room_cameras import safe_eye
    if count<26:raise ValueError('Dense camera plan requires at least 26 cameras')
    room=shapes[0];objects=shapes[1:];lo,hi=np.asarray(room['min'],float),np.asarray(room['max'],float);center=(lo+hi)/2
    rows=worldmesh_cameras(shapes,size,fov);candidates=[]
    for height in [1.15,1.6,2.15]:
        y=hi[1]-height
        if not lo[1]+.3<y<hi[1]-.3:continue
        for scale in [.95,.72,.48]:
            radii=(hi-lo)[[0,2]]/2-.5
            if min(radii)<=0:continue
            for index in range(80):
                angle=2*np.pi*(index+.5)/80
                # Rectangle perimeter keeps the WorldMesh wall-following layout.
                direction=np.array([np.cos(angle),np.sin(angle)]);direction/=np.max(np.abs(direction))
                xz=center[[0,2]]+direction*radii*scale;eye=np.array([xz[0],y,xz[1]])
                if not safe_eye(eye,room,objects):continue
                if any(np.linalg.norm(np.maximum(np.maximum(np.asarray(o['min'])-eye,eye-np.asarray(o['max'])),0))<min_object_distance for o in objects):continue
                for target_height in [.95,1.55]:
                    target=center.copy();target[1]=hi[1]-target_height
                    if np.linalg.norm(target-eye)<.8:continue
                    pose=look_at(eye,target);feature=np.r_[eye[[0,2]]/max((hi-lo)[[0,2]]),height/3,pose[:3,2]*.6]
                    candidates.append((eye,pose,feature,height,scale))
    selected_features=[np.r_[np.array(r['transform_matrix'])[[0,2],3]/max((hi-lo)[[0,2]]),(hi[1]-np.array(r['transform_matrix'])[1,3])/3,np.array(r['transform_matrix'])[:3,2]*.6] for r in rows]
    while len(rows)<count and candidates:
        features=np.array([entry[2] for entry in candidates]);selected=np.array(selected_features)
        distance=((features[:,None,:]-selected[None,:,:])**2).sum(-1).min(1);index=int(distance.argmax());eye,pose,feature,height,scale=candidates.pop(index)
        if any(np.linalg.norm(eye-np.array(r['transform_matrix'])[:3,3])<.12 and pose[:3,2]@np.array(r['transform_matrix'])[:3,2]>.997 for r in rows):continue
        name=f'dense_{len(rows):03d}';rows.append(dict(id=name,file_path=name+'.png',target=name,camera_kind='dense',height_above_floor=height,room_radius_scale=scale,min_object_distance=min_object_distance,K=intrinsics(size,fov).tolist(),transform_matrix=pose.tolist(),split='view'));selected_features.append(feature)
    if len(rows)<count:raise ValueError(f'Only {len(rows)} sufficiently distinct safe cameras fit this room; requested {count}')
    # Preserve the baseline prefix so earlier exact-camera evidence can be reused.
    base=[r for r in rows if r['camera_kind']!='dense'];remaining=[r for r in rows if r['camera_kind']=='dense'];ordered=list(base)
    while remaining:
        rotations=np.array([np.array(r['transform_matrix'])[:3,:3].reshape(-1) for r in remaining]);previous=np.array([np.array(r['transform_matrix'])[:3,:3].reshape(-1) for r in ordered])
        i=int((rotations@previous.T).max(1).argmax())
        ordered.append(remaining.pop(i))
    return ordered
