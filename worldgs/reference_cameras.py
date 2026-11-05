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
