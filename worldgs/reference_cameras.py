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
