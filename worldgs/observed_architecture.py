"""Image-grounded architectural additions to an already placed proxy.

Window/rug bounds come from detections and source-camera ray/plane intersection.
They are coarse hypotheses, never a claim of recovered metric ground truth.
"""
import numpy as np

def ray_plane(uv,K,pose,axis,value):
    rays=np.column_stack([np.asarray(uv,float),np.ones(len(uv))])@np.linalg.inv(K).T@pose[:3,:3].T
    if np.any(np.abs(rays[:,axis])<1e-8):raise ValueError('Plane parallel to source ray')
    t=(value-pose[axis,3])/rays[:,axis]
    if np.any(t<=0):raise ValueError('Plane behind source camera')
    return pose[:3,3]+rays*t[:,None]

def observed_window(box,K,pose,room_size):
    w,h,l=room_size;x0,y0,x1,y1=box
    points=ray_plane([[x0,y0],[x1,y1]],K,pose,2,l/2)
    lo,hi=points.min(0),points.max(0)
    lo[:2]=np.maximum(lo[:2],[-w/2+.15,-h/2+.15]);hi[:2]=np.minimum(hi[:2],[w/2-.15,h/2-.15])
    return dict(x=float((lo[0]+hi[0])/2),width=float(hi[0]-lo[0]),sill=float(h/2-hi[1]),height=float(hi[1]-lo[1]))

def observed_rug(box,K,pose,room_size):
    w,h,l=room_size;x0,y0,x1,y1=box
    points=ray_plane([[x0,y0],[x1,y0],[x0,y1],[x1,y1]],K,pose,1,h/2-.012)
    lo,hi=points.min(0),points.max(0)
    lo=np.maximum(lo,[-w/2+.1,h/2-.012,-l/2+.1]);hi=np.minimum(hi,[w/2-.1,h/2-.004,l/2-.1])
    hi[1]=h/2-.004
    if np.any(hi-lo<=0):raise ValueError('Invalid rug extent')
    return lo,hi
