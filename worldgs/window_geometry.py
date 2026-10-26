"""Explicit back-wall aperture and opaque outdoor geometry for baked GS scenes."""
import numpy as np
import trimesh


def window_bounds(room):
    window=room.get('window')
    if not window:
        return None
    w,h,l=np.asarray(room['size'],float)
    values=np.array([window[k] for k in ('x','width','sill','height')],float)
    x,width,sill,height=values
    if not np.isfinite(values).all() or width<.5 or height<.5 or sill<.15 or sill+height>h-.12 or abs(x)+width/2>w/2-.12:
        raise ValueError('Invalid back-wall window aperture')
    return np.array([x-width/2,h/2-sill-height,l/2]),np.array([x+width/2,h/2-sill,l/2])


def add_window(room,parts,components):
    bounds=window_bounds(room)
    if bounds is None:
        return
    lo,hi=bounds
    w,h,l=room['size']
    def box(a,b,color):
        mesh=trimesh.creation.box(extents=np.asarray(b)-a)
        mesh.apply_translation((np.asarray(a)+b)/2)
        mesh.visual.face_colors=color
        return mesh
    wall=[]
    for a,b in [([-w/2,-h/2,l/2],[lo[0],h/2,l/2+.08]),
                ([hi[0],-h/2,l/2],[w/2,h/2,l/2+.08]),
                ([lo[0],-h/2,l/2],[hi[0],lo[1],l/2+.08]),
                ([lo[0],hi[1],l/2],[hi[0],h/2,l/2+.08])]:
        wall.append(box(a,b,[240,236,223,255]))
    parts[5]=trimesh.util.concatenate(wall)
    components[5].update(faces=len(parts[5].faces),bounds=parts[5].bounds.tolist(),opening=dict(min=lo.tolist(),max=hi.tolist()))
    frames=[]
    for x in (lo[0],(lo[0]+hi[0])/2,hi[0]):
        frames.append(box([x-.035,lo[1]-.04,l/2-.06],[x+.035,hi[1]+.04,l/2+.06],[245,242,231,255]))
    for y in (lo[1],(lo[1]+hi[1])/2,hi[1]):
        frames.append(box([lo[0]-.06,y-.035,l/2-.06],[hi[0]+.06,y+.035,l/2+.06],[245,242,231,255]))
    panel=box([lo[0]-.15,lo[1]-.15,l/2+.48],[hi[0]+.15,hi[1]+.15,l/2+.5],[215,229,234,255])
    rng=np.random.default_rng(71)
    branches=[]
    for _ in range(30):
        x=rng.uniform(lo[0],hi[0])
        y=rng.uniform(lo[1]-.15,hi[1]+.15)
        end=np.array([x+rng.uniform(-.35,.35),y-rng.uniform(.25,.7),l/2+.26])
        start=np.array([x,y,l/2+.25])
        transform=trimesh.geometry.align_vectors([0,0,1],end-start)
        branch=trimesh.creation.cylinder(radius=rng.uniform(.006,.014),height=np.linalg.norm(end-start),sections=6)
        branch.apply_transform(transform)
        branch.apply_translation((start+end)/2)
        branch.visual.face_colors=[89,80,65,255]
        branches.append(branch)
    for name,mesh,material in [('window_frame',trimesh.util.concatenate(frames),'window_frame'),
                               ('window_outdoor',panel,'window_sky'),
                               ('window_branches',trimesh.util.concatenate(branches),'window_branches')]:
        parts.append(mesh)
        components.append(dict(name=name,group='window',material=material,faces=len(mesh.faces),bounds=mesh.bounds.tolist()))
