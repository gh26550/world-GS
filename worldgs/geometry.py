import numpy as np


def intrinsics(size, fov=75):
    f = size / (2 * np.tan(np.deg2rad(fov) / 2))
    return np.array([[f, 0, (size-1)/2], [0, f, (size-1)/2], [0, 0, 1.]])


def camera(yaw):
    """OpenCV camera-to-world: x right, y down, z forward; rotate in place."""
    a = np.deg2rad(yaw)
    pose = np.eye(4)
    pose[:3, :3] = [[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]]
    return pose


def lift(depth, K, pose):
    y, x = np.indices(depth.shape)
    rays = np.stack([x, y, np.ones_like(x)], -1) @ np.linalg.inv(K).T
    return (rays * depth[..., None]) @ pose[:3, :3].T + pose[:3, 3]


def mesh(depth, K, pose, stride=4, edge=0.15):
    """Depth-grid proxy mesh; reject triangles crossing relative depth jumps."""
    pts = lift(depth, K, pose)[::stride, ::stride]
    d = depth[::stride, ::stride]
    h, w = d.shape
    faces = []
    for y in range(h-1):
        for x in range(w-1):
            a = y*w+x
            for tri in ((a, a+1, a+w), (a+1, a+w+1, a+w)):
                z = d.ravel()[list(tri)]
                if np.isfinite(z).all() and z.min() > 0 and np.ptp(z) < edge*z.min():
                    faces.append(tri)
    return pts.reshape(-1, 3), np.asarray(faces, dtype=int).reshape(-1, 3)


def render_mesh(vertices, faces, colors, K, pose, size):
    """CPU triangle rasterizer with perspective-correct RGB and camera-z depth."""
    p = (vertices-pose[:3, 3]) @ pose[:3, :3]
    uvh = p @ K.T
    uv = uvh[:, :2] / np.maximum(p[:, 2:], 1e-8)
    rgb = np.zeros((size, size, 3), np.float32)
    zbuf = np.full((size, size), np.inf)
    for tri in faces:
        z = p[tri, 2]
        if z.min() <= 0.05:
            continue
        t = uv[tri]
        lo = np.maximum(np.floor(t.min(0)).astype(int), 0)
        hi = np.minimum(np.ceil(t.max(0)).astype(int), size-1)
        if (hi < lo).any():
            continue
        yy, xx = np.mgrid[lo[1]:hi[1]+1, lo[0]:hi[0]+1]
        den = np.cross(t[1]-t[0], t[2]-t[0])
        if abs(den) < 1e-8:
            continue
        b = ((xx-t[0, 0])*(t[2, 1]-t[0, 1])-(yy-t[0, 1])*(t[2, 0]-t[0, 0]))/den
        c = ((t[1, 0]-t[0, 0])*(yy-t[0, 1])-(t[1, 1]-t[0, 1])*(xx-t[0, 0]))/den
        weights = np.stack([1-b-c, b, c], -1)
        invz = (weights/z).sum(-1)
        dep = 1/np.maximum(invz, 1e-8)
        patch = zbuf[lo[1]:hi[1]+1, lo[0]:hi[0]+1]
        valid = (weights.min(-1) >= -1e-6) & (dep < patch)
        col = ((weights/z)[..., None]*colors[tri]).sum(-2)*dep[..., None]
        patch[valid] = dep[valid]
        rgb[lo[1]:hi[1]+1, lo[0]:hi[0]+1][valid] = col[valid]
    return np.uint8(np.clip(rgb, 0, 255)), np.where(np.isfinite(zbuf), zbuf, 0)


def align_depth(predicted, reference):
    overlap = (reference > 0) & np.isfinite(predicted) & (predicted > 0)
    if overlap.sum() < 32:
        raise ValueError("Insufficient overlap for depth scale alignment; reduce yaw step or increase FOV")
    scale = np.median(reference[overlap]/predicted[overlap])
    return predicted*scale


def write_obj(path, vertices, faces, colors):
    with open(path, 'w') as f:
        for p, c in zip(vertices, colors/255):
            f.write('v '+' '.join(map(str, [*p, *c]))+'\n')
        for tri in faces+1:
            f.write('f '+' '.join(map(str, tri))+'\n')
