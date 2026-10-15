"""A shared room/object scaffold with analytic visibility and surface textures."""
import numpy as np
from scipy.ndimage import distance_transform_edt
from .geometry import camera


def pose_at(yaw, pitch=0, position=(0, 0, 0)):
    pose = camera(yaw)
    a = np.deg2rad(pitch)
    pose[:3, :3] = pose[:3, :3] @ np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    pose[:3, 3] = position
    return pose


def trajectory(views, baseline=.18):
    result = []
    for pitch, count in [(0, views), (60, (views+1)//2), (-60, (views+1)//2)]:
        for i in range(count):
            yaw = i*360/count
            a = np.deg2rad(yaw)
            pos = [baseline*np.sin(a), 0, baseline*(np.cos(a)-1)]
            result.append((yaw, pitch, pose_at(yaw, pitch, pos)))
    return result


def make_layout(size, detections, relative_depth, K):
    """Fit conservative axis-aligned furniture proxies; scale remains a prior."""
    width, height, length = map(float, size)
    if min(width, height, length) <= 0:
        raise ValueError('Room dimensions must be positive')
    layout = [{'name': 'room', 'kind': 'room', 'min': [-width/2, -height/2, -length/2], 'max': [width/2, height/2, length/2]}]
    n = len(relative_depth)
    scale = (length/2)/np.quantile(relative_depth, .8)
    for index, detection in enumerate(detections):
        x0, y0, x1, y1 = np.clip(detection['box'], 0, n-1)
        if x1-x0 < n*.06 or y1-y0 < n*.06:
            continue
        crop = relative_depth[int(y0):int(y1)+1, int(x0):int(x1)+1]
        label = detection['label'].lower()
        thickness = .3 if ('book' in label or 'shelf' in label) else (.7 if 'sofa' in label or 'couch' in label else .6)
        bottom_ray = (y1-K[1, 2])/K[1, 1]
        # The image's floor contact is a shared scale cue, unlike per-image depth normalization.
        estimated = (height/2)/bottom_ray if bottom_ray > .12 else float(np.quantile(crop, .35)*scale)
        if 'sofa' in label or 'couch' in label:
            estimated += thickness*.35
        elif 'table' in label:
            estimated *= .85
        z = float(np.clip(estimated, .8, length/2-.4))
        dx = float(np.clip((x1-x0)*z/K[0, 0], .3, width*.8))
        top = (y0-K[1, 2])*z/K[1, 1]
        dy = float(np.clip(height/2-top, .3, height*.85))
        cx = float(((x0+x1)/2-K[0, 2])*z/K[0, 0])
        lo = np.array([cx-dx/2, height/2-dy, z])
        hi = np.array([cx+dx/2, height/2, z+thickness])
        lo = np.maximum(lo, np.array(layout[0]['min'])+.05)
        hi = np.minimum(hi, np.array(layout[0]['max'])-.05)
        hi[1] = height/2
        # Ensure no detected proxy encloses the observer corridor.
        if (hi-lo < .15).any() or lo[2] < .5:
            continue
        layout.append({'name': f'object_{index:02d}', 'kind': 'box', 'label': label, 'min': lo.tolist(), 'max': hi.tolist(), 'detection': detection})
    for table in [s for s in layout if 'table' in s.get('label', '')]:
        for sofa in [s for s in layout if 'sofa' in s.get('label', '') or 'couch' in s.get('label', '')]:
            if min(table['max'][0], sofa['max'][0]) > max(table['min'][0], sofa['min'][0]):
                back = min(table['max'][2], sofa['min'][2]-.05)
                if back-table['min'][2] >= .15:
                    table['max'][2] = back
    return layout


def trace(layout, K, pose, size):
    y, x = np.indices((size, size))
    rays = np.stack([x, y, np.ones_like(x)], -1) @ np.linalg.inv(K).T @ pose[:3, :3].T
    origin = pose[:3, 3]
    depth = np.full((size, size), np.inf)
    surfaces = np.full((size, size), -1, int)
    for index, shape in enumerate(layout):
        lo, hi = np.array(shape['min']), np.array(shape['max'])
        inside = bool(((origin > lo) & (origin < hi)).all())
        if index and inside:
            raise ValueError(f'Camera is inside {shape["name"]}; adjust layout or baseline')
        with np.errstate(divide='ignore', invalid='ignore'):
            t0 = (lo-origin)/rays
            t1 = (hi-origin)/rays
        near, far = np.minimum(t0, t1).max(-1), np.maximum(t0, t1).min(-1)
        t = far if inside else near
        valid = (far >= near) & (t > .05) & (t < depth)
        points = origin + rays*np.where(np.isfinite(t), t, 0)[..., None]
        distances = np.concatenate([np.abs(points-lo), np.abs(points-hi)], -1)
        face = distances.argmin(-1)
        depth[valid] = t[valid]
        surfaces[valid] = index*6+face[valid]
    if not np.isfinite(depth).all() or (surfaces < 0).any():
        raise ValueError('Scaffold must enclose all camera rays')
    return depth, origin + rays*depth[..., None], surfaces


def surface_coordinates(layout, points, surfaces, resolution):
    uv = np.zeros((*surfaces.shape, 2), int)
    for sid in np.unique(surfaces):
        index, face = divmod(int(sid), 6)
        axes = [i for i in range(3) if i != face % 3]
        lo, hi = np.array(layout[index]['min']), np.array(layout[index]['max'])
        mask = surfaces == sid
        uv[mask] = np.clip((points[mask][:, axes]-lo[axes])/(hi-lo)[axes]*resolution, 0, resolution-1).astype(int)
    return uv


class Atlas:
    def __init__(self, layout, resolution=192):
        self.layout, self.resolution = layout, resolution
        self.sums = np.zeros((len(layout)*6, resolution, resolution, 3), np.float32)
        self.counts = np.zeros((len(layout)*6, resolution, resolution), np.float32)

    def initialize_materials(self, image):
        """Explicit appearance priors prevent floor/ceiling hallucinations in unseen areas."""
        from PIL import Image
        rgb = np.asarray(image)
        h, w = rgb.shape[:2]
        ceiling = np.median(rgb[int(h*.02):max(int(h*.1), 1), int(w*.3):int(w*.7)], axis=(0, 1))
        floor_patch = image.crop((int(w*.01), int(h*.80), int(w*.12), int(h*.96)))
        n = self.resolution
        tile = np.asarray(floor_patch.resize((max(2, n//6), max(2, n//6)), Image.Resampling.BILINEAR))
        seamless = np.concatenate([np.concatenate([tile, tile[:, ::-1]], 1),
                                   np.concatenate([tile[::-1], tile[::-1, ::-1]], 1)], 0)
        floor = np.tile(seamless, (n//len(seamless)+1, n//len(seamless)+1, 1))[:n, :n]
        # Small weights keep genuinely projected front-image colors dominant.
        self.sums[1] = ceiling*.001
        self.sums[4] = floor*.001
        self.counts[[1, 4]] = .001
        return {'ceiling': 'median color of front image upper-center patch',
                'floor': 'mirrored repeating material from front image lower-left patch',
                'note': 'Appearance priors, not observations of unseen floor or ceiling'}

    def render(self, points, surfaces):
        uv = surface_coordinates(self.layout, points, surfaces, self.resolution)
        colors = np.full((*surfaces.shape, 3), 128., np.float32)
        known = np.zeros(surfaces.shape, bool)
        for sid in np.unique(surfaces):
            mask = surfaces == sid
            valid = self.counts[sid] > 0
            if not valid.any():
                continue
            distance, nearest = distance_transform_edt(~valid, return_indices=True)
            texture = self.sums[sid]/np.maximum(self.counts[sid, ..., None], 1e-8)
            filled = texture[nearest[0], nearest[1]]
            u, v = uv[mask].T
            colors[mask] = filled[u, v]
            known[mask] = distance[u, v] <= 1.5
        return np.uint8(np.clip(colors, 0, 255)), known

    def update(self, points, surfaces, image):
        uv = surface_coordinates(self.layout, points, surfaces, self.resolution)
        s, u, v = surfaces.ravel(), uv[..., 0].ravel(), uv[..., 1].ravel()
        np.add.at(self.sums, (s, u, v), np.asarray(image).reshape(-1, 3))
        np.add.at(self.counts, (s, u, v), 1)

    def sample(self, spacing=.04):
        vertices, normals, ids, minima, maxima = [], [], [], [], []
        for index, shape in enumerate(self.layout):
            lo, hi = np.array(shape['min']), np.array(shape['max'])
            for face in range(6):
                axis = face % 3
                axes = [i for i in range(3) if i != axis]
                lengths = hi-lo
                grids = [lo[a]+(np.arange(max(2, int(np.ceil(lengths[a]/spacing))))+.5)*lengths[a]/max(2, int(np.ceil(lengths[a]/spacing))) for a in axes]
                a, b = np.meshgrid(*grids)
                p = np.zeros((a.size, 3))
                p[:, axes[0]], p[:, axes[1]] = a.ravel(), b.ravel()
                p[:, axis] = lo[axis] if face < 3 else hi[axis]
                normal = np.zeros_like(p)
                normal[:, axis] = (-1 if face < 3 else 1)*(-1 if index == 0 else 1)
                vertices.append(p)
                normals.append(normal)
                ids.append(np.full(len(p), index*6+face))
                minima.append(np.tile(lo, (len(p), 1)))
                maxima.append(np.tile(hi, (len(p), 1)))
        xyz, normal, surfaces = np.concatenate(vertices), np.concatenate(normals), np.concatenate(ids)
        rgb, known = self.render(xyz, surfaces)
        return dict(xyz=xyz.astype('float32'), rgb=rgb, normals=normal.astype('float32'), surface_ids=surfaces,
                    bounds_min=np.concatenate(minima).astype('float32'), bounds_max=np.concatenate(maxima).astype('float32'), color_available=known)


def export_layout(layout, path):
    """Closed boxes with object names; vertex colors are neutral placeholders."""
    from .geometry import write_obj
    vertices, triangles = [], []
    faces = np.array([[0, 2, 3], [0, 3, 1], [4, 5, 7], [4, 7, 6], [0, 1, 5], [0, 5, 4],
                      [2, 6, 7], [2, 7, 3], [0, 4, 6], [0, 6, 2], [1, 3, 7], [1, 7, 5]])
    for shape in layout:
        lo, hi = np.array(shape['min']), np.array(shape['max'])
        v = np.array([[x, y, z] for x in [lo[0], hi[0]] for y in [lo[1], hi[1]] for z in [lo[2], hi[2]]])
        current = faces if shape['kind'] == 'room' else faces[:, ::-1]
        triangles.append(current+len(vertices)*8)
        vertices.append(v)
    v, f = np.concatenate(vertices), np.concatenate(triangles)
    write_obj(path, v, f, np.full_like(v, 160))
