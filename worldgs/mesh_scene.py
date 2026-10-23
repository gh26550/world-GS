"""Volumetric furniture component meshes and visibility-correct triangle references."""
import numpy as np
import trimesh
from .geometry import intrinsics


def look_at(eye, target):
    eye, target = np.array(eye, float), np.array(target, float)
    forward = target-eye
    forward /= np.linalg.norm(forward)
    right = np.cross([0., 1., 0.], forward)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1., 0., 0.])
    right /= np.linalg.norm(right)
    pose = np.eye(4)
    pose[:3, :3] = np.stack([right, np.cross(forward, right), forward], 1)
    pose[:3, 3] = eye
    return pose


def rounded_box(extents, center, radius=.04):
    m = trimesh.creation.box(extents=extents)
    for _ in range(3):
        m = m.subdivide()
    half = np.asarray(extents)/2
    r = min(radius, float(half.min())*.8)
    closest = np.clip(m.vertices, -half+r, half-r)
    delta = m.vertices-closest
    m.vertices = closest+delta/np.maximum(np.linalg.norm(delta, axis=1, keepdims=True), 1e-8)*r
    m.apply_translation(center)
    return m


def build_components(layout, image):
    """Image-conditioned procedural proxies; not learned object reconstruction."""
    geometries, components = [], []
    rgb = np.asarray(image)
    rng = np.random.default_rng(42)
    room = layout[0]
    floor = room['max'][1]
    def add(mesh, name, group, color, material='plain'):
        color = np.uint8(np.clip(color, 0, 255))
        mesh.visual.face_colors = np.tile(np.r_[color, 255], (len(mesh.faces), 1))
        geometries.append(mesh)
        components.append({'name': name, 'group': group, 'material': material, 'faces': len(mesh.faces), 'bounds': mesh.bounds.tolist()})
    def box(lo, hi, name, group, color, rounded=False, material='plain'):
        lo, hi = np.array(lo), np.array(hi)
        if (hi-lo <= .005).any():
            return
        m = rounded_box(hi-lo, (lo+hi)/2) if rounded else trimesh.creation.box(extents=hi-lo)
        if not rounded:
            m.apply_translation((lo+hi)/2)
        add(m, name, group, color, material)
    lo, hi = np.array(room['min']), np.array(room['max'])
    wall = np.median(rgb[:max(1, len(rgb)//8)].reshape(-1, 3), 0)
    for axis in range(3):
        for side in [0, 1]:
            a, b = lo.copy(), hi.copy()
            a[axis], b[axis] = (lo[axis]-.08, lo[axis]) if side == 0 else (hi[axis], hi[axis]+.08)
            mat = 'wood' if axis == 1 and side == 1 else 'plain'
            color = [151, 104, 65] if mat == 'wood' else wall
            box(a, b, f'room_{axis}_{side}', 'room', color, material=mat)
    for shape in layout[1:]:
        a, b = np.array(shape.get('local_min', shape['min'])), np.array(shape.get('local_max', shape['max']))
        width, height, depth = b-a
        group = shape['name']
        label = shape.get('label', '')
        if 'sofa' in label or 'couch' in label:
            box_pixels = shape.get('detection', {}).get('box', [0, 0, len(rgb)-1, len(rgb)-1])
            x0, y0, x1, y1 = np.clip(box_pixels, 0, len(rgb)-1).astype(int)
            patch = rgb[y0:y1+1, x0:x1+1]
            pixels = patch.reshape(-1, 3).astype(float)
            pixels = pixels[(pixels.mean(1) > 25) & (pixels.mean(1) < 180)]
            cloth = np.median(pixels, 0) if len(pixels) else np.array([112, 75, 56])
            seat_y = floor-height*.43
            box([a[0], seat_y, a[2]], [b[0], floor-.12, b[2]], 'base', group, cloth*.8, True, 'fabric')
            box([a[0], a[1], b[2]-.18], [b[0], seat_y+.1, b[2]], 'back', group, cloth, True, 'fabric')
            arm = min(.18, width*.12)
            for x in [a[0], b[0]-arm]:
                box([x, seat_y-.27, a[2]], [x+arm, floor-.12, b[2]], 'arm', group, cloth*.9, True, 'fabric')
            count = max(2, int(width/.7))
            span = (width-2*arm)/count
            for j in range(count):
                x = a[0]+arm+j*span
                box([x+.015, seat_y-.1, a[2]+.03], [x+span-.015, seat_y+.03, b[2]-.18], f'seat_{j}', group, cloth*1.08, True, 'fabric')
                box([x+.03, a[1]+.05, b[2]-.31], [x+span-.03, seat_y-.02, b[2]-.14], f'cushion_{j}', group, cloth*1.03, True, 'fabric')
            for x in [a[0]+.12, b[0]-.12]:
                for z in [a[2]+.12, b[2]-.12]:
                    box([x-.03, floor-.16, z-.03], [x+.03, floor, z+.03], 'foot', group, [55, 43, 34])
        elif 'table' in label:
            center = (a+b)/2
            # Elliptical, closed tabletop and separated legs; clear views underneath.
            top = trimesh.creation.cylinder(radius=1., height=.07, sections=48)
            top.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2, [1, 0, 0]))
            top.apply_scale([width/2, 1, depth/2])
            top.apply_translation([center[0], a[1]+.035, center[2]])
            add(top, 'top', group, [109, 74, 45], 'wood')
            for x in [a[0]+width*.2, b[0]-width*.2]:
                for z in [a[2]+depth*.2, b[2]-depth*.2]:
                    leg = trimesh.creation.cylinder(radius=.025, height=height-.07, sections=16)
                    leg.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2, [1, 0, 0]))
                    leg.apply_translation([x, (a[1]+.07+floor)/2, z])
                    add(leg, 'leg', group, [45, 38, 33])
        elif 'book' in label or 'shelf' in label:
            color = np.array([208, 196, 174])
            for x in [a[0], b[0]-.045]:
                box([x, a[1], a[2]], [x+.045, b[1], b[2]], 'upright', group, color)
            box([a[0], a[1], b[2]-.025], b, 'backboard', group, color*.85)
            levels = max(3, int(height/.36))
            for j in range(levels+1):
                y = a[1]+j*height/levels
                box([a[0], y-.02, a[2]], [b[0], y+.02, b[2]], f'shelf_{j}', group, color)
                if j == 0:
                    continue
                x = a[0]+.06
                k = 0
                while x < b[0]-.1:
                    dx = rng.uniform(.035, .07)
                    dy = rng.uniform(height/levels*.55, height/levels*.85)
                    colorbook = rng.choice([[68, 64, 58], [130, 76, 50], [185, 170, 134], [49, 77, 78], [101, 115, 70]])
                    box([x, y-dy, a[2]+.035], [min(x+dx, b[0]-.05), y-.025, b[2]-.04], f'book_{j}_{k}', group, colorbook)
                    x += dx+.012
                    k += 1
        else:
            box(a, b, label, group, [120, 110, 100], rounded=True)
    for shape in layout[1:]:
        if not shape.get('yaw'):
            continue
        center = (np.asarray(shape['local_min'])+np.asarray(shape['local_max']))/2
        transform = trimesh.transformations.rotation_matrix(np.deg2rad(shape['yaw']), [0,1,0], point=center)
        for geometry, part in zip(geometries, components):
            if part['group'] == shape['name']:
                geometry.apply_transform(transform)
                part['bounds'] = geometry.bounds.tolist()
    combined = trimesh.util.concatenate(geometries)
    owners = np.concatenate([np.full(len(m.faces), i, np.int32) for i, m in enumerate(geometries)])
    return combined, geometries, components, owners


class MeshRenderer:
    def __init__(self, mesh, components, owners):
        from trimesh.ray.ray_pyembree import RayMeshIntersector
        self.mesh, self.components, self.owners = mesh, components, owners
        self.rays = RayMeshIntersector(mesh)

    def shade(self, points, triangles):
        normals = self.mesh.face_normals[triangles]
        color = self.mesh.visual.face_colors[triangles, :3].astype(float)/255
        light = np.array([-.5, -.8, -.3])
        light /= np.linalg.norm(light)
        lighting = .65+.35*np.abs(normals @ light)
        ids = self.owners[triangles]
        for i in np.unique(ids):
            mask = ids == i
            material = self.components[int(i)]['material']
            p = points[mask]
            if material == 'wood':
                grain = .93+.04*np.sin(p[:, 0]*160+2*np.sin(p[:, 2]*5))+.03*np.sin(p[:, 2]*16)
                color[mask] *= grain[:, None]
            elif material == 'fabric':
                color[mask] *= (.98+.02*np.sin(p[:, 0]*700)*np.sin(p[:, 1]*700))[:, None]
        return np.clip(color*lighting[:, None], 0, 1), normals

    def render(self, K, pose, size):
        y, x = np.indices((size, size))
        directions = (np.stack([x, y, np.ones_like(x)], -1) @ np.linalg.inv(K).T @ pose[:3, :3].T).reshape(-1, 3)
        origin = np.tile(pose[:3, 3], (len(directions), 1))
        locations, ray_ids, triangles = self.rays.intersects_location(origin, directions, multiple_hits=False)
        camera_points = (locations-pose[:3, 3]) @ pose[:3, :3]
        keep = camera_points[:, 2] > .03
        locations, ray_ids, triangles = locations[keep], ray_ids[keep], triangles[keep]
        colors, normals = self.shade(locations, triangles)
        rgb = np.full((size*size, 3), .8)
        depth = np.zeros(size*size, np.float32)
        xyz, normal = np.zeros((size*size, 3)), np.zeros((size*size, 3))
        face = np.full(size*size, -1, np.int32)
        rgb[ray_ids], depth[ray_ids] = colors, camera_points[keep, 2]
        xyz[ray_ids], normal[ray_ids], face[ray_ids] = locations, normals, triangles
        return dict(rgb=np.uint8(rgb.reshape(size, size, 3)*255), depth=depth.reshape(size, size),
                    xyz=xyz.reshape(size, size, 3), normals=normal.reshape(size, size, 3), faces=face.reshape(size, size))


def camera_set(layout, views=12):
    room = layout[0]
    lo, hi = np.array(room['min']), np.array(room['max'])
    result = []
    def safe(eye):
        if not ((eye > lo+.12) & (eye < hi-.12)).all():
            return False
        return not any(s['min'][0]-.12 < eye[0] < s['max'][0]+.12 and s['min'][2]-.12 < eye[2] < s['max'][2]+.12 for s in layout[1:])
    # Room-wide translation instead of a 36cm central circle.
    for i in range(views):
        a = 2*np.pi*i/views
        eye = np.array([1.4*np.sin(a), 0., 1.4*np.cos(a)])
        if safe(eye):
            result.append(('room', look_at(eye, eye+np.array([np.sin(a), -.1, np.cos(a)]))))
    for shape in layout[1:]:
        a, b = np.array(shape['min']), np.array(shape['max'])
        target = (a+b)/2
        radius = np.linalg.norm((b-a)[[0, 2]])/2+.8
        for angle in np.linspace(0, 2*np.pi, 12, endpoint=False):
            eye = target+np.array([radius*np.sin(angle), 0, radius*np.cos(angle)])
            eye[1] = -.1
            eye = np.clip(eye, lo+.25, hi-.25)
            if safe(eye):
                result.append((shape['name'], look_at(eye, target)))
    for eye, target in [([0, 0, -1], [0, -1.5, 0]), ([0, 0, -1], [0, 1.5, 0]),
                        ([-2, 0, -2], [1, .5, 2]), ([2, 0, -2], [-1, .5, 2])]:
        if safe(np.array(eye)):
            result.append(('room', look_at(eye, target)))
    return result


def teacher_camera_set(layout, views=24):
    """Full yaw coverage, with translated up/down views at different heights."""
    from .room import pose_at
    lo, hi = np.array(layout[0]['min']), np.array(layout[0]['max'])
    radius = min(1.2, float(np.min((hi-lo)[[0, 2]]))*.18)
    middle = (lo+hi)/2
    def safe(eye):
        if not ((eye > lo+.08) & (eye < hi-.08)).all():
            return False
        return not any(((eye > np.array(s['min'])-.1) & (eye < np.array(s['max'])+.1)).all() for s in layout[1:])
    result = [('front', pose_at(0))] if safe(np.zeros(3)) else []
    for i in range(views):
        yaw = i*360/views
        angle = np.deg2rad(yaw)
        for pitch, height in [(0, 0), (55, -.15*(hi[1]-lo[1])), (-55, .15*(hi[1]-lo[1]))]:
            eye = middle+np.array([radius*np.sin(angle), height, radius*np.cos(angle)])
            if not safe(eye):
                candidates = [middle+np.array([radius*f*np.sin(angle), height, radius*f*np.cos(angle)]) for f in [.65, .3, 0.]]
                eye = next((p for p in candidates if safe(p)), None)
            if eye is None:
                raise ValueError(f'No safe camera for yaw={yaw}, pitch={pitch}; edit layout')
            result.append((f'room_yaw_{yaw:g}_pitch_{pitch:g}', pose_at(yaw, pitch, eye)))
    # Include views around each object to expose sides and accessible backs.
    result.extend((name, pose) for name, pose in camera_set(layout, views) if name != 'room')
    return result
