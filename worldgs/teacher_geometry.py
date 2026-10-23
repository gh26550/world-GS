"""Shared geometry contracts for mesh-teacher rendering and GS fitting."""
import hashlib
import numpy as np


def validate_layout(layout):
    if not layout or layout[0].get('kind') != 'room':
        raise ValueError('Layout must start with an enclosing room')
    for shape in layout:
        a, b = np.asarray(shape['min'], float), np.asarray(shape['max'], float)
        if a.shape != (3,) or b.shape != (3,) or not np.isfinite([a, b]).all() or (b <= a).any():
            raise ValueError('Invalid layout bounds')
        if shape is not layout[0] and ((a < layout[0]['min']).any() or (b > layout[0]['max']).any()):
            raise ValueError('Object is outside the room')


def mesh_digest(vertices, faces):
    digest = hashlib.sha256()
    digest.update(np.asarray(vertices, '<f4').tobytes())
    digest.update(np.asarray(faces, '<i4').tobytes())
    return digest.hexdigest()


def sample_teacher_surface(mesh, renderer, owners, visible_faces, count=240000, seed=0, component_fraction=0.):
    rng = np.random.default_rng(seed)
    weights = mesh.area_faces*np.asarray(visible_faces, bool)
    if weights.sum() <= 0:
        raise ValueError('No mesh surfaces visible in training cameras')
    probability=weights/weights.sum()
    if not 0<=component_fraction<=1:
        raise ValueError('component_fraction must be in [0,1]')
    if component_fraction:
        balanced=np.zeros_like(weights)
        groups=np.unique(owners[weights>0])
        for group in groups:
            selected=owners==group
            balanced[selected]=weights[selected]/weights[selected].sum()/len(groups)
        probability=(1-component_fraction)*probability+component_fraction*balanced
    ids = rng.choice(len(mesh.faces), count, p=probability)
    uv = rng.random((count, 2))
    uv[uv.sum(1) > 1] = 1-uv[uv.sum(1) > 1]
    bary = np.column_stack([1-uv.sum(1), uv])
    xyz = (mesh.triangles[ids]*bary[..., None]).sum(1)
    colors, normals = renderer.shade(xyz, ids)
    return dict(xyz=xyz.astype(np.float32), rgb=np.uint8(colors*255), normals=normals.astype(np.float32),
                triangle_ids=ids, barycentric=bary.astype(np.float32), component_ids=owners[ids])


def structure_report(mesh, components, points):
    checks = {}
    offset = 0
    for part in components:
        faces = mesh.faces[offset:offset+part['faces']]
        offset += part['faces']
        if part['group'] != 'room':
            continue
        axis = int(part['name'].split('_')[1])
        side = int(part['name'].split('_')[2])
        plane = part['bounds'][1-side][axis]
        normal_component = mesh.face_normals[offset-part['faces']:offset, axis]
        inner = normal_component*(1 if side == 0 else -1) > .5
        residual = np.abs(mesh.vertices[faces[inner], axis]-plane)
        checks[part['name']] = dict(inner_plane=float(plane), inner_triangles=int(inner.sum()),
                                   residual_max=float(residual.max()) if residual.size else None)
    expected = (mesh.triangles[points['triangle_ids']]*points['barycentric'][..., None]).sum(1)
    error = np.linalg.norm(points['xyz']-expected, axis=1)
    passed = len(checks) == 6 and all(v['inner_triangles'] == 2 and v['residual_max'] is not None and v['residual_max'] < 1e-5 for v in checks.values()) and error.max() < 1e-5
    return dict(passed=bool(passed), structure=checks, seed_to_assigned_surface_max=float(error.max()),
                note='Checks procedural room planes and seed attachment, not real-world reconstruction accuracy')
