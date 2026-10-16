"""Observation-checked local mesh caps; preserve visible openings and large unknown regions."""
import numpy as np
import trimesh
from scipy.spatial import cKDTree


def supported_patch_points(points, depths, masks, poses, K, tolerance=.04):
    agree, conflict = np.zeros(len(points), int), np.zeros(len(points), int)
    for d, mask, pose in zip(depths, masks, poses):
        p = (points-pose[:3, 3]) @ pose[:3, :3]
        uvh = p @ K.T
        uv = np.rint(uvh[:, :2]/np.maximum(uvh[:, 2:], 1e-8)).astype(int)
        h, w = d.shape
        inside = (p[:, 2] > .2) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        ids = np.flatnonzero(inside)
        x, y = uv[ids].T
        trusted = mask[y, x] & np.isfinite(d[y, x]) & (d[y, x] > .2)
        tol = tolerance+.01*d[y, x]
        agree[ids] += trusted & (np.abs(p[ids, 2]-d[y, x]) <= tol)
        conflict[ids] += trusted & (p[ids, 2] < d[y, x]-tol)
    return (agree >= 1) & (conflict == 0)


def split_long_edges(mesh, max_edge):
    """Conforming bisection: both incident faces share each new midpoint."""
    vertices, faces = np.array(mesh.vertices), np.array(mesh.faces)
    colors = mesh.visual.vertex_colors.astype(float)
    templates = {1: [(0, 3, 2), (3, 1, 2)], 2: [(1, 4, 0), (4, 2, 0)],
        4: [(2, 5, 1), (5, 0, 1)], 3: [(3, 1, 4), (0, 3, 2), (3, 4, 2)],
        6: [(4, 2, 5), (1, 4, 0), (4, 5, 0)], 5: [(5, 0, 3), (2, 5, 1), (5, 3, 1)],
        7: [(0, 3, 5), (3, 1, 4), (5, 4, 2), (3, 4, 5)]}
    for _ in range(12):
        edges = np.sort(faces[:, [[0, 1], [1, 2], [2, 0]]], axis=-1)
        long = np.linalg.norm(vertices[edges[..., 0]]-vertices[edges[..., 1]], axis=-1) > max_edge
        if not long.any():
            break
        unique, inverse = np.unique(edges[long], axis=0, return_inverse=True)
        mid = np.full(long.shape, -1, int)
        mid[long] = len(vertices)+inverse
        vertices = np.concatenate([vertices, vertices[unique].mean(1)])
        colors = np.concatenate([colors, colors[unique].mean(1)])
        expanded = np.column_stack([faces, mid])
        codes = (long*np.array([1, 2, 4])).sum(1)
        new = [faces[codes == 0]]
        for code, template in templates.items():
            selected = expanded[codes == code]
            if len(selected):
                new.append(selected[:, np.array(template)].reshape(-1, 3))
        faces = np.concatenate(new)
    return trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=np.uint8(np.clip(colors, 0, 255)), process=False)


def repair_small_mesh_holes(mesh, depths, masks, poses, K, voxel_size=.025, radius=.18):
    import open3d as o3d
    from .mesh_fusion import open3d_mesh
    # Open3D only proposes caps. Source depths decide whether each proposed triangle is plausible.
    legacy = open3d_mesh(mesh)
    tensor = o3d.t.geometry.TriangleMesh.from_legacy(legacy)
    proposal = tensor.fill_holes(hole_size=radius)
    candidate = proposal.to_legacy()
    vertices = np.asarray(candidate.vertices).copy()
    distance, mapping = cKDTree(mesh.vertices).query(vertices)
    if len(distance) == 0 or distance.max() > 1e-5:
        return mesh, dict(proposed=0, accepted=0, reason='candidate changed boundary positions')
    faces = mapping[np.asarray(candidate.triangles)]
    def keys(f):
        f = np.ascontiguousarray(np.sort(f, axis=1), dtype=np.int64)
        return f.view(np.dtype((np.void, 24))).ravel()
    new = faces[~np.isin(keys(faces), keys(mesh.faces))]
    if len(new):
        _, first = np.unique(keys(new), return_index=True)
        new = new[np.sort(first)]
    if not len(new):
        return mesh, dict(proposed=0, accepted=0)
    triangles = np.asarray(mesh.vertices)[new]
    samples = np.stack([triangles.mean(1), (triangles[:, 0]+triangles[:, 1])/2,
        (triangles[:, 1]+triangles[:, 2])/2, (triangles[:, 2]+triangles[:, 0])/2], 1)
    supported = supported_patch_points(samples.reshape(-1, 3), depths, masks, poses, K).reshape(-1, 4).all(1)
    new = new[supported]
    if not len(new):
        return mesh, dict(proposed=len(supported), accepted=0, radius=radius)
    depth_supported = len(new)
    # A geometric proposal may touch an existing interior edge. Never give an
    # edge a third incident face, even when the source depths agree.
    n = len(mesh.vertices)
    old_edges = np.sort(np.asarray(mesh.faces)[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
    codes, counts = np.unique(old_edges[:, 0]*n+old_edges[:, 1], return_counts=True)
    new_edges = np.sort(new[:, [[0, 1], [1, 2], [2, 0]]], axis=-1)
    edge_codes = new_edges[..., 0]*n+new_edges[..., 1]
    positions = np.searchsorted(codes, edge_codes)
    safe_positions = np.minimum(positions, len(codes)-1)
    base = np.where((positions < len(codes)) & (codes[safe_positions] == edge_codes), counts[safe_positions], 0)
    added, keep = {}, []
    for i, edges in enumerate(edge_codes):
        if all(base[i, k]+added.get(int(e), 0) < 2 for k, e in enumerate(edges)):
            keep.append(i)
            for e in edges:
                added[int(e)] = added.get(int(e), 0)+1
    new, edge_codes, base = new[keep], edge_codes[keep], base[keep]
    # Remove isolated cap islands that no longer connect to the original boundary
    # after depth/topology filtering.
    parent = np.arange(len(new))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    owners = {}
    for i, edges in enumerate(edge_codes):
        for edge in edges:
            edge = int(edge)
            if edge in owners:
                parent[root(i)] = root(owners[edge])
            else:
                owners[edge] = i
    attached = {root(i) for i in np.flatnonzero((base > 0).any(1))}
    new = new[[root(i) in attached for i in range(len(new))]]
    if not len(new):
        return mesh, dict(proposed=len(supported), depth_supported=depth_supported, accepted=0, radius=radius)
    result = trimesh.Trimesh(vertices=mesh.vertices, faces=np.concatenate([mesh.faces, new]),
        vertex_colors=mesh.visual.vertex_colors, process=False)
    result = split_long_edges(result, 3*voxel_size)
    # No global smoothing here: do not move the existing observed surface.
    return result, dict(proposed=len(supported), depth_supported=depth_supported, accepted=len(new), radius=radius,
        policy='all four cap samples need source-depth support and no free-space contradiction')
