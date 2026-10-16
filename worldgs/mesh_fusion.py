"""Global, confidence-filtered TSDF meshing; no concatenation of image sheets."""
import json
from pathlib import Path
import numpy as np
import trimesh
from PIL import Image
from scipy.ndimage import binary_erosion, binary_dilation
from .geometry import lift


def open3d_mesh(mesh):
    import open3d as o3d
    result = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(np.asarray(mesh.vertices)),
                                     o3d.utility.Vector3iVector(np.asarray(mesh.faces)))
    result.vertex_colors = o3d.utility.Vector3dVector(mesh.visual.vertex_colors[:, :3]/255.)
    return result


def mesh_quality(mesh):
    faces, vertices = np.asarray(mesh.faces), np.asarray(mesh.vertices)
    if not len(faces):
        return {'triangles': 0, 'vertices': len(vertices), 'finite': bool(np.isfinite(vertices).all())}
    triangles = vertices[faces]
    edges = triangles[:, [1, 2, 0]]-triangles
    lengths = np.linalg.norm(edges, axis=-1)
    twice_area = np.linalg.norm(np.cross(edges[:, 0], -edges[:, 2]), axis=-1)
    aspect = lengths.max(-1)**2/np.maximum(twice_area, 1e-15)
    labels, counts, areas = open3d_mesh(mesh).cluster_connected_triangles()
    counts, areas = np.array(counts), np.array(areas)
    _, edge_counts = np.unique(np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1), axis=0, return_counts=True)
    return dict(vertices=len(vertices), triangles=len(faces), finite=bool(np.isfinite(vertices).all()),
        components=len(counts), small_components_under_50_faces=int((counts < 50).sum()),
        boundary_edges=int((edge_counts == 1).sum()), nonmanifold_edges=int((edge_counts > 2).sum()),
        boundary_edge_fraction=float((edge_counts == 1).mean()),
        degenerate_faces=int((twice_area < 1e-12).sum()),
        aspect_p95=float(np.quantile(aspect, .95)), needle_face_fraction=float((aspect > 20).mean()),
        max_edge=float(lengths.max()), surface_area=float(mesh.area),
        small_component_area_fraction=float(areas[counts < 50].sum()/max(areas.sum(), 1e-12)))


def observation_masks(depths, valid, poses, K, colors=None, voxel_size=.025, min_views=2,
                      include_provisional=False, return_confidence=False, recover_edges=False):
    """Require depth support from another translated camera, not synthetic re-renders."""
    bases = []
    for d, mask in zip(depths, valid):
        good = mask & np.isfinite(d) & (d > .2)
        edge = np.zeros_like(good)
        for axis in (0, 1):
            a, b = [slice(None)]*2, [slice(None)]*2
            a[axis], b[axis] = slice(None, -1), slice(1, None)
            a, b = tuple(a), tuple(b)
            jump = np.abs(d[a]-d[b]) > .035*np.maximum(np.minimum(d[a], d[b]), .2)
            edge[a] |= jump; edge[b] |= jump
        bases.append(binary_erosion(good, iterations=1) & ~binary_dilation(edge, iterations=1))
    accepted, reports, confidences = [], [], []
    for i, (depth, pose) in enumerate(zip(depths, poses)):
        candidate = np.isfinite(depth) & (depth > .2) if recover_edges else bases[i]
        ids = np.flatnonzero(candidate)
        world = lift(depth, K, pose).reshape(-1, 3)[ids]
        agree, conflict = np.zeros(len(ids), int), np.zeros(len(ids), int)
        for j, other in enumerate(poses):
            if i == j or np.linalg.norm(other[:3, 3]-pose[:3, 3]) < .06:
                continue
            points = (world-other[:3, 3]) @ other[:3, :3]
            uvh = points @ K.T
            uv = np.rint(uvh[:, :2]/np.maximum(uvh[:, 2:], 1e-8)).astype(int)
            h, w = depth.shape
            inside = (points[:, 2] > .2) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
            k = np.flatnonzero(inside)
            x, y = uv[k].T
            trusted = bases[j][y, x]
            measured = depths[j][y, x]
            tolerance = 2*voxel_size+.025*measured
            matched = trusted & (np.abs(points[k, 2]-measured) <= tolerance)
            # Geometry support must not disappear merely because a bookshelf's colours
            # move by a pixel under reprojection. RGB is optimized separately by GS.
            agree[k] += matched
            # A point behind another surface may simply be occluded, not erroneous.
            conflict[k] += trusted & (points[k, 2] < measured-tolerance)
        own_trusted = valid[i].ravel()[ids]
        keep = (agree+own_trusted >= min_views) & (agree >= 2*conflict)
        provisional = np.zeros_like(keep)
        if include_provisional:
            gy, gx = np.gradient(np.log(np.maximum(depth, .2)))
            smooth = (np.hypot(gx, gy).ravel()[ids] < .015)
            if recover_edges:
                # At an occlusion edge, require a coherent one-sided neighbourhood;
                # do not delete both sides of every depth discontinuity.
                padded = np.pad(depth, 1, mode='edge')
                support = np.zeros_like(depth, int)
                for dy in range(3):
                    for dx in range(3):
                        neighbour = padded[dy:dy+depth.shape[0], dx:dx+depth.shape[1]]
                        support += np.isfinite(neighbour) & (np.abs(neighbour-depth) < .02*np.maximum(depth, .2))
                smooth |= support.ravel()[ids] >= 4
            provisional = (~keep) & own_trusted & (agree == 0) & (conflict == 0) & smooth
        confidence = np.zeros_like(depth, np.float32)
        confidence.ravel()[ids[keep]] = 1.
        confidence.ravel()[ids[provisional]] = .25
        keep |= provisional
        mask = np.zeros_like(depth, bool)
        mask.ravel()[ids[keep]] = True
        accepted.append(mask)
        confidences.append(confidence)
        reports.append(dict(index=i, candidates=len(ids), accepted=int(keep.sum()),
            accepted_fraction=float(mask.mean()), conflicting_points=int((conflict > 0).sum()),
            provisional_points=int(provisional.sum()), trusted_points=int((confidence == 1).sum()),
            recovered_edge_points=int((keep & ~bases[i].ravel()[ids]).sum()),
            recovered_previously_masked_points=int((keep & ~own_trusted).sum())))
    return (accepted, reports, confidences) if return_confidence else (accepted, reports)


def fuse_tsdf(depths, colors, poses, K, masks, voxel_size=.025, confidence=None):
    import open3d as o3d
    h, w = depths[0].shape
    intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, K[0, 0], K[1, 1], K[0, 2], K[1, 2])
    volume = o3d.pipelines.integration.ScalableTSDFVolume(voxel_length=voxel_size,
        sdf_trunc=3*voxel_size, color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    for i, (d, rgb, pose, mask) in enumerate(zip(depths, colors, poses, masks)):
        filtered = np.ascontiguousarray(np.where(mask, d, 0), np.float32)
        if not (filtered > 0).any():
            continue
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(rgb, np.uint8)), o3d.geometry.Image(filtered),
            depth_scale=1., depth_trunc=40., convert_rgb_to_intensity=False)
        volume.integrate(rgbd, intrinsic, np.linalg.inv(pose))
        if confidence is not None:
            trusted = mask & (confidence[i] >= .99)
            if trusted.any():
                trusted_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    o3d.geometry.Image(np.ascontiguousarray(rgb, np.uint8)),
                    o3d.geometry.Image(np.ascontiguousarray(np.where(trusted, d, 0), np.float32)),
                    depth_scale=1., depth_trunc=40., convert_rgb_to_intensity=False)
                for _ in range(3):
                    volume.integrate(trusted_rgbd, intrinsic, np.linalg.inv(pose))
        print(f'TSDF integration {i+1}/{len(depths)}: {mask.mean():.1%} supported pixels', flush=True)
    mesh = volume.extract_triangle_mesh()
    if not len(mesh.triangles):
        raise ValueError('No surface supported by multiple translated cameras')
    return mesh


def clean_fused_mesh(mesh, voxel_size=.025, min_faces=50):
    import open3d as o3d
    # Marching cubes can put several isosurface vertices almost on the same voxel corner.
    # Weld only a small fraction of one voxel, rather than retaining tiny sliver triangles.
    mesh.merge_close_vertices(.12*voxel_size)
    mesh.remove_duplicated_vertices().remove_duplicated_triangles().remove_degenerate_triangles()
    labels, counts, areas = mesh.cluster_connected_triangles()
    labels, counts, areas = np.asarray(labels), np.asarray(counts), np.asarray(areas)
    keep_cluster = (counts >= min_faces) & (areas >= 20*voxel_size**2)
    mesh.remove_triangles_by_mask(~keep_cluster[labels])
    mesh.remove_unreferenced_vertices()
    if not len(mesh.triangles):
        raise ValueError('No sufficiently supported surface component remains')
    vertices = np.asarray(mesh.vertices).copy()
    smooth = mesh.filter_smooth_taubin(number_of_iterations=3)
    delta = np.asarray(smooth.vertices)-vertices
    # Limit smoothing displacement; do not freely shrink furniture or fill unknown holes.
    delta *= np.minimum(1., .5*voxel_size/np.maximum(np.linalg.norm(delta, axis=1), 1e-12))[:, None]
    mesh.vertices = o3d.utility.Vector3dVector(vertices+delta)
    mesh.remove_non_manifold_edges().remove_degenerate_triangles().remove_unreferenced_vertices()
    triangles = np.asarray(mesh.vertices)[np.asarray(mesh.triangles)]
    area2 = np.linalg.norm(np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0]), axis=1)
    mesh.remove_triangles_by_mask(area2 < 1e-12)
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    return trimesh.Trimesh(vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.triangles),
        vertex_colors=np.uint8(np.clip(np.asarray(mesh.vertex_colors), 0, 1)*255), process=False)


def surface_points(mesh, count=300000, seed=0):
    rng = np.random.default_rng(seed)
    faces = rng.choice(len(mesh.faces), count, p=mesh.area_faces/mesh.area)
    u, v = rng.random((2, count))
    bary = np.column_stack([1-np.sqrt(u), np.sqrt(u)*(1-v), np.sqrt(u)*v])
    xyz = (mesh.triangles[faces]*bary[..., None]).sum(1)
    rgb = (mesh.visual.vertex_colors[mesh.faces[faces], :3]*bary[..., None]).sum(1)
    normals = mesh.face_normals[faces]
    _, keep = np.unique(np.round(xyz/.01), axis=0, return_index=True)
    return dict(xyz=xyz[keep].astype('float32'), rgb=np.uint8(np.clip(rgb[keep], 0, 255)),
        normals=normals[keep].astype('float32'), triangle_ids=faces[keep], barycentric=bary[keep].astype('float32'))


def rebuild_mesh_scene(path, output, voxel_size=.025, min_views=2, allow_observations=False, strict_support=False,
                       holdout_keyframe=None):
    from .expansion import SharedRGBDRenderer
    from .mesh_pipeline import make_contact
    if not .005 <= voxel_size <= .1 or min_views < 2:
        raise ValueError('voxel-size must be .005..0.1; min-views must be >=2')
    source, output = Path(path), Path(output)
    meta = json.loads((source/'scene.json').read_text())
    baseline_mesh = source/'scene_mesh.glb'
    if meta.get('mode') in ('global_tsdf_v6', 'global_tsdf_v7'):
        source = Path(meta['observation_source'])
        meta = json.loads((source/'scene.json').read_text())
    if meta.get('mode') not in ('progressive_rgbd_v4', 'progressive_rgbd_v5'):
        raise ValueError('rebuild-mesh requires an expand scene')
    output.mkdir(parents=True, exist_ok=True)
    entries = list(output.iterdir())
    permitted = allow_observations and source.resolve() == (output/'observations').resolve() and len(entries) == 1 and entries[0].name == 'observations'
    if entries and not permitted:
        raise FileExistsError('Use an empty output directory')
    keyframes = [meta['frames'][0]]+[f for f in meta['frames'] if 'generated_fraction' in f]
    if len(keyframes) < 4:
        raise ValueError('Insufficient source keyframes')
    # Production uses all available inputs. Leave-one-view-out is an explicit experiment.
    reserved = holdout_keyframe
    if reserved is not None and not 0 <= reserved < len(keyframes):
        raise ValueError('holdout-keyframe is outside the source keyframe list')
    fusion_frames = [f for i, f in enumerate(keyframes) if i != reserved]
    K = np.array(meta['K'])
    colors = [np.asarray(Image.open(source/f['file_path']).convert('RGB')) for f in fusion_frames]
    depths = [np.load(source/f['depth_file']) for f in fusion_frames]
    valid = [np.load(source/f['mask_file']) for f in fusion_frames]
    poses = [np.array(f['transform_matrix']) for f in fusion_frames]
    masks, support, confidences = observation_masks(depths, valid, poses, K, colors, voxel_size, min_views,
        include_provisional=not strict_support, return_confidence=True, recover_edges=True)
    (output/'support_report.json').write_text(json.dumps(support, indent=2))
    fused = fuse_tsdf(depths, colors, poses, K, masks, voxel_size, confidences)
    mesh = clean_fused_mesh(fused, voxel_size)
    from .hole_repair import repair_small_mesh_holes
    mesh, patch_report = repair_small_mesh_holes(mesh, depths, valid, poses, K, voxel_size)
    (output/'hole_repair.json').write_text(json.dumps(patch_report, indent=2))
    after = mesh_quality(mesh)
    before = mesh_quality(trimesh.load(baseline_mesh, force='mesh', process=False))
    passed = after['finite'] and after['triangles'] >= 100 and after['degenerate_faces'] == 0 and after['nonmanifold_edges'] == 0 and after['max_edge'] <= 4*voxel_size
    quality = dict(passed=bool(passed), before=before, after=after, voxel_size=voxel_size,
        min_views=min_views, strict_support=strict_support, reserved_keyframe=reserved, reserved_used_for_fusion=False,
        checks=['finite vertices', '>=100 triangles', 'no degenerate faces', 'no nonmanifold edges', 'max edge <=4 voxels'],
        limitations=['quality checks do not establish semantic correctness or watertightness'])
    (output/'mesh_quality.json').write_text(json.dumps(quality, indent=2))
    mesh.export(output/'scene_mesh.glb')
    mesh.export(output/'proxy_scene.obj')
    if not passed:
        raise ValueError('Fused mesh failed geometric quality checks; inspect mesh_quality.json')
    renderer = SharedRGBDRenderer(mesh)
    frames, heldout = [], []
    def save(rgb, d, mask, pose, records, prefix, kind, confidence=None, **extra):
        i = len(records)
        name = f'{prefix}_{i:03d}.png'
        Image.fromarray(rgb).save(output/name)
        np.save(output/f'{prefix}_depth_{i:03d}.npy', d)
        np.save(output/f'{prefix}_mask_{i:03d}.npy', mask)
        if confidence is not None:
            name_conf = f'{prefix}_confidence_{i:03d}.npy'
            np.save(output/name_conf, confidence)
            extra['confidence_file'] = name_conf
        records.append(dict(file_path=name, depth_file=f'{prefix}_depth_{i:03d}.npy', mask_file=f'{prefix}_mask_{i:03d}.npy',
            transform_matrix=pose.tolist(), reference_source=kind, target='fused_scene', **extra))
    for i, (f, rgb, d, pose, mask) in enumerate(zip(fusion_frames, colors, depths, poses, masks)):
        hit = renderer.render(K, pose, meta['size'])
        accepted = mask & (hit['depth'] > 0) & (np.abs(hit['depth']-d) <= 2*voxel_size+.025*d)
        if accepted.mean() >= .03:
            save(rgb, hit['depth'], accepted, pose, frames, 'view', 'original_keyframe_on_validated_fused_surface',
                 source_keyframe=f['file_path'], valid_fraction=float(accepted.mean()), confidence=confidences[i]*accepted)
    if reserved is not None:
        f = keyframes[reserved]
        pose = np.array(f['transform_matrix'])
        rgb, d = np.asarray(Image.open(source/f['file_path'])), np.load(source/f['depth_file'])
        mask = np.load(source/f['mask_file'])
        hit = renderer.render(K, pose, meta['size'])
        evaluation_mask = mask & (hit['depth'] > 0)
        check = dict(keyframe=reserved, covered_fraction=float(evaluation_mask.mean()),
            median_relative_depth_error=float(np.median(np.abs(hit['depth'][evaluation_mask]/d[evaluation_mask]-1))) if evaluation_mask.any() else None,
            note='source frame excluded from fusion; generated source is not independent real-world ground truth')
        save(rgb, d, evaluation_mask, pose, heldout, 'heldout', 'keyframe_excluded_from_fusion', used_for_fusion=False)
    else:
        check = dict(keyframe=None, note='production fusion uses all input keyframes; render holdouts are not independent ground truth')
    (output/'fusion_holdout.json').write_text(json.dumps(check, indent=2))
    for f in meta.get('heldout_frames', []):
        pose = np.array(f['transform_matrix'])
        hit = renderer.render(K, pose, meta['size'])
        mask = hit['depth'] > 0
        if mask.mean() >= .1:
            save(hit['rgb'], hit['depth'], mask, pose, heldout, 'heldout', 'fused_mesh_render')
    if len(frames) < 3:
        raise ValueError('Too few supported training views; do not train from unsupported surfaces')
    points = surface_points(mesh, seed=meta['seed'])
    np.savez_compressed(output/'points.npz', **points)
    # Barycentric seeds lie exactly on exported triangles, not on rejected RGB-D sheets.
    point_check = np.max(np.linalg.norm(points['xyz']-(mesh.triangles[points['triangle_ids']]*points['barycentric'][..., None]).sum(1), axis=1))
    quality['seed_to_assigned_surface_max'] = float(point_check)
    quality['passed'] = quality['passed'] and bool(np.isfinite(point_check) and point_check <= 1e-4)
    (output/'mesh_quality.json').write_text(json.dumps(quality, indent=2))
    if not quality['passed']:
        raise ValueError('GS seeds do not agree with the validated fused mesh')
    new_meta = dict(mode='global_tsdf_v7', prompt=meta['prompt'], seed=meta['seed'], demo=meta['demo'],
        size=meta['size'], K=meta['K'], frames=frames, heldout_frames=heldout,
        convention=meta['convention'], observation_source=str(source.resolve()), voxel_size=voxel_size, min_views=min_views,
        room_size_prior=meta.get('room_size_prior'), mesh_triangles=len(mesh.faces), point_count=len(points['xyz']),
        strict_support=strict_support, provisional_policy='none' if strict_support else 'smooth uncontradicted single-view surface, weight .25',
        render_min_incidence=.02, coverage_loss_weight=.05, holdout_keyframe=reserved,
        geometry_constraint='no hard projection or mesh position prior; seeds sampled on validated fused surface',
        limitations=['relative generated depth and camera assumptions remain', 'provisional single-view surfaces are uncertain; contradicted surfaces excluded',
            'fine details below voxel resolution may be lost', 'view support is not independent real-world evidence', 'not dynamically relightable'])
    (output/'scene.json').write_text(json.dumps(new_meta, indent=2))
    Image.open(source/'front.png').save(output/'front.png')
    make_contact(output, frames, 'references.jpg')
    print(json.dumps(dict(mesh_quality=after, point_count=len(points['xyz']), train_views=len(frames)), indent=2), flush=True)
    return new_meta
