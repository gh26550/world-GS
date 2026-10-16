"""Progressive, shared RGB-D scene completion. Scaffold is a guide, not teacher RGB."""
import gc
import json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion
import trimesh
from .geometry import intrinsics, camera, mesh as depth_mesh
from .room import make_layout, pose_at
from .mesh_scene import build_components, MeshRenderer
from .mesh_pipeline import make_contact


def calibrated_depth(predicted, reference, observed, bounds):
    """Keep shared observations exact; estimate new depth without fixing proxy surfaces."""
    good = observed & (reference > 0) & np.isfinite(predicted) & (predicted > 0)
    if good.sum() >= 32:
        scale = np.median(reference[good]/predicted[good])
        method = 'shared_rgbd_overlap'
    else:
        # Relative depth has no metric scale. Enclosing room supplies only a scale prior.
        scale = np.quantile(bounds[bounds > 0], .65)/max(np.quantile(predicted, .85), .01)
        method = 'room_scale_prior'
    depth = np.clip(predicted*scale, .25, np.maximum(bounds, .26))
    depth[observed] = reference[observed]
    return depth, method


def patch_geometry(rgb, depth, K, pose, accept, stride=2):
    vertices, faces = depth_mesh(depth, K, pose, stride=stride, edge=.10)
    accepted = accept[::stride, ::stride].ravel()
    faces = faces[accepted[faces].all(-1)]
    if not len(faces):
        return None
    used, remap = np.unique(faces, return_inverse=True)
    return trimesh.Trimesh(vertices=vertices[used], faces=remap.reshape(-1, 3),
        vertex_colors=rgb[::stride, ::stride].reshape(-1, 3)[used], process=False)


def filter_visibility_conflicts(patch, observations, K):
    """Do not insert geometry in free space already observed by an earlier camera."""
    allowed = np.ones(len(patch.vertices), bool)
    for pose, depth, valid in observations:
        points = (patch.vertices-pose[:3, 3]) @ pose[:3, :3]
        uvh = points @ K.T
        uv = np.rint(uvh[:, :2]/np.maximum(uvh[:, 2:], 1e-8)).astype(int)
        h, w = depth.shape
        inside = (points[:, 2] > .15) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        ids = np.flatnonzero(inside)
        x, y = uv[ids].T
        conflict = valid[y, x] & (points[ids, 2] < depth[y, x]-.08*np.maximum(depth[y, x], 1.))
        allowed[ids[conflict]] = False
    keep = allowed[patch.faces].all(-1)
    patch.update_faces(keep)
    patch.remove_unreferenced_vertices()
    return patch


class SharedRGBDRenderer:
    """One world geometry/colour store; barycentric colour, no independently painted views."""
    def __init__(self, geometry):
        from trimesh.ray.ray_pyembree import RayMeshIntersector
        self.mesh = geometry
        self.rays = RayMeshIntersector(geometry)

    def render(self, K, pose, size):
        y, x = np.indices((size, size))
        rays = (np.stack([x, y, np.ones_like(x)], -1) @ np.linalg.inv(K).T @ pose[:3, :3].T).reshape(-1, 3)
        locations, ids, triangles = self.rays.intersects_location(np.tile(pose[:3, 3], (len(rays), 1)), rays, multiple_hits=False)
        depth = ((locations-pose[:3, 3]) @ pose[:3, :3])[:, 2]
        normals = self.mesh.face_normals[triangles]
        incidence = np.abs((normals*(-rays[ids]/np.linalg.norm(rays[ids], axis=1)[:, None])).sum(-1))
        good = (depth > .15) & (incidence > .18)
        locations, ids, triangles, depth, normals = locations[good], ids[good], triangles[good], depth[good], normals[good]
        bary = trimesh.triangles.points_to_barycentric(self.mesh.triangles[triangles], locations)
        colors = (self.mesh.visual.vertex_colors[self.mesh.faces[triangles], :3]*bary[..., None]).sum(1)
        rgb = np.zeros((size*size, 3), np.uint8)
        z = np.zeros(size*size, np.float32)
        xyz, normal = np.zeros((size*size, 3), np.float32), np.zeros((size*size, 3), np.float32)
        rgb[ids], z[ids], xyz[ids], normal[ids] = np.uint8(np.clip(colors, 0, 255)), depth, locations, normals
        return dict(rgb=rgb.reshape(size, size, 3), depth=z.reshape(size, size),
                    xyz=xyz.reshape(size, size, 3), normals=normal.reshape(size, size, 3))


def expansion_poses(views):
    # Small translations during completion introduce parallax; all directions share one store.
    poses = []
    for i in range(max(12, views)):
        yaw = i*360/max(12, views)
        a = np.deg2rad(yaw)
        poses.append((yaw, 0, pose_at(yaw, position=(.25*np.sin(a), 0, .25*(np.cos(a)-1)))))
    for pitch in (-55, 55):
        for yaw in (0, 90, 180, 270):
            poses.append((yaw, pitch, pose_at(yaw, pitch)))
    return poses


def direction_prompt(yaw, pitch):
    # pose_at positive pitch looks toward negative camera/world Y (ceiling).
    if pitch > 0:
        return 'upward looking ceiling photograph, ceiling and upper walls'
    if pitch < 0:
        return 'downward looking floor photograph, floor and lower furniture'
    return f'room direction {yaw:.0f} degrees away from the original view'


def generate_expanded_scene(prompt, output, size, views, seed, demo, front_image, room_size):
    from .pipeline import Models
    if size < 64 or size % 8 or views < 8:
        raise ValueError('size >=64 and divisible by 8; views >=8')
    if not demo and size < 256:
        raise ValueError('Diffusion completion requires --size >=256')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError('Use a new output directory')
    (output/'meshes').mkdir()
    K = intrinsics(size)
    models = None if demo else Models(seed)
    if demo:
        yy, xx = np.indices((size, size))
        image = Image.fromarray(np.uint8(np.stack([40+xx/size*150, 40+yy/size*130, np.full_like(xx, 100)], -1)))
        predicted = np.full((size, size), 3.)
        detections = []
    else:
        image = Image.open(front_image).convert('RGB').resize((size, size)) if front_image else models.front(prompt, size)
        predicted = models.depth(image)
        detections = models.detect(image, prompt)
    image.save(output/'front.png')
    np.save(output/'front_relative_depth.npy', predicted)
    layout = make_layout(room_size, detections, predicted, K)
    proxy, _, components, owners = build_components(layout, image)
    proxy.export(output/'initial_scaffold.glb')
    guide = MeshRenderer(proxy, components, owners)
    first = guide.render(K, camera(0), size)
    depth, _ = calibrated_depth(predicted, first['depth'], np.zeros_like(predicted, bool), first['depth'])
    shared = patch_geometry(np.asarray(image), depth, K, camera(0), np.ones_like(depth, bool))
    if shared is None:
        raise ValueError('Front depth has no valid triangles')
    frames, heldout, history = [], [], []
    observations = [(camera(0), depth, depth > 0)]

    def save_frame(rgb, d, valid, pose, prefix, records, **extra):
        i = len(records)
        filename = f'{prefix}_{i:03d}.png'
        Image.fromarray(rgb).save(output/filename)
        np.save(output/f'{prefix}_depth_{i:03d}.npy', d)
        np.save(output/f'{prefix}_mask_{i:03d}.npy', valid)
        records.append(dict(file_path=filename, depth_file=f'{prefix}_depth_{i:03d}.npy',
            mask_file=f'{prefix}_mask_{i:03d}.npy', transform_matrix=pose.tolist(), target='generated_scene', **extra))

    save_frame(np.asarray(image), depth, depth > 0, camera(0), 'view', frames, reference_source='front_image_predicted_depth')
    for index, (yaw, pitch, pose) in enumerate(expansion_poses(views)[1:], 1):
        observation = SharedRGBDRenderer(shared).render(K, pose, size)
        bounds = guide.render(K, pose, size)
        observed = binary_erosion(observation['depth'] > 0, iterations=2)
        missing = ~observed
        # Fill cracks from neighbouring shared pixels, but leave their mask open for completion.
        guide_rgb = bounds['rgb'].copy()
        guide_rgb[observed] = observation['rgb'][observed]
        if demo:
            generated = guide_rgb.copy()
            generated[missing] = np.uint8([(seed+index*43)%180+40, 90+index%4*20, 110])
            new_prediction = np.maximum(bounds['depth']*.8, .3)
        elif missing.mean() > .005:
            direction = direction_prompt(yaw, pitch)
            generated = np.asarray(models.controlled_complete(
                prompt+', coherent interior photograph, '+direction+', realistic furniture and architecture',
                guide_rgb, missing, np.where(observed, observation['depth'], bounds['depth']),
                index, control_strength=.35))
            new_prediction = models.depth(Image.fromarray(generated))
        else:
            generated, new_prediction = guide_rgb, observation['depth']
        fused_depth, alignment = calibrated_depth(new_prediction, observation['depth'], observed, bounds['depth'])
        generated = generated.copy()
        generated[observed] = observation['rgb'][observed]
        Image.fromarray(guide_rgb).save(output/f'guide_{index:03d}.png')
        Image.fromarray(np.uint8(missing)*255).save(output/f'completion_mask_{index:03d}.png')
        patch = patch_geometry(generated, fused_depth, K, pose, missing)
        added = 0
        if patch is not None:
            patch = filter_visibility_conflicts(patch, observations, K)
            added = len(patch.faces)
            if added:
                patch.export(output/'meshes'/f'generated_patch_{index:03d}.glb')
                shared = trimesh.util.concatenate([shared, patch])
        supported = SharedRGBDRenderer(shared).render(K, pose, size)['depth']
        consistent = observed | ((supported > 0) & (np.abs(supported-fused_depth) < .08*np.maximum(fused_depth, 1.)))
        observations.append((pose, fused_depth, consistent))
        save_frame(generated, fused_depth, consistent, pose, 'view', frames,
            reference_source='shared_reprojection_and_diffusion_rgbd' if not demo else 'synthetic_fixture',
            yaw=yaw, pitch=pitch, generated_fraction=float(missing.mean()))
        history.append(dict(yaw=yaw, pitch=pitch, generated_fraction=float(missing.mean()),
            added_triangles=added, consistent_fraction=float(consistent.mean()), depth_alignment=alignment))
        (output/'completion_history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
        print(f'Completion {index}/{len(expansion_poses(views))-1}: yaw={yaw:.0f} pitch={pitch} new={missing.mean():.1%} added_triangles={added}', flush=True)
    del models
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except ImportError:
        pass
    shared.export(output/'scene_mesh.glb')
    shared.export(output/'proxy_scene.obj')
    renderer = SharedRGBDRenderer(shared)
    clouds, colors, normals = [], [], []
    for frame in frames:
        # Lift actual teacher RGB-D, not procedural scaffold RGB.
        from .geometry import lift
        d = np.load(output/frame['depth_file'])
        rgb = np.asarray(Image.open(output/frame['file_path']))
        valid = np.load(output/frame['mask_file'])
        xyz = lift(d, K, np.array(frame['transform_matrix']))
        normal = np.cross(np.gradient(xyz, axis=0), np.gradient(xyz, axis=1))
        normal /= np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-8)
        take = valid.copy()
        take[1::2] = False
        take[:, 1::2] = False
        clouds.append(xyz[take]); colors.append(rgb[take]); normals.append(normal[take])
    # Additional translated cameras use the UPDATED RGB-D geometry, never the initial proxy.
    for i in range(views):
        yaw = i*360/views+15
        a = np.deg2rad(yaw)
        pose = pose_at(yaw, position=(.5*np.sin(a), 0, .5*np.cos(a)))
        hit = renderer.render(K, pose, size)
        valid = hit['depth'] > 0
        if valid.mean() < .3:
            continue
        if i % 4 == 0:
            save_frame(hit['rgb'], hit['depth'], valid, pose, 'heldout', heldout,
                reference_source='updated_generated_rgbd_mesh_render')
        else:
            save_frame(hit['rgb'], hit['depth'], valid, pose, 'view', frames,
                reference_source='updated_generated_rgbd_mesh_render')
    if not heldout:
        raise ValueError('No supported translated evaluation view; inspect generated geometry')
    xyz, rgb, normal = np.concatenate(clouds), np.concatenate(colors), np.concatenate(normals)
    _, keep = np.unique(np.round(xyz/.015), axis=0, return_index=True)
    np.savez_compressed(output/'points.npz', xyz=xyz[keep].astype('float32'), rgb=rgb[keep], normals=normal[keep].astype('float32'))
    poses = np.array([f['transform_matrix'] for f in frames])
    meta = dict(mode='progressive_rgbd_v4', prompt=prompt, seed=seed, demo=demo, size=size, K=K.tolist(),
        convention='OpenCV camera-to-world, x right y down z forward', frames=frames, heldout_frames=heldout,
        geometry_constraint='none; generated depth cue annealed during GS fitting', point_count=len(keep),
        reference_source='progressive diffusion completion with shared RGBD reprojection and updated geometry',
        camera_baseline=float(np.linalg.norm(poses[:, None, :3, 3]-poses[None, :, :3, 3], axis=-1).max()),
        mesh_triangles=len(shared.faces), initial_scaffold_is_teacher=False,
        room_size_prior=list(room_size), controlnet_strength=.35,
        depth_model='depth-anything/Depth-Anything-V2-Small-hf' if not demo else 'synthetic_fixture',
        limitations=['unseen content is generated, not recovered truth', 'relative monocular depth can warp geometry',
            'reprojection preserves observed pixels but does not guarantee global semantic consistency',
            'heldout references derive from generated geometry, not real photographs', 'not dynamically relightable'])
    (output/'scene.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    make_contact(output, frames, 'references.jpg')
    return meta
