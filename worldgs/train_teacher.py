"""Fit Gaussian appearance on an immutable, explicitly sampled mesh surface."""
import json
import hashlib
from pathlib import Path
from shutil import copyfile
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from .coverage_sampling import stratified_seed_ids
from .teacher_geometry import mesh_digest
from .train_mesh import normal_quaternions
from .ply import FULL_FIELDS, write_gaussian_ply


def validate_teacher(path):
    path = Path(path)
    meta = json.loads((path/'scene.json').read_text())
    from .front_alignment import validate_front_audit
    validate_front_audit(path,meta)
    if hashlib.sha256((path/'scene_mesh.glb').read_bytes()).hexdigest() != meta['mesh_file_sha256']:
        raise ValueError('Exported teacher mesh changed; regenerate references before training')
    geometry = np.load(path/'mesh_geometry.npz')
    if mesh_digest(geometry['vertices'], geometry['faces']) != meta['mesh_digest']:
        raise ValueError('Teacher mesh changed; regenerate references before training')
    quality = json.loads((path/'structure_quality.json').read_text())
    if not quality['passed']:
        raise ValueError('Teacher structure validation failed')
    points = np.load(path/'points.npz')
    triangles = geometry['vertices'][geometry['faces'][points['triangle_ids']]]
    bary = points['barycentric']
    expected = (triangles*bary[..., None]).sum(1)
    if not np.isfinite(points['xyz']).all() or not np.isfinite(bary).all() or (bary < -1e-6).any() or not np.allclose(bary.sum(1), 1., atol=1e-6):
        raise ValueError('Invalid teacher surface samples')
    if np.linalg.norm(points['xyz']-expected, axis=1).max() > 1e-5:
        raise ValueError('GS seeds no longer lie on their teacher triangles')
    from .geometry import lift
    vertices, faces = geometry['vertices'], geometry['faces']
    triangles = vertices[faces]
    normals = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    for frame in meta['frames']+meta['heldout_frames']:
        depth = np.load(path/frame['depth_file'])
        mask = np.load(path/frame['mask_file'])
        face_ids = np.load(path/frame['face_file'])
        if not mask.any() or not np.isfinite(depth[mask]).all() or (depth[mask] <= 0).any() or (face_ids[mask] < 0).any() or (face_ids[mask] >= len(faces)).any():
            raise ValueError('Invalid mesh reference depth or triangle IDs')
        xyz = lift(depth, np.asarray(meta['K']), np.asarray(frame['transform_matrix']))
        ids = face_ids[mask]
        residual = np.abs(((xyz[mask]-triangles[ids, 0])*normals[ids]).sum(1))
        if residual.max() > 1e-5:
            raise ValueError('Reference depth or camera no longer agrees with teacher mesh')
    return meta, points


def train_teacher(path, steps, max_points):
    if steps < 1 or max_points < 2:
        raise ValueError('Positive steps and at least two Gaussians required')
    path = Path(path)
    meta, source = validate_teacher(path)
    import torch
    from .runtime import validate_torch_location
    validate_torch_location(torch.__file__)
    from gsplat import rasterization
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for Gaussian training')
    torch.manual_seed(meta['seed'])
    def tensor(x):
        return torch.tensor(x, dtype=torch.float32, device='cuda')
    def cpu(x):
        return x.detach().cpu().numpy()
    ids = stratified_seed_ids(source['xyz'], max_points, meta['seed'])
    initial = source['xyz'][ids].copy()
    n = len(initial)
    # Overlapping tangential kernels cover the surface; their normal thickness
    # is bounded separately and never optimized into a raised floor/ceiling.
    spacing = cKDTree(initial).query(initial, k=min(7, n))[0][:, 1:].mean(1)
    radius = np.clip(spacing*.85, .008, .12)
    means = tensor(initial)
    quats = tensor(normal_quaternions(source['normals'][ids]))
    scales = tensor(np.log(np.column_stack([radius, radius, np.minimum(radius*.12, .006)])))
    appearance = torch.nn.ParameterDict({
        'opacities': torch.nn.Parameter(tensor(np.full(n, .0))),
        'sh0': torch.nn.Parameter(tensor((source['rgb'][ids]/255-.5)/.28209479177387814)[:, None]),
        'shN': torch.nn.Parameter(torch.zeros(n, 15, 3, device='cuda'))})
    optimizer = torch.optim.Adam([{'params': [appearance['opacities']], 'lr': .03},
                                  {'params': [appearance['sh0']], 'lr': .004},
                                  {'params': [appearance['shN']], 'lr': .0004}])
    K = tensor(meta['K'])[None]
    size = meta['size']
    frames = meta['frames']
    targets = [tensor(np.asarray(Image.open(path/f['file_path']).convert('RGB'))/255) for f in frames]
    masks = [tensor(np.load(path/f['mask_file'])).bool() for f in frames]
    depths = [tensor(np.load(path/f['depth_file'])) for f in frames]
    poses = [torch.linalg.inv(tensor(f['transform_matrix']))[None] for f in frames]
    def render(pose, degree):
        return rasterization(means=means, quats=quats, scales=scales.exp(),
            opacities=appearance['opacities'].sigmoid(), colors=torch.cat([appearance['sh0'], appearance['shN']], 1),
            sh_degree=degree, viewmats=pose, Ks=K, width=size, height=size, packed=False, render_mode='RGB+ED')
    log = []
    for step in range(steps):
        i = step % len(frames)
        degree = min(3, step//300)
        rendered, alpha, _ = render(poses[i], degree)
        mask = masks[i]
        rgb_loss = (rendered[0, ..., :3][mask]-targets[i][mask]).abs().mean()
        coverage_loss = -alpha[0, ..., 0][mask].clamp_min(1e-4).log().mean()
        supported = mask & (alpha[0, ..., 0].detach() > .1)
        depth_loss = (rendered[0, ..., 3][supported].clamp_min(.05).log()-depths[i][supported].clamp_min(.05).log()).abs().mean() if supported.any() else rgb_loss*0
        loss = rgb_loss+.05*coverage_loss+.1*depth_loss
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite teacher training loss')
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if step % 100 == 0 or step == steps-1:
            record = dict(step=step, rgb_l1=float(rgb_loss.detach()), depth_log_l1=float(depth_loss.detach()), coverage_loss=float(coverage_loss.detach()))
            log.append(record)
            print(json.dumps(record), flush=True)
    degree = min(3, (steps-1)//300)
    geometry_unchanged = np.array_equal(cpu(means), initial)
    if not geometry_unchanged:
        raise RuntimeError('Immutable Gaussian centers changed')
    fields = dict(means=cpu(means), quats=cpu(quats), scales=cpu(scales), **{key: cpu(value) for key, value in appearance.items()})
    values = np.concatenate([fields['means'], np.zeros_like(initial), fields['sh0'][:, 0],
        fields['shN'].transpose(0, 2, 1).reshape(n, 45), fields['opacities'][:, None], fields['scales'], fields['quats']], 1)
    write_gaussian_ply(path/'scene.ply', values, FULL_FIELDS)
    copyfile(path/'scene.ply', path/'scene_unity.ply')
    np.savez_compressed(path/'gaussians.npz', **fields)
    np.savez_compressed(path/'gaussian_surface.npz', seed_ids=ids, triangle_ids=source['triangle_ids'][ids], barycentric=source['barycentric'][ids])
    evaluation, previews = [], []
    for i, frame in enumerate(meta['heldout_frames']):
        with torch.no_grad():
            result, alpha, _ = render(torch.linalg.inv(tensor(frame['transform_matrix']))[None], degree)
        predicted, opacity = cpu(result[0, ..., :3]), cpu(alpha[0, ..., 0])
        target = np.asarray(Image.open(path/frame['file_path']))/255
        depth, mask = np.load(path/frame['depth_file']), np.load(path/frame['mask_file'])
        valid = mask & (opacity > .5)
        evaluation.append(dict(camera=i, rgb_l1=float(np.abs(predicted[mask]-target[mask]).mean()),
            alpha_below_half_on_target=float((opacity[mask] < .5).mean()),
            depth_relative_error=float((np.abs(cpu(result[0, ..., 3])[valid]-depth[valid])/depth[valid]).mean()) if valid.any() else None))
        image = Image.fromarray(np.uint8(np.clip(predicted, 0, 1)*255))
        image.save(path/f'heldout_render_{i:03d}.png')
        np.save(path/f'heldout_alpha_{i:03d}.npy', opacity)
        if len(previews) < 8:
            previews.append((Image.fromarray(np.uint8(target*255)), image))
    canvas = Image.new('RGB', (4*256, max(1, (len(previews)+3)//4)*512))
    for i, pair in enumerate(previews):
        for row, image in enumerate(pair):
            canvas.paste(image.resize((256, 256)), (i%4*256, i//4*512+row*256))
    canvas.save(path/'heldout_preview.png')
    training = dict(trainer='fixed_mesh_teacher_v8', steps=steps, initial_points=n, final_points=n,
        trainable=list(appearance.keys()), fixed=['means', 'quats', 'scales'], geometry_unchanged=geometry_unchanged,
        mesh_digest=meta['mesh_digest'], normal_sigma_max=float(np.exp(fields['scales'][:, 2]).max()),
        densification=False, sh_degree=degree, log=log,
        limitations=['Geometry follows procedural mesh; fine geometric detail is not learned from RGB', 'Baked appearance, not dynamically relightable'])
    (path/'training.json').write_text(json.dumps(training, indent=2))
    (path/'evaluation.json').write_text(json.dumps(dict(reference='heldout renders of same teacher mesh; not independent real-world ground truth', frames=evaluation), indent=2))
    print(f'Saved {n} mesh-attached Gaussians; geometry unchanged={geometry_unchanged}', flush=True)
