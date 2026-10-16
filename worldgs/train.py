import json
from pathlib import Path
import numpy as np
from PIL import Image


def train(path, steps=1000, max_points=100000):
    meta_file = Path(path)/'scene.json'
    if json.loads(meta_file.read_text()).get('mode') in ('mesh_components_v3', 'progressive_rgbd_v4', 'progressive_rgbd_v5', 'global_tsdf_v6', 'global_tsdf_v7'):
        from .train_mesh import train_mesh
        return train_mesh(path, steps, max_points)
    import torch
    from .runtime import validate_torch_location
    validate_torch_location(torch.__file__)
    from gsplat import rasterization
    if steps < 1 or max_points < 1:
        raise ValueError('steps and max_points must be positive')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU is required for gsplat training')
    path = Path(path)
    meta = json.loads((path/'scene.json').read_text())
    torch.manual_seed(meta['seed'])
    points = np.load(path/'points.npz')
    ids = np.random.default_rng(meta['seed']).permutation(len(points['xyz']))[:max_points]
    if not len(ids):
        raise ValueError('No initial points')
    def tensor(a):
        return torch.tensor(a, dtype=torch.float32, device='cuda')
    means = torch.nn.Parameter(tensor(points['xyz'][ids]))
    initial = means.detach().clone()
    constrained = meta.get('geometry_constraint') == 'axis_aligned_surface'
    normals = tensor(points['normals'][ids]) if constrained else None
    normal_axes = normals.abs().argmax(-1) if constrained else None
    row_ids = torch.arange(len(ids), device='cuda')
    bounds_min = tensor(points['bounds_min'][ids]) if constrained else None
    bounds_max = tensor(points['bounds_max'][ids]) if constrained else None
    colors = torch.nn.Parameter(torch.logit(tensor(points['rgb'][ids]/255).clamp(.001, .999)))
    scales = torch.nn.Parameter(torch.full_like(means, -3.5))
    quats = torch.nn.Parameter(tensor(np.tile([1, 0, 0, 0], (len(ids), 1))), requires_grad=not constrained)
    opacity = torch.nn.Parameter(torch.zeros(len(ids), device='cuda'))
    optimizer = torch.optim.Adam([{'params': [means], 'lr': .0002}, {'params': [colors, scales, quats, opacity], 'lr': .005}])
    K = tensor(meta['K'])[None]
    size = meta['size']
    losses = []
    depth_losses = []
    targets = [tensor(np.array(Image.open(path/f['file_path']))/255)[None] for f in meta['frames']]
    views = [torch.linalg.inv(tensor(f['transform_matrix']))[None] for f in meta['frames']]
    depths = [tensor(np.load(path/f'depth_{i:03d}.npy')) for i in range(len(meta['frames']))] if constrained else None
    def render(view):
        return rasterization(means=means, quats=quats, scales=scales.exp(), opacities=opacity.sigmoid(),
            colors=colors.sigmoid(), viewmats=view, Ks=K, width=size, height=size, packed=False,
            render_mode='RGB+ED' if constrained else 'RGB')
    for step in range(steps):
        index = step % len(meta['frames'])
        rendered, alpha, _ = render(views[index])
        photometric = (rendered[..., :3]-targets[index]).abs().mean()
        depth_loss = means.new_tensor(0.)
        if constrained:
            mask = alpha[0, ..., 0].detach() > .1
            depth_error = (rendered[0, ..., 3].clamp_min(.05).log()-depths[index].log()).abs()
            depth_loss = depth_error[mask].mean() if mask.any() else depth_error.mean()*0
        loss = photometric + .1*depth_loss + .01*(means-initial).square().mean()
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite training loss')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            scales.clamp_(-7, np.log(.12) if constrained else 0)
            if constrained:
                means.copy_(torch.maximum(bounds_min, torch.minimum(bounds_max, means)))
                means.sub_(((means-initial)*normals).sum(-1, keepdim=True)*normals)
                scales[row_ids, normal_axes] = np.log(.008)
        losses.append(float(photometric.detach()))
        depth_losses.append(float(depth_loss.detach()))
        if step % 100 == 0:
            print(f'{step}/{steps} L1={losses[-1]:.5f} depth={depth_losses[-1]:.5f}', flush=True)
    def cpu(t):
        return t.detach().cpu().numpy()
    torch.save({'means': cpu(means), 'log_scales': cpu(scales), 'quats': cpu(quats), 'opacity_logits': cpu(opacity), 'color_logits': cpu(colors)}, path/'checkpoint.pt')
    # Standard 3DGS degree-0 SH PLY: scales and opacity remain logits/log-space.
    from .ply import write_gaussian_ply
    normalized = torch.nn.functional.normalize(quats, dim=-1)
    values = np.concatenate([cpu(means), np.zeros((len(ids), 3)), (cpu(colors.sigmoid())-.5)/.28209479177387814, cpu(opacity)[:, None], cpu(scales), cpu(normalized)], -1)
    write_gaussian_ply(path/'scene.ply', values)
    from shutil import copyfile
    copyfile(path/'scene.ply', path/'scene_unity.ply')
    with torch.no_grad():
        preview, _, _ = render(views[0])
    Image.fromarray(np.uint8(np.clip(cpu(preview[0, ..., :3]), 0, 1)*255)).save(path/'train_preview.png')
    metrics = {'steps': steps, 'points': len(ids), 'first_l1': losses[0], 'last_l1': losses[-1], 'last_depth_log_l1': depth_losses[-1],
               'geometry_constrained': constrained, 'demo': meta['demo'], 'note': 'fixed point count, SH0; no densification; losses are training-view errors'}
    if constrained:
        metrics['max_surface_normal_displacement'] = float(((means-initial)*normals).sum(-1).abs().max().detach())
    (path/'training.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    if constrained:
        evaluate(meta, path, render, tensor, cpu, torch)


def evaluate(meta, path, render, tensor, cpu, torch):
    """Held-out camera reprojection against the proxy, not real-world ground truth."""
    images, records = [], []
    for i, frame in enumerate(meta['heldout_frames']):
        with torch.no_grad():
            prediction, alpha, _ = render(torch.linalg.inv(tensor(frame['transform_matrix']))[None])
        rgb = cpu(prediction[0, ..., :3])
        target = np.asarray(Image.open(path/frame['file_path']))/255
        expected = np.load(path/f'heldout_depth_{i:03d}.npy')
        observed = np.load(path/f'heldout_mask_{i:03d}.npy')
        valid = (cpu(alpha[0, ..., 0]) > .5) & observed
        predicted_depth = cpu(prediction[0, ..., 3])
        records.append({'camera': i, 'evaluated_fraction': float(valid.mean()), 'rgb_l1_observed': float(np.abs(rgb[observed]-target[observed]).mean()) if observed.any() else None,
                        'depth_relative_error': float((np.abs(predicted_depth[valid]-expected[valid])/expected[valid]).mean()) if valid.any() else None})
        predicted = Image.fromarray(np.uint8(np.clip(rgb, 0, 1)*255))
        predicted.save(path/f'heldout_render_{i:03d}.png')
        images.append((Image.fromarray(np.uint8(target*255)), predicted))
    canvas = Image.new('RGB', (len(images)*256, 512))
    for i, (target, prediction) in enumerate(images):
        canvas.paste(target.resize((256, 256)), (i*256, 0))
        canvas.paste(prediction.resize((256, 256)), (i*256, 256))
    canvas.save(path/'heldout_preview.png')
    (path/'evaluation.json').write_text(json.dumps({'reference': 'proxy atlas; not independent ground truth', 'cameras_excluded_from_training': True, 'frames': records}, indent=2), encoding='utf-8')
