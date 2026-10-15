"""Free 3D Gaussian fitting from actual multi-position mesh render references."""
import json
from pathlib import Path
from shutil import copyfile
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from .ply import FULL_FIELDS, write_gaussian_ply


def normal_quaternions(normals):
    q = np.column_stack([1+normals[:, 2], -normals[:, 1], normals[:, 0], np.zeros(len(normals))])
    q[normals[:, 2] < -.99999] = [0, 1, 0, 0]
    return q/np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-8)


def train_mesh(path, steps, max_points):
    import torch
    from .runtime import validate_torch_location
    validate_torch_location(torch.__file__)
    from gsplat import DefaultStrategy, rasterization
    from gsplat.strategy.ops import remove
    path = Path(path)
    meta = json.loads((path/'scene.json').read_text())
    torch.manual_seed(meta['seed'])
    if not torch.cuda.is_available() or steps < 1 or max_points < 1:
        raise ValueError('CUDA and positive steps/point limit required')
    def t(x):
        return torch.tensor(x, dtype=torch.float32, device='cuda')
    def cpu(x):
        return x.detach().cpu().numpy()
    source = np.load(path/'points.npz')
    ids = np.random.default_rng(meta['seed']).permutation(len(source['xyz']))[:max_points]
    initial = source['xyz'][ids]
    n = len(initial)
    distance = cKDTree(initial).query(initial, k=2)[0][:, 1]
    radius = np.clip(distance*.8, .008, .08)
    parameters = torch.nn.ParameterDict({
        'means': torch.nn.Parameter(t(initial)),
        'scales': torch.nn.Parameter(t(np.log(np.column_stack([radius, radius, radius*.5])))),
        'quats': torch.nn.Parameter(t(normal_quaternions(source['normals'][ids]))),
        'opacities': torch.nn.Parameter(t(np.full(n, -1.4))),
        'sh0': torch.nn.Parameter(t((source['rgb'][ids]/255-.5)/.28209479177387814)[:, None]),
        'shN': torch.nn.Parameter(torch.zeros(n, 15, 3, device='cuda'))})
    learning_rates = {'means': .0003, 'scales': .005, 'quats': .001, 'opacities': .03, 'sh0': .004, 'shN': .0004}
    optimizers = {key: torch.optim.Adam([value], lr=learning_rates[key], eps=1e-15) for key, value in parameters.items()}
    strategy = DefaultStrategy(refine_start_iter=150, refine_stop_iter=min(steps-100, 1200), refine_every=100,
        reset_every=5000, grow_grad2d=.0005, prune_opa=.01, absgrad=True, verbose=True)
    strategy.check_sanity(parameters, optimizers)
    state = strategy.initialize_state(scene_scale=4.)
    K = t(meta['K'])[None]
    size = meta['size']
    frames = meta['frames']
    targets = [t(np.asarray(Image.open(path/f['file_path']))/255)[None] for f in frames]
    depths = [t(np.load(path/f['depth_file'])) for f in frames]
    masks = [t(np.load(path/f['mask_file'])).bool() for f in frames]
    poses = [torch.linalg.inv(t(f['transform_matrix']))[None] for f in frames]
    tree = cKDTree(source['xyz'])
    anchors, normals = None, None
    last_anchor_step = -100
    degree = 0
    def render(view):
        return rasterization(means=parameters['means'], quats=parameters['quats'], scales=parameters['scales'].exp(),
            opacities=parameters['opacities'].sigmoid(), colors=torch.cat([parameters['sh0'], parameters['shN']], 1),
            sh_degree=degree, viewmats=view, Ks=K, width=size, height=size, packed=False, absgrad=True, render_mode='RGB+ED')
    log = []
    for step in range(steps):
        degree = min(3, step//300)
        count = len(parameters['means'])
        if anchors is None or len(anchors) != count or step-last_anchor_step >= 100:
            nearest = tree.query(cpu(parameters['means']), workers=2)[1]
            anchors, normals = t(source['xyz'][nearest]), t(source['normals'][nearest])
            last_anchor_step = step
        i = step % len(frames)
        rendered, alpha, info = render(poses[i])
        valid = masks[i]
        rgb_loss = (rendered[0, ..., :3][valid]-targets[i][0][valid]).abs().mean()
        supported = valid & (alpha[0, ..., 0].detach() > .1)
        depth_loss = ((rendered[0, ..., 3].clamp_min(.05).log()-depths[i].clamp_min(.05).log()).abs()[supported].mean()
                      if supported.any() else rgb_loss*0)
        # Soft guidance with tolerance. No projection, locked plane, fixed rotation or normal thickness.
        offset = ((parameters['means']-anchors)*normals).sum(-1).abs()
        prior = torch.relu(offset-.04).mean()
        loss = rgb_loss+.1*depth_loss+.1*prior
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite mesh GS training loss')
        strategy.step_pre_backward(parameters, optimizers, state, step, info)
        loss.backward()
        for optimizer in optimizers.values():
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        strategy.step_post_backward(parameters, optimizers, state, step, info, packed=False)
        if step > strategy.refine_start_iter and step % strategy.refine_every == 0:
            anchors = None
        with torch.no_grad():
            parameters['scales'].clamp_(np.log(.0015), np.log(.25))
        if len(parameters['means']) > 180000:
            with torch.no_grad():
                top = parameters['opacities'].topk(180000).indices
                drop = torch.ones(len(parameters['means']), device='cuda', dtype=torch.bool)
                drop[top] = False
                remove(parameters, optimizers, state, drop)
            strategy.refine_stop_iter = step
        if step % 100 == 0 or step == steps-1:
            record = {'step': step, 'points': len(parameters['means']), 'rgb_l1': float(rgb_loss.detach()), 'depth_log_l1': float(depth_loss.detach()), 'soft_mesh_penalty': float(prior.detach()), 'sh_degree': degree}
            log.append(record)
            print(json.dumps(record), flush=True)
    degree = min(3, (steps-1)//300)
    means, scales = cpu(parameters['means']), cpu(parameters['scales'])
    q = cpu(torch.nn.functional.normalize(parameters['quats'], dim=-1))
    values = np.concatenate([means, np.zeros_like(means), cpu(parameters['sh0'])[:, 0],
        cpu(parameters['shN']).transpose(0, 2, 1).reshape(len(means), 45), cpu(parameters['opacities'])[:, None], scales, q], 1)
    write_gaussian_ply(path/'scene.ply', values, FULL_FIELDS)
    copyfile(path/'scene.ply', path/'scene_unity.ply')
    np.savez_compressed(path/'gaussians.npz', **{key: cpu(value) for key, value in parameters.items()})
    evaluation = []
    previews = []
    for i, frame in enumerate(meta['heldout_frames']):
        with torch.no_grad():
            result, alpha, _ = render(torch.linalg.inv(t(frame['transform_matrix']))[None])
        predicted = cpu(result[0, ..., :3])
        target = np.asarray(Image.open(path/frame['file_path']))/255
        depth = np.load(path/frame['depth_file'])
        mask = np.load(path/frame['mask_file'])
        valid = mask & (cpu(alpha[0, ..., 0]) > .5)
        evaluation.append({'camera': i, 'target': frame['target'], 'valid_fraction': float(valid.mean()),
            'rgb_l1': float(np.abs(predicted[mask]-target[mask]).mean()),
            'depth_relative_error': float((np.abs(cpu(result[0, ..., 3])[valid]-depth[valid])/depth[valid]).mean()) if valid.any() else None})
        image = Image.fromarray(np.uint8(np.clip(predicted, 0, 1)*255))
        image.save(path/f'heldout_render_{i:03d}.png')
        if len(previews) < 8:
            previews.append((Image.fromarray(np.uint8(target*255)), image))
    canvas = Image.new('RGB', (4*256, ((len(previews)+3)//4)*2*256))
    for i, (target, prediction) in enumerate(previews):
        x, y = i % 4*256, i//4*512
        canvas.paste(target.resize((256, 256)), (x, y))
        canvas.paste(prediction.resize((256, 256)), (x, y+256))
    canvas.save(path/'heldout_preview.png')
    nearest = tree.query(means, workers=2)[0]
    training = {'trainer': 'free_3dgs_mesh_references_v3', 'steps': steps, 'initial_points': n, 'final_points': len(means),
        'trainable': list(parameters.keys()), 'hard_projection': False, 'densification': True, 'sh_degree': degree,
        'nearest_source_distance_median': float(np.median(nearest)), 'nearest_source_distance_p95': float(np.quantile(nearest, .95)), 'log': log}
    (path/'training.json').write_text(json.dumps(training, indent=2), encoding='utf-8')
    (path/'evaluation.json').write_text(json.dumps({'reference': 'heldout triangle mesh renders; not real-world ground truth', 'frames': evaluation}, indent=2), encoding='utf-8')
    print(f'Saved {len(means)} freely optimized Gaussians; SH degree {degree}', flush=True)
