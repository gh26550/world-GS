"""Free 3D Gaussian fitting from actual multi-position mesh render references."""
import json
import re
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


def confidence_loss(error, confidence):
    # Normalize by pixel count, NOT weight sum: an entirely provisional view must
    # have less influence than a fully supported one, not regain full weight.
    return (error*confidence).sum()/max(error.numel(), 1)


def foreground_coverage_loss(alpha, confidence, mask):
    return confidence_loss(-alpha[mask].clamp(1e-4, 1.).log(), confidence[mask])


def train_mesh(path, steps, max_points):
    path = Path(path)
    meta = json.loads((path/'scene.json').read_text())
    prompt_mesh = meta.get('mode') in ('prompt_mesh_radiance_v9','learned_mesh_v10')
    expanded = prompt_mesh or meta.get('mode') in ('progressive_rgbd_v4', 'progressive_rgbd_v5', 'global_tsdf_v6', 'global_tsdf_v7')
    coverage_mode = prompt_mesh or meta.get('mode')=='global_tsdf_v7'
    if meta.get('mode') in ('global_tsdf_v6', 'global_tsdf_v7'):
        quality = json.loads((path/'mesh_quality.json').read_text())
        if not quality.get('passed') or quality.get('seed_to_assigned_surface_max', 1.) > 1e-4:
            raise ValueError('Mesh or GS seed quality check failed; rebuild before training')
    import torch
    from .runtime import validate_torch_location
    validate_torch_location(torch.__file__)
    from gsplat import DefaultStrategy, rasterization
    from gsplat.strategy.ops import remove
    torch.manual_seed(meta['seed'])
    if not torch.cuda.is_available() or steps < 1 or max_points < 1:
        raise ValueError('CUDA and positive steps/point limit required')
    def t(x):
        return torch.tensor(x, dtype=torch.float32, device='cuda')
    def cpu(x):
        return x.detach().cpu().numpy()
    source = np.load(path/'points.npz')
    matte_components=[]
    if meta.get('mode')=='learned_mesh_v10':
        room=json.loads((path/'render_job.json').read_text())['plan']['room']
        from .physical_render import matte_room_finish
        if matte_room_finish(room['walls']):
            matte_components.extend([0,1,4,5])
        if matte_room_finish(room['ceiling']):
            matte_components.append(2)
    from .coverage_sampling import stratified_seed_ids, coverage_drop_mask,component_stratified_seed_ids
    ids = stratified_seed_ids(source['xyz'], max_points, meta['seed']) if coverage_mode else np.random.default_rng(meta['seed']).permutation(len(source['xyz']))[:max_points]
    if meta.get('mode')=='learned_mesh_v10':
        # Preserve small objects when reducing the area-weighted cloud.
        ids=component_stratified_seed_ids(source['xyz'],source['component_ids'],max_points,meta['seed'])
    initial = source['xyz'][ids]
    n = len(initial)
    distance = cKDTree(initial).query(initial, k=2)[0][:, 1]
    radius = np.clip(distance*.8, .008, .08)
    if meta.get('mode')=='learned_mesh_v10':
        neighbours=cKDTree(initial).query(initial,k=min(7,n))[0][:,1:]
        radius=np.clip(neighbours.mean(1)*.85,.002,.12)
    normal_radius=np.minimum(radius*.25,.01) if meta.get('mode')=='learned_mesh_v10' else radius*.5
    parameters = torch.nn.ParameterDict({
        'means': torch.nn.Parameter(t(initial)),
        'scales': torch.nn.Parameter(t(np.log(np.column_stack([radius, radius, normal_radius])))),
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
    confidences = [t(np.load(path/f['confidence_file'])) if f.get('confidence_file') else torch.ones_like(masks[i], dtype=torch.float32)
                   for i, f in enumerate(frames)]
    poses = [torch.linalg.inv(t(f['transform_matrix']))[None] for f in frames]
    tree = cKDTree(source['xyz'])
    anchors, normals = None, None
    matte_mask,anchor_colors=None,None
    last_anchor_step = -100
    degree = 0
    def render(view):
        shN=parameters['shN']
        if matte_mask is not None:
            shN=shN*(~matte_mask)[:,None,None]
        return rasterization(means=parameters['means'], quats=parameters['quats'], scales=parameters['scales'].exp(),
            opacities=parameters['opacities'].sigmoid(), colors=torch.cat([parameters['sh0'], shN], 1),
            sh_degree=degree, viewmats=view, Ks=K, width=size, height=size, packed=False, absgrad=True, render_mode='RGB+ED')
    log = []
    for step in range(steps):
        degree = min(3, step//300)
        count = len(parameters['means'])
        if (not expanded or prompt_mesh) and (anchors is None or len(anchors) != count or step-last_anchor_step >= 100):
            nearest = tree.query(cpu(parameters['means']), workers=2)[1]
            anchors, normals = t(source['xyz'][nearest]), t(source['normals'][nearest])
            if matte_components:
                matte_mask=t(np.isin(source['component_ids'][nearest],matte_components)).bool()
                anchor_colors=t((source['rgb'][nearest]/255-.5)/.28209479177387814)
            last_anchor_step = step
        i = step % len(frames)
        rendered, alpha, info = render(poses[i])
        valid = masks[i]
        weight = confidences[i][valid]
        rgb_loss = confidence_loss((rendered[0, ..., :3][valid]-targets[i][0][valid]).abs().mean(-1), weight)
        supported = valid & (alpha[0, ..., 0].detach() > .1)
        depth_loss = (confidence_loss((rendered[0, ..., 3].clamp_min(.05).log()-depths[i].clamp_min(.05).log()).abs()[supported], confidences[i][supported])
                      if supported.any() else rgb_loss*0)
        # Soft guidance with tolerance. No projection, locked plane, fixed rotation or normal thickness.
        if prompt_mesh:
            delta = parameters['means']-anchors
            offset = (delta*normals).sum(-1).abs()
            structural = t(source['component_ids'][nearest] < 6).bool()
            tolerance = torch.where(structural, .015, .06)
            prior = torch.relu(offset-tolerance).mean()+.1*torch.relu(delta.norm(dim=-1)-.15).mean()
            # Normal thickness is softly bounded on room structure. Rotation,
            # covariance and furniture geometry are still trainable.
            q = torch.nn.functional.normalize(parameters['quats'],dim=-1)
            local = normals+2*torch.linalg.cross(q[:,1:],torch.linalg.cross(q[:,1:],normals)-q[:,:1]*normals)
            sigma = ((local.square()*parameters['scales'].exp().square()).sum(-1)+1e-12).sqrt()
            prior = prior+torch.relu(sigma[structural]-.015).mean() if structural.any() else prior
        elif expanded:
            prior = rgb_loss*0
        else:
            offset = ((parameters['means']-anchors)*normals).sum(-1).abs()
            prior = torch.relu(offset-.04).mean()
        # Generated-depth scenes have no constraint to their initial scaffold.
        # Predicted depth is an uncertain cue; reduce its weight during fitting.
        depth_weight = .2 if prompt_mesh else .1*(1-.8*step/max(steps-1, 1)) if expanded else .1
        coverage_loss = foreground_coverage_loss(alpha[0, ..., 0], confidences[i], valid) if coverage_mode else rgb_loss*0
        loss = rgb_loss+depth_weight*depth_loss+(.3 if prompt_mesh else 0. if expanded else .1)*prior+.05*coverage_loss
        if matte_mask is not None and matte_mask.any():
            # Rough plaster is approximately view-independent. Do not invent
            # coloured high-order lobes to fit low-texture room surfaces.
            loss=loss+.03*(parameters['sh0'][matte_mask,0]-anchor_colors[matte_mask]).abs().mean()
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
                if coverage_mode:
                    drop = coverage_drop_mask(parameters['means'], parameters['opacities'], 180000)
                else:
                    top = parameters['opacities'].topk(180000).indices
                    drop = torch.ones(len(parameters['means']), device='cuda', dtype=torch.bool)
                    drop[top] = False
                remove(parameters, optimizers, state, drop)
            strategy.refine_stop_iter = step
        if step % 100 == 0 or step == steps-1:
            record = {'step': step, 'points': len(parameters['means']), 'rgb_l1': float(rgb_loss.detach()), 'depth_log_l1': float(depth_loss.detach()), 'coverage_loss': float(coverage_loss.detach()), 'soft_mesh_penalty': float(prior.detach()), 'sh_degree': degree}
            log.append(record)
            print(json.dumps(record), flush=True)
    degree = min(3, (steps-1)//300)
    if matte_components:
        nearest=tree.query(cpu(parameters['means']),workers=2)[1]
        matte_mask=t(np.isin(source['component_ids'][nearest],matte_components)).bool()
        with torch.no_grad():
            parameters['shN'][matte_mask]=0
    means, scales = cpu(parameters['means']), cpu(parameters['scales'])
    q = cpu(torch.nn.functional.normalize(parameters['quats'], dim=-1))
    values = np.concatenate([means, np.zeros_like(means), cpu(parameters['sh0'])[:, 0],
        cpu(parameters['shN']).transpose(0, 2, 1).reshape(len(means), 45), cpu(parameters['opacities'])[:, None], scales, q], 1)
    write_gaussian_ply(path/'scene.ply', values, FULL_FIELDS)
    copyfile(path/'scene.ply', path/'scene_unity.ply')
    np.savez_compressed(path/'gaussians.npz', **{key: cpu(value) for key, value in parameters.items()})
    evaluation = []
    with torch.inference_mode():
        front_result,_,_=render(poses[0])
    Image.fromarray(np.uint8(np.clip(cpu(front_result[0,...,:3]),0,1)*255)).save(path/'front_render.png')
    previews = []
    for i, frame in enumerate(meta['heldout_frames']):
        with torch.no_grad():
            result, alpha, _ = render(torch.linalg.inv(t(frame['transform_matrix']))[None])
        predicted = cpu(result[0, ..., :3])
        target = np.asarray(Image.open(path/frame['file_path']))/255
        depth = np.load(path/frame['depth_file'])
        mask = np.load(path/frame['mask_file'])
        valid = mask & (cpu(alpha[0, ..., 0]) > .5)
        np.save(path/f'heldout_alpha_{i:03d}.npy', cpu(alpha[0, ..., 0]))
        evaluation.append({'camera': i, 'target': frame['target'], 'valid_fraction': float(valid.mean()),
            'rgb_l1': float(np.abs(predicted[mask]-target[mask]).mean()),
            'alpha_below_half_on_target': float(((cpu(alpha[0, ..., 0]) < .5) & mask).sum()/max(mask.sum(), 1)),
            'alpha_below_95_on_target': float(((cpu(alpha[0, ..., 0]) < .95) & mask).sum()/max(mask.sum(), 1)),
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
    training = {'trainer': meta['mode'] if prompt_mesh else ('coverage_3dgs_v7' if coverage_mode else 'fused_tsdf_3dgs_v6' if meta.get('mode') == 'global_tsdf_v6' else 'generated_rgbd_3dgs_v5' if meta.get('mode') == 'progressive_rgbd_v5' else 'generated_rgbd_3dgs_v4') if expanded else 'free_3dgs_mesh_references_v3', 'steps': steps, 'initial_points': n, 'final_points': len(means),
        'coverage_loss_weight': .05 if coverage_mode else 0., 'spatially_stratified_sampling': coverage_mode, 'spatial_budget_pruning': coverage_mode,
        'mesh_prior_weight': .3 if prompt_mesh else 0. if expanded else .1, 'depth_weight_schedule': [.2, .2] if prompt_mesh else [.1, .02] if expanded else [.1, .1],
        'trainable': list(parameters.keys()), 'hard_projection': False, 'densification': True, 'sh_degree': degree,
        'matte_room_components':matte_components,'matte_room_sh_degree':0 if matte_components else None,
        'nearest_source_distance_median': float(np.median(nearest)), 'nearest_source_distance_p95': float(np.quantile(nearest, .95)), 'log': log}
    (path/'training.json').write_text(json.dumps(training, indent=2), encoding='utf-8')
    (path/'evaluation.json').write_text(json.dumps({'reference': 'heldout physically rendered UV-mesh views; not independent real-world ground truth' if meta.get('mode')=='learned_mesh_v10' else 'heldout shared photographic surface renders on the same prompt mesh; not independent ground truth' if prompt_mesh else 'heldout updated generated RGBD geometry renders; not independent ground truth' if expanded else 'heldout triangle mesh renders; not real-world ground truth', 'frames': evaluation}, indent=2), encoding='utf-8')
    print(f'Saved {len(means)} freely optimized Gaussians; SH degree {degree}', flush=True)
