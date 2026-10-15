import json
from pathlib import Path
import numpy as np
from PIL import Image


def train(path, steps=1000, max_points=30000):
    import torch
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
    colors = torch.nn.Parameter(torch.logit(tensor(points['rgb'][ids]/255).clamp(.001, .999)))
    scales = torch.nn.Parameter(torch.full_like(means, -3.5))
    quats = torch.nn.Parameter(tensor(np.tile([1, 0, 0, 0], (len(ids), 1))))
    opacity = torch.nn.Parameter(torch.zeros(len(ids), device='cuda'))
    optimizer = torch.optim.Adam([{'params': [means], 'lr': .0002}, {'params': [colors, scales, quats, opacity], 'lr': .005}])
    K = tensor(meta['K'])[None]
    size = meta['size']
    losses = []
    for step in range(steps):
        frame = meta['frames'][step % len(meta['frames'])]
        target = tensor(np.array(Image.open(path/frame['file_path']))/255)[None]
        view = torch.linalg.inv(tensor(frame['transform_matrix']))[None]
        rendered, _, _ = rasterization(means=means, quats=quats, scales=scales.exp(), opacities=opacity.sigmoid(), colors=colors.sigmoid(), viewmats=view, Ks=K, width=size, height=size, packed=False)
        photometric = (rendered-target).abs().mean()
        loss = photometric + .01*(means-initial).square().mean()
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite training loss')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            scales.clamp_(-7, 0)
        losses.append(float(photometric.detach()))
        if step % 100 == 0:
            print(f'{step}/{steps} L1={losses[-1]:.5f}', flush=True)
    def cpu(t):
        return t.detach().cpu().numpy()
    torch.save({'means': cpu(means), 'log_scales': cpu(scales), 'quats': cpu(quats), 'opacity_logits': cpu(opacity), 'color_logits': cpu(colors)}, path/'checkpoint.pt')
    # Standard 3DGS degree-0 SH PLY: scales and opacity remain logits/log-space.
    fields = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity', 'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3']
    normalized = torch.nn.functional.normalize(quats, dim=-1)
    values = np.concatenate([cpu(means), np.zeros((len(ids), 3)), (cpu(colors.sigmoid())-.5)/.28209479177387814, cpu(opacity)[:, None], cpu(scales), cpu(normalized)], -1)
    with open(path/'scene.ply', 'w') as f:
        f.write('ply\nformat ascii 1.0\nelement vertex '+str(len(ids))+'\n')
        for name in fields:
            f.write('property float '+name+'\n')
        f.write('end_header\n')
        np.savetxt(f, values, fmt='%.7g')
    Image.fromarray(np.uint8(np.clip(cpu(rendered[0]), 0, 1)*255)).save(path/'train_preview.png')
    (path/'training.json').write_text(json.dumps({'steps': steps, 'points': len(ids), 'first_l1': losses[0], 'last_l1': losses[-1], 'demo': meta['demo'], 'note': 'fixed point count, SH0; no densification; losses are training-view errors'}), encoding='utf-8')
