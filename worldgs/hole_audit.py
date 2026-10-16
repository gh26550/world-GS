"""Separate missing geometry, grazing-angle rejection and Gaussian transparency."""
import json
from pathlib import Path
import numpy as np
import trimesh
from PIL import Image
from .expansion import SharedRGBDRenderer


def audit_holes(path, output):
    import torch
    from gsplat import rasterization
    path, output = Path(path), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    meta = json.loads((path/'scene.json').read_text())
    mesh = trimesh.load(path/'scene_mesh.glb', force='mesh', process=False)
    renderer = SharedRGBDRenderer(mesh)
    source = Path(meta['observation_source'])
    source_meta = json.loads((source/'scene.json').read_text())
    keyframes = [source_meta['frames'][0]]+[f for f in source_meta['frames'] if 'generated_fraction' in f]
    canonical = [(f, source/f['mask_file'], f['file_path']) for f in keyframes]
    for f in meta['heldout_frames']:
        if not any(np.allclose(f['transform_matrix'], k['transform_matrix']) for k in keyframes):
            canonical.append((f, path/f['mask_file'], None))
    data = np.load(path/'gaussians.npz')
    def t(x):
        return torch.as_tensor(x, dtype=torch.float32, device='cuda')
    means, quats, scales, opacity = t(data['means']), t(data['quats']), t(data['scales']).exp(), t(data['opacities']).sigmoid()
    colors = torch.cat([t(data['sh0']), t(data['shN'])], 1)
    rows = []
    for i, (f, expected_file, source_name) in enumerate(canonical):
        pose = np.array(f['transform_matrix'])
        raw = renderer.render(np.array(meta['K']), pose, meta['size'], min_incidence=0.)
        gated = renderer.render(np.array(meta['K']), pose, meta['size'], min_incidence=meta.get('render_min_incidence', .18))
        hit, kept = raw['depth'] > 0, gated['depth'] > 0
        with torch.no_grad():
            _, alpha, _ = rasterization(means=means, quats=quats, scales=scales, opacities=opacity,
                colors=colors, sh_degree=3, viewmats=torch.linalg.inv(t(pose))[None], Ks=t(meta['K'])[None],
                width=meta['size'], height=meta['size'], packed=False, absgrad=True, render_mode='RGB+ED')
        a = alpha[0, ..., 0].cpu().numpy()
        expected = np.load(expected_file)
        row = dict(index=i, source_keyframe=source_name, mesh_hit_fraction=float(hit.mean()),
            grazing_rejected_fraction=float((hit & ~kept).mean()),
            mesh_missing_on_source_fraction=float((expected & ~hit).sum()/max(expected.sum(), 1)),
            gs_alpha_below_half_on_mesh=float(((a < .5) & hit).sum()/max(hit.sum(), 1)),
            gs_alpha_below_95_on_mesh=float(((a < .95) & hit).sum()/max(hit.sum(), 1)))
        rows.append(row)
        np.save(output/f'alpha_{i:03d}.npy', a)
        # Red=missing source geometry, yellow=grazing rejection, blue=GS opacity deficit.
        rgb = np.full((*hit.shape, 3), 235, np.uint8)
        rgb[expected & ~hit] = [220, 55, 55]
        rgb[hit & ~kept] = [240, 195, 30]
        rgb[hit & (a < .5)] = [40, 90, 230]
        Image.fromarray(rgb).save(output/f'causes_{i:03d}.png')
    report = dict(path=str(path), reference='source generated-depth validity, not real-world ground truth', frames=rows,
        mean_mesh_missing_on_source=float(np.mean([r['mesh_missing_on_source_fraction'] for r in rows[:len(keyframes)]])),
        mean_grazing_rejected_fraction=float(np.mean([r['grazing_rejected_fraction'] for r in rows])),
        mean_gs_alpha_below_half_on_mesh=float(np.mean([r['gs_alpha_below_half_on_mesh'] for r in rows])),
        mean_gs_alpha_below_95_on_mesh=float(np.mean([r['gs_alpha_below_95_on_mesh'] for r in rows])))
    (output/'audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != 'frames'}, indent=2), flush=True)
    return report
