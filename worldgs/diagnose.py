"""Inspect existing geometry and training data without GPU or regeneration."""
import argparse
import json
from pathlib import Path
import numpy as np
from .ply import read_gaussian_ply


def quantiles(values):
    return dict(zip(['min', 'p05', 'median', 'p95', 'max'], map(float, np.quantile(values, [0, .05, .5, .95, 1]))))


def mesh_stats(path):
    vertices = 0
    faces = []
    with path.open() as stream:
        for line in stream:
            if line.startswith('v '):
                vertices += 1
            elif line.startswith('f '):
                faces.append([int(x.split('/')[0])-1 for x in line.split()[1:]])
    triangles = np.asarray(faces, dtype=np.int64)
    edges = np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]])
    _, counts = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
    return {'path': str(path.resolve()), 'vertices': vertices, 'faces': len(faces),
            'referenced_vertices': int(len(np.unique(triangles))), 'boundary_edges': int((counts == 1).sum()),
            'nonmanifold_edges': int((counts > 2).sum()), 'closed_surface': bool((counts == 2).all())}


def diagnose(folder, plots=False):
    folder = Path(folder).resolve()
    out = folder/'diagnostics'
    out.mkdir(exist_ok=True)
    meta = json.loads((folder/'scene.json').read_text())
    training = json.loads((folder/'training.json').read_text())
    poses = np.asarray([frame['transform_matrix'] for frame in meta['frames']])
    centers = poses[:, :3, 3]
    baseline = float(np.linalg.norm(centers[:, None]-centers[None], axis=-1).max())
    points = np.load(folder/'points.npz')
    fields, final = read_gaussian_ply(folder/'scene.ply')
    xyz = final[:, [fields.index(x) for x in ['x', 'y', 'z']]]
    ids = np.random.default_rng(meta['seed']).permutation(len(points['xyz']))[:len(xyz)]
    initial = points['xyz'][ids]
    radius = np.linalg.norm(initial, axis=1)
    displacement = np.linalg.norm(xyz-initial, axis=1)
    scales = np.exp(final[:, [fields.index(f'scale_{i}') for i in range(3)]])
    opacity = 1/(1+np.exp(-np.clip(final[:, fields.index('opacity')], -80, 80)))
    visits = np.bincount(np.arange(training['steps']) % len(poses), minlength=len(poses))
    K = np.asarray(meta['K'])
    fov = float(np.rad2deg(2*np.arctan(meta['size']/(2*K[1, 1]))))
    # Demonstrate radial-depth ambiguity for a point visible in views 0 and 1.
    point = np.array([.1, .2, 3.])
    errors = []
    for pose in poses[:2]:
        def project(p):
            cam = (p-pose[:3, 3]) @ pose[:3, :3]
            pixel = cam @ K.T
            return pixel[:2]/pixel[2]
        for scale in [.5, 2., 4.]:
            errors.append(float(np.linalg.norm(project(point*scale)-project(point))))
    elevation = np.rad2deg(np.arctan2(initial[:, 1], np.linalg.norm(initial[:, [0, 2]], axis=1)))
    depths = []
    for index, frame in enumerate(meta['frames']):
        d = np.load(folder/f'depth_{index:03d}.npy')
        depths.append({'yaw': frame['yaw'], 'generated_fraction': frame['generated_fraction'], 'camera_z': quantiles(d)})
    report = {
        'scene': str(folder), 'frames': len(poses), 'camera_centers': centers.tolist(),
        'max_camera_baseline_scene_units': baseline, 'training_visits_per_view': visits.tolist(),
        'vertical_fov_degrees_assumed': fov,
        'ideal_max_yaw_only_spherical_area_fraction': float(np.sin(np.deg2rad(fov/2))),
        'radial_rescale_reprojection_difference_pixels': max(errors),
        'initial_point_radius': quantiles(radius), 'point_displacement_after_training': quantiles(displacement),
        'displacement_over_initial_radius': quantiles(displacement/np.maximum(radius, 1e-8)),
        'largest_gaussian_scale': quantiles(scales.max(axis=1)),
        'gaussian_scale_anisotropy': quantiles(scales.max(axis=1)/scales.min(axis=1)),
        'largest_gaussian_scale_over_camera_distance': quantiles(scales.max(axis=1)/np.maximum(np.linalg.norm(xyz, axis=1), 1e-8)),
        'gaussian_opacity': quantiles(opacity),
        'initial_elevation_degrees': quantiles(elevation), 'per_view_depth': depths,
        'initial_mesh': mesh_stats(folder/'proxy_front.obj'), 'expanded_mesh': mesh_stats(folder/'proxy_scene.obj'),
        'training': training,
        'notes': ['Scene units are arbitrary, not meters.', 'The projection test concerns point centers, not complete Gaussian images.',
                  'Spherical coverage is an ideal pinhole bound, not measured surface coverage.',
                  'Initial/final point correspondence assumes the current fixed-count trainer with the stored seed.',
                  'Unity camera pose and transforms were not measured.']}
    (out/'geometry_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    if plots:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from PIL import Image
        rng = np.random.default_rng(0)
        selected = rng.choice(len(xyz), min(25000, len(xyz)), replace=False)
        color = points['rgb'][ids][selected]/255
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharex=True, sharey=True)
        for ax, cloud, title in zip(axes, [initial, xyz], ['Before GS optimization', 'After GS optimization']):
            ax.scatter(cloud[selected, 0], cloud[selected, 2], c=color, s=.5, rasterized=True)
            ax.scatter(centers[:, 0], centers[:, 2], marker='x', c='red', s=80, label=f'Camera centers ({len(poses)} views)')
            ax.set(title=title, xlabel='World x (arbitrary units)', ylabel='World z (arbitrary units)')
            ax.set_aspect('equal')
            ax.legend(loc='upper left')
        fig.suptitle('Top view: point geometry before and after training; not an occupancy map')
        fig.tight_layout()
        fig.savefig(out/'initial_vs_trained_top.png', dpi=160)
        plt.close(fig)
        depth = np.load(folder/'front_depth.npy')
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        axes[0].imshow(Image.open(folder/'front.png'))
        axes[0].set_title('Generated front image')
        axes[0].axis('off')
        im = axes[1].imshow(depth, cmap='viridis')
        axes[1].set_title('Initial depth: arbitrary camera-z units')
        axes[1].axis('off')
        fig.colorbar(im, ax=axes[1], fraction=.046)
        for fraction in [.25, .5, .75]:
            axes[2].plot(depth[int(fraction*len(depth))], label=f'row {fraction:.0%}')
        axes[2].set(title='Depth profiles (not wall segmentation)', xlabel='Image x (pixels)', ylabel='Arbitrary depth')
        axes[2].legend()
        fig.tight_layout()
        fig.savefig(out/'front_and_depth.png', dpi=160)
        plt.close(fig)
    compact = {k: v for k, v in report.items() if k not in ('camera_centers', 'per_view_depth', 'notes')}
    print(json.dumps(compact, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder')
    parser.add_argument('--plots', action='store_true', help='Requires matplotlib')
    args = parser.parse_args()
    diagnose(args.folder, args.plots)
