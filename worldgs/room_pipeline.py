import json
from pathlib import Path
import numpy as np
from PIL import Image
from .geometry import intrinsics, mesh, write_obj
from .room import Atlas, make_layout, trace, trajectory, pose_at, export_layout
from .pipeline import Models


def generate_room(prompt, output, size, views, seed, demo, front_image, room_size):
    if size < 64 or size % 8 or views < 8 or len(room_size) != 3 or min(room_size) < 2:
        raise ValueError('size >=64 and divisible by 8; views >=8; room dimensions >=2')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError('Use a new, empty output directory')
    K = intrinsics(size)
    models = None if demo else Models(seed)
    if demo:
        y, x = np.indices((size, size))
        image = Image.fromarray(np.uint8(np.stack([x/size*180+30, y/size*180+30, np.full_like(x, 120)], -1)))
        relative = np.full((size, size), 3.)
        detections = []
    else:
        image = Image.open(front_image).convert('RGB').resize((size, size)) if front_image else models.front(prompt, size)
        old_depth = Path(front_image).with_name('front_depth.npy') if front_image else None
        relative = np.load(old_depth) if old_depth is not None and old_depth.exists() else models.depth(image)
        if relative.shape != (size, size):
            relative = models.depth(image)
        print('Detecting foreground furniture for coarse mesh proxies', flush=True)
        detections = models.detect(image, prompt)
    image.save(output/'front.png')
    np.save(output/'front_relative_depth.npy', relative)
    layout = make_layout(room_size, detections, relative, K)
    (output/'layout.json').write_text(json.dumps({'assumed_room_size': list(room_size), 'scale': 'arbitrary prior; not metric reconstruction', 'shapes': layout, 'detections': detections}, indent=2), encoding='utf-8')
    (output/'meshes').mkdir()
    for shape in layout:
        export_layout([shape], output/'meshes'/f'{shape["name"]}.obj')
    export_layout(layout, output/'proxy_scene.obj')
    atlas = Atlas(layout)
    material_priors = atlas.initialize_materials(image)
    schedule = trajectory(views)
    frames = []
    for i, (yaw, pitch, pose) in enumerate(schedule):
        depth, points, surfaces = trace(layout, K, pose, size)
        guide, known = atlas.render(points, surfaces)
        missing = ~known
        conditioning_prompt, negative = prompt, ''
        visible_labels = {layout[int(s)//6].get('label', '') for s in np.unique(surfaces) if int(s)//6 > 0}
        if pitch > 30:
            conditioning_prompt = 'Looking upward at the plain white ceiling and upper walls of a cozy room, architectural photograph, ceiling fills the image.'
            negative = 'floor, rug, carpet, furniture on the ceiling'
        elif pitch < -30:
            conditioning_prompt = 'Looking downward at the floor and lower parts of furniture in the same cozy living room, architectural photograph.'
            negative = 'ceiling, skylight'
        if not any('sofa' in label or 'couch' in label for label in visible_labels):
            negative += ', sofa, couch'
        if not any('table' in label for label in visible_labels):
            negative += ', table'
        if i == 0:
            current = image
            v, f = mesh(depth, K, pose)
            write_obj(output/'proxy_front.obj', v, f, np.asarray(image)[::4, ::4].reshape(-1, 3))
            np.save(output/'front_depth.npy', depth)
        elif demo:
            current = Image.fromarray(np.where(missing[..., None], np.array(image), guide).astype('uint8'))
        elif missing.mean() > .002:
            current = models.controlled_complete(conditioning_prompt, guide, missing, depth, i, negative)
        else:
            current = Image.fromarray(guide)
        atlas.update(points, surfaces, current)
        current.save(output/f'view_{i:03d}.png')
        Image.fromarray(guide).save(output/f'guide_{i:03d}.png')
        Image.fromarray(np.uint8(missing)*255).save(output/f'mask_{i:03d}.png')
        np.save(output/f'depth_{i:03d}.npy', depth.astype('float32'))
        frames.append({'file_path': f'view_{i:03d}.png', 'transform_matrix': pose.tolist(), 'yaw': yaw, 'pitch': pitch,
                       'generated_fraction': float(missing.mean()), 'conditioning_prompt': conditioning_prompt, 'negative_prompt': negative})
        print(f'Reference {i+1}/{len(schedule)} yaw={yaw:.0f} pitch={pitch:.0f} new={missing.mean():.1%}', flush=True)
    samples = atlas.sample()
    np.savez_compressed(output/'points.npz', **samples)
    heldout = []
    for i, yaw in enumerate([45, 135, 225, 315]):
        pose = pose_at(yaw, 0, [.3*np.cos(np.deg2rad(yaw)), -.12, .3*np.sin(np.deg2rad(yaw))])
        depth, points, surfaces = trace(layout, K, pose, size)
        target, known = atlas.render(points, surfaces)
        Image.fromarray(target).save(output/f'heldout_{i:03d}.png')
        np.save(output/f'heldout_depth_{i:03d}.npy', depth.astype('float32'))
        np.save(output/f'heldout_mask_{i:03d}.npy', known)
        heldout.append({'file_path': f'heldout_{i:03d}.png', 'transform_matrix': pose.tolist(), 'known_fraction': float(known.mean())})
    meta = {'mode': 'room_scaffold_v2', 'prompt': prompt, 'seed': seed, 'demo': demo, 'size': size, 'K': K.tolist(),
            'frames': frames, 'heldout_frames': heldout, 'geometry_constraint': 'axis_aligned_surface',
            'texture_available_fraction': float(samples['color_available'].mean()), 'material_priors': material_priors,
            'front_image_source': str(front_image) if front_image else 'generated',
            'convention': 'OpenCV camera-to-world; x right, y down, z forward',
            'limitations': ['single box room prior', 'furniture are heuristic boxes from image detections',
                           'heldout images are proxy-rendered references, not independent ground truth', 'arbitrary scale']}
    (output/'scene.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(f'Saved {len(samples["xyz"])} scaffold points; texture available {meta["texture_available_fraction"]:.1%}', flush=True)
    return meta
