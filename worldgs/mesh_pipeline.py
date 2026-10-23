import json
import hashlib
from pathlib import Path
import numpy as np
from PIL import Image
import trimesh
from .geometry import intrinsics
from .pipeline import Models
from .room import make_layout
from .mesh_scene import MeshRenderer, build_components, teacher_camera_set


def generate_mesh_scene(prompt, output, size, views, seed, demo, front_image, room_size, layout_file=None):
    if size < 64 or size % 8 or views < 8:
        raise ValueError('size >=64 and divisible by 8; views >=8')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError('Use a new output directory')
    K = intrinsics(size)
    layout_record = json.loads(Path(layout_file).read_text()) if layout_file else None
    saved_layout = layout_record['shapes'] if layout_record else None
    if saved_layout is not None:
        from .teacher_geometry import validate_layout
        validate_layout(saved_layout)
        if not front_image and not demo:
            raise ValueError('--layout-file requires --front-image (or --demo)')
        original_size = layout_record.get('source_image_size') or (Image.open(front_image).width if front_image else size)
        for shape in saved_layout[1:]:
            if 'detection' in shape:
                shape['detection']['box'] = (np.asarray(shape['detection']['box'])*size/original_size).tolist()
        image = Image.open(front_image).convert('RGB').resize((size, size)) if front_image else Image.new('RGB', (size, size), 'tan')
        depth = None
    elif demo:
        image = Image.fromarray(np.full((size, size, 3), [130, 95, 70], np.uint8))
        depth = np.full((size, size), 3.)
        detections = [{'box': [size*.2, size*.5, size*.8, size*.95], 'label': 'sofa', 'score': 1.},
                      {'box': [size*.35, size*.7, size*.65, size*.98], 'label': 'table', 'score': 1.}]
    else:
        models = Models(seed)
        image = Image.open(front_image).convert('RGB').resize((size, size)) if front_image else models.front(prompt, size)
        depth = models.depth(image)
        detections = models.detect(image, prompt)
        del models
    image.save(output/'front.png')
    if depth is not None:
        np.save(output/'front_relative_depth.npy', depth)
    layout = saved_layout if saved_layout is not None else make_layout(room_size, detections, depth, K)
    room_size = np.array(layout[0]['max'])-np.array(layout[0]['min'])
    mesh, geometries, components, owners = build_components(layout, image)
    scene = trimesh.Scene()
    (output/'meshes').mkdir()
    groups = {}
    for geometry, component in zip(geometries, components):
        name = f'{component["group"]}_{component["name"]}_{len(scene.geometry)}'
        scene.add_geometry(geometry, node_name=name, geom_name=name)
        groups.setdefault(component['group'], []).append(geometry)
    scene.export(output/'scene_mesh.glb')
    mesh.export(output/'proxy_scene.obj')
    for name, group in groups.items():
        trimesh.util.concatenate(group).export(output/'meshes'/f'{name}.obj')
    (output/'layout.json').write_text(json.dumps({'room_size_prior': list(room_size), 'source_image_size': size, 'shapes': layout, 'components': components,
        'geometry_source': 'procedural 3D components inferred from front detections; not learned object reconstruction'}, indent=2), encoding='utf-8')
    renderer = MeshRenderer(mesh, components, owners)
    cameras = teacher_camera_set(layout, views)
    if len(cameras) < 12:
        raise ValueError('Insufficient collision-free mesh reference cameras')
    # Every fifth camera is truly excluded from training, not an atlas reprojection.
    training, heldout = [], []
    visible_faces = np.zeros(len(mesh.faces), bool)
    for i, (target, pose) in enumerate(cameras):
        hit = renderer.render(K, pose, size)
        valid = hit['depth'] > 0
        split = 'heldout' if i % 5 == 2 else 'view'
        records = heldout if split == 'heldout' else training
        index = len(records)
        filename = f'{split}_{index:03d}.png'
        Image.fromarray(hit['rgb']).save(output/filename)
        np.save(output/f'{split}_depth_{index:03d}.npy', hit['depth'])
        np.save(output/f'{split}_mask_{index:03d}.npy', valid)
        np.save(output/f'{split}_normals_{index:03d}.npy', hit['normals'].astype(np.float32))
        np.save(output/f'{split}_faces_{index:03d}.npy', hit['faces'])
        records.append({'file_path': filename, 'depth_file': f'{split}_depth_{index:03d}.npy', 'mask_file': f'{split}_mask_{index:03d}.npy',
                        'normal_file': f'{split}_normals_{index:03d}.npy', 'face_file': f'{split}_faces_{index:03d}.npy',
                        'target': target, 'transform_matrix': pose.tolist(), 'reference_source': 'fixed_triangle_mesh_render'})
        if split == 'view':
            visible_faces[hit['faces'][valid]] = True
        print(f'Mesh camera {i+1}/{len(cameras)} {split} target={target} valid={valid.mean():.1%}', flush=True)
    from .teacher_geometry import sample_teacher_surface, mesh_digest, structure_report
    points = sample_teacher_surface(mesh, renderer, owners, visible_faces, seed=seed)
    np.savez_compressed(output/'points.npz', **points)
    np.savez_compressed(output/'mesh_geometry.npz', vertices=np.asarray(mesh.vertices, np.float32), faces=np.asarray(mesh.faces, np.int32), owners=owners)
    centers = np.array([f['transform_matrix'] for f in training])[:, :3, 3]
    metadata = {'mode': 'mesh_teacher_v8', 'prompt': prompt, 'seed': seed, 'demo': demo, 'size': size,
        'K': K.tolist(), 'frames': training, 'heldout_frames': heldout, 'geometry_constraint': 'fixed_surface_gaussians',
        'convention': 'OpenCV camera-to-world, x right y down z forward',
        'reference_source': 'multi-position triangle mesh renders; no photo/atlas painted on room walls',
        'point_count': len(points['xyz']), 'mesh_components': len(components), 'mesh_triangles': len(mesh.faces),
        'mesh_digest': mesh_digest(mesh.vertices, mesh.faces), 'room_size_prior': room_size.tolist(),
        'mesh_file_sha256': hashlib.sha256((output/'scene_mesh.glb').read_bytes()).hexdigest(),
        'depth_source': 'ray-triangle intersection, camera-z; never re-estimated from reference images',
        'front_depth_usage': 'initial object placement only' if depth is not None else 'none; saved layout reused',
        'requested_yaw_views': views, 'yaw_coverage_degrees': 360,
        'camera_baseline': float(np.linalg.norm(centers[:, None]-centers[None], axis=-1).max()),
        'visible_triangle_area_fraction': float(mesh.area_faces[visible_faces].sum()/mesh.area),
        'limitations': ['procedural furniture proxies, not exact image-to-object reconstruction',
                        'mesh-render references, not independent real photographs', 'hidden surfaces may be unobserved',
                        'appearance uses procedural materials and fixed baked lighting; not dynamic relighting',
                        'undetected and hidden objects are not automatically invented']}
    (output/'structure_quality.json').write_text(json.dumps(structure_report(mesh, components, points), indent=2))
    (output/'scene.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    make_contact(output, training, 'references.jpg')
    make_contact(output, heldout, 'heldout_references.jpg')
    print(json.dumps({k: metadata[k] for k in ['point_count', 'mesh_components', 'mesh_triangles', 'camera_baseline', 'visible_triangle_area_fraction']}), flush=True)
    return metadata


def make_contact(folder, frames, name):
    from PIL import ImageDraw
    rows = (len(frames)+3)//4
    canvas = Image.new('RGB', (4*240, rows*168), 'white')
    draw = ImageDraw.Draw(canvas)
    for i, frame in enumerate(frames):
        x, y = i % 4*240, i//4*168
        canvas.paste(Image.open(folder/frame['file_path']).resize((240, 144)), (x, y))
        draw.text((x+4, y+146), f'{i}: {frame.get("target", "camera")}', fill='black')
    canvas.save(folder/name)
