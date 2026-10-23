"""Prompt inventory -> component mesh -> shared photographic surface -> GS refs."""
import gc
import json
import re
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image
from .scene_plan import plan_scene, NAMES
from .pipeline import Models
from .mesh_pipeline import generate_mesh_scene, make_contact
from .mesh_scene import build_components, MeshRenderer
from .surface_appearance import SurfaceAppearance
from .teacher_geometry import sample_teacher_surface


class AppearanceRenderer(MeshRenderer):
    def __init__(self, mesh, components, owners, appearance):
        super().__init__(mesh, components, owners)
        self.appearance = appearance

    def shade(self, points, triangles):
        colors, normals = super().shade(points, triangles)
        colors, _ = self.appearance.sample(points, triangles, colors*255)
        return colors/255, normals


def generate_prompt_scene(prompt, output, size, views, seed, demo, front_image, room_size):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use an empty output directory')
    if not demo and size < 256:
        raise ValueError('Photographic generation requires size >=256')
    plan = plan_scene(prompt, room_size)
    models = None if demo else Models(seed)
    style = (Image.open(front_image).convert('RGB').resize((size,size)) if front_image else
             Image.new('RGB',(size,size),'tan') if demo else models.front(prompt,size))
    with tempfile.TemporaryDirectory() as temp:
        layout_file, image_file = Path(temp)/'layout.json', Path(temp)/'front.png'
        layout_file.write_text(json.dumps(dict(shapes=plan['shapes'],source_image_size=size)))
        style.save(image_file)
        meta = generate_mesh_scene(prompt,output,size,views,seed,demo,str(image_file),room_size,str(layout_file))
    style.save(output/'style_front.png')
    (output/'scene_plan.json').write_text(json.dumps(plan,indent=2,ensure_ascii=False),encoding='utf-8')
    mesh, _, parts, owners = build_components(plan['shapes'], style)
    atlas = SurfaceAppearance(mesh)
    renderer = AppearanceRenderer(mesh,parts,owners,atlas)
    K = np.asarray(meta['K'])
    (output/'appearance_keyframes').mkdir()
    history = []
    # All appearance keyframes come from training cameras. Held-out cameras
    # contribute neither generated textures nor pixels to this shared surface.
    for index, frame in enumerate(meta['frames']):
        pose = np.asarray(frame['transform_matrix'])
        hit = renderer.render(K,pose,size)
        valid = hit['faces'] >= 0
        xyz, faces = hit['xyz'][valid],hit['faces'][valid]
        _, known = atlas.sample(xyz,faces,hit['rgb'][valid])
        missing = np.zeros((size,size),bool)
        missing[valid] = ~known
        rgb = hit['rgb'].copy()
        visible = set()
        for owner in np.unique(owners[faces]):
            group = parts[int(owner)]['group']
            for shape in plan['shapes'][1:]:
                if shape['name'] == group:
                    visible.add(shape['label'])
        forward_y = pose[1,2]
        direction = ('looking upward at ceiling and upper walls' if forward_y < -.5 else
                     'looking downward at floor and lower furniture' if forward_y > .5 else 'eye level interior photograph')
        if not demo and missing.mean() > .01:
            absent = [name for name in NAMES if name not in visible]
            style_prompt = prompt
            for name in absent:
                style_prompt = re.sub(NAMES[name], '', style_prompt, flags=re.I)
            details = ', '.join(sorted(visible)) or 'architectural surfaces only'
            view_prompt = ('Photorealistic architectural interior photograph, physically plausible materials, natural light, '
                           'detailed wood grain, realistic fabric, coherent cozy living room, '+direction+'. Visible objects: '+details+'. Scene style: '+style_prompt)
            rgb = np.asarray(models.controlled_complete(view_prompt,rgb,missing,hit['depth'],index,
                negative_prompt=', '.join(absent)+', cartoon, low poly, flat color, plastic, duplicated objects, warped architecture',
                control_strength=1.1)).copy()
        Image.fromarray(rgb).save(output/'appearance_keyframes'/f'proposal_{index:03d}.png')
        Image.fromarray(np.uint8(missing)*255).save(output/'appearance_keyframes'/f'mask_{index:03d}.png')
        atlas.update(xyz,faces,rgb[valid])
        history.append(dict(index=index,missing_fraction=float(missing.mean()),visible_objects=sorted(visible)))
        print(f'Photographic surface {index+1}/{len(meta["frames"])} new={missing.mean():.1%}',flush=True)
    visible_faces = np.zeros(len(mesh.faces),bool)
    coverage = []
    for frame in meta['frames']+meta['heldout_frames']:
        hit = renderer.render(K,np.asarray(frame['transform_matrix']),size)
        valid = hit['faces'] >= 0
        _, known = atlas.sample(hit['xyz'][valid],hit['faces'][valid],hit['rgb'][valid])
        Image.fromarray(hit['rgb']).save(output/frame['file_path'])
        coverage.append(float(known.mean()))
        frame['reference_source'] = 'shared_generated_surface_on_prompt_mesh'
        if frame in meta['frames']:
            visible_faces[hit['faces'][valid]] = True
    points = sample_teacher_surface(mesh,renderer,owners,visible_faces,seed=seed)
    np.savez_compressed(output/'points.npz',**points)
    atlas.save(output/'surface_appearance.npz')
    Image.open(output/meta['frames'][0]['file_path']).save(output/'front.png')
    meta.update(mode='prompt_mesh_radiance_v9',geometry_constraint='soft_mesh_structure_prior',
                experimental=True, general_prompt_compliance_verified=False,
                front_depth_usage='none; object inventory and layout come from prompt',
                reference_source='depth-conditioned generated appearance shared on 3D triangle texture memory',
                planner=plan['planner'],requirements=plan['requirements'],
                limitations=plan['limitations']+['Generated appearance can contain hallucinated details not represented by coarse geometry',
                  'Procedural component meshes, not learned text-to-object reconstruction',
                  'Baked illumination; dynamic Unity lighting is unsupported'])
    (output/'scene.json').write_text(json.dumps(meta,indent=2))
    (output/'appearance_report.json').write_text(json.dumps(dict(history=history,
        reference_texel_coverage_min=min(coverage),reference_texel_coverage_mean=float(np.mean(coverage)),
        heldout_used_for_texture=False,depth_reestimated=False,
        visual_quality_automatically_verified=False),indent=2))
    make_contact(output,meta['frames'],'references.jpg')
    make_contact(output,meta['heldout_frames'],'heldout_references.jpg')
    del models
    gc.collect()
    return meta
