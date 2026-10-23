"""Run with Blender Python: UV-textured learned meshes, PBR materials, lighting."""
import hashlib
import json
import re
from pathlib import Path
import sys
import numpy as np


def blender_camera_matrix(pose):
    pose=np.asarray(pose)
    transform=np.array([[1.,0,0],[0,0,1],[0,-1,0]])
    camera_axes=np.diag([1.,-1,-1])
    matrix=np.eye(4)
    matrix[:3,:3]=transform@pose[:3,:3]@camera_axes
    matrix[:3,3]=transform@pose[:3,3]
    return matrix


def render(job_file):
    import bpy
    from mathutils import Matrix
    job=json.loads(Path(job_file).read_text())
    folder=Path(job['folder'])
    signature=hashlib.sha256((json.dumps(job,sort_keys=True)+Path(__file__).read_text()).encode())
    signature.update((folder/'scene_mesh.glb').read_bytes())
    for filename in sorted((folder/'materials').glob('*.png')):
        signature.update(filename.read_bytes())
    signature=signature.hexdigest()
    state=folder/'render_state.json'
    reuse=state.exists() and json.loads(state.read_text()).get('signature')==signature
    state.write_text(json.dumps(dict(signature=signature,complete=False)))
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(folder/'scene_mesh.glb'))
    # glTF importer maps y-up to z-up. Our exported world is camera-style y-down:
    # an additional 180-degree x rotation yields (x,z,-y) in Blender world.
    correction=Matrix(np.diag([1.,-1.,-1.,1.]).tolist())
    for obj in list(bpy.context.scene.objects):
        if obj.parent is None:
            obj.matrix_world=correction@obj.matrix_world
    def room_material(name):
        material=bpy.data.materials.new(name)
        material.use_nodes=True
        nodes=material.node_tree.nodes
        links=material.node_tree.links
        bsdf=nodes.get('Principled BSDF')
        bsdf.inputs['Roughness'].default_value=.48 if name=='floor' else .8
        coordinate=nodes.new('ShaderNodeTexCoord')
        texture=nodes.new('ShaderNodeTexImage')
        texture.image=bpy.data.images.load(str(folder/'materials'/f'{name}.png'))
        texture.projection='BOX'
        texture.projection_blend=.15
        links.new(coordinate.outputs['Object'],texture.inputs['Vector'])
        description=job['plan']['room'][name].lower()
        if re.search(r'plaster|paint|stucco',description):
            # A generic generated tile can invent lace/wallpaper on plain paint.
            # Keep the specified plain material physically plain, with micro-bump.
            tint=(.72,.69,.63,1.) if name=='walls' else (.8,.8,.78,1.)
            for word,color in [('black',(.025,.025,.025,1)),('white',(.82,.82,.8,1)),
                               ('red',(.5,.05,.03,1)),('blue',(.05,.15,.4,1)),('green',(.1,.3,.12,1)),
                               ('gray',(.4,.4,.4,1)),('grey',(.4,.4,.4,1))]:
                if re.search(r'\b'+word+r'\b',description):
                    tint=color
                    break
            bsdf.inputs['Base Color'].default_value=tint
        else:
            links.new(texture.outputs['Color'],bsdf.inputs['Base Color'])
        noise=nodes.new('ShaderNodeTexNoise')
        noise.inputs['Scale'].default_value=80.
        bump=nodes.new('ShaderNodeBump')
        bump.inputs['Strength'].default_value=.12
        bump.inputs['Distance'].default_value=.008
        links.new(coordinate.outputs['Object'],noise.inputs['Vector'])
        links.new(noise.outputs['Fac'],bump.inputs['Height'])
        links.new(bump.outputs['Normal'],bsdf.inputs['Normal'])
        return material
    materials={name:room_material(name) for name in ['floor','walls','ceiling']}
    for obj in bpy.context.scene.objects:
        if obj.type!='MESH':
            continue
        if obj.name.startswith('room_'):
            name='floor' if obj.name.startswith('room_1_1') else 'ceiling' if obj.name.startswith('room_1_0') else 'walls'
            obj.data.materials.clear()
            obj.data.materials.append(materials[name])
    scene=bpy.context.scene
    scene.render.engine='CYCLES'
    device='CPU'
    try:
        prefs=bpy.context.preferences.addons['cycles'].preferences
        prefs.compute_device_type='OPTIX'
        prefs.get_devices()
        for d in prefs.devices:
            d.use=d.type=='OPTIX'
        if any(d.use for d in prefs.devices):
            scene.cycles.device='GPU'
            device='OPTIX'
    except Exception as error:
        print('GPU selection unavailable, using CPU:',error,flush=True)
    scene.cycles.samples=32
    scene.cycles.use_denoising=True
    scene.cycles.max_bounces=6
    scene.render.use_persistent_data=True
    scene.render.resolution_x=scene.render.resolution_y=job['size']
    scene.render.resolution_percentage=100
    scene.render.image_settings.file_format='PNG'
    scene.render.image_settings.color_mode='RGB'
    scene.view_settings.view_transform='AgX'
    world=bpy.data.worlds.new('Ambient')
    world.use_nodes=True
    world.node_tree.nodes['Background'].inputs[0].default_value=(.8,.85,1.,1.)
    world.node_tree.nodes['Background'].inputs[1].default_value=.07
    scene.world=world
    w,h,l=job['plan']['room']['size']
    power=float(job['plan']['lighting']['power'])
    color=job['plan']['lighting']['color']
    for i,(x,z) in enumerate([(-w/4,-l/4),(-w/4,l/4),(w/4,-l/4),(w/4,l/4)]):
        light=bpy.data.lights.new(f'CeilingArea_{i}','AREA')
        light.energy=power/4
        light.shape='DISK'
        light.size=2.2
        light.color=color
        obj=bpy.data.objects.new(light.name,light)
        scene.collection.objects.link(obj)
        obj.location=(x,z,h/2-.1)
    camera=bpy.data.cameras.new('ReferenceCamera')
    obj=bpy.data.objects.new(camera.name,camera)
    scene.collection.objects.link(obj)
    scene.camera=obj
    camera.sensor_fit='HORIZONTAL'
    camera.sensor_width=36.
    camera.lens=float(job['K'][0][0])*36/job['size']
    camera.clip_start=.03
    camera.clip_end=100.
    for index,frame in enumerate(job['frames']):
        pose=np.asarray(frame['transform_matrix'])
        matrix=blender_camera_matrix(pose)
        obj.matrix_world=Matrix(matrix.tolist())
        scene.render.filepath=str(folder/frame['file_path'])
        if not reuse or not (folder/frame['file_path']).exists():
            bpy.ops.render.render(write_still=True)
        print(f'Physical reference {index+1}/{len(job["frames"])} {frame["file_path"]}',flush=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(folder/'physical_scene.blend'))
    state.write_text(json.dumps(dict(signature=signature,complete=True,device=device,samples=32,
        reference_count=len(job['frames']),lighting=job['plan']['lighting'],
        note='UV assets and physically rendered baked lighting; not a claim of intrinsic albedo recovery'),indent=2))


if __name__=='__main__':
    render(sys.argv[sys.argv.index('--')+1])
