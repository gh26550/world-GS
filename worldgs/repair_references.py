"""Rebuild physically rendered supervision without regenerating or changing meshes."""
import json
import shutil
from pathlib import Path
import numpy as np
import trimesh
from .geometry import intrinsics
from .learned_scene import backend_paths,run_worker
from .learned_references import save_references
from .reference_coverage import protect_training_coverage


def repair_references(path,output):
    source=Path(path).resolve();folder=Path(output).resolve()
    if source==folder or source in folder.parents or folder in source.parents:
        raise ValueError('Use a separate sibling output directory')
    if folder.exists() and any(folder.iterdir()):raise FileExistsError('Use an empty output directory')
    folder.mkdir(parents=True,exist_ok=True);(folder/'logs').mkdir()
    for name in ('assets','planning','materials','room_images'):
        shutil.copytree(source/name,folder/name)
    for name in ('source_front.png','image_room_provenance.json','request.json','scene_mesh.glb','mesh_geometry.npz','layout.json','front_alignment.json'):
        if (source/name).exists():shutil.copy2(source/name,folder/name)
    request=json.loads((folder/'request.json').read_text())
    request['geometry_pipeline_version']=14
    (folder/'request.json').write_text(json.dumps(request,indent=2))
    old=json.loads((source/'scene.json').read_text());job=json.loads((source/'render_job.json').read_text())
    job['folder']=str(folder)
    # Retain the previous heldout poses AND intrinsics for a paired comparison.
    for frame in job['frames']:
        frame['K']=frame.get('K',job['K']) if frame['split']=='heldout' or frame['target']=='front' else intrinsics(job['size']).tolist()
    job['K']=intrinsics(job['size']).tolist()
    geometry=np.load(folder/'mesh_geometry.npz')
    mesh=trimesh.Trimesh(geometry['vertices'],geometry['faces'],process=False)
    components=json.loads((folder/'layout.json').read_text())['components'];owners=geometry['owners']
    coverage=protect_training_coverage(mesh,components,owners,job['frames'],job['size'])
    (folder/'reference_coverage.json').write_text(json.dumps(coverage,indent=2))
    (folder/'render_job.json').write_text(json.dumps(job,indent=2))
    print(json.dumps(coverage),flush=True)
    blender=backend_paths()[-1]
    run_worker([blender,'-b','--python',Path(__file__).with_name('physical_render.py'),'--',folder/'render_job.json'],folder/'logs/render.log')
    meta=save_references(folder,old['prompt'],old['seed'],job['size'],mesh,components,owners,job)
    for key in ('mode','generation_source','input_image','proposal_images_are_training_targets','front_alignment'):
        if key in old:meta[key]=old[key]
    meta['reference_repair']={'source':str(source),'mesh_unchanged':True,'heldout_projection_preserved':True}
    (folder/'scene.json').write_text(json.dumps(meta,indent=2))
    if meta.get('front_alignment'):
        from .front_alignment import audit_front
        if not audit_front(folder)['passed']:raise ValueError('Front silhouette gate failed')
    (folder/'status.json').write_text(json.dumps({'stage':'references_ready','success':True}))
    return meta
