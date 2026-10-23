"""Isolated workers for the installed SAM3/SAM3D environments.

This file intentionally depends only on each backend's own environment. It is
launched by argument arrays, never a model-generated shell command.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import numpy as np
from PIL import Image


def segment(job, root):
    import torch
    sys.path.insert(0,str(root.parent/'sam3'))
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    model = build_sam3_image_model(checkpoint_path=str(root.parent/'sam3/checkpoints/sam3.pt'),device='cuda',eval_mode=True)
    processor = Sam3Processor(model,confidence_threshold=.35,device='cuda')
    for entry in job['objects']:
        folder = Path(entry['folder'])
        if (folder/'segmentation.json').exists():
            continue
        errors=[]
        for candidate in sorted(folder.glob('candidate_*.png')):
            image = Image.open(candidate).convert('RGB')
            with torch.inference_mode():
                state = processor.set_image(image)
                state = processor.set_text_prompt(prompt=entry['label'],state=state)
            masks = state.get('masks')
            if masks is None or not len(masks):
                errors.append(candidate.name+': object not detected')
                continue
            scores = state['scores'].detach().cpu().numpy().reshape(-1)
            masks = masks.detach().cpu().numpy().reshape(-1,image.height,image.width) > .5
            order = np.argsort(scores)[::-1]
            mask = masks[order[0]]
            y,x = np.where(mask)
            if not len(x) or mask.mean() < .02 or mask.mean() > .88 or x.min()<3 or y.min()<3 or x.max()>image.width-4 or y.max()>image.height-4:
                errors.append(candidate.name+': tiny, oversized or cropped object')
                continue
            extra = [m for m in masks[order[1:]] if (m & ~mask).sum() > mask.sum()*.3]
            image.save(folder/'image.png')
            Image.fromarray(np.uint8(mask)*255).save(folder/'mask.png')
            rgba = np.dstack([np.asarray(image),np.uint8(mask)*255])
            Image.fromarray(rgba).save(folder/'object.png')
            (folder/'segmentation.json').write_text(json.dumps(dict(candidate=candidate.name,label=entry['label'],
                confidence=float(scores[order[0]]),foreground_fraction=float(mask.mean()),
                detected_instances=len(masks),unselected_instances_masked_out=len(extra),backend='SAM3'),indent=2))
            print('Segmented '+entry['id'],flush=True)
            break
        else:
            raise RuntimeError('Could not isolate '+entry['id']+': '+'; '.join(errors))


def reconstruct(job, root):
    os.environ.setdefault('CONDA_PREFIX',sys.prefix)
    os.environ['LIDRA_SKIP_INIT']='true'
    os.environ.setdefault('ATTN_BACKEND','sdpa')
    os.environ.setdefault('SPARSE_ATTN_BACKEND','sdpa')
    os.environ.setdefault('WORLDMESH_LOW_VRAM','1')
    os.environ['XFORMERS_DISABLED']='1'
    sys.path.insert(0,str(root))
    sys.path.insert(0,str(root/'notebook'))
    os.chdir(root)
    from inference import Inference
    import torch
    (Path(job['objects'][0]['folder']).parent/'backend_runtime.json').write_text(json.dumps(dict(
        python=sys.executable,torch=torch.__version__,cuda=torch.version.cuda,source_root=str(root),
        attention='sdpa; DINO xformers disabled',low_vram=True),indent=2))
    inference = Inference(str(root/'checkpoints/hf/pipeline.yaml'),compile=False)
    for entry in job['objects']:
        folder = Path(entry['folder'])
        if (folder/'asset.json').exists() and (folder/'generated.glb').exists():
            continue
        image = np.asarray(Image.open(folder/'image.png').convert('RGB'))
        mask = np.asarray(Image.open(folder/'mask.png')) > 127
        rgba = inference.merge_mask_to_rgba(image,mask)
        with torch.inference_mode():
            result = inference._pipeline.run(rgba,None,seed=job['seed'],stage1_only=False,
                with_mesh_postprocess=True,with_texture_baking=True,with_layout_postprocess=False,
                use_vertex_color=False,pointmap=None)
        mesh = result.get('glb')
        if mesh is None or len(mesh.faces)<100 or not np.isfinite(mesh.vertices).all():
            raise RuntimeError('SAM3D did not return a valid mesh for '+entry['id'])
        mesh.export(folder/'generated.glb')
        (folder/'asset.json').write_text(json.dumps(dict(id=entry['id'],backend='SAM3D',
            vertices=len(mesh.vertices),triangles=len(mesh.faces),bounds=mesh.bounds.tolist(),
            canonical_coordinates='SAM3D GLB y-up; converted at assembly',
            visual_kind=mesh.visual.kind,seed=job['seed']),indent=2))
        del result,mesh
        torch.cuda.empty_cache()
        print('Reconstructed '+entry['id'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['segment','reconstruct'])
    parser.add_argument('job')
    parser.add_argument('--root',required=True)
    args=parser.parse_args()
    job=json.loads(Path(args.job).read_text())
    globals()[args.stage](job,Path(args.root).resolve())
