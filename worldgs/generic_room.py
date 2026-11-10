"""Prompt-only scene generation without a previous room's correction plan."""
from pathlib import Path
import json


def generate(prompt, output, seed=0, size=512, steps=12000, max_points=180000,
             resume=False, prepare_only=False):
    from .local_room import create_local_proxy
    from .regenerate_surface import run
    folder=Path(output).resolve()
    request=dict(prompt=prompt,seed=seed,size=size,steps=steps,max_points=max_points,
                 cameras=233,rgb_target=200,scene_specific_corrections=False)
    marker=folder.with_name(folder.name+'_generic_request.json')
    if marker.exists() and (not resume or json.loads(marker.read_text())!=request):
        raise FileExistsError('Use a new output, or --resume with the identical generic request')
    marker.parent.mkdir(parents=True,exist_ok=True)
    marker.write_text(json.dumps(request,indent=2),encoding='utf-8')
    proxy=create_local_proxy(prompt,folder,size=size,views=16,seed=seed,resume=resume)
    return run(proxy,folder,steps=steps,max_points=max_points,
               validated_proxy=True,prepare_only=prepare_only)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prompt',required=True);p.add_argument('--output',required=True)
    p.add_argument('--seed',type=int,default=0);p.add_argument('--size',type=int,default=512)
    p.add_argument('--steps',type=int,default=12000);p.add_argument('--max-points',type=int,default=180000)
    p.add_argument('--resume',action='store_true');p.add_argument('--prepare-only',action='store_true')
    generate(**vars(p.parse_args()))
