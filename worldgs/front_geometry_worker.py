"""MoGe-1 geometry in its installed isolated environment (no model installs)."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image


def estimate(source, output):
    import torch
    from moge.model.v1 import MoGeModel
    image=np.asarray(Image.open(source).convert('RGB'))
    model=MoGeModel.from_pretrained('Ruicheng/moge-vitl').to('cuda').eval()
    tensor=torch.tensor(image,dtype=torch.float32,device='cuda').permute(2,0,1)/255
    with torch.inference_mode():
        result=model.infer(tensor)
    np.savez_compressed(output,**{key:value.detach().float().cpu().numpy() for key,value in result.items()
                                 if key in ('points','depth','mask','intrinsics')})
    print('Estimated front pointmap and intrinsics: '+str(output),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('source',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    estimate(args.source,args.output)
