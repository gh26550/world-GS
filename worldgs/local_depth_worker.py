"""Separate Depth Pro environment; fixed known focal length, no per-view mesh fusion."""
import json,sys
from pathlib import Path
import numpy as np
import torch

def load_model(request):
    import depth_pro
    from depth_pro.depth_pro import DepthProConfig,DEFAULT_MONODEPTH_CONFIG_DICT as defaults
    config=DepthProConfig(patch_encoder_preset=defaults.patch_encoder_preset,image_encoder_preset=defaults.image_encoder_preset,decoder_features=defaults.decoder_features,use_fov_head=defaults.use_fov_head,fov_encoder_preset=defaults.fov_encoder_preset,checkpoint_uri=request['checkpoint'])
    model,transform=depth_pro.create_model_and_transforms(config=config,device=torch.device('cuda'),precision=torch.half);model.eval();return model,transform

def run(request,bundle=None):
    import depth_pro
    model,transform=bundle if bundle is not None else load_model(request);model.to('cuda');image,_,_=depth_pro.load_rgb(Path(request['image']))
    with torch.inference_mode():prediction=model.infer(transform(image),f_px=torch.tensor(float(request['focal_px']),device='cuda'))
    np.save(request['output'],prediction['depth'].float().cpu().numpy().squeeze())
if __name__=='__main__':run(json.loads(Path(sys.argv[1]).read_text()))
