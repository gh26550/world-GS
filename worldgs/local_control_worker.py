"""Native depth-conditioned generation; spatial control is a model input."""
import json,sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from diffusers import ControlNetModel,StableDiffusionControlNetInpaintPipeline

def load_model(request=None):
    control=ControlNetModel.from_pretrained('lllyasviel/control_v11f1p_sd15_depth',torch_dtype=torch.float16,local_files_only=True);boundary=ControlNetModel.from_pretrained('lllyasviel/control_v11p_sd15_canny',torch_dtype=torch.float16,local_files_only=True);pipe=StableDiffusionControlNetInpaintPipeline.from_pretrained('stable-diffusion-v1-5/stable-diffusion-inpainting',controlnet=[control,boundary],torch_dtype=torch.float16,local_files_only=True).to('cuda');pipe.load_ip_adapter('h94/IP-Adapter',subfolder='models',weight_name='ip-adapter_sd15.bin',local_files_only=True);pipe.set_ip_adapter_scale(.2)
    return pipe


def run(r,pipe=None):
    pipe=pipe if pipe is not None else load_model(r)
    pipe.to('cuda')
    options=r.get('generation_options',{})
    strength=float(options.get('strength',.7));ip_scale=float(options.get('ip_scale',.2));depth_scale=float(options.get('depth_scale',1.4));boundary_scale=float(options.get('boundary_scale',1.0))
    if not 0<strength<=1 or not 0<=ip_scale<=1 or not 0<=depth_scale<=2 or not 0<=boundary_scale<=2:raise ValueError('Invalid generation controls')
    pipe.set_ip_adapter_scale(ip_scale)
    labels=list(dict.fromkeys(r.get('visible_labels') or []));words={'bookshelf':'bookshelves filled with books','sofa':'leather sofa','table':'wooden coffee table','rug':'woven area rug','window':'window with daylight through glass'};visible=', '.join(words.get(label,label) for label in labels) if labels else 'empty room surfaces, no furniture'
    prompt='Photorealistic interior photograph, '+visible+'. '+r.get('appearance','beige plaster, wood floor')+'. Warm daylight, realistic materials. Preserve the existing camera, positions, sizes and clipped furniture.'
    prompt=pipe.tokenizer.decode(pipe.tokenizer.encode(prompt,add_special_tokens=False)[:75])
    absent=[label for label in ['sofa','chair','table','bookshelf','bed','cabinet','plant'] if label not in labels];negative='additional furniture, duplicate furniture, camera change, zoom, distorted geometry, illustration, cartoon, text, watermark, '+', '.join(absent)
    if 'NO windows' in r['prompt']:negative+=', window'
    image=Image.open(r['guide']).convert('RGB');depth=Image.open(r['depth_guide']).convert('RGB');style=Image.open(r['style']).convert('RGB');size=r['size']
    edit_mask=Image.open(options['edit_mask']).convert('L') if options.get('edit_mask') else Image.new('L',image.size,255)
    if edit_mask.size!=image.size:raise ValueError('Edit mask changed image calibration')
    result=pipe(prompt=prompt,negative_prompt=negative,image=image,mask_image=edit_mask,control_image=[depth,Image.open(r['edge_guide']).convert('RGB')],ip_adapter_image=style,width=size,height=size,strength=strength,num_inference_steps=35,guidance_scale=6.5,controlnet_conditioning_scale=[depth_scale,boundary_scale],generator=torch.Generator('cuda').manual_seed(r['seed'])).images[0]
    if options.get('preserve_observed'):
        pixels=np.array(result);known=np.asarray(edit_mask)==0;pixels[known]=np.asarray(image)[known];result=Image.fromarray(pixels)
    if np.asarray(result).mean()<1:raise RuntimeError('Local generator rejected the image')
    result.save(r['output']);Path(r['output']+'.conditioning.json').write_text(json.dumps(dict(prompt=prompt,negative_prompt=negative,depth_control_scale=depth_scale,boundary_control_scale=boundary_scale,ip_scale=ip_scale,strength=strength),indent=2))


if __name__=='__main__':run(json.loads(Path(sys.argv[1]).read_text()))
