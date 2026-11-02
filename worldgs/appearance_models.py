"""Local depth ControlNet inpainting + reference-image IP-Adapter."""
import numpy as np
from PIL import Image
from .appearance_geometry import depth_condition


class AppearanceModels:
    def __init__(self,seed):
        import torch
        from diffusers import ControlNetModel,StableDiffusionControlNetInpaintPipeline
        from transformers import pipeline
        self.torch=torch;self.seed=seed
        dtype=torch.float16
        self.depth_model=pipeline('depth-estimation',model='depth-anything/Depth-Anything-V2-Small-hf',device='cuda')
        control=ControlNetModel.from_pretrained('lllyasviel/control_v11f1p_sd15_depth',torch_dtype=dtype)
        self.pipe=StableDiffusionControlNetInpaintPipeline.from_pretrained(
            'stable-diffusion-v1-5/stable-diffusion-inpainting',controlnet=control,torch_dtype=dtype).to('cuda')
        self.pipe.load_ip_adapter('h94/IP-Adapter',subfolder='models',weight_name='ip-adapter_sd15.bin')
        self.pipe.set_ip_adapter_scale(.2)
        # Keep the IP-Adapter SDPA processors; generic attention slicing replaces them.
        from transformers import AutoProcessor,AutoModelForZeroShotObjectDetection
        self.detector_processor=AutoProcessor.from_pretrained('IDEA-Research/grounding-dino-tiny')
        self.detector=AutoModelForZeroShotObjectDetection.from_pretrained('IDEA-Research/grounding-dino-tiny').to('cuda')
        print('Appearance models ready: depth ControlNet + IP-Adapter',flush=True)

    def generate(self,prompt,guide,known,depth,style,index,attempt,regions=None):
        if regions is None:
            return self._sample(prompt,guide,known,depth,style,index,attempt)
        rgb=guide.copy()
        for n,region in enumerate(regions):
            editable=region['mask'] & ~known
            if not editable.any():continue
            if region['id']=='background':
                local_prompt='Photorealistic room architecture, floor, walls and ceiling, same room materials and lighting as reference. Empty architectural surfaces in the edited region. No furniture or additional objects.'
                extra_negative='sofa, chair, table, bookshelf, cabinet, bed, furniture'
            else:
                local_prompt='Photorealistic '+region.get('description',region['label'])+'. Same materials and lighting as the reference room photograph. One existing '+region['label']+'. Keep its placement, size, orientation and camera perspective.'
                extra_negative='additional furniture, duplicate objects'
            rgb=self._sample(local_prompt,rgb,~editable,depth,style,index*31+n,attempt,negative_extra=extra_negative)
        rgb[known]=guide[known]
        return rgb

    def _sample(self,prompt,guide,known,depth,style,index,attempt,negative_extra=""):
        image=self.pipe(prompt=prompt,
            negative_prompt='illustration, cartoon, plastic toy, distorted furniture, duplicate furniture, warped walls, fisheye, text, watermark, '+negative_extra,
            image=Image.fromarray(guide),mask_image=Image.fromarray(np.uint8(~known)*255),
            control_image=Image.fromarray(depth_condition(depth)).convert('RGB'),ip_adapter_image=style,
            width=guide.shape[1],height=guide.shape[0],strength=.75-.1*attempt,num_inference_steps=30,guidance_scale=6.,
            controlnet_conditioning_scale=1.2+.3*attempt,
            generator=self.torch.Generator('cuda').manual_seed(self.seed+index*17+attempt)).images[0]
        rgb=np.asarray(image).copy()
        if rgb.mean()<1:raise RuntimeError('Appearance generator returned a black/rejected image')
        rgb[known]=guide[known]
        return rgb

    def inverse_depth(self,rgb):
        prediction=self.depth_model(Image.fromarray(rgb))['predicted_depth']
        prediction=self.torch.nn.functional.interpolate(prediction.reshape(1,1,*prediction.shape[-2:]),size=rgb.shape[:2],mode='bicubic',align_corners=False)
        return prediction[0,0].detach().cpu().numpy()

    def detections(self,rgb,labels=None):
        import inspect
        inputs=self.detector_processor(images=Image.fromarray(rgb),text='. '.join(labels or ['sofa','table','bookshelf','chair'])+'.',return_tensors='pt').to('cuda')
        with self.torch.inference_mode():outputs=self.detector(**inputs)
        key='box_threshold' if 'box_threshold' in inspect.signature(self.detector_processor.post_process_grounded_object_detection).parameters else 'threshold'
        result=self.detector_processor.post_process_grounded_object_detection(outputs,inputs.input_ids,target_sizes=[rgb.shape[:2]],text_threshold=.25,**{key:.25 if labels else .4})[0]
        names=result.get('text_labels',result.get('labels'))
        return [dict(label=str(label),box=box.detach().cpu().tolist(),score=float(score)) for label,box,score in zip(names,result['boxes'],result['scores'])]
