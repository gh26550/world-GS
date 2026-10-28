import gc
import json
from pathlib import Path
import numpy as np
from PIL import Image
from .geometry import intrinsics, camera, mesh, render_mesh, align_depth, write_obj


class Models:
    def __init__(self, seed):
        import torch
        self.torch = torch
        self.seed = seed
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dtype = torch.float16 if self.device == 'cuda' else torch.float32

    def release(self, model):
        del model
        gc.collect()
        if self.device == 'cuda':
            self.torch.cuda.empty_cache()

    def front(self, prompt, size):
        from diffusers import StableDiffusionPipeline
        pipe = StableDiffusionPipeline.from_pretrained('stable-diffusion-v1-5/stable-diffusion-v1-5', torch_dtype=self.dtype).to(self.device)
        image = pipe(prompt, width=size, height=size, generator=self.torch.Generator(self.device).manual_seed(self.seed)).images[0]
        del pipe
        gc.collect()
        self.torch.cuda.empty_cache()
        return image

    def depth(self, image):
        from transformers import pipeline
        estimator = pipeline('depth-estimation', model='depth-anything/Depth-Anything-V2-Small-hf', device=self.device)
        result = estimator(image)['predicted_depth']
        result = result.reshape(1, 1, *result.shape[-2:])
        d = self.torch.nn.functional.interpolate(result, size=(image.height, image.width), mode='bicubic', align_corners=False)[0, 0].cpu().numpy()
        # Relative inverse depth: arbitrary scene units, not metric geometry.
        lo, hi = np.percentile(d, [2, 98])
        depth = 1/(0.15+0.85*np.clip((d-lo)/max(hi-lo, 1e-6), 0, 1))
        del estimator
        gc.collect()
        self.torch.cuda.empty_cache()
        return depth

    def complete(self, prompt, rgb, missing, index):
        from diffusers import StableDiffusionInpaintPipeline
        pipe = StableDiffusionInpaintPipeline.from_pretrained('stable-diffusion-v1-5/stable-diffusion-inpainting', torch_dtype=self.dtype).to(self.device)
        result = pipe(prompt=prompt, image=Image.fromarray(rgb), mask_image=Image.fromarray(np.uint8(missing)*255), width=rgb.shape[1], height=rgb.shape[0], generator=self.torch.Generator(self.device).manual_seed(self.seed+index)).images[0]
        result = np.asarray(result).copy()
        result[~missing] = rgb[~missing]
        del pipe
        gc.collect()
        self.torch.cuda.empty_cache()
        return Image.fromarray(result)

    def detect(self, image, prompt, labels=None, confidence=.30, text_confidence=.25, max_detections=8):
        import inspect
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        vocabulary = [('sofa', ('sofa', 'couch')), ('table', ('table',)), ('bookshelf', ('bookshel', 'bookcase')),
                      ('chair', ('chair',)), ('bed', ('bed',)), ('cabinet', ('cabinet',))]
        labels = labels or [label for label, words in vocabulary if any(word in prompt.lower() for word in words)] or ['sofa', 'table', 'bookshelf']
        processor = AutoProcessor.from_pretrained('IDEA-Research/grounding-dino-tiny')
        model = AutoModelForZeroShotObjectDetection.from_pretrained('IDEA-Research/grounding-dino-tiny').to(self.device)
        inputs = processor(images=image, text='. '.join(labels)+'.', return_tensors='pt').to(self.device)
        with self.torch.no_grad():
            outputs = model(**inputs)
        threshold = 'box_threshold' if 'box_threshold' in inspect.signature(processor.post_process_grounded_object_detection).parameters else 'threshold'
        result = processor.post_process_grounded_object_detection(outputs, inputs.input_ids, target_sizes=[image.size[::-1]], text_threshold=text_confidence, **{threshold: confidence})[0]
        names = result.get('text_labels', result.get('labels'))
        detections = [{'box': box.cpu().tolist(), 'label': str(label), 'score': float(score)} for box, label, score in zip(result['boxes'], names, result['scores'])]
        selected = []
        for detection in sorted(detections, key=lambda d: -d['score']):
            a = np.array(detection['box'])
            duplicate = False
            for other in selected:
                b = np.array(other['box'])
                inter = np.maximum(0, np.minimum(a[2:], b[2:])-np.maximum(a[:2], b[:2])).prod()
                union = (a[2:]-a[:2]).prod()+(b[2:]-b[:2]).prod()-inter
                duplicate |= inter/max(union, 1) > .6
            if not duplicate:
                selected.append(detection)
        del model, processor, outputs, inputs
        gc.collect()
        self.torch.cuda.empty_cache()
        return selected[:max_detections]

    def controlled_complete(self, prompt, rgb, missing, depth, index, negative_prompt='', control_strength=1.):
        from diffusers import ControlNetModel, StableDiffusionControlNetInpaintPipeline
        if not hasattr(self, 'control_pipe'):
            control = ControlNetModel.from_pretrained('lllyasviel/control_v11f1p_sd15_depth', torch_dtype=self.dtype)
            self.control_pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
                'stable-diffusion-v1-5/stable-diffusion-inpainting', controlnet=control, torch_dtype=self.dtype).to(self.device)
            self.control_pipe.enable_attention_slicing()
        inverse = 1/np.maximum(depth, .05)
        lo, hi = np.percentile(inverse, [1, 99])
        condition = np.uint8(np.clip((inverse-lo)/max(hi-lo, 1e-6), 0, 1)*255)
        result = self.control_pipe(prompt=prompt, negative_prompt='distorted walls, warped room, duplicate furniture, fisheye, text, '+negative_prompt,
            image=Image.fromarray(rgb), mask_image=Image.fromarray(np.uint8(missing)*255),
            control_image=Image.fromarray(condition).convert('RGB'), width=rgb.shape[1], height=rgb.shape[0],
            num_inference_steps=25, guidance_scale=6., controlnet_conditioning_scale=control_strength,
            generator=self.torch.Generator(self.device).manual_seed(self.seed+index)).images[0]
        image = np.asarray(result).copy()
        image[~missing] = rgb[~missing]
        return Image.fromarray(image)


def generate_legacy(prompt, output, size=512, views=24, seed=0, demo=False):
    if size < 64 or size % 8 or views < 8:
        raise ValueError('size must be >=64 and divisible by 8; views must be >=8')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError('Use an empty output directory to preserve existing results')
    models = None if demo else Models(seed)
    K = intrinsics(size)
    if demo:
        y, x = np.indices((size, size))
        image = Image.fromarray(np.uint8(np.stack([x/size*200+30, y/size*180+40, np.full_like(x, 110)], -1)))
        depth = np.full((size, size), 3.)
    else:
        image = models.front(prompt, size)
        depth = models.depth(image)
    image.save(output/'front.png')
    np.save(output/'front_depth.npy', depth)
    vertices, faces = mesh(depth, K, camera(0))
    colors = np.asarray(image)[::4, ::4].reshape(-1, 3)
    write_obj(output/'proxy_front.obj', vertices, faces, colors)
    frames = []
    for i in range(views):
        pose = camera(i*360/views)
        if i == 0:
            current = image
            reference = depth
            missing = np.zeros_like(depth, bool)
        else:
            rgb, reference = render_mesh(vertices, faces, colors, K, pose, size)
            missing = reference == 0
            Image.fromarray(rgb).save(output/f'guide_{i:03d}.png')
            if demo:
                # Explicit synthetic geometry for integration testing only.
                current = Image.fromarray(np.where(missing[..., None], np.asarray(image), rgb).astype('uint8'))
                aligned = np.full_like(reference, 3.)
            else:
                current = models.complete(prompt, rgb, missing, i)
                aligned = align_depth(models.depth(current), reference)
            fused = np.where(missing, aligned, reference)
            v, f = mesh(fused, K, pose)
            c = np.asarray(current)[::4, ::4].reshape(-1, 3)
            # Add only newly observed triangles; preserve the existing surface.
            new = missing[::4, ::4].ravel()
            f = f[new[f].all(-1)]
            faces = np.concatenate([faces, f+len(vertices)])
            vertices = np.concatenate([vertices, v])
            colors = np.concatenate([colors, c])
            reference = fused
        name = f'view_{i:03d}.png'
        current.save(output/name)
        np.save(output/f'depth_{i:03d}.npy', reference)
        Image.fromarray(np.uint8(missing)*255).save(output/f'mask_{i:03d}.png')
        frames.append({'file_path': name, 'transform_matrix': pose.tolist(), 'yaw': i*360/views, 'generated_fraction': float(missing.mean())})
    # Use referenced mesh vertices only and quantize to limit duplicate seeds.
    used = np.unique(faces)
    xyz, rgb = vertices[used], colors[used]
    _, indices = np.unique(np.round(xyz/0.02), axis=0, return_index=True)
    np.savez_compressed(output/'points.npz', xyz=xyz[indices].astype('float32'), rgb=rgb[indices])
    write_obj(output/'proxy_scene.obj', vertices, faces, colors)
    metadata = {'prompt': prompt, 'seed': seed, 'demo': demo, 'size': size, 'K': K.tolist(), 'convention': 'OpenCV camera-to-world; x right, y down, z forward', 'frames': frames, 'limitations': ['relative scale', 'rotation-only views constrain parallax poorly', 'unseen geometry is hallucinated', 'no semantic object separation']}
    (output/'scene.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return metadata


def generate(prompt, output, size=512, views=24, seed=0, demo=False, mode='image-room', front_image=None, room_size=None, layout_file=None, resume=False, planner_model=None):
    if mode == 'image-room':
        if demo or layout_file:
            raise ValueError('image-room requires real room images; demo/layout overrides are not supported')
        from .learned_scene import generate_learned_scene
        return generate_learned_scene(prompt,output,size,views,seed,room_size,resume,planner_model,
                                      image_room=True,front_image=front_image)
    if mode=='learned':
        if demo or layout_file or front_image:
            raise ValueError('learned mode uses prompt-only generation; demo/layout/front-image overrides are not supported')
        from .learned_scene import generate_learned_scene
        return generate_learned_scene(prompt,output,size,views,seed,room_size,resume,planner_model)
    if resume or planner_model:
        raise ValueError('--resume/--planner-model require --mode learned')
    if layout_file and mode != 'mesh':
        raise ValueError('--layout-file requires --mode mesh')
    room_size = room_size or ((8., 3., 8.) if mode in ('scene', 'mesh', 'expand') else (6., 3., 6.))
    if mode == 'scene':
        from .prompt_scene import generate_prompt_scene
        return generate_prompt_scene(prompt,output,size,views,seed,demo,front_image,room_size)
    if mode == 'expand':
        from .expansion import generate_expanded_scene
        from .mesh_fusion import rebuild_mesh_scene
        folder = Path(output)
        if folder.exists() and any(folder.iterdir()):
            raise FileExistsError('Use an empty output directory')
        observations = folder/'observations'
        generate_expanded_scene(prompt, observations, size, views, seed, demo, front_image, room_size)
        return rebuild_mesh_scene(observations, folder, allow_observations=True)
    if mode == 'mesh':
        from .mesh_pipeline import generate_mesh_scene
        return generate_mesh_scene(prompt, output, size, views, seed, demo, front_image, room_size, layout_file)
    if mode == 'legacy':
        if front_image is not None:
            raise ValueError('--front-image is not supported in legacy mode')
        return generate_legacy(prompt, output, size, views, seed, demo)
    from .room_pipeline import generate_room
    return generate_room(prompt, output, size, views, seed, demo, front_image, room_size)
