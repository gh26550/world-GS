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
        d = self.torch.nn.functional.interpolate(result[:, None], size=(image.height, image.width), mode='bicubic', align_corners=False)[0, 0].cpu().numpy()
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


def generate(prompt, output, size=512, views=24, seed=0, demo=False):
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
