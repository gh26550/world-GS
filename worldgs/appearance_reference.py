"""Appearance-only reference preprocessing, never independent object T2I."""
import numpy as np
from PIL import Image

def appearance_atlas(image,seed,grid=8):
    rgb=np.asarray(image.convert('RGB'));h,w=rgb.shape[:2]
    if h%grid or w%grid:raise ValueError('Image dimensions must be divisible by atlas grid')
    tiles=[rgb[y*h//grid:(y+1)*h//grid,x*w//grid:(x+1)*w//grid] for y in range(grid) for x in range(grid)];indices=np.random.default_rng(seed).permutation(grid*grid);out=np.empty_like(rgb)
    for k,index in enumerate(indices):out[(k//grid)*h//grid:(k//grid+1)*h//grid,(k%grid)*w//grid:(k%grid+1)*w//grid]=tiles[index]
    return Image.fromarray(out)
