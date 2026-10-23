"""Triangle-local texture memory: generated appearance is shared in 3D."""
import numpy as np
import trimesh


class SurfaceAppearance:
    def __init__(self, mesh, texel_size=.04):
        self.mesh = mesh
        self.resolutions = np.clip(np.ceil(np.sqrt(mesh.area_faces*2)/texel_size), 2, 256).astype(int)
        self.offsets = np.r_[0, np.cumsum((self.resolutions+1)**2)]
        self.colors = np.zeros((self.offsets[-1],3), np.float32)
        self.weights = np.zeros(self.offsets[-1], np.float32)

    def indices(self, xyz, faces):
        bary = trimesh.triangles.points_to_barycentric(self.mesh.triangles[faces], xyz)
        n = self.resolutions[faces]
        uv = np.clip(np.rint(bary[:,1:]*n[:,None]), 0, n[:,None]).astype(int)
        return self.offsets[faces]+uv[:,0]*(n+1)+uv[:,1]

    def sample(self, xyz, faces, fallback):
        ids = self.indices(xyz, faces)
        known = self.weights[ids] > 0
        color = np.asarray(fallback).copy()
        color[known] = self.colors[ids[known]]/self.weights[ids[known],None]
        return color, known

    def update(self, xyz, faces, rgb):
        ids = self.indices(xyz, faces)
        # Existing observations remain authoritative: no repeated per-view averaging
        # of incompatible furniture textures into the same surface.
        new = self.weights[ids] == 0
        np.add.at(self.colors, ids[new], rgb[new])
        np.add.at(self.weights, ids[new], 1.)

    def save(self, path):
        np.savez_compressed(path, resolutions=self.resolutions, offsets=self.offsets, colors=self.colors, weights=self.weights)
