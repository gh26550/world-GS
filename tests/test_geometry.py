import unittest
import numpy as np
from worldgs.geometry import intrinsics, camera, lift, mesh, render_mesh, align_depth


class GeometryTest(unittest.TestCase):
    def test_projection_roundtrip(self):
        K = intrinsics(32)
        depth = np.full((32, 32), 3.)
        pose = camera(60)
        world = lift(depth, K, pose)
        cam = world @ pose[:3, :3]
        uv = cam @ K.T
        uv = uv[..., :2]/uv[..., 2:]
        y, x = np.indices(depth.shape)
        np.testing.assert_allclose(uv, np.stack([x, y], -1), atol=1e-10)

    def test_raster_depth_and_occlusion(self):
        K = intrinsics(32)
        v, f = mesh(np.full((32, 32), 3.), K, camera(0), stride=1)
        c = np.full((len(v), 3), 170)
        rgb, d = render_mesh(v, f, c, K, camera(0), 32)
        self.assertGreater((d > 0).mean(), .9)
        np.testing.assert_allclose(d[d > 0], 3.)
        self.assertTrue((rgb[d > 0] >= 169).all())

    def test_alignment_rejects_empty(self):
        with self.assertRaises(ValueError):
            align_depth(np.ones((10, 10)), np.zeros((10, 10)))
        np.testing.assert_allclose(align_depth(np.ones((10, 10)), np.full((10, 10), 4)), 4)

    def test_discontinuity_filter(self):
        d = np.ones((8, 8))
        d[:, 4:] = 20
        v, f = mesh(d, intrinsics(8), camera(0), stride=1)
        self.assertTrue(np.all(np.ptp(v[f, 2], axis=1) == 0))


if __name__ == '__main__':
    unittest.main()
