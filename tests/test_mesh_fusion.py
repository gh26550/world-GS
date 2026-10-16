import unittest
import numpy as np
import trimesh
import tempfile
import json
from pathlib import Path
from worldgs.geometry import camera, intrinsics
from worldgs.mesh_fusion import observation_masks, fuse_tsdf, clean_fused_mesh, surface_points


class MeshFusionTest(unittest.TestCase):
    def test_same_camera_repeats_do_not_count_as_multiview_support(self):
        depth = np.full((32, 32), 3.)
        masks, _ = observation_masks([depth]*2, [np.ones_like(depth, bool)]*2, [camera(0)]*2, intrinsics(32))
        self.assertFalse(any(m.any() for m in masks))

    def test_translated_consistent_views_are_accepted(self):
        depth = np.full((32, 32), 3.)
        a, b = camera(0), camera(0)
        b[0, 3] = .15
        masks, _ = observation_masks([depth]*2, [np.ones_like(depth, bool)]*2, [a, b], intrinsics(32))
        self.assertTrue(all(m[16, 16] for m in masks))

    def test_single_view_is_only_retained_as_low_confidence_provisional(self):
        depth = np.full((32, 32), 3.)
        masks, reports, confidence = observation_masks([depth], [np.ones_like(depth, bool)], [camera(0)], intrinsics(32),
            include_provisional=True, return_confidence=True)
        self.assertTrue(masks[0][16, 16])
        self.assertEqual(confidence[0][16, 16], .25)
        self.assertEqual(reports[0]['trusted_points'], 0)

    def test_conflicting_foreground_is_not_retained_as_provisional(self):
        a, b = camera(0), camera(0)
        b[0, 3] = .15
        depths = [np.full((32, 32), d) for d in (3., 1.5)]
        masks, _, _ = observation_masks(depths, [np.ones((32, 32), bool)]*2, [a, b], intrinsics(32),
            include_provisional=True, return_confidence=True)
        self.assertFalse(masks[1][8:24, 8:24].any())

    def test_contradictory_depth_does_not_gain_support(self):
        a, b = camera(0), camera(0)
        b[0, 3] = .15
        depths = [np.full((32, 32), d) for d in (3., 1.5)]
        masks, _ = observation_masks(depths, [np.ones((32, 32), bool)]*2, [a, b], intrinsics(32))
        self.assertFalse(any(m.any() for m in masks))

    def test_tsdf_recovers_known_plane_and_camera_transform(self):
        a, b = camera(0), camera(0)
        b[0, 3] = .15
        depth = np.full((64, 64), 3.017)
        rgb = np.full((64, 64, 3), [160, 70, 30], np.uint8)
        result = fuse_tsdf([depth]*2, [rgb]*2, [a, b], intrinsics(64), [np.ones_like(depth, bool)]*2, .04)
        self.assertGreater(len(result.triangles), 100)
        self.assertLess(np.median(np.abs(np.asarray(result.vertices)[:, 2]-3.017)), .03)

    def test_cleanup_removes_small_floating_component(self):
        large = trimesh.creation.box().subdivide().subdivide()
        small = trimesh.creation.box(extents=[.01]*3).subdivide().subdivide()
        small.apply_translation([2, 0, 0])
        from worldgs.mesh_fusion import open3d_mesh
        result = clean_fused_mesh(open3d_mesh(trimesh.util.concatenate([large, small])))
        self.assertLess(result.bounds[1, 0], 1.)

    def test_gaussian_seeds_lie_on_assigned_triangles(self):
        mesh = trimesh.creation.box()
        data = surface_points(mesh, count=1000)
        reconstructed = (mesh.triangles[data['triangle_ids']]*data['barycentric'][..., None]).sum(1)
        np.testing.assert_allclose(data['xyz'], reconstructed, atol=1e-6)

    def test_invalid_mesh_is_rejected_before_cuda_training(self):
        from worldgs.train_mesh import train_mesh
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path/'scene.json').write_text(json.dumps({'mode': 'global_tsdf_v6'}))
            (path/'mesh_quality.json').write_text(json.dumps({'passed': False}))
            with self.assertRaisesRegex(ValueError, 'quality check failed'):
                train_mesh(path, 1, 100)

    def test_provisional_view_has_quarter_loss_and_gradient(self):
        import torch
        from worldgs.train_mesh import confidence_loss
        error = torch.ones(8, requires_grad=True)
        trusted = confidence_loss(error, torch.ones(8))
        provisional = confidence_loss(error, torch.full((8,), .25))
        self.assertAlmostEqual(float(provisional), float(trusted)*.25)
        provisional.backward()
        torch.testing.assert_close(error.grad, torch.full((8,), .25/8))


if __name__ == '__main__':
    unittest.main()
