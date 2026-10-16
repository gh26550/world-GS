import unittest
import numpy as np
import trimesh
from worldgs.expansion import calibrated_depth, patch_geometry, SharedRGBDRenderer, expansion_poses, filter_visibility_conflicts, direction_prompt
from worldgs.geometry import camera, intrinsics


class ExpansionTest(unittest.TestCase):
    def test_depth_updates_unknown_geometry_and_preserves_shared_overlap(self):
        predicted = np.full((16, 16), 2.)
        predicted[:, 8:] = 1.
        reference = np.full((16, 16), 3.)
        observed = np.zeros((16, 16), bool)
        observed[:, :8] = True
        depth, method = calibrated_depth(predicted, reference, observed, np.full((16, 16), 5.))
        self.assertEqual(method, 'shared_rgbd_overlap')
        np.testing.assert_allclose(depth[:, :8], 3.)
        np.testing.assert_allclose(depth[:, 8:], 1.5)

    def test_patch_only_adds_unobserved_triangles(self):
        d = np.full((32, 32), 3.)
        mask = np.zeros_like(d, bool)
        mask[:, 16:] = True
        m = patch_geometry(np.full((32, 32, 3), 123, np.uint8), d, intrinsics(32), camera(0), mask)
        self.assertGreater(len(m.faces), 0)
        self.assertTrue((m.vertices[:, 0] > 0).all())
        np.testing.assert_allclose(m.vertices[:, 2], 3.)

    def test_shared_render_uses_nearest_geometry_and_actual_colors(self):
        K = intrinsics(32)
        mask = np.ones((32, 32), bool)
        a = patch_geometry(np.full((32, 32, 3), [220, 30, 40], np.uint8), np.full((32, 32), 3.), K, camera(0), mask)
        b = patch_geometry(np.full((32, 32, 3), [10, 190, 60], np.uint8), np.full((32, 32), 2.), K, camera(0), mask)
        hit = SharedRGBDRenderer(trimesh.util.concatenate([a, b])).render(K, camera(0), 32)
        self.assertAlmostEqual(hit['depth'][16, 16], 2., places=5)
        np.testing.assert_allclose(hit['rgb'][16, 16], [10, 190, 60], atol=1)

    def test_completion_covers_all_directions_with_translation(self):
        poses = expansion_poses(12)
        self.assertEqual({int(y) for y, p, _ in poses if p == 0}, set(range(0, 360, 30)))
        centers = np.array([m[:3, 3] for _, _, m in poses])
        self.assertGreater(np.ptp(centers[:, 0]), .4)
        self.assertEqual({p for _, p, _ in poses}, {-55, 0, 55})

    def test_new_geometry_cannot_occlude_previously_observed_free_space(self):
        K = intrinsics(32)
        mask = np.ones((32, 32), bool)
        patch = patch_geometry(np.full((32, 32, 3), 100, np.uint8), np.full((32, 32), 2.), K, camera(0), mask)
        result = filter_visibility_conflicts(patch, [(camera(0), np.full((32, 32), 3.), mask)], K)
        self.assertEqual(len(result.faces), 0)

    def test_direction_text_matches_camera_forward(self):
        for yaw, pitch, pose in expansion_poses(12):
            if pitch:
                expected = 'ceiling' if pose[1, 2] < 0 else 'floor'
                self.assertIn(expected, direction_prompt(yaw, pitch))


if __name__ == '__main__':
    unittest.main()
