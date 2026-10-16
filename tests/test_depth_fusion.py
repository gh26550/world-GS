import unittest
import numpy as np
from worldgs.depth_fusion import align_inverse_depth, match_depth_seam, reject_depth_spikes, feather_color_seam, refine_prediction
from worldgs.expansion import patch_geometry
from worldgs.geometry import intrinsics, camera


class DepthFusionTest(unittest.TestCase):
    def test_inverse_shift_and_outliers_are_corrected_outside_overlap(self):
        y, x = np.indices((64, 64))
        truth = 2.+y/24+x/96
        predicted = 1/(1.7/truth+.12)
        reference = truth.copy()
        reference[(x+y*3) % 11 == 0] *= 1.8
        observed = x < 32
        result, report = align_inverse_depth(predicted, reference, observed, np.full_like(truth, 12.))
        self.assertEqual(report['method'], 'robust_inverse_affine_overlap')
        self.assertLess(np.median(np.abs(result[~observed]/truth[~observed]-1)), .02)
        self.assertLess(report['heldout_selected_log_error'], report['heldout_scale_log_error'])

    def test_invalid_depth_is_not_an_alignment_anchor(self):
        pred = np.full((16, 16), 2.)
        pred[0, :3] = [np.nan, np.inf, -1]
        depth, report = align_inverse_depth(pred, np.full((16, 16), 3.), np.ones((16, 16), bool), np.full((16, 16), 5.))
        self.assertTrue(np.isfinite(depth).all())
        self.assertEqual(report['anchors'], 253)

    def test_seam_correction_keeps_anchors_and_decays_away(self):
        depth = np.full((48, 64), 3.2)
        reference = np.full_like(depth, 3.)
        observed = np.indices(depth.shape)[1] < 24
        rgb = np.full((*depth.shape, 3), 130, np.uint8)
        result = match_depth_seam(depth, reference, observed, rgb)
        np.testing.assert_allclose(result[observed], 3.)
        self.assertLess(abs(result[24, 24]-3.), .08)
        self.assertAlmostEqual(result[24, -1], 3.2)

    def test_real_occlusion_edge_is_not_smoothed_away(self):
        depth = np.full((32, 64), 6.)
        depth[:, :32] = 2.
        observed = np.indices(depth.shape)[1] < 32
        rgb = np.zeros((*depth.shape, 3), np.uint8)
        rgb[:, 32:] = 240
        result = match_depth_seam(depth, depth, observed, rgb)
        np.testing.assert_allclose(result, depth)

    def test_isolated_spike_is_replaced_without_removing_depth_step(self):
        depth = np.full((32, 32), 3.)
        depth[:, 20:] = 5.
        depth[12, 12] = 12.
        rgb = np.full((32, 32, 3), 110, np.uint8)
        cleaned, valid, count = reject_depth_spikes(depth, rgb)
        self.assertEqual(cleaned[12, 12], 3.)
        np.testing.assert_allclose(cleaned[:, 20:], 5.)
        self.assertEqual(count, 1)
        self.assertTrue(valid.all())

    def test_stitch_triangles_bridge_overlap_instead_of_leaving_grid_gap(self):
        depth = np.full((32, 32), 3.)
        rgb = np.full((32, 32, 3), 100, np.uint8)
        accept = np.indices(depth.shape)[1] >= 16
        old = patch_geometry(rgb, depth, intrinsics(32), camera(0), accept)
        joined = patch_geometry(rgb, depth, intrinsics(32), camera(0), accept, stitch=True)
        self.assertGreater(len(joined.faces), len(old.faces))
        self.assertLess(joined.vertices[:, 0].min(), 0)

    def test_color_blending_preserves_sharp_edges_and_observed_pixels(self):
        rgb = np.full((16, 32, 3), 130, np.uint8)
        reference = np.full_like(rgb, 120)
        observed = np.indices((16, 32))[1] < 16
        depth = np.full((16, 32), 3.)
        result = feather_color_seam(rgb, reference, observed, depth, depth)
        self.assertTrue((result[observed] == 120).all())
        self.assertLess(result[8, 16, 0], 130)
        rgb[:, 16:] = 240
        result = feather_color_seam(rgb, reference, observed, depth, depth)
        self.assertEqual(result[8, 16, 0], 240)

    def test_refinement_does_not_move_unknown_depth_outside_enclosure(self):
        depth = np.full((32, 32), 3.)
        ref = np.full((32, 32), 3.3)
        observed = np.indices((32, 32))[1] < 16
        result, valid, _ = refine_prediction(depth, ref, observed, np.full_like(depth, 3.1), np.full((32, 32, 3), 120, np.uint8))
        self.assertTrue((result[~observed] <= 3.1).all())
        np.testing.assert_allclose(result[observed], 3.3)


if __name__ == '__main__':
    unittest.main()
