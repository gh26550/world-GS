import unittest
import numpy as np
from worldgs.coverage_sampling import stratified_seed_ids, coverage_drop_mask


class CoverageSamplingTest(unittest.TestCase):
    def test_small_remote_region_gets_an_initial_seed(self):
        rng = np.random.default_rng(10)
        xyz = np.concatenate([rng.random((10000, 3)), [[20, 0, 0]]])
        ids = stratified_seed_ids(xyz, 100)
        self.assertIn(10000, ids)
        self.assertEqual(len(np.unique(ids)), 100)

    def test_budget_keeps_low_opacity_remote_region(self):
        import torch
        xyz = torch.cat([torch.rand(1000, 3), torch.tensor([[20., 0., 0.]])])
        scores = torch.cat([torch.ones(1000), torch.tensor([-5.])])
        drop = coverage_drop_mask(xyz, scores, 100)
        self.assertFalse(bool(drop[-1]))
        self.assertEqual(int((~drop).sum()), 100)

    def test_coverage_pushes_opaque_known_surface_but_not_unknown_pixels(self):
        import torch
        from worldgs.train_mesh import foreground_coverage_loss
        alpha = torch.tensor([.1, .1], requires_grad=True)
        loss = foreground_coverage_loss(alpha, torch.ones(2), torch.tensor([True, False]))
        loss.backward()
        self.assertLess(float(alpha.grad[0]), 0.)
        self.assertEqual(float(alpha.grad[1]), 0.)


if __name__ == '__main__':
    unittest.main()
