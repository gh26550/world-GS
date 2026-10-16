import unittest
import numpy as np
import trimesh
from worldgs.geometry import camera, intrinsics
from worldgs.hole_repair import repair_small_mesh_holes, split_long_edges
from worldgs.expansion import SharedRGBDRenderer
from worldgs.mesh_fusion import observation_masks


def plane_with_hole(radius=.1):
    x, y = np.meshgrid(np.linspace(-1, 1, 25), np.linspace(-1, 1, 25))
    vertices = np.column_stack([x.ravel(), y.ravel(), np.full(x.size, 3.)])
    faces = []
    for row in range(24):
        for col in range(24):
            i = row*25+col
            for tri in [(i, i+1, i+25), (i+1, i+26, i+25)]:
                if np.linalg.norm(vertices[list(tri), :2].mean(0)) > radius:
                    faces.append(tri)
    return trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=np.tile([120, 90, 70, 255], (len(vertices), 1)), process=False)


class HoleRepairTest(unittest.TestCase):
    def test_supported_small_hole_gets_a_surface(self):
        mesh = plane_with_hole()
        K = intrinsics(64)
        fixed, report = repair_small_mesh_holes(mesh, [np.full((64, 64), 3.)], [np.ones((64, 64), bool)], [camera(0)], K)
        self.assertGreater(report['accepted'], 0)
        hit = SharedRGBDRenderer(fixed).render(K, camera(0), 64)
        self.assertAlmostEqual(hit['depth'][32, 32], 3., places=4)

    def test_visible_opening_is_not_capped_in_front_of_background(self):
        mesh = plane_with_hole()
        fixed, report = repair_small_mesh_holes(mesh, [np.full((64, 64), 6.)], [np.ones((64, 64), bool)], [camera(0)], intrinsics(64))
        self.assertEqual(report['accepted'], 0)
        self.assertEqual(len(fixed.faces), len(mesh.faces))

    def test_large_unknown_hole_is_not_closed(self):
        mesh = plane_with_hole(.5)
        _, report = repair_small_mesh_holes(mesh, [np.full((64, 64), 3.)], [np.ones((64, 64), bool)], [camera(0)], intrinsics(64))
        self.assertEqual(report['accepted'], 0)

    def test_edge_subdivision_has_no_cracks_or_changed_surface(self):
        mesh = trimesh.creation.box()
        fixed = split_long_edges(mesh, .2)
        self.assertTrue(fixed.is_watertight)
        self.assertLessEqual(fixed.edges_unique_length.max(), .2+1e-6)
        np.testing.assert_allclose(fixed.bounds, mesh.bounds)

    def test_depth_edges_are_retained_without_mixing_foreground_background(self):
        d = np.full((32, 32), 3.)
        d[:, 16:] = 5.
        masks, _, conf = observation_masks([d], [np.ones_like(d, bool)], [camera(0)], intrinsics(32),
            include_provisional=True, return_confidence=True, recover_edges=True)
        self.assertTrue(masks[0][16, 15])
        self.assertTrue(masks[0][16, 16])
        self.assertEqual(conf[0][16, 15], .25)

    def test_grazing_surface_is_not_deleted_by_old_angle_cutoff(self):
        u, v, center = np.array([-.1, 0., .995]), np.array([0., 1., 0.]), np.array([0., 0., 3.])
        vertices = np.array([center-2*u-v, center+2*u-v, center+2*u+v, center-2*u+v])
        mesh = trimesh.Trimesh(vertices=vertices, faces=[[0, 1, 2], [0, 2, 3]],
            vertex_colors=np.full((4, 4), 128, np.uint8), process=False)
        renderer = SharedRGBDRenderer(mesh)
        old = renderer.render(intrinsics(64), camera(0), 64, min_incidence=.18)
        new = renderer.render(intrinsics(64), camera(0), 64)
        self.assertEqual(old['depth'][32, 32], 0.)
        self.assertGreater(new['depth'][32, 32], 0.)


if __name__ == '__main__':
    unittest.main()
