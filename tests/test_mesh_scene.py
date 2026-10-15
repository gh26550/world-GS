import unittest
import numpy as np
from PIL import Image
from worldgs.geometry import intrinsics
from worldgs.mesh_scene import build_components, MeshRenderer, look_at, camera_set
from worldgs.train_mesh import normal_quaternions


class MeshSceneTest(unittest.TestCase):
    def setUp(self):
        self.layout = [{'name': 'room', 'kind': 'room', 'min': [-4, -1.5, -4], 'max': [4, 1.5, 4]},
                       {'name': 'table', 'kind': 'box', 'label': 'table', 'min': [-.5, .8, 1], 'max': [.5, 1.5, 1.8]}]

    def test_table_is_not_an_opaque_box(self):
        mesh, pieces, components, owners = build_components(self.layout, Image.new('RGB', (64, 64), 'tan'))
        self.assertEqual(len([c for c in components if c['name'] == 'leg']), 4)
        self.assertGreater(len(mesh.faces), 100)
        renderer = MeshRenderer(mesh, components, owners)
        hit = renderer.render(intrinsics(64), look_at([0, 1.2, 0], [0, 1.2, 2]), 64)
        self.assertGreater(hit['depth'][32, 32], 3.)
        self.assertTrue((hit['depth'] > 0).all())

    def test_reference_cameras_have_large_baseline(self):
        cameras = camera_set(self.layout, 12)
        positions = np.array([p[:3, 3] for _, p in cameras])
        self.assertGreater(np.linalg.norm(positions[:, None]-positions[None], axis=-1).max(), 2.5)
        for _, pose in cameras:
            np.testing.assert_allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-7)

    def test_normal_quaternions_are_normalized(self):
        normals = np.array([[0, 0, 1.], [0, 0, -1.], [1, 0, 0], [0, 1, 0]])
        q = normal_quaternions(normals)
        np.testing.assert_allclose(np.linalg.norm(q, axis=1), 1.)
