import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
from worldgs.geometry import intrinsics, lift
from worldgs.mesh_scene import build_components, MeshRenderer, teacher_camera_set
from worldgs.room import pose_at
from worldgs.teacher_geometry import structure_report, sample_teacher_surface
from worldgs.train_teacher import validate_teacher


class MeshTeacherTest(unittest.TestCase):
    def setUp(self):
        self.layout = [dict(name='room', kind='room', min=[-4, -1.5, -4], max=[4, 1.5, 4])]
        self.mesh, _, self.parts, self.owners = build_components(self.layout, Image.new('RGB', (64, 64), 'tan'))
        self.renderer = MeshRenderer(self.mesh, self.parts, self.owners)

    def test_floor_and_ceiling_depth_are_exact_planes(self):
        for pitch, height in [(55, -1.5), (-55, 1.5)]:
            pose = pose_at(0, pitch)
            hit = self.renderer.render(intrinsics(64), pose, 64)
            xyz = lift(hit['depth'], intrinsics(64), pose)
            np.testing.assert_allclose(xyz[24:40, 24:40, 1], height, atol=1e-6)

    def test_all_yaws_and_translated_vertical_cameras(self):
        cameras = teacher_camera_set(self.layout, 8)
        for pitch in [0, 55, -55]:
            group = [p for label, p in cameras if label.startswith('room_yaw_') and label.endswith(f'pitch_{pitch}')]
            self.assertEqual(len(group), 8)
            self.assertGreater(np.ptp(np.array(group)[:, :3, 3], axis=0)[0], 2.)
        self.assertGreater(np.ptp(np.array([p[:3, 3] for _, p in cameras])[:, 1]), .8)

    def test_structure_gate_rejects_sagging_ceiling(self):
        points = sample_teacher_surface(self.mesh, self.renderer, self.owners, np.ones(len(self.mesh.faces), bool), count=100)
        self.assertTrue(structure_report(self.mesh, self.parts, points)['passed'])
        ceiling = np.flatnonzero(np.isclose(self.mesh.vertices[:, 1], -1.5))[0]
        self.mesh.vertices[ceiling, 1] += .2
        self.assertFalse(structure_report(self.mesh, self.parts, points)['passed'])

    def test_default_generation_uses_mesh_and_rejects_modified_teacher(self):
        from worldgs.pipeline import generate
        with tempfile.TemporaryDirectory() as folder, patch('worldgs.pipeline.Models', side_effect=AssertionError('Unexpected model inference')):
            meta = generate('test', folder, size=64, views=8, demo=True)
            self.assertEqual(meta['mode'], 'mesh_teacher_v8')
            self.assertTrue(all(f['reference_source'] == 'fixed_triangle_mesh_render' for f in meta['frames']))
            validate_teacher(folder)
            path = Path(folder)/'points.npz'
            with np.load(path) as saved:
                points = {k: saved[k] for k in saved.files}
            points['xyz'][0, 1] += .1
            np.savez_compressed(path, **points)
            with self.assertRaisesRegex(ValueError, 'teacher triangles'):
                validate_teacher(folder)

    def test_training_rejects_inconsistent_reference_depth(self):
        from worldgs.pipeline import generate
        with tempfile.TemporaryDirectory() as folder:
            meta = generate('test', folder, size=64, views=8, demo=True)
            path = Path(folder)/meta['frames'][0]['depth_file']
            depth = np.load(path)
            depth[32, 32] *= .5
            np.save(path, depth)
            with self.assertRaisesRegex(ValueError, 'Reference depth or camera'):
                validate_teacher(folder)

    def test_saved_layout_does_not_invoke_depth_estimation(self):
        from worldgs.pipeline import generate
        with tempfile.TemporaryDirectory() as folder:
            layout = Path(folder)/'layout.json'
            layout.write_text(json.dumps(dict(shapes=self.layout)))
            front = Path(folder)/'front.png'
            Image.new('RGB', (64, 64), 'tan').save(front)
            with patch('worldgs.pipeline.Models', side_effect=AssertionError('Unexpected depth estimation')):
                meta = generate('test', Path(folder)/'run', size=64, views=8, layout_file=str(layout), front_image=str(front))
            self.assertEqual(meta['front_depth_usage'], 'none; saved layout reused')
            self.assertFalse((Path(folder)/'run'/'front_relative_depth.npy').exists())
