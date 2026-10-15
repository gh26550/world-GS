import unittest
import numpy as np
from worldgs.geometry import intrinsics
from worldgs.room import Atlas, make_layout, trace, pose_at, trajectory


class RoomTest(unittest.TestCase):
    def setUp(self):
        self.K = intrinsics(64)
        self.layout = make_layout([6, 3, 6], [], np.full((64, 64), 3.), self.K)

    def test_closed_room_rays_cover_up_down_and_translations(self):
        for yaw, pitch, pose in trajectory(8):
            depth, xyz, sid = trace(self.layout, self.K, pose, 64)
            self.assertTrue((depth > 0).all() and np.isfinite(depth).all())
            self.assertTrue((np.abs(xyz[:, :, [0, 2]]) <= 3+1e-8).all())
            self.assertTrue((np.abs(xyz[:, :, 1]) <= 1.5+1e-8).all())
        centers = np.array([p[:3, 3] for _, _, p in trajectory(8)])
        self.assertGreater(np.linalg.norm(centers[0]-centers[4]), .3)

    def test_foreground_occludes_wall(self):
        self.layout.append({'name': 'object', 'min': [-1, -1, 1], 'max': [1, 1, 2]})
        depth, _, sid = trace(self.layout, self.K, pose_at(0), 64)
        self.assertAlmostEqual(depth[32, 32], 1.)
        self.assertGreaterEqual(sid[32, 32], 6)

    def test_camera_inside_furniture_is_rejected(self):
        self.layout.append({'name': 'bad', 'min': [-1, -1, -1], 'max': [1, 1, 1]})
        with self.assertRaises(ValueError):
            trace(self.layout, self.K, pose_at(0), 64)

    def test_atlas_preserves_world_surface_color(self):
        depth, points, sid = trace(self.layout, self.K, pose_at(0), 64)
        atlas = Atlas(self.layout, resolution=32)
        atlas.update(points, sid, np.full((64, 64, 3), [25, 80, 130], np.uint8))
        color, known = atlas.render(points, sid)
        self.assertTrue(known.all())
        np.testing.assert_array_equal(color[0, 0], [25, 80, 130])
        samples = atlas.sample(spacing=.2)
        self.assertTrue((samples['xyz'] >= samples['bounds_min']-1e-7).all())
        self.assertTrue((samples['xyz'] <= samples['bounds_max']+1e-7).all())

    def test_detected_proxy_top_projects_to_image_box(self):
        detections = [{'box': [10, 35, 50, 62], 'label': 'sofa', 'score': .9}]
        layout = make_layout([6, 3, 6], detections, np.full((64, 64), 3.), self.K)
        self.assertEqual(len(layout), 2)
        shape = layout[1]
        top = np.array([shape['min'][0], shape['min'][1], shape['min'][2]]) @ self.K.T
        self.assertAlmostEqual(top[1]/top[2], 35)
        self.assertEqual(shape['max'][1], 1.5)

    def test_ceiling_and_floor_materials_are_available(self):
        from PIL import Image
        atlas = Atlas(self.layout, resolution=32)
        atlas.initialize_materials(Image.fromarray(np.full((64, 64, 3), 200, np.uint8)))
        for pitch in [-90, 90]:
            _, points, sid = trace(self.layout, self.K, pose_at(0, pitch), 64)
            color, known = atlas.render(points, sid)
            self.assertTrue(known.all())
            self.assertTrue((color >= 199).all())
