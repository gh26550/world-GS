import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from worldgs.scene_plan import plan_scene, requirements
from worldgs.mesh_scene import build_components, MeshRenderer
from worldgs.surface_appearance import SurfaceAppearance


class PromptSceneTest(unittest.TestCase):
    def test_inventory_includes_objects_absent_from_front_image(self):
        plan = plan_scene('A cozy living room with a sofa, bookshelves and a wooden table')
        self.assertEqual({r['category']:r['count'] for r in plan['requirements']},dict(sofa=1,bookshelf=2,table=1))
        self.assertEqual(len(plan['shapes']),5)
        self.assertTrue(all(s['source']=='prompt_inventory' for s in plan['shapes'][1:]))

    def test_counts_and_negation(self):
        self.assertEqual(requirements('two sofas and three tables without bookshelves'),
            [dict(category='sofa',count=2,source='explicit prompt requirement'),dict(category='table',count=3,source='explicit prompt requirement')])
        self.assertEqual(requirements('ソファ1台，本棚2台とテーブル1台')[1]['count'],1)
        with self.assertRaises(ValueError):
            requirements('a cathedral with a dragon')
        with self.assertRaisesRegex(ValueError,'no implemented mesh backend'):
            requirements('a sofa and a bed')

    def test_prompt_mesh_contains_volumetric_parts_and_rotated_shelves(self):
        plan = plan_scene('a sofa, bookshelves and a table')
        mesh,_,parts,owners = build_components(plan['shapes'],Image.new('RGB',(64,64),'tan'))
        self.assertEqual(sum(p['name']=='leg' for p in parts),4)
        self.assertGreater(sum(p['name'].startswith('book_') for p in parts),10)
        for shape in plan['shapes'][1:]:
            vertices = np.concatenate([mesh.triangles[owners==i].reshape(-1,3) for i,p in enumerate(parts) if p['group']==shape['name']])
            self.assertTrue((vertices.min(0) >= np.asarray(shape['min'])-.08).all())
            self.assertTrue((vertices.max(0) <= np.asarray(shape['max'])+.08).all())

    def test_shared_texture_preserves_first_observation(self):
        plan = plan_scene('a table')
        mesh,_,parts,owners = build_components(plan['shapes'],Image.new('RGB',(64,64),'tan'))
        atlas = SurfaceAppearance(mesh)
        xyz,faces = mesh.triangles_center[:2],np.arange(2)
        atlas.update(xyz,faces,np.array([[200,10,20],[20,180,30]]))
        atlas.update(xyz,faces,np.full((2,3),255))
        colors,known = atlas.sample(xyz,faces,np.zeros((2,3)))
        self.assertTrue(known.all())
        np.testing.assert_allclose(colors,[[200,10,20],[20,180,30]])

    def test_experimental_scene_builds_prompt_inventory_without_depth(self):
        from worldgs.pipeline import generate
        from worldgs.train_teacher import validate_teacher
        with tempfile.TemporaryDirectory() as folder:
            meta = generate('a sofa, bookshelves and a table',folder,size=64,views=8,demo=True,mode='scene')
            self.assertEqual(meta['mode'],'prompt_mesh_radiance_v9')
            self.assertFalse((Path(folder)/'front_relative_depth.npy').exists())
            self.assertEqual(len(meta['requirements']),3)
            validate_teacher(folder)
