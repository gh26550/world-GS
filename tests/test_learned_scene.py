import copy
import json
from pathlib import Path
import tempfile
import sys
import unittest
from unittest.mock import patch
import numpy as np
import trimesh
from worldgs.semantic_plan import validate_plan,repair_layout,extract_json
from worldgs.learned_scene import assemble,backend_environment,learned_cameras,run_worker
from worldgs.physical_render import blender_camera_matrix
from worldgs.room import pose_at


def sample_plan():
    return dict(room=dict(size=[8,3,8],floor='wood',walls='plaster',ceiling='plaster'),style='cozy',
        objects=[dict(id='chair_00',label='chair',description='a green wooden chair',dimensions=[.6,1.,.6],position=[0.,2.],yaw=0),
                 dict(id='drum_00',label='drum',description='one small red drum',dimensions=[.5,.5,.5],position=[1.,1.],yaw=35)],
        requirements=[dict(id='r0',text='a chair and a drum',object_ids=['chair_00','drum_00'],kind='object')],
        relations=[],lighting=dict(power=1000,color=[1.,.9,.8],description='warm'),assumptions=[],unsupported=[])


class LearnedSceneTest(unittest.TestCase):
    def test_arbitrary_object_categories_are_not_a_three_object_template(self):
        self.assertTrue(validate_plan(sample_plan())['passed'])

    def test_layout_solver_preserves_objects_and_meets_relations(self):
        plan=sample_plan()
        plan['objects'][0]['position']=[5.,6.]
        plan['objects'][1]['position']=[5.,6.]
        plan['relations']=[dict(subject='drum_00',relation='in_front_of',object='chair_00')]
        before=copy.deepcopy(plan['requirements'])
        fixed=repair_layout(plan)
        self.assertTrue(validate_plan(fixed)['passed'])
        self.assertEqual(fixed['requirements'],before)
        self.assertEqual(len(fixed['objects']),2)
        self.assertTrue(fixed['layout_adjustments'])

    def test_missing_and_unsupported_requirements_stop_generation(self):
        plan=sample_plan()
        plan['requirements'][0]['object_ids']=['chair_00']
        with self.assertRaisesRegex(ValueError,'without requirement provenance'):
            validate_plan(plan)
        plan=sample_plan()
        plan['unsupported']=['two storey room']
        with self.assertRaisesRegex(ValueError,'Unsupported'):
            validate_plan(plan)
        plan=sample_plan()
        del plan['unsupported']
        with self.assertRaisesRegex(ValueError,'Missing or invalid scene field'):
            validate_plan(plan)
        plan=sample_plan()
        plan['objects'][0]['id']='../escape'
        with self.assertRaisesRegex(ValueError,'safe identifiers'):
            validate_plan(plan)

    def test_scene_json_is_data_and_not_executed(self):
        value=extract_json('```json\n{"style":"$(rm anything)"}\n```')
        self.assertEqual(value['style'],'$(rm anything)')

    def test_front_relation_respects_reference_object_heading(self):
        plan=sample_plan()
        plan['objects'][0].update(position=[1.,1.],yaw=90)
        plan['objects'][1].update(position=[-.2,1.],yaw=0)
        plan['relations']=[dict(subject='drum_00',relation='in_front_of',object='chair_00',frame='object')]
        self.assertTrue(validate_plan(plan)['passed'])
        plan['relations'][0]['frame']='world'
        with self.assertRaisesRegex(ValueError,'Unsatisfied spatial relation'):
            validate_plan(plan)

    def test_opencv_blender_camera_projection_agrees(self):
        world_to_blender=np.array([[1,0,0],[0,0,1],[0,-1,0]])
        for yaw,pitch in [(0,0),(90,55),(210,-55)]:
            pose=pose_at(yaw,pitch,[.3,-.2,.5])
            camera_point=np.array([.4,.2,3.])
            world=pose[:3,:3]@camera_point+pose[:3,3]
            transform=blender_camera_matrix(pose)
            local=transform[:3,:3].T@(world_to_blender@world-transform[:3,3])
            np.testing.assert_allclose(local,[.4,-.2,-3.],atol=1e-7)
            self.assertAlmostEqual(np.linalg.det(transform[:3,:3]),1.)

    def test_assembly_preserves_generated_shape_and_contacts_floor(self):
        plan=sample_plan()
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)
            for obj in plan['objects']:
                path=folder/'assets'/obj['id']
                path.mkdir(parents=True)
                mesh=trimesh.creation.icosphere(subdivisions=2)
                mesh.apply_scale([.7,1.,.4])
                mesh.visual.vertex_colors=np.tile([60,180,70,255],(len(mesh.vertices),1))
                mesh.export(path/'generated.glb')
            mesh,parts,owners,layout=assemble(plan,folder)
            self.assertEqual(len(parts),8)
            for obj in plan['objects']:
                placed=trimesh.load(folder/'assets'/obj['id']/'placed.glb',force='mesh')
                self.assertAlmostEqual(placed.bounds[1,1],1.5,places=6)
                self.assertGreater(len(placed.faces),100)
                self.assertEqual(placed.visual.kind,'vertex')
            self.assertTrue((folder/'scene_mesh.glb').exists())
            self.assertEqual(owners.max(),7)

    def test_backend_cuda_search_is_scoped_to_selected_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix=Path(directory)
            library=prefix/'lib/python3.12/site-packages/nvidia/cu13/lib'
            library.mkdir(parents=True)
            result=backend_environment(prefix/'bin/python',{'PATH':'original'})
            self.assertTrue(result['LD_LIBRARY_PATH'].startswith(str(library)))
            self.assertEqual(result['PATH'],'original')

    def test_room_edges_have_direct_structure_reference_cameras(self):
        layout=[dict(name='room',kind='room',min=[-4,-1.5,-4],max=[4,1.5,4])]
        cameras=learned_cameras(layout,8)
        self.assertEqual(sum(name.startswith('structure_corner_') for name,_ in cameras),8)
        self.assertEqual(sum(name.startswith('structure_grid_') for name,_ in cameras),18)
        for name,pose in cameras:
            if name.startswith('structure_grid_'):
                self.assertAlmostEqual(abs(pose[1,2]),1.)

    def test_backend_failure_is_not_replaced_by_placeholder_success(self):
        with tempfile.TemporaryDirectory() as folder:
            log=Path(folder)/'worker.log'
            with self.assertRaisesRegex(RuntimeError,'No placeholder mesh'):
                run_worker([sys.executable,'-c','print("failure evidence"); raise SystemExit(17)'],log)
            self.assertIn('failure evidence',log.read_text())
            run_worker([sys.executable,'-c','print("Reconstructed verified")'],log)
            self.assertTrue((Path(folder)/'worker.attempt_01.log').exists())

    def test_default_cli_uses_learned_generation(self):
        from worldgs.cli import main
        with patch('sys.argv',['worldgs','generate','--prompt','a bedroom','--output','unused']),patch('worldgs.pipeline.generate') as generation:
            main()
        self.assertEqual(generation.call_args.kwargs['mode'],'learned')
