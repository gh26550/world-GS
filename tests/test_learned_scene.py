import copy
import json
from pathlib import Path
import tempfile
import sys
import unittest
from unittest.mock import patch
import numpy as np
import trimesh
from worldgs.semantic_plan import validate_plan,repair_layout,extract_json,normalize_size_locks
from worldgs.learned_scene import assemble,backend_environment,learned_cameras,run_worker,contact_height,configured_front_camera
from worldgs.physical_render import blender_camera_matrix,matte_room_finish
from worldgs.room import pose_at


def sample_plan():
    return dict(room=dict(size=[8,3,8],floor='wood',walls='plaster',ceiling='plaster'),style='cozy',
        objects=[dict(id='chair_00',label='chair',description='a green wooden chair',dimensions=[.6,1.,.6],position=[0.,2.],yaw=0),
                 dict(id='drum_00',label='drum',description='one small red drum',dimensions=[.5,.5,.5],position=[1.,1.],yaw=35)],
        requirements=[dict(id='r0',text='a chair and a drum',object_ids=['chair_00','drum_00'],kind='object')],
        relations=[],lighting=dict(power=1000,color=[1.,.9,.8],description='warm'),assumptions=[],unsupported=[])


class LearnedSceneTest(unittest.TestCase):
    def test_reference_camera_rejects_furniture_and_outside_room(self):
        layout=[dict(min=[-3,-1.5,-3],max=[3,1.5,3]),dict(min=[1,.5,1],max=[2,1.5,2])]
        pose=configured_front_camera(layout,dict(eye=[0,0,-2],target=[0,.3,1]))
        np.testing.assert_allclose(pose[:3,3],[0,0,-2])
        for eye in ([1.5,1.,1.5],[0,0,-4]):
            with self.assertRaisesRegex(ValueError,'Reference camera'):
                configured_front_camera(layout,dict(eye=eye,target=[0,0,1]))

    def test_window_is_an_actual_wall_aperture(self):
        from worldgs.window_geometry import add_window,window_bounds
        from worldgs.mesh_scene import build_components
        from PIL import Image
        from trimesh.ray.ray_pyembree import RayMeshIntersector
        room=dict(size=[6.,3.,6.],window=dict(x=0.,width=1.8,sill=.8,height=1.8))
        _,parts,components,_=build_components([dict(name='room',kind='room',min=[-3,-1.5,-3],max=[3,1.5,3])],Image.new('RGB',(32,32),'white'))
        add_window(room,parts,components)
        rays=RayMeshIntersector(parts[5])
        hits,ids,_=rays.intersects_location(np.array([[.2,0.,0.],[2.,0.,0.]]),np.array([[0.,0.,1.],[0.,0.,1.]]),multiple_hits=False)
        self.assertEqual(ids.tolist(),[1])
        self.assertEqual(len(parts),9)
        self.assertGreater(components[5]['faces'],12)
        room['window']['width']=7
        with self.assertRaisesRegex(ValueError,'window aperture'):
            window_bounds(room)

    def test_floor_rug_does_not_enclose_camera(self):
        plan=sample_plan()
        plan['objects'][0].update(label='rug',dimensions=[2.,.012,2.],position=[0.,0.])
        self.assertTrue(validate_plan(repair_layout(plan))['passed'])
        self.assertLess(np.linalg.norm(plan['objects'][0]['position']),.01)

    def test_window_preserves_room_plane_and_surface_contracts(self):
        from worldgs.window_geometry import add_window
        from worldgs.mesh_scene import build_components
        from worldgs.teacher_geometry import sample_teacher_surface,structure_report
        from PIL import Image
        room=dict(size=[6.,3.,6.],window=dict(x=0.,width=1.8,sill=.8,height=1.8))
        _,parts,components,_=build_components([dict(name='room',kind='room',min=[-3,-1.5,-3],max=[3,1.5,3])],Image.new('RGB',(32,32),'white'))
        add_window(room,parts,components)
        mesh=trimesh.util.concatenate(parts)
        owners=np.concatenate([np.full(len(part.faces),i,int) for i,part in enumerate(parts)])
        class Renderer:
            def shade(self,xyz,ids):
                return np.ones_like(xyz)*.5,mesh.face_normals[ids]
        points=sample_teacher_surface(mesh,Renderer(),owners,np.ones(len(mesh.faces),bool),count=2000)
        report=structure_report(mesh,components,points)
        self.assertTrue(report['passed'])
        self.assertEqual(report['structure']['room_2_1']['inner_triangles'],8)

    def test_inferred_dimensions_are_not_locked_by_object_counts(self):
        plan=sample_plan()
        plan['objects'][0]['size_locked']=True
        normalize_size_locks(plan,'A room with exactly ten separate objects.')
        self.assertFalse(plan['objects'][0]['size_locked'])
        self.assertTrue(plan['assumptions'])
        for prompt in ('A table 2m wide.','A table two meters wide.','A table six feet wide.'):
            plan=sample_plan()
            plan['objects'][0]['size_locked']=True
            normalize_size_locks(plan,prompt)
            self.assertTrue(plan['objects'][0]['size_locked'])

    def test_small_object_budget_keeps_room_spatial_coverage(self):
        from worldgs.coverage_sampling import component_stratified_seed_ids
        rng=np.random.default_rng(0)
        room=rng.uniform(-4,4,(30000,3))
        prop=rng.uniform(-.02,.02,(3000,3))
        xyz=np.r_[room,prop]
        groups=np.r_[np.zeros(len(room),int),np.ones(len(prop),int)]
        ids=component_stratified_seed_ids(xyz,groups,4000,0)
        self.assertEqual(len(np.unique(ids)),4000)
        self.assertGreaterEqual((groups[ids]==1).sum(),1000)
        for axis in range(3):
            self.assertLess(xyz[ids,axis].min(),-3.9)
            self.assertGreater(xyz[ids,axis].max(),3.9)

    def test_colour_only_walls_are_plain_and_wood_is_not(self):
        self.assertTrue(matte_room_finish('warm cream'))
        self.assertTrue(matte_room_finish('white plaster'))
        self.assertFalse(matte_room_finish('white wood panelling'))
        self.assertFalse(matte_room_finish('blue patterned wallpaper'))
    def test_supported_accessory_is_placed_on_parent_and_cycles_rejected(self):
        plan=sample_plan()
        plan['objects'][0].update(dimensions=[2.,.8,1.],position=[0.,2.])
        plan['objects'][1].update(dimensions=[.2,.3,.2],position=[0.,2.],support='chair_00')
        self.assertTrue(validate_plan(repair_layout(plan))['passed'])
        self.assertAlmostEqual(plan['objects'][1]['bottom_y'],.7)
        plan['objects'][0]['support']='drum_00'
        with self.assertRaisesRegex(ValueError,'Cyclic support'):
            validate_plan(plan)

    def test_actual_support_mesh_rejects_voids_and_places_contact(self):
        table=trimesh.creation.box(extents=[2.,.1,1.])
        child=trimesh.creation.box(extents=[.2,.3,.2])
        self.assertAlmostEqual(contact_height(table,child),-.05)
        child.apply_translation([2.,0.,0.])
        with self.assertRaisesRegex(ValueError,'not fully supported'):
            contact_height(table,child)

    def test_assembly_places_accessory_on_actual_mesh(self):
        plan=sample_plan()
        plan['objects'][0].update(dimensions=[2.,.2,1.],position=[0.,2.])
        plan['objects'][1].update(dimensions=[.2,.1,.2],position=[0.,2.],yaw=0,support='chair_00')
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)
            for obj in plan['objects']:
                path=folder/'assets'/obj['id']
                path.mkdir(parents=True)
                mesh=trimesh.creation.box(extents=obj['dimensions']).subdivide().subdivide()
                mesh.export(path/'generated.glb')
            assemble(plan,folder)
            parent=trimesh.load(folder/'assets/chair_00/placed.glb',force='mesh')
            child=trimesh.load(folder/'assets/drum_00/placed.glb',force='mesh')
            self.assertAlmostEqual(child.bounds[1,1],parent.bounds[0,1],places=5)
            self.assertEqual(len(json.loads((folder/'assembly.json').read_text())['support_contacts']),1)

    def test_small_components_receive_seed_budget(self):
        from worldgs.teacher_geometry import sample_teacher_surface
        large=trimesh.creation.box(extents=[10.,10.,10.])
        tiny=trimesh.creation.box(extents=[.05,.05,.05])
        mesh=trimesh.util.concatenate([large,tiny])
        owners=np.r_[np.zeros(len(large.faces),int),np.ones(len(tiny.faces),int)]
        class Renderer:
            def shade(self,xyz,ids):
                return np.ones_like(xyz)*.5,mesh.face_normals[ids]
        points=sample_teacher_surface(mesh,Renderer(),owners,np.ones(len(mesh.faces),bool),
                                     count=4000,seed=0,component_fraction=.35)
        self.assertGreater((points['component_ids']==1).sum(),500)

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

    def test_default_cli_uses_whole_room_image_generation(self):
        from worldgs.cli import main
        with patch('sys.argv',['worldgs','generate','--prompt','a bedroom','--output','unused']),patch('worldgs.pipeline.generate') as generation:
            main()
        self.assertEqual(generation.call_args.kwargs['mode'],'image-room')
