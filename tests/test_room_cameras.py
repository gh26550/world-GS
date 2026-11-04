import unittest,tempfile,json,hashlib
from pathlib import Path
import numpy as np
import trimesh
from worldgs.room_cameras import room_camera_plan,safe_eye,nudge_eye,rotational_order
from worldgs.room_verification import verify_depth

class RoomCameraTest(unittest.TestCase):
 def setUp(self):self.shapes=[dict(name='room',min=[-3,-1.5,-3],max=[3,1.5,3]),dict(name='cabinet',min=[-1,-.2,2.3],max=[1,1.5,2.95])]
 def test_room_views_have_two_bootstraps_and_multiple_heights(self):
  frames=room_camera_plan(self.shapes)
  self.assertEqual(sum(f['camera_kind']=='bootstrap' for f in frames),2)
  self.assertEqual(sum(f['camera_kind']=='overhead' for f in frames),8)
  self.assertGreaterEqual(len(frames),24)
  self.assertGreater(np.ptp([np.array(f['transform_matrix'])[1,3] for f in frames]),.5)
  for f in frames:self.assertTrue(safe_eye(np.array(f['transform_matrix'])[:3,3],self.shapes[0],self.shapes[1:]))
 def test_colliding_camera_is_moved_without_moving_furniture(self):
  original=json.dumps(self.shapes);eye=nudge_eye([0,1,2.7],self.shapes[0],self.shapes[1:])
  self.assertTrue(safe_eye(eye,self.shapes[0],self.shapes[1:]));self.assertEqual(original,json.dumps(self.shapes))
 def test_camera_rotation_is_proper_and_looks_into_room(self):
  for f in room_camera_plan(self.shapes):
   r=np.array(f['transform_matrix'])[:3,:3];np.testing.assert_allclose(r.T@r,np.eye(3),atol=1e-7);self.assertAlmostEqual(np.linalg.det(r),1.)
 def test_bootstraps_remain_first_in_progressive_order(self):
  rows=rotational_order(room_camera_plan(self.shapes));self.assertEqual([f['camera_kind'] for f in rows[:2]],['bootstrap','bootstrap'])
 def test_depth_gate_rejects_missing_furniture_edge(self):
  depth=np.ones((64,64))*4;depth[20:45,20:45]=2
  self.assertTrue(verify_depth(depth,depth*3)['passed'])
  self.assertFalse(verify_depth(depth,np.ones_like(depth))['passed'])
 def test_flat_wall_does_not_claim_informative_depth_check(self):
  result=verify_depth(np.ones((64,64)),np.ones((64,64)));self.assertTrue(result['passed']);self.assertFalse(result['informative'])
 def test_incompatible_depth_resolution_is_rejected(self):
  with self.assertRaises(ValueError):verify_depth(np.ones((64,64)),np.ones((32,32)))
 def test_local_reference_tampering_is_rejected(self):
  from worldgs.local_room import validate_local
  from worldgs.appearance import projection_fingerprint
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)
   for name in ['scene_mesh.glb','source_front.png','room.png']:(p/name).write_bytes(name.encode())
   sha=lambda name:hashlib.sha256((p/name).read_bytes()).hexdigest()
   meta=dict(local_room_audit='audit.json',frames=[dict(file_path='room.png',K=[],transform_matrix=[])],heldout_frames=[])
   audit=dict(complete=True,passed=True,projection_sha256=projection_fingerprint(meta),mesh_sha256=sha('scene_mesh.glb'),source_sha256=sha('source_front.png'),views=[dict(file_path='room.png',passed=True,teacher_sha256=sha('room.png'))]);(p/'audit.json').write_text(json.dumps(audit));validate_local(p,meta)
   (p/'room.png').write_bytes(b'changed')
   with self.assertRaisesRegex(ValueError,'RGB changed'):validate_local(p,meta)
 def test_depth_success_does_not_override_wrong_camera_scale(self):
  from worldgs.room_verification import verify_layout
  obj=dict(id='sofa',label='sofa',box=[20,20,40,40],requires_detection=True,mask=np.zeros((64,64),bool))
  result=verify_layout([obj],[dict(label='sofa',box=[10,10,50,50],score=.9)],np.full((64,64,3),120,np.uint8))
  self.assertFalse(result['passed']);self.assertEqual(result['missing_or_displaced'],['sofa'])
 def test_blank_wall_full_frame_label_is_not_furniture_evidence(self):
  from worldgs.room_verification import verify_layout
  result=verify_layout([],[dict(label='table',box=[0,0,64,64],score=.4)],np.full((64,64,3),120,np.uint8))
  self.assertTrue(result['passed']);self.assertEqual(len(result['unlocalized_flat_wall_detections']),1)
 def test_exterior_wall_faces_do_not_drive_camera_coverage(self):
  from worldgs.room_cameras import coverage_weights
  wall=trimesh.creation.box([.08,3,6]);wall.apply_translation([-3.04,0,0]);owners=np.zeros(len(wall.faces),int);components=[dict(name='room_0_0',group='room',bounds=wall.bounds.tolist())]
  weights,eligible=coverage_weights(wall,owners,components)
  self.assertEqual(int(eligible.sum()),2);self.assertAlmostEqual(float(weights.sum()),1.)
 def test_appearance_atlas_preserves_colors_without_preserving_layout(self):
  from worldgs.appearance_reference import appearance_atlas
  from PIL import Image
  a=np.arange(64*64*3,dtype=np.uint8).reshape(64,64,3);b=np.asarray(appearance_atlas(Image.fromarray(a),84))
  self.assertFalse(np.array_equal(a,b))
  for c in range(3):np.testing.assert_array_equal(np.sort(a[:,:,c].ravel()),np.sort(b[:,:,c].ravel()))
 def test_same_depth_objects_keep_a_boundary_condition(self):
  from worldgs.room_verification import instance_boundaries
  faces=np.zeros((32,32),int);faces[16:]=1
  edge=instance_boundaries(dict(faces=faces),np.array([3,13]));self.assertTrue(edge[16].all());self.assertFalse(edge[5].any())
 def test_unobservable_depth_requires_independent_rgb_evidence(self):
  depth=np.ones((64,64))*4;depth[20:45,20:45]=2;flat=np.ones_like(depth)
  self.assertFalse(verify_depth(depth,flat,proxy_depth=flat)['passed'])
  image=np.full((64,64,3),50,np.uint8);image[20:45,20:45]=220
  self.assertTrue(verify_depth(depth,flat,proxy_depth=flat,rgb=image)['passed'])
 def test_plain_architecture_is_not_a_localized_furniture_detection(self):
  from worldgs.room_verification import verify_layout
  hit=dict(faces=np.zeros((64,64),int));depth=np.ones((64,64))*4
  result=verify_layout([],[dict(label='bookshelf',box=[10,10,54,54])],np.full((64,64,3),120,np.uint8),hit=hit,owners=np.array([4]),predicted_depth=depth)
  self.assertTrue(result['passed']);self.assertEqual(len(result['planar_architecture_ambiguities']),1)
