import hashlib,json,tempfile,unittest
from pathlib import Path
import numpy as np
from worldgs.observed_architecture import ray_plane,observed_window,observed_rug
from worldgs.external_references import validate_external
from worldgs.appearance import projection_fingerprint

class DualReferenceTest(unittest.TestCase):
 def test_window_reprojects_to_detected_rectangle(self):
  K=np.array([[100,0,50],[0,100,50],[0,0,1.]])
  pose=np.eye(4);box=[30,20,70,60];w=observed_window(box,K,pose,[6,3,6])
  pts=np.array([[w['x']-w['width']/2,1.5-w['sill']-w['height'],3],[w['x']+w['width']/2,1.5-w['sill'],3]])
  uv=pts@K.T;np.testing.assert_allclose((uv[:,:2]/uv[:,2:]).ravel(),box)
 def test_plane_behind_camera_is_rejected(self):
  with self.assertRaises(ValueError):ray_plane([[50,50]],np.array([[100,0,50],[0,100,50],[0,0,1.]]),np.eye(4),2,-3)
 def test_rug_is_thin_and_on_room_floor(self):
  pose=np.eye(4);pose[2,3]=-2.8
  lo,hi=observed_rug([20,75,80,95],np.array([[100,0,50],[0,100,50],[0,0,1.]]),pose,[6,3,6])
  self.assertAlmostEqual(hi[1]-lo[1],.008);self.assertLess(hi[1],1.5);self.assertTrue(np.all(hi>lo))
 def test_modified_external_teacher_is_blocked(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp);files={'scene_mesh.glb':b'mesh','source_front.png':b'source','photo.png':b'original'}
   for name,value in files.items():(p/name).write_bytes(value)
   sha=lambda name:hashlib.sha256((p/name).read_bytes()).hexdigest()
   meta=dict(external_reference_audit='audit.json',frames=[dict(file_path='photo.png',K=[],transform_matrix=[])],heldout_frames=[])
   a=dict(complete=True,passed=True,projection_sha256=projection_fingerprint(meta),mesh_sha256=sha('scene_mesh.glb'),source_sha256=sha('source_front.png'),views=[dict(file_path='photo.png',passed=True,teacher_sha256=sha('photo.png'))])
   (p/'audit.json').write_text(json.dumps(a));validate_external(p,meta)
   (p/'photo.png').write_bytes(b'replaced')
   with self.assertRaisesRegex(ValueError,'teacher changed'):validate_external(p,meta)

 def test_shape_only_views_do_not_displace_photo_sampling(self):
  from worldgs.train_mesh import training_frame_index
  frames=[dict(rgb_weight=1.),dict(rgb_weight=1.)]+[dict(rgb_weight=0.) for _ in range(20)]
  indices=[training_frame_index(frames,i) for i in range(80)]
  self.assertEqual(sum(i<2 for i in indices),60)
  self.assertEqual(len({i for i in indices if i>=2}),20)
 def test_invalid_rgb_supervision_weight_is_rejected(self):
  from worldgs.train_mesh import training_frame_index
  with self.assertRaises(ValueError):training_frame_index([dict(rgb_weight=-1.)],0)
