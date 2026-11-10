import unittest
import numpy as np
from worldgs.controlled_revision import assess_v4,project_overlap,camera_quality
class ControlledRevisionTest(unittest.TestCase):
 def scene(self):
  d=np.full((96,96),4.);d[24:70,25:69]=2.;faces=np.zeros_like(d,int);faces[24:70,25:69]=1
  rgb=np.full((96,96,3),40,np.uint8);rgb[24:70,25:69]=220
  h=dict(depth=d,faces=faces)
  return h,rgb
 def test_missing_furniture_depth_is_not_rescued_by_texture(self):
  h,rgb=self.scene();c=[dict(group='room'),dict(group='sofa',material='generated_asset')]
  self.assertFalse(assess_v4(h,h['depth'],np.full((96,96),4.),rgb,np.array([0,1]),c)['passed'])
 def test_coplanar_surface_uses_rgb_not_invented_depth(self):
  h,rgb=self.scene();h['depth'][:]=4.;c=[dict(group='room'),dict(group='thin_surface')]
  r=assess_v4(h,h['depth'],h['depth'],rgb,np.array([0,1]),c)
  self.assertTrue(r['passed']);self.assertTrue(any(g['reason']=='coplanar_or_estimator_unreliable_rgb' for g in r['groups']))
 def test_no_boundary_evidence_is_not_automatic_acceptance(self):
  h=dict(depth=np.full((96,96),4.),faces=np.zeros((96,96),int))
  self.assertFalse(assess_v4(h,h['depth'],h['depth'],np.full((96,96,3),100,np.uint8),np.array([0]),[dict(group='room')])['passed'])
 def test_occluded_reference_cannot_transfer_appearance(self):
  yy,xx=np.indices((16,16));xyz=np.stack([xx*2,yy*2,np.full_like(xx,2)],-1).astype(float);hit=dict(xyz=xyz,faces=np.zeros((16,16),int));f=dict(K=np.eye(3).tolist(),transform_matrix=np.eye(4).tolist())
  _,_,visible=project_overlap(hit,f,f,np.full((16,16),1.),16);self.assertFalse(visible.any())
  _,_,visible=project_overlap(hit,f,f,np.full((16,16),2.),16);self.assertTrue(visible.any())
 def test_near_camera_is_flagged(self):
  h,rgb=self.scene();h['depth'][24:70,25:69]=.3
  q=camera_quality(h,np.array([0,1]),[dict(group='room'),dict(group='sofa',material='generated_asset')]);self.assertFalse(q['safe'])

 def test_distant_partial_furniture_is_not_rejected(self):
  h,rgb=self.scene();h['faces'][24:70,0:69]=1;h['depth'][24:70,0:69]=2.
  q=camera_quality(h,np.array([0,1]),[dict(group='room'),dict(group='sofa',material='generated_asset')]);self.assertTrue(q['safe'])

 def test_comparison_copy_does_not_mutate_original_points(self):
  import tempfile
  from pathlib import Path
  from worldgs.controlled_revision import clone_tree
  with tempfile.TemporaryDirectory() as tmp:
   src=Path(tmp)/'source';dst=Path(tmp)/'copy';src.mkdir();(src/'points.npz').write_bytes(b'original')
   clone_tree(src,dst);(dst/'points.npz').write_bytes(b'revised')
   self.assertEqual((src/'points.npz').read_bytes(),b'original')
