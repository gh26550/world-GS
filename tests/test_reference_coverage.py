import unittest
import numpy as np
from worldgs.reference_coverage import observed,scaled_intrinsics

class ReferenceCoverageTest(unittest.TestCase):
    def test_intrinsics_pixel_centers(self):
        K=np.array([[100,0,49.5],[0,100,49.5],[0,0,1.]])
        np.testing.assert_allclose(scaled_intrinsics(K,100,50),[[50,0,24.5],[0,50,24.5],[0,0,1]])
    def test_visibility_rejects_occluded_and_outside(self):
        K=[[10,0,2],[0,10,2],[0,0,1]]
        frames=[dict(K=K,transform_matrix=np.eye(4).tolist())]
        hits=[dict(depth=np.ones((5,5)),faces=np.zeros((5,5),int))]
        result=observed(np.array([[0,0,1],[0,0,2],[10,0,1],[0,0,-1]]),frames,hits,5,5)
        np.testing.assert_array_equal(result,[True,False,False,False])

    def test_wide_teacher_covers_surface_outside_front_camera(self):
        hit=dict(depth=np.ones((5,5)),faces=np.zeros((5,5),int))
        point=np.array([[.5,0,1]])
        narrow=dict(K=[[10,0,2],[0,10,2],[0,0,1]],transform_matrix=np.eye(4).tolist())
        wide=dict(K=[[2,0,2],[0,2,2],[0,0,1]],transform_matrix=np.eye(4).tolist())
        self.assertFalse(observed(point,[narrow],[hit],5,5)[0])
        self.assertTrue(observed(point,[wide],[hit],5,5)[0])
