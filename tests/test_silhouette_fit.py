import unittest
import numpy as np


class SilhouetteFitTest(unittest.TestCase):
    def test_rigid_fit_improves_visible_mask_without_deforming_mesh(self):
        import trimesh
        from worldgs.geometry import intrinsics
        from worldgs.mesh_scene import MeshRenderer,look_at
        from worldgs.silhouette_fit import refine
        mesh=trimesh.creation.box(extents=[1,1,1]);original=mesh.vertices.copy()
        obj=dict(id='generic_furniture',yaw=0,bottom_y=1.5)
        K=intrinsics(32,65);pose=look_at([0,0,-3],[0,0,1])
        truth=mesh.copy();truth.apply_translation([0,1,1])
        hit=MeshRenderer(truth,[dict(material='generated_asset')],np.zeros(len(truth.faces),int)).render(K,pose,32)
        values=np.array([65,0,-3,0,.6,1,0,0.])
        result,report=refine(values,[40,-1,-4,-.3,-1,.5,-.2,-1],[90,1,-2,.3,1,1.5,.2,1],
            [mesh],[(0,obj)],[hit['faces']>=0],lambda v:(K,pose),lambda v:np.zeros(1),32,resolution=32)
        self.assertLess(report['after_score'],report['before_score'])
        np.testing.assert_array_equal(mesh.vertices,original)
        self.assertTrue(np.isfinite(result).all())
