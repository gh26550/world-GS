import unittest
import numpy as np
from worldgs.room_method import protection_mask,parallax_audit,whole_room_prompt

class FakeRenderer:
    def render(self,K,pose,size):
        return dict(faces=np.zeros((size,size),int),xyz=np.broadcast_to([0.,0.,2.],(size,size,3)))

class RoomMethodTest(unittest.TestCase):
    def frame(self,x=0,rotation=False):
        pose=np.eye(4);pose[0,3]=x
        if rotation:pose[:3,:3]=np.diag([-1,1,-1])
        return dict(file_path=str(x)+str(rotation)+'.png',K=[[32,0,31.5],[0,32,31.5],[0,0,1]],transform_matrix=pose.tolist())
    def test_rotation_alone_is_not_multiview_shape_support(self):
        result=parallax_audit(FakeRenderer(),[dict(name='sofa',material='generated_asset')],np.array([0]),[self.frame(),self.frame(rotation=True)])
        self.assertFalse(result['passed'])
    def test_translated_views_support_object(self):
        result=parallax_audit(FakeRenderer(),[dict(name='sofa',material='generated_asset')],np.array([0]),[self.frame(-.5),self.frame(.5)])
        self.assertTrue(result['passed'])
    def test_translated_furniture_is_not_frozen_to_source_projection(self):
        hit=dict(faces=np.array([[0,1]]));known=np.ones((1,2),bool)
        protected=protection_mask(hit,self.frame(.5),known,np.eye(4),np.array([0,6]))
        self.assertEqual(protected.tolist(),[[False,False]])
        self.assertTrue(protection_mask(hit,self.frame(),known,np.eye(4),np.array([0,6])).all())
    def test_source_prompt_retains_user_requirements(self):
        prompt=whole_room_prompt('A room with red chairs')
        self.assertIn('red chairs',prompt);self.assertIn('three walls',prompt)
        self.assertIn('no fisheye',prompt)

    def test_single_chair_does_not_select_leftmost_low_confidence_seat(self):
        from worldgs.image_room import choose_front_detection
        detections=[dict(box=[0,0,10,10],score=.33),dict(box=[90,0,100,10],score=.75)]
        self.assertEqual(choose_front_detection([0,1],detections,True),1)
        self.assertEqual(choose_front_detection([0,1],detections,False),0)
