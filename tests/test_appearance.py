import hashlib
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from worldgs.appearance_geometry import transfer_appearance,geometry_check
from worldgs.appearance import validate_appearance,projection_fingerprint

class AppearanceTest(unittest.TestCase):
    def scene(self):
        size=16;y,x=np.indices((size,size));K=np.array([[10,0,7.5],[0,10,7.5],[0,0,1.]])
        xyz=np.stack([(x-7.5)/10,(y-7.5)/10,np.ones_like(x)],-1)
        hit=dict(xyz=xyz,normals=np.broadcast_to([0,0,-1],xyz.shape),depth=np.ones((size,size)),faces=np.zeros((size,size),int))
        frame=dict(K=K.tolist(),transform_matrix=np.eye(4).tolist())
        entry=dict(frame=frame,depth=hit['depth'].copy(),valid=np.ones((size,size),bool),rgb=np.full((size,size,3),127,np.uint8),confidence=np.ones((size,size)))
        return hit,frame,entry
    def test_shared_surface_inherits_color(self):
        hit,frame,entry=self.scene();rgb,known,confidence=transfer_appearance(hit,frame,[entry])
        self.assertTrue(known[8,8]);self.assertEqual(rgb[8,8,0],127);self.assertEqual(confidence[8,8],1)
    def test_occluded_source_does_not_paint_surface(self):
        hit,frame,entry=self.scene();entry['depth'][:]=.5
        _,known,_=transfer_appearance(hit,frame,[entry]);self.assertFalse(known.any())
    def test_depth_gate_rejects_missing_object_boundary(self):
        depth=np.ones((48,48));depth[16:32,16:32]=.5
        faces=np.zeros((48,48),int);faces[16:32,16:32]=1
        self.assertTrue(geometry_check(depth,1/depth,faces,np.array([0,1]),tolerance=3)['passed'])
        self.assertFalse(geometry_check(depth,np.ones_like(depth),faces,np.array([0,1]),tolerance=3)['passed'])
    def test_modified_photo_cannot_be_trained(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root);(path/'photo.png').write_bytes(b'original');(path/'appearance_request.json').write_text('{}')
            audit=dict(complete=True,passed=True,mesh_digest='mesh',request_sha256=hashlib.sha256(b'{}').hexdigest(),
                       views=[dict(frame='photo.png',passed=True,image_sha256=hashlib.sha256(b'original').hexdigest())])
            (path/'appearance_audit.json').write_text(json.dumps(audit))
            meta=dict(appearance_audit='appearance_audit.json',mesh_digest='mesh',frames=[dict(file_path='photo.png')],heldout_frames=[])
            audit['projection_sha256']=projection_fingerprint(meta)
            (path/'appearance_audit.json').write_text(json.dumps(audit))
            validate_appearance(path,meta);(path/'photo.png').write_bytes(b'modified')
            with self.assertRaisesRegex(ValueError,'RGB changed'):validate_appearance(path,meta)

    def test_extra_furniture_without_mesh_support_is_rejected(self):
        from worldgs.appearance_geometry import semantic_support
        faces=np.zeros((32,32),int)
        detection=[dict(label='sofa',box=[4,4,28,28])]
        result=semantic_support(detection,faces,np.array([0]),[dict(name='room_0_0')])
        self.assertFalse(result['passed'])
        result=semantic_support(detection,faces,np.array([0]),[dict(name='sofa_00')])
        self.assertTrue(result['passed'])
    def test_structural_depth_plane_rejects_large_surface_relief(self):
        from worldgs.appearance_geometry import structural_planarity
        y,x=np.indices((48,48));faces=np.zeros((48,48),int);mask=np.ones_like(faces,bool)
        planar=x+y*.5
        self.assertTrue(structural_planarity(planar,faces,np.array([0]),mask)['passed'])
        checker=(x//8+y//8)%2
        self.assertFalse(structural_planarity(checker,faces,np.array([0]),mask)['passed'])

    def test_incomplete_photo_packaging_cannot_use_mesh_training(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root);(path/'appearance_request.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'incomplete'):validate_appearance(path,{})
    def test_blank_full_frame_detection_is_not_treated_as_furniture(self):
        from worldgs.appearance_geometry import semantic_support
        faces=np.zeros((32,32),int);detection=[dict(label='table',box=[0,0,32,32])]
        result=semantic_support(detection,faces,np.array([0]),[dict(name='room_0_0')],rgb=np.full((32,32,3),220,np.uint8))
        self.assertTrue(result['passed']);self.assertEqual(len(result['ignored_unlocalized_detections']),1)

    def test_chair_detection_on_sofa_geometry_uses_seating_family(self):
        from worldgs.appearance_geometry import semantic_support
        faces=np.zeros((32,32),int);detection=[dict(label='chair',box=[4,4,28,28])]
        result=semantic_support(detection,faces,np.array([0]),[dict(name='sofa_00')])
        self.assertTrue(result['passed'])

    def test_discarding_images_cannot_hide_missing_camera_coverage(self):
        from worldgs.reference_coverage import audit_planned_coverage
        from unittest.mock import patch
        frame=dict(file_path='view.png',target='wall',K=[[8,0,3.5],[0,8,3.5],[0,0,1]],transform_matrix=np.eye(4).tolist())
        hit=dict(depth=np.ones((8,8)),faces=np.zeros((8,8),int))
        with patch('worldgs.reference_coverage.MeshRenderer') as renderer:
            renderer.return_value.render.return_value=hit
            self.assertTrue(audit_planned_coverage(None,[],None,[frame],[frame],8,size=8)['passed'])
            self.assertFalse(audit_planned_coverage(None,[],None,[frame],[],8,size=8)['passed'])

    def test_guide_detector_error_is_not_attributed_to_generation(self):
        from worldgs.appearance_geometry import semantic_support
        faces=np.zeros((32,32),int);detection=[dict(label='table',box=[0,24,32,32])]
        result=semantic_support(detection,faces,np.array([0]),[dict(name='room_1_1')],baseline_detections=detection)
        self.assertTrue(result['passed'])
        result=semantic_support(detection,faces,np.array([0]),[dict(name='room_1_1')])
        self.assertFalse(result['passed'])
