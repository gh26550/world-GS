import unittest
import numpy as np
from worldgs.layout_consistency import layout_check,synthesis_regions,projected_objects,layout_displacement_loss

class LayoutTest(unittest.TestCase):
    def obj(self,name='sofa_00',label='sofa',box=(10,10,40,40)):
        mask=np.zeros((100,100),bool);x0,y0,x1,y1=box;mask[y0:y1,x0:x1]=True
        return dict(id=name,label=label,box=box,mask=mask,pixels=int(mask.sum()))
    def det(self,label='sofa',box=(10,10,40,40)):
        return dict(label=label,box=box,score=.9)
    def test_displaced_object_is_rejected(self):
        self.assertFalse(layout_check([self.obj()],[self.det(box=(60,60,90,90))],(100,100))['passed'])
    def test_missing_and_extra_instances_are_rejected(self):
        self.assertFalse(layout_check([self.obj()],[],(100,100))['passed'])
        self.assertFalse(layout_check([self.obj()],[self.det(),self.det(box=(60,60,90,90))],(100,100))['passed'])
    def test_local_shape_variation_is_allowed(self):
        self.assertTrue(layout_check([self.obj()],[self.det(box=(9,8,43,42))],(100,100))['passed'])
    def test_chair_cannot_replace_sofa(self):
        self.assertFalse(layout_check([self.obj()],[self.det('chair')],(100,100))['passed'])
    def test_one_detection_cannot_cover_two_instances(self):
        objects=[self.obj(),self.obj('sofa_01',box=(60,10,90,40))]
        self.assertFalse(layout_check(objects,[self.det()],(100,100))['passed'])
    def test_detector_part_duplicate_does_not_change_count(self):
        self.assertTrue(layout_check([self.obj()],[self.det(),self.det(box=(15,15,30,30))],(100,100))['passed'])
    def test_background_and_object_regions_preserve_other_objects(self):
        a=self.obj();b=self.obj('chair_00','chair',(42,10,60,40));regions=synthesis_regions([a,b],(100,100))
        self.assertFalse(regions[0]['mask'][a['mask']].any())
        self.assertFalse(regions[1]['mask'][b['mask']].any())
        self.assertTrue(regions[1]['mask'][a['mask']].any())
    def test_fully_occluded_object_is_not_required_in_image(self):
        hit=dict(faces=np.zeros((100,100),int));components=[dict(name='sofa_00',material='generated_asset'),dict(name='chair_00',material='generated_asset')]
        objects=projected_objects(hit,np.array([0]),components,dict(sofa_00='sofa',chair_00='chair'))
        self.assertEqual([o['id'] for o in objects],['sofa_00'])
    def test_coherent_translation_is_penalized_but_local_detail_is_free(self):
        import torch
        local=torch.tensor([[.1,0,0],[-.1,0,0]],requires_grad=True)
        self.assertEqual(float(layout_displacement_loss(local,[6,6])),0.)
        shifted=torch.tensor([[.2,0,0],[.2,0,0]],requires_grad=True)
        loss=layout_displacement_loss(shifted,[6,6]);self.assertGreater(float(loss),0.1)
        loss.backward();self.assertTrue(torch.isfinite(shifted.grad).all())

    def test_background_prompt_does_not_request_room_furniture(self):
        from worldgs.appearance_models import AppearanceModels
        from types import MethodType
        model=AppearanceModels.__new__(AppearanceModels);calls=[]
        def sample(self,prompt,guide,known,depth,style,index,attempt,negative_extra=''):
            calls.append((prompt,negative_extra));return guide.copy()
        model._sample=MethodType(sample,model)
        obj=self.obj();regions=synthesis_regions([obj],(100,100))
        model.generate('A room with sofa and bookshelves',np.zeros((100,100,3),np.uint8),np.zeros((100,100),bool),np.ones((100,100)),None,0,0,regions=regions)
        self.assertNotIn('sofa',calls[0][0]);self.assertNotIn('bookshel',calls[0][0])
        self.assertIn('sofa',calls[1][0])
    def test_modified_layout_contract_cannot_be_trained(self):
        import tempfile,json,hashlib
        from pathlib import Path
        from worldgs.appearance import validate_appearance
        with tempfile.TemporaryDirectory() as root:
            p=Path(root);(p/'layout_contract.json').write_text('changed')
            audit=dict(version=17,complete=True,passed=True,mesh_digest='mesh',layout_contract_sha256=hashlib.sha256(b'original').hexdigest())
            (p/'audit.json').write_text(json.dumps(audit))
            with self.assertRaisesRegex(ValueError,'layout contract'):
                validate_appearance(p,dict(appearance_audit='audit.json',mesh_digest='mesh'))

    def test_nested_source_part_requires_mask_and_matched_parent(self):
        from worldgs.layout_consistency import calibrate_source_parts
        obj=self.obj();detections=[self.det(),self.det('chair',(15,15,30,30))]
        calibrated=calibrate_source_parts([obj],detections,(100,100),{'sofa_00':obj['mask']})
        self.assertTrue(layout_check([obj],detections,(100,100),calibrated['parts'])['passed'])
        self.assertFalse(layout_check([obj],[detections[1]],(100,100),calibrated['parts'])['passed'])
        self.assertFalse(layout_check([obj],[self.det(),self.det('chair',(60,60,90,90))],(100,100),calibrated['parts'])['passed'])
        no_mask=calibrate_source_parts([obj],detections,(100,100),{'sofa_00':np.zeros((100,100),bool)})
        self.assertEqual(no_mask['parts'],{})
    def test_structural_swatch_cannot_carry_furniture_texture(self):
        from worldgs.layout_consistency import structure_palette_guide
        rgb=np.full((100,100,3),200,np.uint8);rgb[10:30,10:30]=0;hit={'faces':np.zeros((100,100),int)}
        result=structure_palette_guide(rgb,hit,np.array([0]))
        self.assertTrue((result==200).all())

    def test_aggregate_recognition_cannot_hide_missing_parent_or_new_location(self):
        a=self.obj();detections=[self.det(),self.det('bed',(11,11,39,39))]
        self.assertTrue(layout_check([a],detections,(100,100))['passed'])
        self.assertFalse(layout_check([a],[detections[1]],(100,100))['passed'])
        self.assertFalse(layout_check([a],[self.det(),self.det('bed',(60,60,90,90))],(100,100))['passed'])
    def test_subresolution_object_is_preserved_not_resynthesized(self):
        a=self.obj();a['requires_detection']=False
        regions=synthesis_regions([a],(100,100))
        self.assertEqual(len(regions),1)
        self.assertFalse(regions[0]['mask'][a['mask']].any())
        self.assertTrue(layout_check([a],[],(100,100))['passed'])

    def test_world_surface_edits_cannot_change_architecture_or_other_instance(self):
        from worldgs.layout_consistency import surface_regions,ownership_check
        a=self.obj();b=self.obj('chair_00','chair',(42,10,60,40));regions=surface_regions([a,b],(100,100))
        self.assertFalse(regions[0]['mask'][b['mask']].any())
        guide=np.zeros((100,100,3),np.uint8);rgb=guide.copy();rgb[regions[0]['mask']]=100
        self.assertTrue(ownership_check(rgb,guide,regions)['passed'])
        rgb[90,90]=100
        self.assertFalse(ownership_check(rgb,guide,regions)['passed'])

    def test_matched_table_box_does_not_remove_floor_between_legs(self):
        from worldgs.layout_consistency import source_exclusion_mask
        obj=dict(id='table_00',label='coffee table',evidence='front',source_box=[.2,.2,.8,.8])
        detections=[dict(label='table',box=[20,20,80,80]),dict(label='chair',box=[0,0,15,20])]
        mask=source_exclusion_mask((100,100),[obj],detections,100)
        self.assertFalse(mask[60,50]);self.assertTrue(mask[10,5])
