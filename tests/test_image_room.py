import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
from worldgs.image_room import checked_box, source_crop, prepare_image_room, plan_front
from worldgs.pipeline import generate


class ImageRoomTest(unittest.TestCase):
    def test_invalid_boxes_rejected(self):
        for box in ([0,0,1000,1000], [0,0,0,.3], [0,0,float('nan'),.4], None):
            with self.assertRaises((ValueError,TypeError)):
                checked_box(box)

    def test_source_crop_preserves_pixels_with_margin(self):
        image=Image.new('RGB',(200,100),'red')
        crop=source_crop(image,[.2,.2,.8,.8])
        self.assertEqual(crop.getpixel((0,0)),(255,255,255))
        self.assertEqual(crop.getpixel((crop.width//2,crop.height//2)),(255,0,0))

    def test_deep_sofa_does_not_remove_downward_camera_coverage(self):
        from worldgs.mesh_scene import teacher_camera_set
        lo=np.array([-2.,.4,-1.6]);hi=np.array([2.,1.5,1.6])
        layout=[dict(min=[-3,-1.5,-3],max=[3,1.5,3]),dict(name='deep_sofa',min=lo.tolist(),max=hi.tolist())]
        cameras=teacher_camera_set(layout,8)
        downward=[pose for name,pose in cameras if name.startswith('room_yaw') and 'pitch_-55' in name]
        self.assertEqual(len(downward),8)
        for pose in downward:
            eye=pose[:3,3]
            self.assertFalse(((eye>lo-.1)&(eye<hi+.1)).all())

    def test_default_routes_to_image_first_without_object_generation(self):
        with patch('worldgs.learned_scene.generate_learned_scene',return_value='mesh') as backend,patch('worldgs.appearance.synthesize_appearance',return_value='ok') as appearance:
            self.assertEqual(generate('room','out'),'ok')
            self.assertEqual(backend.call_args.args[1],Path('out_scaffold'))
            appearance.assert_called_once_with(Path('out_scaffold'),'out',size=512,resume=False)
            self.assertTrue(backend.call_args.kwargs['image_room'])

    def test_new_mode_uses_validated_mesh_trainer(self):
        import json
        from worldgs.train import train
        with tempfile.TemporaryDirectory() as temp:
            (Path(temp)/'scene.json').write_text(json.dumps({'mode':'image_room_mesh_v12'}))
            with patch('worldgs.train_teacher.validate_teacher') as validate,patch('worldgs.train_mesh.train_mesh',return_value='trained') as trainer:
                self.assertEqual(train(temp,20,2000),'trained')
                validate.assert_called_once_with(temp)
                trainer.assert_called_once_with(temp,20,2000)
            self.assertEqual(json.loads((Path(temp)/'status.json').read_text())['stage'],'trained')

    def test_visual_observation_is_not_promoted_to_original_user_request(self):
        from test_learned_scene import sample_plan
        with tempfile.TemporaryDirectory() as temp:
            with patch('worldgs.image_room.VisualPlanner') as vision,patch('worldgs.pipeline.Models') as models,patch('worldgs.semantic_plan.plan_with_llm',return_value=sample_plan()) as metric:
                vision.return_value.ask.return_value='A green chair is visible on the left; rear wall is unseen.'
                models.return_value.detect.return_value=[dict(label='chair',box=[10,20,40,80],score=.9)]
                plan=plan_front('a chair and drum',Image.new('RGB',(100,100),'blue'),Path(temp),None,'vision-test')
            self.assertEqual(metric.call_args.args[0],'a chair and drum')
            self.assertIn('green chair',metric.call_args.kwargs['visual_observation'])
            self.assertEqual(plan['objects'][0]['evidence'],'front')
            self.assertEqual(plan['objects'][1]['evidence'],'inferred')
            self.assertIsNone(plan['objects'][1]['source_box'])
            self.assertLess(plan['objects'][0]['position'][0],0)

    def test_crop_provenance_and_no_t2i_fallback(self):
        plan=dict(style='white',room=dict(size=[6,3,6]),objects=[dict(id='chair',evidence='front',source_box=[.2,.2,.8,.8])])
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            front=folder/'input.png'
            Image.new('RGB',(64,64),'blue').save(front)
            (folder/'room_images').mkdir()
            source=folder/'room_images/front.png'
            Image.new('RGB',(64,64),'blue').save(source)
            with patch('worldgs.image_room.plan_front',return_value=plan),patch('worldgs.image_room.completion_views',return_value=[source]):
                prepare_image_room('room',folder,64,42,front_image=front)
            self.assertTrue((folder/'assets/chair/reference_asset.png').exists())
            self.assertFalse(list(folder.glob('assets/*/candidate_*.png')))
            import json
            record=json.loads((folder/'assets/chair/source.json').read_text())
            self.assertEqual(record['source_image'],'room_images/front.png')
            self.assertFalse(record['isolated_text_to_image'])

    def test_missing_inferred_object_cannot_be_silently_replaced(self):
        plan=dict(style='white',room=dict(size=[6,3,6]),objects=[dict(id='chair',evidence='inferred',source_box=None)])
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            front=folder/'input.png'
            Image.new('RGB',(64,64),'blue').save(front)
            (folder/'planning').mkdir()
            (folder/'planning/completion_grounding.json').write_text('{"objects": [{"id":"chair","box":null}]}')
            with patch('worldgs.image_room.plan_front',return_value=plan),patch('worldgs.image_room.completion_views',return_value=[front]):
                with self.assertRaisesRegex(ValueError,'No isolated image fallback'):
                    prepare_image_room('room',folder,64,42,front_image=front)


if __name__=='__main__':
    unittest.main()
