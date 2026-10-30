import tempfile
from pathlib import Path
import unittest
import numpy as np
from PIL import Image, ImageOps
import trimesh
from worldgs.front_alignment import source_mask,crop_region,projection,box_iou,estimate_floor,fit_front,refit_scene
from worldgs.geometry import intrinsics
from worldgs.mesh_scene import look_at
from worldgs.front_alignment import projection_fingerprint, validate_front_audit


class FrontAlignmentTest(unittest.TestCase):
    def test_crop_mask_is_mapped_back_to_original_coordinates(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);asset=folder/'assets/a';asset.mkdir(parents=True)
            obj=dict(id='a',source_box=[.2,.3,.6,.8])
            lo,hi=crop_region(obj['source_box'],[100,100])
            mask=np.zeros((100,100),np.uint8);mask[30:80,20:60]=255
            crop=Image.fromarray(mask).crop((*lo,*hi))
            ImageOps.expand(crop,32,fill=0).save(asset/'mask.png')
            self.assertTrue(np.array_equal(source_mask(folder,obj,(100,100)),mask>127))

    def test_plane_prior_ignores_masked_objects(self):
        y,x=np.indices((80,80));points=np.stack([x/30,y*0+1.3,1+y/15],-1)
        excluded=np.zeros((80,80),bool);excluded[55:75,20:60]=True
        points[excluded,1]=.3
        floor=estimate_floor(points,np.ones((80,80),bool),excluded)
        self.assertAlmostEqual(floor['distance'],1.3,places=5)
        self.assertGreater(floor['inlier_fraction'],.95)

    def test_reprojection_fit_improves_image_alignment_without_warping_mesh(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);asset=folder/'assets/a';asset.mkdir(parents=True)
            size=256
            Image.new('RGB',(size,size),'white').save(folder/'source_front.png')
            target=trimesh.creation.box(extents=[1,1,1]);target.apply_translation([.8,1.,1.])
            pose=look_at([0,.2,-2.5],[0,.2,1])
            box,_=projection(target.vertices,intrinsics(size,70),pose,[size,size])
            pixel=np.rint(box*size).astype(int)
            lo,hi=crop_region(box,[size,size])
            mask=np.zeros((size,size),np.uint8);mask[pixel[1]:pixel[3],pixel[0]:pixel[2]]=255
            ImageOps.expand(Image.fromarray(mask).crop((*lo,*hi)),32,fill=0).save(asset/'mask.png')
            plan=dict(room=dict(size=[6,3,6]),objects=[dict(id='a',evidence='front',support='floor',source_box=box.tolist(),position=[-.7,.2],dimensions=[1.4,1.4,1.4],yaw=0)])
            mesh=trimesh.creation.box(extents=[1.4,1.4,1.4])
            fit_front(plan,[mesh],folder)
            report=plan['front_alignment']
            self.assertGreater(report['mean_box_iou_after'],.85)
            self.assertGreater(report['mean_box_iou_after'],report['mean_box_iou_before']+.2)
            np.testing.assert_allclose(mesh.extents/mesh.extents[0],[1,1,1])
            self.assertTrue(np.isfinite(plan['reference_intrinsics']).all())
            # Explicit metric dimensions must remain fixed, including a very
            # narrow optimizer bound; this previously made x0 infeasible.
            plan['objects'][0].update(position=[-.7,.2],dimensions=[1.4,1.4,1.4],yaw=0,size_locked=True)
            locked=trimesh.creation.box(extents=[1.4,1.4,1.4])
            fit_front(plan,[locked],folder)
            np.testing.assert_allclose(locked.extents,[1.4,1.4,1.4],atol=1e-6)

    def test_refit_cannot_overwrite_or_nest_inside_source(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):refit_scene(temp,Path(temp)/'child')

    def test_refit_external_image_uses_saved_png_identity_for_resume(self):
        import json,hashlib
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as temp:
            source=Path(temp)/'source';target=Path(temp)/'replay';source.mkdir()
            for name in ('assets','planning','materials','room_images'):(source/name).mkdir()
            Image.new('RGB',(64,64),'blue').save(source/'source_front.png')
            (source/'image_room_provenance.json').write_text('{}')
            (source/'request.json').write_text(json.dumps(dict(mode='image-room',prompt='room',size=64,views=8,seed=0,room_size=None,planner_model=None,front_image_sha256='original-jpeg-hash')))
            with patch('worldgs.learned_scene.generate_learned_scene',return_value='ok') as generate:
                self.assertEqual(refit_scene(source,target),'ok')
            actual=json.loads((target/'request.json').read_text())
            self.assertEqual(actual['front_image_sha256'],hashlib.sha256((target/'source_front.png').read_bytes()).hexdigest())
            self.assertEqual(generate.call_args.kwargs['front_image'],str(target/'source_front.png'))

    def test_failed_or_stale_alignment_cannot_be_used_for_gs_training(self):
        import json,hashlib
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            (folder/'source_front.png').write_bytes(b'original image bytes')
            meta=dict(front_alignment='front_alignment.json',K=np.eye(3).tolist(),frames=[dict(transform_matrix=np.eye(4).tolist())],mesh_digest='geometry')
            audit=dict(passed=False,projection_fingerprint=projection_fingerprint(meta),source_sha256=hashlib.sha256((folder/'source_front.png').read_bytes()).hexdigest())
            filename=folder/'front_silhouette_audit.json'
            filename.write_text(json.dumps(audit))
            with self.assertRaisesRegex(ValueError,'quality gate'):validate_front_audit(folder,meta)
            audit['passed']=True;filename.write_text(json.dumps(audit))
            validate_front_audit(folder,meta)
            meta['K'][0][0]=2
            with self.assertRaisesRegex(ValueError,'stale'):validate_front_audit(folder,meta)


if __name__=='__main__':unittest.main()
