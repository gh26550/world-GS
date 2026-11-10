import unittest,numpy as np
from worldgs.regenerate_surface import surface_cameras,box
class SurfaceRegenerationTest(unittest.TestCase):
 def test_zero_depth_is_reported_before_placement_solver(self):
  from worldgs.semantic_plan import repair_layout
  plan=dict(style='neutral',room=dict(size=[6,3,6]),objects=[dict(id='furniture',dimensions=[2,1,0],position=[0,2],yaw=0,support='floor')])
  with self.assertRaisesRegex(ValueError,'Invalid dimensions for furniture'):repair_layout(plan)
 def test_hotel_furniture_gets_surface_views_without_livingroom_labels(self):
  layout={'shapes':[dict(name='room',kind='room',min=[-4,-1.5,-4],max=[4,1.5,4]),dict(name='bed_00',kind='generated_asset',label='king bed',min=[-1,.5,1],max=[1,1.5,3]),dict(name='desk_00',kind='generated_asset',label='writing desk',min=[2,.6,-1],max=[3,1.5,0]),dict(name='chair_00',kind='generated_asset',label='upholstered chair',min=[1,.5,-2],max=[1.5,1.5,-1.5])]}
  cameras=surface_cameras(layout,96);self.assertEqual(len(cameras),233)
  self.assertEqual({f['focus_object'] for f in cameras if f['camera_kind']=='surface_targeted'},{'bed_00','desk_00','chair_00'})
 def test_surface_camera_budget_and_object_targets(self):
  from worldgs.room_cameras import safe_eye
  layout={'shapes':[dict(name='room',kind='room',min=[-3,-1.5,-3],max=[3,1.5,3]),dict(name='sofa',label='sofa',min=[-1,.5,1],max=[1,1.5,2]),dict(name='shelf',label='bookshelf',min=[-2,-1,2],max=[-1.3,1.5,2.7]),dict(name='table',label='coffee table',min=[-.6,.9,-.6],max=[.6,1.5,.4])]}
  cameras=surface_cameras(layout,96);self.assertEqual(len(cameras),233);targeted=[f for f in cameras if f['camera_kind']=='surface_targeted'];self.assertEqual(set(f['focus_object'] for f in targeted),{'sofa','shelf','table'})
  for f in targeted:
   self.assertTrue(safe_eye(np.array(f['transform_matrix'])[:3,3],layout['shapes'][0],layout['shapes'][1:]));self.assertTrue(f['file_path'].endswith('.png'))
 def test_architectural_box_is_closed_and_rejects_empty_extent(self):
  m=box([0,0,0],[1,2,.08],[240,240,240,255]);self.assertTrue(m.is_watertight)
  with self.assertRaises(ValueError):box([0,0,0],[0,1,1],[255]*4)

 def test_coplanar_different_owners_cannot_exchange_rgb(self):
  from worldgs.appearance_geometry import transfer_appearance
  size=32;yy,xx=np.indices((size,size));K=np.array([[32,0,15.5],[0,32,15.5],[0,0,1.]])
  rays=np.stack([xx,yy,np.ones_like(xx)],-1)@np.linalg.inv(K).T;xyz=rays*2
  hit=dict(xyz=xyz,normals=np.broadcast_to([0,0,1.],xyz.shape),depth=np.full((size,size),2.),faces=np.zeros((size,size),int),owner_ids=np.zeros((size,size),int));frame=dict(K=K.tolist(),transform_matrix=np.eye(4).tolist())
  entry=dict(frame=frame,rgb=np.full((size,size,3),128,np.uint8),depth=hit['depth'],valid=np.ones((size,size),bool),confidence=np.ones((size,size)),owner_ids=np.ones((size,size),int))
  _,known,_=transfer_appearance(hit,frame,[entry]);self.assertFalse(known.any())
  entry['owner_ids'][:]=0;_,known,_=transfer_appearance(hit,frame,[entry]);self.assertGreater(known.mean(),.5)
 def test_new_geometry_requires_inspection_backed_plan(self):
  import tempfile
  from pathlib import Path
  from worldgs.regenerate_surface import correct_geometry
  with tempfile.TemporaryDirectory() as tmp:
   with self.assertRaisesRegex(ValueError,'correction plan'):correct_geometry(Path(tmp)/'source',Path(tmp)/'out')

 def test_failed_generator_request_never_reuses_stale_photo(self):
  import tempfile,json
  from pathlib import Path
  from unittest.mock import patch
  from worldgs.local_runtime import LocalRuntime
  with tempfile.TemporaryDirectory() as tmp:
   d=Path(tmp);runtime=LocalRuntime(d,image_model='controlnet');output=d/'photo.png';calls=[]
   def failed_worker(python,script,request,record):
    calls.append(request);record=Path(record);record.write_text(json.dumps(request));record.with_suffix('.done.json').write_text(json.dumps({'success':False}));output.write_bytes(b'previous image')
   with patch.object(runtime,'persistent_worker',side_effect=failed_worker):
    runtime.generate(d/'guide.png',d/'style.png','room',1,output,64)
    runtime.generate(d/'guide.png',d/'style.png','room',1,output,64)
   self.assertEqual(len(calls),2)

 def test_changed_guide_invalidates_successful_image_cache(self):
  import tempfile,json
  from pathlib import Path
  from unittest.mock import patch
  from worldgs.local_runtime import LocalRuntime
  with tempfile.TemporaryDirectory() as tmp:
   d=Path(tmp);runtime=LocalRuntime(d,image_model='controlnet');guide=d/'guide.png';guide.write_bytes(b'first geometry');output=d/'photo.png';calls=[]
   def successful_worker(python,script,request,record):
    calls.append(request);record=Path(record);record.write_text(json.dumps(request));record.with_suffix('.done.json').write_text(json.dumps({'success':True}));output.write_bytes(b'image')
   with patch.object(runtime,'persistent_worker',side_effect=successful_worker):
    runtime.generate(guide,d/'style.png','room',1,output,64)
    runtime.generate(guide,d/'style.png','room',1,output,64)
    self.assertEqual(len(calls),1)
    guide.write_bytes(b'corrected geometry');runtime.generate(guide,d/'style.png','room',1,output,64)
   self.assertEqual(len(calls),2)
