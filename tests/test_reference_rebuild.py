import json,tempfile,unittest,shutil
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
from worldgs.reference_cameras import benchmark_cameras
from worldgs.reference_geometry import assess_reference
from worldgs.reference_pipeline import collect_candidates,run_reference_experiment

class ReferenceRebuildTest(unittest.TestCase):
 def test_four_room_cameras_without_moving_geometry(self):
  shapes=[dict(min=[-3,-1.5,-3],max=[3,1.5,3])];old=json.dumps(shapes);frames=benchmark_cameras(shapes,64)
  self.assertEqual([f['id'] for f in frames],['front','left','right','rear']);self.assertEqual(old,json.dumps(shapes))
  self.assertGreater(np.ptp([np.array(f['transform_matrix'])[0,3] for f in frames]),4)
  for f in frames:np.testing.assert_allclose(np.array(f['transform_matrix'])[:3,:3].T@np.array(f['transform_matrix'])[:3,:3],np.eye(3),atol=1e-7)
 def scene(self):
  d=np.ones((64,64))*4;d[20:45,20:45]=2;faces=np.zeros((64,64),int);faces[20:45,20:45]=1;rgb=np.full((64,64,3),50,np.uint8);rgb[20:45,20:45]=220
  return d,faces,rgb
 def test_aligned_geometry_passes_without_semantic_labels(self):
  d,faces,rgb=self.scene();self.assertTrue(assess_reference(d,d,d,rgb,faces,np.array([0,6]),64)['passed'])
 def test_missing_foreground_is_rejected(self):
  d,faces,rgb=self.scene();self.assertFalse(assess_reference(d,d,np.ones_like(d)*4,np.full_like(rgb,50),faces,np.array([0,6]),64)['passed'])
 def test_large_view_displacement_is_rejected(self):
  d,faces,rgb=self.scene();moved=np.roll(d,24,axis=1);image=np.roll(rgb,24,axis=1);self.assertFalse(assess_reference(d,d,moved,image,faces,np.array([0,6]),64)['passed'])
 def test_unobservable_boundaries_require_image_evidence(self):
  d,faces,rgb=self.scene();flat=np.ones_like(d)*4
  self.assertFalse(assess_reference(d,flat,flat,np.full_like(rgb,50),faces,np.array([0,6]),64)['passed'])
  self.assertTrue(assess_reference(d,flat,flat,rgb,faces,np.array([0,6]),64)['passed'])
 def test_bad_view_does_not_stop_other_camera_collection(self):
  with tempfile.TemporaryDirectory() as tmp:
   folder=Path(tmp);(folder/'planning').mkdir();(folder/'planning/image_room_plan.json').write_text('{"objects":[],"style":"beige"}');Image.new('RGB',(64,64),'white').save(folder/'source_front.png')
   frames=[dict(id=name,file_path=name+'.png',K=np.eye(3).tolist(),transform_matrix=np.eye(4).tolist()) for name in ['front','left','right','rear']]
   for f in frames:
    p=folder/'references'/f['id'];p.mkdir(parents=True)
    for n in ['proxy.png','depth.png','boundaries.png']:Image.new('RGB',(64,64),'white').save(p/n)
    np.save(p/'mesh_depth.npy',np.ones((64,64)))
   class Runtime:
    count=0
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def depth(self,image,focal,output):np.save(output,np.ones((64,64)))
    def generate(self,guide,style,prompt,seed,photo,size,**kw):self.count+=1;shutil.copy2(guide,photo)
   class Mesh:vertices=np.zeros((3,3));faces=np.zeros((1,3),int)
   class Renderer:
    mesh=Mesh();owners=np.array([0]);components=[]
    def render(self,*args):return dict(depth=np.ones((64,64)),faces=np.zeros((64,64),int))
   runtime=Runtime();checks=[dict(passed=i!=0,score=.8) for i in range(4)]
   with patch('worldgs.layout_consistency.layout_contract',return_value={'objects':[]}),patch('worldgs.room_verification.camera_context',return_value=('empty room',[])),patch('worldgs.reference_pipeline.assess_reference',side_effect=checks):
    results=collect_candidates(folder,Renderer(),frames,runtime,dict(seed=1,size=64,attempts=1,mesh_sha256='mesh',source_sha256='source'))
   self.assertEqual(runtime.count,4);self.assertEqual(len(results),4);self.assertFalse(json.loads((folder/'assessment.json').read_text())['ready'])
 def test_unresolved_experiment_never_exports_training_dataset(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp);out=p/'out';out.mkdir();results=[dict(assessment=dict(passed=False))]
   with patch('worldgs.reference_pipeline.prepare_reference_experiment',return_value=({'size':64},None,{'components':[]},None,[],None,{})),patch('worldgs.reference_pipeline.initialize_experiment'),patch('worldgs.reference_pipeline.MeshRenderer'),patch('worldgs.reference_pipeline.render_benchmark'),patch('worldgs.reference_pipeline.collect_candidates',return_value=results),patch('worldgs.reference_pipeline.write_review'),patch('worldgs.reference_pipeline.package_dataset') as export:
    run_reference_experiment(p/'source',out)
   export.assert_not_called();self.assertFalse(json.loads((out/'status.json').read_text())['trainable'])
 def test_mesh_parts_of_one_object_are_evaluated_as_one_group(self):
  d,faces,rgb=self.scene();faces[20:45,32:45]=2
  result=assess_reference(d,d,d,rgb,faces,np.array([0,6,7]),64,components=[dict(group='room') for _ in range(6)]+[dict(group='furniture'),dict(group='furniture')])
  groups=[r for r in result['components'] if r['group']=='furniture'];self.assertEqual(len(groups),1);self.assertEqual(groups[0]['owners'],[6,7]);self.assertTrue(result['passed'])
