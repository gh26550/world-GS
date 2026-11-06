import unittest
import numpy as np
from worldgs.front_inspection import silhouette_metrics

class FrontInspectionTest(unittest.TestCase):
 def test_recall_does_not_hide_excess_mesh_area(self):
  target=np.zeros((8,8),bool);target[2:4,2:4]=True
  predicted=np.zeros_like(target);predicted[2:4,2:6]=True
  result=silhouette_metrics(target,predicted)
  self.assertEqual(result['iou'],.5);self.assertEqual(result['target_recall'],1.);self.assertEqual(result['predicted_precision'],.5)
  self.assertEqual(result['predicted_only_pixels'],4)
 def test_empty_masks_do_not_report_perfect_alignment(self):
  result=silhouette_metrics(np.zeros((4,4),bool),np.zeros((4,4),bool))
  self.assertEqual(result['iou'],0.);self.assertIsNone(result['centroid_offset_pixels'])
