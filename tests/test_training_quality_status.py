import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from worldgs.train import train

class TrainingQualityStatusTest(unittest.TestCase):
    def test_failed_coverage_is_not_overwritten_by_success(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)
            (path/'scene.json').write_text(json.dumps({'mode':'image_room_mesh_v12'}))
            (path/'gs_coverage_quality.json').write_text(json.dumps({'passed':False,'mean_missing_fraction':.2}))
            with patch('worldgs.train_teacher.validate_teacher'),patch('worldgs.train_mesh.train_mesh'):
                train(path,steps=20,max_points=100)
            status=json.loads((path/'status.json').read_text())
            self.assertFalse(status['success'])
            self.assertEqual(status['stage'],'training_quality_failed')
