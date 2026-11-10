import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch


class PlanningBenchmarkTest(unittest.TestCase):
    def test_records_failure_and_rejects_changed_protocol(self):
        from worldgs.planning_benchmark import run
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);suite=folder/'suite.json'
            spec=dict(scope='test planning',models=[dict(id='model',quantization='none')],cases=[dict(id='a',prompt='room A'),dict(id='b',prompt='room B')])
            suite.write_text(json.dumps(spec))
            with patch('worldgs.semantic_plan.plan_with_llm',side_effect=[ValueError('invalid dimensions'),dict(objects=[{}])]) as planner:
                result=run(suite,folder/'results')
                self.assertEqual(result['summary'][0]['passed'],1)
                self.assertEqual(result['summary'][0]['total'],2)
                self.assertIn('invalid dimensions',result['rows'][0]['error'])
                self.assertEqual(planner.call_args_list[0].kwargs,planner.call_args_list[1].kwargs)
            spec['cases'][0]['prompt']='changed room';suite.write_text(json.dumps(spec))
            with self.assertRaisesRegex(ValueError,'Protocol/code changed'):run(suite,folder/'results')
