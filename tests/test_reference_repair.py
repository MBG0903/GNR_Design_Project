import unittest
from pathlib import Path
import numpy as np
from scipy import ndimage
from textile.reference_repair import _fit_reference_rejected_experiment as experimental_fit
from textile.reference_repair import fit_reference
from textile.pixel_hierarchy import trace_on_boundary
from textile.separation import conflicts

class RejectedExperimentCharacterizationTests(unittest.TestCase):
    """Narrow former guards still pass; they do not prove full-curve compliance."""
    def test_structured_boundary_fit_improves_match_and_preserves_source(self):
        labels=np.zeros((24,18),np.int32);labels[3:21,3:6]=1
        original=labels.copy();target=np.zeros_like(labels,bool);target[3:21,3:7]=True
        result,report=experimental_fit(labels,target)
        np.testing.assert_array_equal(labels,original)
        np.testing.assert_array_equal(result>0,target)
        self.assertGreater(report['mismatch_reduction'],0)
        self.assertFalse(report['unseen_inference_accuracy_claim'])
        self.assertFalse(report['full_semantic_rule_compliance'])
        self.assertEqual(report['source_mismatches']-report['result_mismatches'],sum(o['reference_mismatch_reduction'] for o in report['operations']))

    def test_target_is_not_copied_when_it_would_close_two_pixel_clearance(self):
        labels=np.zeros((24,16),np.int32);labels[3:21,2:5]=1;labels[3:21,8:10]=2
        target=labels>0;target[:,5:8]=True
        result,report=experimental_fit(labels,target)
        self.assertFalse(conflicts(result))
        self.assertGreater(report['result_mismatches'],0)
        self.assertGreater(report['rejections'].get('two_pixel_neighbour_clearance',0),0)
        self.assertEqual(set(np.unique(result)),{0,1,2})

    def test_verified_staircase_survives_reference_pressure_to_fill_it(self):
        root=Path(__file__).resolve().parents[1]
        original=np.load(root/'examples/pixel_hierarchy_pass/piece_labels.npy')
        labels=np.where(original==40,40,0)
        target=ndimage.binary_dilation(labels>0,iterations=1)
        result,report=experimental_fit(labels,target)
        trace=[[204,139],[204,148],[203,148],[203,151],[202,151],[202,153],[201,153],[201,155],
               [200,155],[200,156],[199,156],[199,157],[198,157],[198,158]]
        self.assertTrue(trace_on_boundary(result==40,trace))
        self.assertTrue(report['verified_traces_preserved'])


class DisabledReferenceRepairTests(unittest.TestCase):
    def test_public_pipeline_blocks_rejected_fitting(self):
        with self.assertRaisesRegex(RuntimeError,'disabled after designer rejection'):
            fit_reference(None,None)

    def test_cli_cannot_regenerate_rejected_outputs(self):
        import subprocess,sys,hashlib
        root=Path(__file__).resolve().parents[1]
        path=root/'examples/reference_assisted_repair/step_3_reference_repaired.png'
        before=hashlib.sha256(path.read_bytes()).hexdigest()
        process=subprocess.run([sys.executable,'-m','scripts.run_reference_repair'],cwd=root,capture_output=True,text=True)
        self.assertNotEqual(process.returncode,0)
        self.assertIn('disabled after designer rejection',process.stderr)
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),before)
