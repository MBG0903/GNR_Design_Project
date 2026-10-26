import unittest
from pathlib import Path
import numpy as np
from scipy import ndimage
from textile.reference_repair import fit_reference
from textile.pixel_hierarchy import trace_on_boundary
from textile.separation import conflicts

class ReferenceRepairTests(unittest.TestCase):
    def test_structured_boundary_fit_improves_match_and_preserves_source(self):
        labels=np.zeros((24,18),np.int32);labels[3:21,3:6]=1
        original=labels.copy();target=np.zeros_like(labels,bool);target[3:21,3:7]=True
        result,report=fit_reference(labels,target)
        np.testing.assert_array_equal(labels,original)
        np.testing.assert_array_equal(result>0,target)
        self.assertGreater(report['mismatch_reduction'],0)
        self.assertFalse(report['unseen_inference_accuracy_claim'])
        self.assertFalse(report['full_semantic_rule_compliance'])
        self.assertEqual(report['source_mismatches']-report['result_mismatches'],sum(o['reference_mismatch_reduction'] for o in report['operations']))

    def test_target_is_not_copied_when_it_would_close_two_pixel_clearance(self):
        labels=np.zeros((24,16),np.int32);labels[3:21,2:5]=1;labels[3:21,8:10]=2
        target=labels>0;target[:,5:8]=True
        result,report=fit_reference(labels,target)
        self.assertFalse(conflicts(result))
        self.assertGreater(report['result_mismatches'],0)
        self.assertGreater(report['rejections'].get('two_pixel_neighbour_clearance',0),0)
        self.assertEqual(set(np.unique(result)),{0,1,2})

    def test_verified_staircase_survives_reference_pressure_to_fill_it(self):
        root=Path(__file__).resolve().parents[1]
        original=np.load(root/'examples/pixel_hierarchy_pass/piece_labels.npy')
        labels=np.where(original==40,40,0)
        target=ndimage.binary_dilation(labels>0,iterations=1)
        result,report=fit_reference(labels,target)
        trace=[[204,139],[204,148],[203,148],[203,151],[202,151],[202,153],[201,153],[201,155],
               [200,155],[200,156],[199,156],[199,157],[198,157],[198,158]]
        self.assertTrue(trace_on_boundary(result==40,trace))
        self.assertTrue(report['verified_traces_preserved'])
