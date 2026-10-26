import unittest
from pathlib import Path
import numpy as np
from textile.global_hierarchy import boundary_loops,run_global
from textile.pixel_hierarchy import trace_on_boundary,check_runs
from textile.separation import conflicts

class WholeImageTests(unittest.TestCase):
    def test_outer_and_hole_boundaries_are_both_scanned(self):
        mask=np.ones((3,3),bool);mask[1,1]=False
        loops=boundary_loops(mask)
        self.assertEqual(len(loops),2)
        self.assertEqual(sum(len(p)-1 for p in loops),16)
        self.assertTrue(all(trace_on_boundary(mask,p) for p in loops))

    def test_diagonal_regions_have_closed_noninterleaved_loops(self):
        mask=np.eye(2,dtype=bool)
        loops=boundary_loops(mask)
        self.assertEqual(len(loops),2)
        self.assertTrue(all(len(p)==5 for p in loops))

    def test_real_feather_scanned_in_both_directions_and_safely_corrected(self):
        root=Path(__file__).resolve().parents[1]
        source=np.load(root/'examples/separation_2px_pass/provisional_piece_labels.npy')
        labels=np.where(source==40,40,0)
        result,report=run_global(labels)
        self.assertEqual(report['scanned_regions'],1)
        self.assertGreaterEqual(report['corrected_arcs'],1)
        self.assertTrue(report['two_pixel_clearance_passed'])
        self.assertFalse(report['image_wide_rule_complete'])
        self.assertFalse(conflicts(result))
        for arc in report['arcs']:
            if arc['status']=='corrected':
                self.assertTrue(check_runs(arc['after_runs'],allow_21=True)['passed'])
                self.assertTrue(trace_on_boundary(result==40,arc['after_points']))

    def test_existing_gap_failure_blocks_processing(self):
        labels=np.zeros((10,10),np.int32);labels[2,2]=1;labels[2,4]=2
        with self.assertRaises(ValueError):run_global(labels)
