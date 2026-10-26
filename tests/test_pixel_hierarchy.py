import unittest
import numpy as np
from textile.pixel_hierarchy import check_runs,contour_runs,check_pair,plan_counts,trace_on_boundary


def trace(runs,x=0,y=0,axis='y',sx=1,sy=1):
    points=[[x,y]]
    for i,n in enumerate(runs):
        if i:
            if axis=='y':x+=sx
            else:y+=sy
            points.append([x,y])
        if axis=='y':y+=sy*n
        else:x+=sx*n
        points.append([x,y])
    return points

class HierarchyTests(unittest.TestCase):
    def test_counts_are_step_counts_not_stroke_widths(self):
        result=check_runs([6,3,3,2,2,2,1,1,1,1,1])
        self.assertTrue(result['passed'])
        self.assertEqual(result['counts'],{'3':2,'2':3,'1':5})

    def test_skips_equal_counts_and_reversals_fail(self):
        for runs in [[8,3,1,1,1],[8,3,2,1],[8,3,2,2,1,1],[8,3,2,1,2,1,1,1],[8,4,3,2,2,1,1,1]]:
            self.assertFalse(check_runs(runs)['passed'])

    def test_all_four_rotations_and_reflections(self):
        runs=[6,3,2,2,1,1,1]
        for axis in ['x','y']:
            for sx in [-1,1]:
                for sy in [-1,1]:
                    self.assertEqual(contour_runs(trace(runs,axis=axis,sx=sx,sy=sy),axis),runs)

    def test_front_must_be_strictly_shorter(self):
        a=trace([6,3,2,2,1,1,1]);b=trace([8,3,2,2,1,1,1],x=20)
        self.assertTrue(check_pair(a,b,'y')['sequence_checks_passed'])
        self.assertFalse(check_pair(a,b,'y')['annotated_raster_checks_passed'])
        self.assertFalse(check_pair(b,a,'y')['sequence_checks_passed'])

    def test_false_boundary_trace_rejected(self):
        mask=np.zeros((6,6),bool);mask[1:5,1:5]=True
        self.assertTrue(trace_on_boundary(mask,[[1,1],[1,5]]))
        self.assertFalse(trace_on_boundary(mask,[[2,1],[2,5]]))
        self.assertFalse(trace_on_boundary(mask,[[-1,1],[-1,5]]))

    def test_endpoint_preserving_count_proposal_and_impossible_case(self):
        plans=plan_counts(span=16,steps=10)
        self.assertTrue(plans)
        for p in plans:
            self.assertEqual(sum(p['runs']),16);self.assertEqual(len(p['runs']),10)
            self.assertTrue(check_runs([8]+p['runs'])['passed'])
        self.assertEqual(plan_counts(5,3),[])

    def test_non_unit_connector_and_turn_need_new_annotation(self):
        for points in [[[0,0],[0,6],[2,6],[2,9]],[[0,0],[0,6],[1,6],[1,3]]]:
            with self.assertRaises(ValueError):contour_runs(points,'y')

class NativeCorrectionTests(unittest.TestCase):
    def test_real_selected_curve_reorders_steps_preserving_spacing_and_endpoints(self):
        from pathlib import Path
        from textile.pixel_hierarchy import replace_trace
        root=Path(__file__).resolve().parents[1]
        labels=np.load(root/'examples/separation_2px_pass/provisional_piece_labels.npy')
        old=[[204,139],[204,148],[203,148],[203,150],[202,150],[202,153],[201,153],[201,154],
             [200,154],[200,156],[199,156],[199,157],[198,157],[198,158]]
        new=[[204,139],[204,148],[203,148],[203,151],[202,151],[202,153],[201,153],[201,155],
             [200,155],[200,156],[199,156],[199,157],[198,157],[198,158]]
        result,report=replace_trace(labels,40,old,new,'y')
        self.assertEqual(result.shape,labels.shape)
        self.assertEqual(report['changed_pixels'],2)
        self.assertEqual(report['after_runs'],[9,3,2,2,1,1,1])
        self.assertTrue(report['two_pixel_clearance_preserved'])
        self.assertFalse(report['image_wide_approval'])

class FallbackTests(unittest.TestCase):
    def test_two_level_counts_order_and_explicit_opt_in(self):
        self.assertFalse(check_runs([8,2,1,1])['passed'])
        self.assertEqual(check_runs([8,2,1,1],allow_21=True)['mode'],'2-1')
        for runs in [[8,2,1],[8,2,2,1,1],[8,1,2,1],[8,1,1,1],[8,3,1,1,1]]:
            self.assertFalse(check_runs(runs,allow_21=True)['passed'])
        self.assertEqual(check_runs([8,3,2,2,1,1,1],allow_21=True)['mode'],'3-2-1')

    def test_fallback_preserves_required_span_and_count(self):
        from textile.pixel_hierarchy import plan_counts_21
        plan=plan_counts_21(4,3)[0]
        self.assertEqual(plan['runs'],[2,1,1])
        self.assertEqual(plan_counts_21(6,4),[])
