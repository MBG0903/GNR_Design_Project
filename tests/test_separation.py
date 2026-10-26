import unittest
import numpy as np
from textile.separation import separate, conflicts

class SeparationTests(unittest.TestCase):
    def test_one_pixel_gap_becomes_two_without_eroding_other_boundaries(self):
        labels=np.zeros((12,14),np.int32);labels[2:10,2:6]=1;labels[2:10,7:11]=2
        result,report=separate(labels)
        self.assertTrue(report['clearance_passed'])
        self.assertFalse(conflicts(result))
        self.assertEqual(set(np.unique(result)),{0,1,2})
        # Existing exterior silhouette on the other side is untouched.
        np.testing.assert_array_equal(result[:,2],labels[:,2])
        self.assertEqual(np.flatnonzero(result[5]==2).min()-np.flatnonzero(result[5]==1).max()-1,2)
        again,record=separate(result)
        np.testing.assert_array_equal(again,result)
        self.assertEqual(record['removed_red_pixels'],0)

    def test_existing_wide_gaps_and_same_piece_connections_unchanged(self):
        labels=np.zeros((10,20),np.int32);labels[2:8,2:6]=1;labels[2:8,12:16]=2
        result,_=separate(labels);np.testing.assert_array_equal(result,labels)
        single=np.ones((5,5),np.int32);result,_=separate(single)
        np.testing.assert_array_equal(single,result)

    def test_diagonal_contact_separated_but_single_pixels_not_deleted(self):
        labels=np.zeros((7,7),np.int32);labels[1,1]=1;labels[2,2]=2
        result,report=separate(labels)
        self.assertFalse(report['clearance_passed'])
        np.testing.assert_array_equal(result,labels)

    def test_disconnected_annotation_rejected(self):
        labels=np.zeros((6,6),np.int32);labels[1,1]=1;labels[4,4]=1
        with self.assertRaises(ValueError):separate(labels)
