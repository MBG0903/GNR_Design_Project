import unittest
import numpy as np
from PIL import Image
from textile.core import render
from textile.comparison import compare

class ComparisonTests(unittest.TestCase):
    def test_known_confusion_counts_and_similarity(self):
        truth=render(np.array([[1,1,1],[0,0,0]],bool))
        pred=render(np.array([[1,1,0],[1,0,0]],bool))
        report,diff=compare(truth,pred)
        self.assertEqual(report['matched_red_pixels'],2)
        self.assertEqual(report['matched_white_pixels'],2)
        self.assertEqual(report['missing_red_pixels'],1)
        self.assertEqual(report['excess_red_pixels'],1)
        self.assertEqual(report['red_iou'],0.5)
        self.assertAlmostEqual(report['red_dice'],2/3)
        self.assertAlmostEqual(report['pixel_agreement'],2/3)
        self.assertEqual(diff.getpixel((2,0)),(0,100,255))
        self.assertEqual(diff.getpixel((0,1)),(255,160,0))

    def test_identical_image_is_exact_match(self):
        im=render(np.array([[1,0],[0,1]],bool))
        report,_=compare(im,im)
        self.assertEqual(report['pixel_agreement'],1)
        self.assertEqual(report['red_iou'],1)
        self.assertEqual(report['boundary']['symmetric_mean_distance_pixels'],0)

    def test_no_implicit_resize_or_palette_normalization(self):
        with self.assertRaises(ValueError):compare(Image.new('RGB',(4,4),'white'),Image.new('RGB',(5,4),'white'))
        with self.assertRaises(ValueError):compare(Image.new('RGB',(4,4),'white'),Image.new('RGB',(4,4),'gray'))
