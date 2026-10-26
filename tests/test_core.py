import tempfile
import unittest
import json
from pathlib import Path
import numpy as np
from PIL import Image
from textile.core import RULES, exact_palette, validate, export_reviewed

ROOT = Path(__file__).resolve().parents[1]

class RasterTests(unittest.TestCase):
    def setUp(self):
        self.source = Image.open(ROOT/'examples/step_2.bmp')
        self.reference = Image.open(ROOT/'examples/step_3_reference.bmp')
        self.addCleanup(self.source.close)
        self.addCleanup(self.reference.close)

    def test_reference_exact_palette_and_dimensions(self):
        result = validate(self.reference, self.source.size)
        self.assertTrue(result['automated_checks_passed'])
        self.assertEqual((result['width'], result['height']), (401, 340))

    def test_normalization_composites_alpha_and_is_idempotent(self):
        a = np.array([[[0,0,0,0], [255,50,50,255]]], dtype=np.uint8)
        normalized = exact_palette(a)
        np.testing.assert_array_equal(np.asarray(normalized), [[[255,255,255],[255,0,0]]])
        np.testing.assert_array_equal(normalized, exact_palette(normalized))

    def test_export_rejects_wrong_size_colour_and_blank(self):
        for im in [self.reference.resize((400,340)), self.source, Image.new('RGB', self.source.size, 'white')]:
            with self.assertRaises(ValueError), tempfile.TemporaryDirectory() as out:
                export_reviewed(im, self.source, RULES, 'Tester', '', out)

    def test_export_requires_all_rules_reviewer_and_no_uncertainty(self):
        for rules, name, pending in [(RULES[:-1], 'Tester', ''), (RULES, '', ''), (RULES, 'Tester', 'Uncertain eye')]:
            with self.assertRaises(ValueError), tempfile.TemporaryDirectory() as out:
                export_reviewed(self.reference, self.source, rules, name, pending, out)

    def test_reviewed_export_roundtrips_exact_pixels(self):
        with tempfile.TemporaryDirectory() as out:
            paths = export_reviewed(self.reference, self.source, RULES, 'Test reviewer', '', out)
            for p in paths[:2]:
                np.testing.assert_array_equal(np.asarray(Image.open(p)), np.asarray(self.reference))
            record = json.loads(Path(paths[2]).read_text())
            self.assertEqual(record['status'], 'designer_reviewed_raster')
            self.assertEqual(len(record['reviewed_rules']), 16)

if __name__ == '__main__':
    unittest.main()
