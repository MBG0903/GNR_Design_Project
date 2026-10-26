import json
from pathlib import Path
import tempfile
import unittest
from PIL import Image
from textile.colab_edits import apply_plan

class ColabEditTests(unittest.TestCase):
    def test_local_correction_retains_dimensions_palette_uncertainties_and_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source.png'
            im=Image.new('RGB',(100,100),'white')
            im.paste('red',(10,10,30,30));im.save(source)
            original=source.read_bytes()
            plan=dict(findings=[],unresolved=['Review curve taper'],operations=[
                dict(rule='R2',reason='Separate confirmed adjacent pieces',colour='white',points=[[19,10],[20,10],[20,29],[19,29]])])
            path=apply_plan(source,plan,root/'iteration_01',source=source)
            with Image.open(path) as result:
                self.assertEqual(result.size,(100,100))
                self.assertEqual(result.getpixel((19,15)),(255,255,255))
                self.assertEqual(result.getpixel((15,15)),(255,0,0))
            self.assertEqual(source.read_bytes(),original)
            record=json.loads((root/'iteration_01/audit.json').read_text())
            self.assertFalse(record['semantic_approval'])
            self.assertEqual(record['unresolved'],['Review curve taper'])
            with self.assertRaises(ValueError):apply_plan(source,plan,root/'iteration_01',source=source)

    def test_wrong_source_dimensions_block_before_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); source=root/'original.png';current=root/'current.png'
            Image.new('RGB',(401,340),'white').save(source)
            im=Image.new('RGB',(400,340),'white');im.putpixel((5,5),(255,0,0));im.save(current)
            with self.assertRaises(ValueError):
                apply_plan(current,dict(findings=[],unresolved=[],operations=[]),root/'out',source=source)
            self.assertFalse((root/'out').exists())
