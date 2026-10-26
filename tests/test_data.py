import csv
import json
import tempfile
import unittest
from pathlib import Path
from PIL import Image
from textile.data import pair_key, audit_dataset

class PairingTests(unittest.TestCase):
    def test_sample_names_match(self):
        self.assertEqual(pair_key('362 GNR 2.bmp',2),pair_key('362 GNR 3.bmp',3))

    def fixture(self,root):
        a=root/'Stage_2'; b=root/'Stage_3'; a.mkdir(); b.mkdir()
        for i in range(8):
            source=Image.new('RGB',(16,16),'white'); source.putpixel((i,1),(0,0,0))
            target=Image.new('RGB',(16,16),'white'); target.putpixel((i,2),(255,0,0))
            source.save(a/f'{i} GNR 2.bmp');target.save(b/f'{i} GNR 3.bmp')
        return a,b

    def test_required_groups_block_until_complete_and_keep_variants_together(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); a,b=self.fixture(root); out=root/'audit'
            with self.assertRaises(ValueError):
                audit_dataset(a,b,out,require_groups=True)
            self.assertTrue((out/'groups_template.csv').exists())
            self.assertFalse((out/'manifest.json').exists())
            groups=root/'groups.csv'
            with groups.open('w',newline='') as handle:
                writer=csv.writer(handle);writer.writerow(['key','design_id'])
                for i in range(8):writer.writerow([f'{i} gnr',str(i//2)])
            result=audit_dataset(a,b,out,groups_csv=groups,require_groups=True)
            self.assertEqual(result['pairs'],8)
            rows=json.loads((out/'manifest.json').read_text())
            self.assertEqual(rows[0]['split'],rows[1]['split'])
            self.assertTrue(all(result['counts'].values()))
            # A new unmatched image invalidates the old successful manifest.
            Image.new('RGB',(16,16),'white').save(a/'unmatched 2.bmp')
            with self.assertRaises(ValueError):audit_dataset(a,b,out)
            self.assertFalse((out/'manifest.json').exists())
