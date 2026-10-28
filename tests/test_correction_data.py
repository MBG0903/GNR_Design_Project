import json,tempfile,unittest,zipfile
from pathlib import Path
from PIL import Image
import numpy as np
from textile.tiles import split,save_proposal
from textile.correction_data import capture,export_dataset

class CorrectionDataTests(unittest.TestCase):
    def setup_run(self,tmp):
        run=Path(tmp)/'run';base=Image.new('RGB',(40,30),'white');base.putpixel((8,8),(255,0,0))
        m=split(base,base,run,rows=1,columns=2,overlap=4);return run,m
    def review(self):
        return dict(reviewer='Designer',unresolved=[],complete_front_back_and_tips_reviewed=True,
                    hierarchy_reviewed=True,two_pixel_gaps_reviewed=True,design_group_id='original-A',
                    rules_corrected=['R7','R8'],correction_notes='Corrected the annotated taper.',draft_generator_model='unknown')
    def test_capture_preserves_pair_masks_group_and_rule_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            run,m=self.setup_run(tmp);tile=m['tiles'][0];before=Image.open(run/'tiles'/f"{tile['id']}_draft.png").convert('RGB')
            edited=before.copy();edited.putpixel((9,8),(255,0,0));path=Path(save_proposal(run,tile['id'],edited,self.review()))
            out=capture(run,tile['id'],path.with_suffix('.json'));data=json.loads((out/'example.json').read_text())
            self.assertEqual(data['changed_pixels'],1);self.assertEqual(data['design_group_id'],'original-A')
            self.assertEqual(data['training_status'],'collected_only_not_trained')
            self.assertFalse(data['whole_image_production_approved'])
            with Image.open(out/'before.png') as image:np.testing.assert_array_equal(image,before)
            with Image.open(out/'after.png') as image:np.testing.assert_array_equal(image,edited)
            with Image.open(out/'core_loss_mask.png') as image:
                self.assertEqual(image.getpixel((19,8)),255);self.assertEqual(image.getpixel((20,8)),0)
            self.assertEqual(capture(run,tile['id'],path.with_suffix('.json')),out)
    def test_export_only_latest_revision_and_no_automatic_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            run,m=self.setup_run(tmp);tile=m['tiles'][0]
            with Image.open(run/'tiles'/f"{tile['id']}_draft.png") as opened:edited=opened.convert('RGB')
            for x in [9,10]:
                edited.putpixel((x,8),(255,0,0));save_proposal(run,tile['id'],edited,self.review())
            report,archive=export_dataset(run,Path(tmp)/'export')
            self.assertEqual(report['samples'],1);self.assertEqual(report['training_status'],'not_trained')
            rows=[json.loads(line) for line in (Path(tmp)/'export/dataset.jsonl').read_text().splitlines()]
            metadata=json.loads((Path(tmp)/'export'/rows[0]['metadata']).read_text())
            self.assertEqual(metadata['revision'],'0002');self.assertEqual(metadata['changed_pixels'],2)
            with zipfile.ZipFile(archive) as z:self.assertIn('dataset.jsonl',z.namelist())
    def test_missing_learning_notes_are_not_silently_treated_as_training_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            run,m=self.setup_run(tmp);tile=m['tiles'][0]
            with Image.open(run/'tiles'/f"{tile['id']}_draft.png") as opened:edited=opened.convert('RGB')
            edited.putpixel((9,8),(255,0,0));review=self.review();review['correction_notes']=''
            path=Path(save_proposal(run,tile['id'],edited,review))
            with self.assertRaisesRegex(ValueError,'short explanation'):capture(run,tile['id'],path.with_suffix('.json'))
    def test_inconsistent_family_ids_in_one_design_block_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            run,m=self.setup_run(tmp)
            for i,tile in enumerate(m['tiles']):
                with Image.open(run/'tiles'/f"{tile['id']}_draft.png") as opened:edited=opened.convert('RGB')
                x,y=tile['core_local'][:2];edited.putpixel((x,y),(255,0,0))
                review=self.review();review['design_group_id']=f'conflicting-family-{i}'
                save_proposal(run,tile['id'],edited,review)
            with self.assertRaisesRegex(ValueError,'inconsistent group IDs'):
                export_dataset(run,Path(tmp)/'bad-export')
