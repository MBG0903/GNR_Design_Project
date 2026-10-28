import json,tempfile,unittest
from pathlib import Path
import numpy as np
from PIL import Image
from textile.tiles import split,merge,save_proposal,merge_saved

class TileTests(unittest.TestCase):
    def image(self):
        a=np.full((340,401,3),255,np.uint8);a[5:330:3,7:390:2]=[255,0,0]
        return Image.fromarray(a)
    def review(self):
        return dict(reviewer='Test reviewer',unresolved=[],complete_front_back_and_tips_reviewed=True,hierarchy_reviewed=True,two_pixel_gaps_reviewed=True)
    def test_uneven_9_and_12_tile_roundtrips_preserve_every_pixel(self):
        with tempfile.TemporaryDirectory() as tmp:
            for columns in [3,4]:
                root=Path(tmp)/str(columns);original=self.image();m=split(original,original,root,columns=columns)
                proposals={t['id']:root/'tiles'/f"{t['id']}_draft.png" for t in m['tiles']}
                merged,report=merge(root,proposals)
                np.testing.assert_array_equal(merged,original)
                self.assertEqual(merged.size,(401,340));self.assertEqual(report['changed_pixels'],0)
                self.assertFalse(report['automatic_staircase_compliance'])
    def test_halo_edits_and_unsigned_core_edits_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'tiles';im=self.image();m=split(im,im,root);tile=m['tiles'][5]
            with Image.open(root/'tiles'/f"{tile['id']}_draft.png") as image:proposal=image.convert('RGB')
            x,y=tile['core_local'][:2];old=proposal.getpixel((x,y));proposal.putpixel((x,y),(255,0,0) if old==(255,255,255) else (255,255,255))
            with self.assertRaises(ValueError):merge(root,{tile['id']:proposal})
            proposal.putpixel((0,0),(255,0,0) if proposal.getpixel((0,0))==(255,255,255) else (255,255,255))
            with self.assertRaisesRegex(ValueError,'halo edits'):merge(root,{tile['id']:proposal},{tile['id']:self.review()})
    def test_signed_core_edit_lands_at_global_coordinate_and_versions_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'tiles';im=self.image();m=split(im,im,root);t=m['tiles'][5]
            with Image.open(root/'tiles'/f"{t['id']}_draft.png") as opened:proposal=opened.convert('RGB')
            x,y=t['core_local'][:2];old=proposal.getpixel((x,y));value=(255,0,0) if old==(255,255,255) else (255,255,255)
            proposal.putpixel((x,y),value)
            first=save_proposal(root,t['id'],proposal,self.review());second=save_proposal(root,t['id'],proposal,self.review())
            self.assertNotEqual(first,second);self.assertTrue(Path(first).exists())
            merged,report=merge_saved(root);self.assertEqual(report['changed_pixels'],1)
            self.assertEqual(merged.getpixel(tuple(t['core'][:2])),value)
            self.assertTrue(report['whole_image_validation_required'])
    def test_stale_source_bad_palette_resizing_and_partition_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'tiles';im=self.image();m=split(im,im,root);t=m['tiles'][0]
            for proposal in [Image.new('RGB',(1,1),'white'),Image.new('RGB',(t['context'][2],t['context'][3]),'pink')]:
                with self.assertRaises(ValueError):merge(root,{t['id']:proposal},{t['id']:self.review()})
            m['tiles'][0]['core']=[1,0,100,113];(root/'manifest.json').write_text(json.dumps(m))
            with self.assertRaises(ValueError):merge(root)

    def test_modified_source_is_not_used_as_a_silent_new_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'tiles';im=self.image();split(im,im,root)
            with Image.open(root/'original.png') as opened:changed=opened.convert('RGB')
            changed.putpixel((0,0),(255,0,0));changed.save(root/'original.png')
            with self.assertRaisesRegex(ValueError,'does not match'):merge(root)
