import base64,io,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from PIL import Image
import numpy as np
from textile.tiles import split
from textile.tile_genai import propose,proposal_input

class TileGenerationTests(unittest.TestCase):
    def test_generated_halo_is_not_trusted_and_native_core_is_owned(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'run';im=Image.new('RGB',(40,30),'white');im.putpixel((8,8),(255,0,0))
            manifest=split(im,im,root,rows=1,columns=2,overlap=4);tile=manifest['tiles'][0]
            with Image.open(root/'tiles'/f"{tile['id']}_draft.png") as opened:current=opened.convert('RGB')
            raw=io.BytesIO();Image.new('RGB',(1024,1024),'red').save(raw,format='PNG');calls=[]
            def edit(**kwargs):
                calls.append(kwargs);return SimpleNamespace(data=[SimpleNamespace(b64_json=base64.b64encode(raw.getvalue()).decode())])
            client=SimpleNamespace(images=SimpleNamespace(edit=edit))
            candidate,metadata=propose(root,tile['id'],current,client)
            self.assertEqual(candidate.size,current.size)
            x0,y0,x1,y1=tile['core_local']
            self.assertEqual(candidate.getpixel((x0,y0)),(255,0,0))
            outside=np.ones((candidate.height,candidate.width),bool);outside[y0:y1,x0:x1]=False
            np.testing.assert_array_equal(np.asarray(candidate)[outside],np.asarray(current)[outside])
            self.assertIn('mask',calls[0]);self.assertEqual(calls[0]['model'],'gpt-image-2.5-sunburst')
            self.assertFalse(metadata['automatic_merge_allowed'])
            before,_=proposal_input(root,tile['id'],metadata['revision'],manifest)
            np.testing.assert_array_equal(candidate,before)
            from textile.tiles import save_proposal
            from textile.correction_data import capture
            human=candidate.copy();human.putpixel((1,1),(255,255,255))
            review=dict(reviewer='Designer',unresolved=[],complete_front_back_and_tips_reviewed=True,
                        hierarchy_reviewed=True,two_pixel_gaps_reviewed=True,design_group_id='design-A',
                        rules_corrected=['R7'],correction_notes='Corrected a tip.',ai_proposal_revision=metadata['revision'])
            path=Path(save_proposal(root,tile['id'],human,review))
            sample=capture(root,tile['id'],path.with_suffix('.json'))
            record=json.loads((sample/'example.json').read_text())
            self.assertEqual(record['changed_pixels'],1)
            self.assertEqual(record['draft_generator_model'],'gpt-image-2.5-sunburst')
            with Image.open(sample/'before.png') as image:np.testing.assert_array_equal(image,candidate)
            with Image.open(sample/'initial_ai_draft.png') as image:np.testing.assert_array_equal(image,current)
