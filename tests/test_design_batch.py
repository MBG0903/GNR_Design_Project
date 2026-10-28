import base64
import hashlib
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
from textile.design_batch import run_batch


class DesignBatchTests(unittest.TestCase):
    def setup_inputs(self, root):
        inputs=root/'inputs';inputs.mkdir();records=[]
        for index,size in enumerate([(40,72),(36,60),(36,60),(48,19)],3001):
            key=f'GNR_{index}';folder=inputs/key;folder.mkdir()
            sketch=Image.new('RGB',size,'white');sketch.putpixel((3,3),(0,0,0));sketch.save(folder/'step_2.bmp')
            truth=Image.new('RGB',size,'white');truth.paste('red',(4,4,size[0]-4,size[1]-4));truth.save(folder/'designer_step_3.bmp')
            sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
            records.append(dict(design_id=key,source_sha256=sha(folder/'step_2.bmp'),designer_sha256=sha(folder/'designer_step_3.bmp')))
        (inputs/'manifest.json').write_text(json.dumps(records));return inputs

    def client(self):
        image=Image.new('RGB',(1024,1024),'white');image.paste('red',(250,250,700,700))
        stream=io.BytesIO();image.save(stream,format='PNG');calls=[]
        def edit(**kwargs):
            calls.append(kwargs);return SimpleNamespace(data=[SimpleNamespace(b64_json=base64.b64encode(stream.getvalue()).decode())])
        return SimpleNamespace(images=SimpleNamespace(edit=edit)),calls

    def test_four_native_outputs_real_comparisons_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);inputs=self.setup_inputs(root);client,calls=self.client();output=root/'out'
            rows=run_batch(inputs,output,client,progress=None)
            self.assertEqual(len(calls),4);self.assertEqual(len(rows),4)
            for row,call in zip(rows,calls):
                folder=output/row['design_id'];report=json.loads((folder/'comparison.json').read_text())
                self.assertFalse(report['designer_target_sent_to_generation'])
                self.assertFalse(report['production_ready']);self.assertEqual(len(call['image']),1)
                self.assertIn(f"{row['width']} columns by {row['height']} rows",call['prompt'])
                ai=Image.open(folder/'ai_step_3.png').convert('RGB');truth=Image.open(folder/'designer_step_3.bmp').convert('RGB')
                self.assertEqual(ai.size,truth.size)
                direct=100*np.mean(np.all(np.asarray(ai)==np.asarray(truth),axis=2))
                self.assertAlmostEqual(row['pixel_agreement_pct'],direct)
                np.testing.assert_array_equal(ai,Image.open(folder/'ai_step_3.bmp').convert('RGB'))
            with zipfile.ZipFile(output/'all_designs_review.zip') as archive:self.assertIsNone(archive.testzip())
            run_batch(inputs,output,client,progress=None);self.assertEqual(len(calls),4)
            with self.assertRaisesRegex(ValueError,'configuration changed'):
                run_batch(inputs,output,client,model='other-model',progress=None)

    def test_held_out_query_target_cannot_be_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);inputs=self.setup_inputs(root);client,calls=self.client()
            pair=(inputs/'GNR_3001/step_2.bmp',inputs/'GNR_3001/designer_step_3.bmp')
            with self.assertRaisesRegex(ValueError,'cannot be a generation reference'):
                run_batch(inputs,root/'out',client,references=[pair],progress=None)
            self.assertEqual(calls,[])

    def test_interruption_retains_prior_generation_without_false_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);inputs=self.setup_inputs(root);client,calls=self.client();edit=client.images.edit
            def interrupted(**kwargs):
                if len(calls)==1:raise RuntimeError('synthetic API interruption')
                return edit(**kwargs)
            client.images.edit=interrupted
            with self.assertRaisesRegex(RuntimeError,'Earlier completed designs are retained'):
                run_batch(inputs,root/'out',client,progress=None)
            summary=json.loads((root/'out/comparison_summary.json').read_text());self.assertEqual(len(summary),1)
            self.assertFalse((root/'out/GNR_3002/ai_step_3.png').exists())
            client.images.edit=edit
            rows=run_batch(inputs,root/'out',client,progress=None)
            self.assertEqual(len(calls),4);self.assertEqual(len(rows),4)
