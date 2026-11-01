import ast
import base64
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from textile.paired_training import (split_80_20,native_patches,prepare_pairs,pixel_hash,
    native_inference_canvas,restore_native,evaluate_split)


def fixture(root):
    rows=[]
    for i in range(10):
        s=Image.new('RGB',(97,73),'white');s.putpixel((i,2),(0,0,0))
        t=Image.new('RGB',s.size,'white');t.putpixel((i,3),(255,0,0))
        sp,tp=root/f's{i}.png',root/f't{i}.png';s.save(sp);t.save(tp)
        rows.append(dict(key=f'k{i}',source=str(sp),target=str(tp),group=str(i//2),
            source_hash=pixel_hash(sp),target_hash=pixel_hash(tp),width=97,height=73,split='old'))
    manifest=root/'manifest.json';manifest.write_text(json.dumps(rows))
    return rows,manifest


class TrainingDataTests(unittest.TestCase):
    def test_family_split_deterministic_and_duplicate_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows,_=fixture(Path(tmp));a=split_80_20(rows)
            self.assertEqual(a,split_80_20(rows))
            self.assertEqual(sum(r['split']=='train' for r in a),8)
            self.assertEqual(sum(r['split']=='test' for r in a),2)
            for i in range(0,10,2):self.assertEqual(a[i]['split'],a[i+1]['split'])
            train=next(r for r in a if r['split']=='train')
            test=next(r for r in a if r['split']=='test')
            rows[int(test['key'][1:])]['target_hash']=train['target_hash']
            with self.assertRaisesRegex(ValueError,'crosses'):split_80_20(rows)

    def test_native_patch_geometry_without_interpolation(self):
        # Distinct native pixels let us detect even a one-pixel shift or resize.
        rng=np.random.default_rng(3)
        a=rng.integers(0,256,(83,137,3),dtype=np.uint8)
        b=np.full_like(a,255);b[rng.random((83,137))>.5]=[255,0,0]
        source,target=Image.fromarray(a),Image.fromarray(b)
        seen=np.zeros((83,137),bool)
        for s,t,g in native_patches(source,target,64):
            x0,y0,x1,y1=g['source_box'];x,y=g['padding_left'],g['padding_top']
            self.assertEqual(s.crop((x,y,x+x1-x0,y+y1-y0)).tobytes(),source.crop((x0,y0,x1,y1)).tobytes())
            self.assertEqual(t.crop((x,y,x+x1-x0,y+y1-y0)).tobytes(),target.crop((x0,y0,x1,y1)).tobytes())
            seen[y0:y1,x0:x1]=True
            self.assertEqual(set(t.getdata()),{(255,0,0),(255,255,255)})
        self.assertTrue(seen.all())
        small=Image.new('RGB',(11,17),'red')
        s,_,g=next(native_patches(small,small,64))
        self.assertEqual(s.crop((g['padding_left'],g['padding_top'],g['padding_left']+11,g['padding_top']+17)).tobytes(),small.tobytes())

    def test_prepare_excludes_test_patches_and_rejects_changed_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rows,manifest=fixture(root)
            summary=prepare_pairs(manifest,root/'prepared',size=64,progress=None)
            self.assertEqual((summary['training_pairs'],summary['test_pairs']),(8,2))
            split=json.loads((root/'prepared/split.json').read_text())
            test_keys={r['key'] for r in split if r['split']=='test'}
            patches=[json.loads(l) for l in (root/'prepared/train_patches.jsonl').read_text().splitlines()]
            self.assertTrue(all(p['pair_key'] not in test_keys for p in patches))
            self.assertEqual(prepare_pairs(manifest,root/'prepared',size=64,progress=None),summary)
            with self.assertRaisesRegex(ValueError,'changed'):
                prepare_pairs(manifest,root/'prepared',size=128,progress=None)

    def test_native_inference_padding_and_refuse_resize(self):
        source=Image.new('RGB',(401,340),'red')
        canvas,box=native_inference_canvas(source)
        self.assertEqual(canvas.size,(448,384))
        result=restore_native(canvas,canvas.size,box)
        self.assertEqual(result.tobytes(),source.tobytes())
        self.assertEqual(result.size,source.size)
        with self.assertRaisesRegex(ValueError,'refusing'):
            restore_native(Image.new('RGB',(512,512)),canvas.size,box)

    def test_evaluation_predicts_only_test_sketches_and_scores_independently(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rows,manifest=fixture(root)
            prepare_pairs(manifest,root/'prepared',size=64,progress=None)
            calls=[]
            def predict(sketch,seed):
                calls.append((sketch.tobytes(),seed))
                return Image.new('RGB',sketch.size,'white')
            summary=evaluate_split(root/'prepared',root/'results',predict,'finetuned',{'test':True},progress=None)
            self.assertEqual(len(calls),2)
            self.assertEqual(summary['mean_percentages']['red_iou'],0)
            self.assertAlmostEqual(summary['mean_percentages']['pixel_agreement'],100*(97*73-1)/(97*73))
            split=json.loads((root/'prepared/split.json').read_text())
            expected={Image.open(r['source']).tobytes() for r in split if r['split']=='test'}
            self.assertEqual({c[0] for c in calls},expected)
            evaluate_split(root/'prepared',root/'results',predict,'finetuned',{'test':True},progress=None)
            self.assertEqual(len(calls),2)
            split[0]['split']='changed'
            (root/'prepared/split.json').write_text(json.dumps(split))
            with self.assertRaisesRegex(ValueError,'split changed'):
                evaluate_split(root/'prepared',root/'results',predict,'finetuned',{},progress=None)

    def test_notebook_and_vendor_contract(self):
        root=Path(__file__).resolve().parents[1]
        nb=json.loads((root/'notebooks/Paired_ControlNet_Training.ipynb').read_text())
        for i,cell in enumerate(nb['cells']):
            if cell['cell_type']!='code':continue
            source=''.join(cell['source']);compile(source,f'cell_{i}','exec')
            for node in ast.parse(source).body:
                if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PAYLOAD' for t in node.targets):
                    with zipfile.ZipFile(io.BytesIO(base64.b64decode(ast.literal_eval(node.value)))) as z:
                        for name in z.namelist():self.assertEqual(z.read(name),(root/name).read_bytes())
        script=(root/'vendor/controlnet/train_controlnet.py').read_text()
        compile(script,'vendor_trainer','exec')
        self.assertNotIn('transforms.Resize(',script)
        self.assertNotIn('transforms.CenterCrop(',script)
        self.assertIn('load_from_disk(args.train_data_dir)',script)
        self.assertIn('unet.requires_grad_(False)',script)
        self.assertIn('controlnet.train()',script)
