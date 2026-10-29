import base64
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from textile.data import audit_dataset, pair_key
from textile.retrieval_experiment import (suggest_groups, pixel_hash, embed_sources,
    plan_pilot, run_pilot, preview_plan)


class FakeImages:
    def __init__(self):
        self.calls = []
        self.fail_at = None

    def edit(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == self.fail_at:
            raise RuntimeError('sensitive error text must not be saved')
        result = Image.new('RGB', (1024, 1024), 'white')
        for x in range(300, 700):
            for y in range(300, 700):
                result.putpixel((x, y), (255, 0, 0))
        buffer = io.BytesIO()
        result.save(buffer, 'PNG')
        return SimpleNamespace(data=[SimpleNamespace(b64_json=base64.b64encode(buffer.getvalue()).decode())])


def fixture(root):
    rows = []
    for i in range(8):
        sketch = Image.new('RGB', (18, 22), 'white')
        sketch.putpixel((i, 1), (0, 0, 0))
        target = Image.new('RGB', sketch.size, 'white')
        target.putpixel((i, 2), (255, 0, 0))
        s, t = root/f'{i}_source.png', root/f'{i}_target.png'
        sketch.save(s); target.save(t)
        rows.append({'key': str(i), 'source': str(s), 'target': str(t),
                     'source_hash': pixel_hash(s), 'target_hash': pixel_hash(t),
                     'group': str(i), 'width': 18, 'height': 22,
                     'split': 'reference' if i < 5 else 'validation' if i < 7 else 'test'})
    return rows


class RetrievalTests(unittest.TestCase):
    def test_file_names_grouping_and_transitive_duplicate_split(self):
        self.assertEqual(pair_key('GNR 3001 ( FILE 2 ).bmp', 2), 'gnr 3001')
        self.assertEqual(pair_key('GNR 3001 ( FILE 2 ) cropA.bmp', 2),
                         pair_key('GNR 3001 ( FILE 3 ) cropA.bmp', 3))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a, b = root/'Stage_2', root/'Stage_3'
            a.mkdir(); b.mkdir()
            for i in range(12):
                im = Image.new('RGB', (16, 16), 'white')
                im.putpixel((i, 1), (0, 0, 0))
                im.save(a/f'GNR {i//2} ( FILE 2 ) crop{i%2}.bmp')
                im.putpixel((i, 1), (255, 0, 0))
                im.save(b/f'GNR {i//2} ( FILE 3 ) crop{i%2}.bmp')
            groups = suggest_groups(a, root/'groups.csv')
            content = groups.read_bytes()
            suggest_groups(a, groups)
            self.assertEqual(groups.read_bytes(), content)
            audit_dataset(a, b, root/'audit', groups_csv=groups, require_groups=True, hold_fraction=.1)
            rows = json.loads((root/'audit/manifest.json').read_text())
            for i in range(0, 12, 2):
                self.assertEqual(rows[i]['split'], rows[i+1]['split'])
                self.assertEqual(rows[i]['group'], rows[i+1]['group'])
            self.assertEqual({r['split'] for r in rows}, {'reference','validation','test'})

    def test_embedding_cache_and_reference_exclusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); rows = fixture(root)
            class Encoder:
                def encode(self, imgs, **kwargs):
                    return np.arange(len(imgs)*3, dtype=np.float32).reshape(-1, 3)+1
            cache = root/'cache.npz'
            vectors = embed_sources(rows, cache, Encoder(), model_name='test-encoder')
            restored = embed_sources(rows, cache, model_name='test-encoder')
            np.testing.assert_allclose(vectors, restored)
            plan = plan_pilot(rows, vectors, count=2, references=2)
            for job in plan['jobs']:
                self.assertIn(job['query'], ('5','6'))
                self.assertTrue(all(int(r['key']) < 5 for r in job['references']))
            preview_plan(rows, plan, root/'preview.png')
            changed = [dict(r) for r in rows]
            changed[0]['group'] = changed[5]['group']
            with self.assertRaisesRegex(ValueError, 'leakage'):
                plan_pilot(changed, vectors)
            im = Image.open(rows[0]['source']).convert('RGB')
            im.putpixel((0,0), (0,0,0)); im.save(rows[0]['source'])
            with self.assertRaisesRegex(ValueError, 'changed'):
                embed_sources(rows, cache, model_name='test-encoder')

    def test_real_generation_contract_metrics_resume_and_targets_held_out(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); rows = fixture(root)
            vectors = np.eye(8, dtype=np.float32)
            plan = plan_pilot(rows, vectors, count=1, references=2)
            api = FakeImages(); client = SimpleNamespace(images=api)
            out = root/'run'
            report = run_pilot(rows, plan, out, client, progress=lambda _: None)
            self.assertEqual(len(api.calls), 2)
            self.assertEqual([len(call['image']) for call in api.calls], [1,5])
            # Compare actual submitted reference payloads against expected training images.
            refs = [r['key'] for r in plan['jobs'][0]['references']]
            for i, key in enumerate(refs):
                for j, field in enumerate(('source','target')):
                    uploaded = Image.open(io.BytesIO(api.calls[1]['image'][1+2*i+j][1])).convert('RGB')
                    expected = Image.open(rows[int(key)][field]).convert('RGB')
                    self.assertEqual(uploaded.tobytes(), expected.tobytes())
            self.assertEqual(report['completed_design_pairs'], 1)
            self.assertEqual(report['delta_percentage_points']['red_iou_pct'], 0)
            self.assertFalse(report['production_ready'])
            self.assertTrue((out/'pilot_review.zip').exists())
            for method in ('rules_only','similar_examples'):
                folder = out/'design_001'/method
                output = Image.open(folder/'draft.png').convert('RGB')
                target = Image.open(rows[int(plan['jobs'][0]['query'])]['target']).convert('RGB')
                expected = float((np.asarray(output)==np.asarray(target)).all(axis=2).mean())
                metric = json.loads((folder/'comparison.json').read_text())
                self.assertEqual(metric['pixel_agreement'], expected)
                self.assertEqual(output.size, target.size)
                self.assertEqual(output.tobytes(), Image.open(folder/'step_3.bmp').convert('RGB').tobytes())
            run_pilot(rows, plan, out, client, progress=lambda _: None)
            self.assertEqual(len(api.calls), 2)
            with self.assertRaisesRegex(ValueError, 'changed'):
                run_pilot(rows, plan, out, client, model='changed', progress=lambda _: None)
            self.assertEqual(len(api.calls), 2)

    def test_failure_keeps_completed_and_blocks_automatic_rebilling(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); rows = fixture(root)
            plan = plan_pilot(rows, np.eye(8), count=1)
            api = FakeImages(); api.fail_at = 2
            client = SimpleNamespace(images=api)
            with self.assertRaisesRegex(RuntimeError, 'request failed'):
                run_pilot(rows, plan, root/'run', client, progress=lambda _: None)
            partial = json.loads((root/'run/summary.json').read_text())
            self.assertEqual(partial['completed_design_pairs'], 0)
            self.assertNotIn('methods', partial)
            with self.assertRaisesRegex(ValueError, 'incomplete request'):
                run_pilot(rows, plan, root/'run', client, progress=lambda _: None)
            self.assertEqual(len(api.calls), 2)
            self.assertNotIn('sensitive error', (root/'run/design_001/similar_examples/failure.json').read_text())

    def test_mutated_reference_or_leaking_plan_stops_before_api(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); rows = fixture(root)
            plan = plan_pilot(rows, np.eye(8), count=1)
            api = FakeImages(); client = SimpleNamespace(images=api)
            bad = json.loads(json.dumps(plan))
            bad['jobs'][0]['references'][0]['key'] = bad['jobs'][0]['query']
            with self.assertRaisesRegex(ValueError, 'leakage'):
                run_pilot(rows, bad, root/'run', client)
            source = rows[int(plan['jobs'][0]['references'][0]['key'])]['source']
            Image.new('RGB', (18,22), 'black').save(source)
            with self.assertRaisesRegex(ValueError, 'changed'):
                run_pilot(rows, plan, root/'run', client)
            self.assertEqual(len(api.calls), 0)

    def test_notebook_code_and_embedded_runtime(self):
        root = Path(__file__).resolve().parents[1]
        notebook = json.loads((root/'notebooks/Reference_Guided_Pilot.ipynb').read_text())
        import ast
        import zipfile
        for i, cell in enumerate(notebook['cells']):
            if cell['cell_type'] != 'code': continue
            code = ''.join(cell['source'])
            compile(code, f'cell_{i}', 'exec')
            for node in ast.parse(code).body:
                if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'PAYLOAD' for t in node.targets):
                    with zipfile.ZipFile(io.BytesIO(base64.b64decode(ast.literal_eval(node.value)))) as archive:
                        self.assertTrue(all(name.endswith('.py') for name in archive.namelist()))
                        self.assertEqual(archive.read('textile/retrieval_experiment.py'), (root/'textile/retrieval_experiment.py').read_bytes())
