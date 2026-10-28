import base64
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
from textile.tiles import split, merge_saved, _digest
from textile.tile_batch import generate_all, assemble_preview


class BatchPreviewTests(unittest.TestCase):
    def client(self):
        stream = io.BytesIO()
        Image.new('RGB', (1024, 1024), 'red').save(stream, format='PNG')
        calls = []
        def edit(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(data=[SimpleNamespace(b64_json=base64.b64encode(stream.getvalue()).decode())])
        return SimpleNamespace(images=SimpleNamespace(edit=edit)), calls

    def test_complete_preview_resume_and_review_separation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'run'; source = Image.new('RGB', (41, 31), 'white')
            split(source, source, root, rows=3, columns=4, overlap=3)
            client, calls = self.client()
            output, report = generate_all(root, client)
            self.assertEqual(len(calls), 12)
            self.assertEqual(report['tile_count'], 12)
            self.assertEqual(report['changed_pixels'], 41*31)
            self.assertFalse(report['production_ready'])
            with Image.open(output/'combined_unreviewed.bmp') as combined:
                self.assertEqual(combined.size, source.size)
                self.assertTrue(np.all(np.asarray(combined.convert('RGB')) == (255, 0, 0)))
            with zipfile.ZipFile(output/'all_tiles_and_combined_review.zip') as archive:
                self.assertEqual(len([n for n in archive.namelist() if n.endswith('_core.png')]), 12)
            second, _ = generate_all(root, client)
            self.assertNotEqual(output, second)
            self.assertEqual(len(calls), 12)
            reviewed, _ = merge_saved(root)
            np.testing.assert_array_equal(reviewed, source)
            self.assertFalse((root/'proposals').exists())
            with self.assertRaisesRegex(ValueError, 'different model'):
                generate_all(root, client, model='different-model')
            self.assertEqual(len(calls), 12)

    def test_missing_tiles_and_halo_tampering_block_preview(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'run'; source = Image.new('RGB', (40, 30), 'white')
            manifest = split(source, source, root, rows=1, columns=2, overlap=4)
            with self.assertRaisesRegex(ValueError, 'missing AI proposal'):
                assemble_preview(root)
            client, _ = self.client(); generate_all(root, client)
            tile = manifest['tiles'][0]; folder = root/'ai_proposals'/tile['id']/'0001'
            with Image.open(folder/'candidate.png') as opened: image = opened.convert('RGB')
            image.putpixel((image.width-1, 0), (255, 0, 0)); image.save(folder/'candidate.png')
            metadata = json.loads((folder/'proposal.json').read_text())
            metadata['candidate_rgb_sha256'] = _digest(image)
            (folder/'proposal.json').write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, 'read-only halo'):
                assemble_preview(root)

    def test_interrupted_generation_reuses_completed_requests(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'run'; source = Image.new('RGB', (40, 30), 'white')
            split(source, source, root, rows=1, columns=2, overlap=4)
            client, calls = self.client(); original = client.images.edit
            def fail_second(**kwargs):
                if len(calls) == 1: raise RuntimeError('API interrupted')
                return original(**kwargs)
            client.images.edit = fail_second
            with self.assertRaisesRegex(RuntimeError, 'API interrupted'):
                generate_all(root, client)
            self.assertEqual(len(calls), 1)
            client.images.edit = original
            _, report = generate_all(root, client)
            self.assertEqual(len(calls), 2)
            self.assertEqual(report['tile_count'], 2)
