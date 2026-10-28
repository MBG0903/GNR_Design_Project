import hashlib
import tempfile
import unittest
from pathlib import Path
from textile.drive_inputs import stage_files


class DriveInputTests(unittest.TestCase):
    def asset(self,name,content,upload=None):
        return dict(path=name,sha256=hashlib.sha256(content).hexdigest(),upload_name=upload,url='https://example.test/'+name)

    def test_uploaded_sketch_and_downloaded_target_are_cached_in_drive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);uploads=root/'content';uploads.mkdir();drive=root/'drive'
            name='GNR 3001 ( FILE 2 ).bmp';(uploads/name).write_bytes(b'sketch')
            assets=[self.asset('GNR_3001/step_2.bmp',b'sketch',name),self.asset('GNR_3001/designer_step_3.bmp',b'target')]
            calls=[]
            def fetch(url):calls.append(url);return b'target'
            report=stage_files(drive,assets,uploads,fetch)
            self.assertEqual(len(calls),1);self.assertEqual({r['origin'] for r in report},{'colab_upload','pinned_github_download'})
            self.assertEqual((drive/'GNR_3001/step_2.bmp').read_bytes(),b'sketch')
            def no_fetch(url):raise AssertionError('A cached Drive file must not be downloaded again.')
            report=stage_files(drive,assets,uploads,no_fetch)
            self.assertTrue(all(r['origin']=='existing_drive_file' for r in report))

    def test_existing_conflict_is_not_overwritten_or_hidden_by_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            drive=Path(tmp);(drive/'source.bmp').write_bytes(b'user edits')
            with self.assertRaisesRegex(ValueError,'not overwritten'):
                stage_files(drive,[self.asset('new.bmp',b'new'),self.asset('source.bmp',b'original')],fetch=lambda url:b'new')
            self.assertEqual((drive/'source.bmp').read_bytes(),b'user edits');self.assertFalse((drive/'new.bmp').exists())

    def test_bad_upload_or_download_is_not_saved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);uploads=root/'content';uploads.mkdir();(uploads/'source.bmp').write_bytes(b'wrong upload')
            with self.assertRaisesRegex(ValueError,'hash differs'):
                stage_files(root/'drive',[self.asset('source.bmp',b'original','source.bmp')],uploads,lambda url:b'original')
            self.assertFalse((root/'drive/source.bmp').exists())
            with self.assertRaisesRegex(ValueError,'hash differs'):
                stage_files(root/'drive',[self.asset('source.bmp',b'original')],fetch=lambda url:b'wrong download')
            self.assertFalse((root/'drive/source.bmp').exists())
