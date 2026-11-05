import ast
import base64
import json
import tempfile
import types
import unittest
import warnings
import zlib
from pathlib import Path
import importlib.util
import io
from unittest.mock import patch

for dependency in ('cv2','skimage','IPython','nbformat'):
    if importlib.util.find_spec(dependency) is None:
        raise unittest.SkipTest('Optional v5 geometry checks require requirements-staircase.txt')

import cv2
import numpy as np
from PIL import Image
from textile.staircase_v5 import (build_corner,corner_check,curve_ok,curve_audit,construct_corner,
    repair_curves,transition_guard,export_review,install_bindings,self_test,stage_ai_draft)


def unchecked(h,v,fh=11,fv=7):
    p=[[x,0] for x in range(fh)];x,y=fh-1,0
    for n in h:
        y+=1
        for _ in range(n):x+=1;p.append([x,y])
    x+=1;y+=1;p.append([x,y])
    for n in v[::-1]:
        x+=1
        for _ in range(n):y+=1;p.append([x,y])
    x+=1
    for _ in range(fv):y+=1;p.append([x,y])
    return np.array(p)


def specimen(bad=False):
    q=build_corner([3,2,2],[3,2,2],11,7)
    tr=(unchecked([3,3,1],[3,2,2]) if bad else q)+[22,0]
    other=q+[22,0]
    parts=[tr,other[::-1]*[1,-1]+[0,36],other*[-1,-1]+[44,36],other[::-1]*[-1,1]+[44,0]]
    p=np.concatenate([v[:-1] for v in parts])+5
    mask=np.zeros((50,58),np.uint8);cv2.fillPoly(mask,[p.astype(np.int32)],1)
    return mask.astype(bool)


class StaircaseV5Tests(unittest.TestCase):
    def test_conservative_pass_rejects_broad_fill_and_preserves_holes(self):
        from textile.staircase_v5 import finish_from_draft,baseline_protection
        baseline=specimen(True)
        baseline[20:25,20:25]=False
        protected=baseline_protection(baseline)
        self.assertTrue(protected[20:25,20:25].all())
        out,report=finish_from_draft(baseline,learned_proposal=np.ones_like(baseline),max_repeat=3)
        self.assertFalse(report['learned_proposal']['accepted'])
        self.assertFalse(out[20:25,20:25].any())
        self.assertLessEqual(report['changed_pixels'],report['pixel_budget'])
        self.assertFalse((out&protected).any())
        unchanged,report=finish_from_draft(baseline,max_repeat=3,max_changed_fraction=0)
        np.testing.assert_array_equal(unchanged,baseline)

    def test_final_cell_ignores_destructive_intermediate_variables(self):
        import textile.staircase_v5 as core
        root=Path(__file__).resolve().parents[1]
        nb=json.loads((root/'notebooks/GNR_Step3_v5_1_Conservative_Staircase.ipynb').read_text())
        source=next(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code' and '# CELL 10: FINAL' in ''.join(c['source']))
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);step2=folder/'step2.bmp';sketch=Image.new('RGB',(58,50),'white');sketch.save(step2)
            draft=folder/'draft.bmp';mask=specimen()
            a=np.full((*mask.shape,3),255,np.uint8);a[mask]=[255,0,0];Image.fromarray(a).save(draft)
            ns=dict(PIPE=folder,S2={'1862':step2},DESIGN='1862',sketch=sketch,W=58,H=50,
                RED_PS=mask,SOFT_PS=mask.astype(np.float32),REF=None,
                RED_FINAL=np.ones_like(mask),RED_SEP=np.ones_like(mask),RED_LEARN=np.ones_like(mask),
                V5_CORE=core,V5_USE_LEARNED_FINISHER=False,V5_MAX_CHANGED_FRACTION=.01,V5_PROTECTED_WHITE=None,
                V5_STAGE_REPORTS={},DESIGN_DIR=folder,np=np,Image=Image,json=json)
            ns['V5_ACTIVE_INPUT']=core.capture_input_context(ns,draft)
            core.record_stage(ns,'Photoshop original',mask)
            helpers=next(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code' and '# CELL 3: IMPORT' in ''.join(c['source']))
            nodes=[node for node in ast.parse(helpers).body if isinstance(node,ast.FunctionDef) and node.name=='rgb_of']
            exec(compile(ast.Module(body=nodes,type_ignores=[]),'cell_3_rgb_helper','exec'),ns)
            exec(compile(source[:source.index('\nviews =')],'conservative_final','exec'),ns)
            np.testing.assert_array_equal(ns['RED_DONE'],mask)
            self.assertTrue(ns['pure_ok'])
            self.assertEqual(ns['V5_LAST_FINISHING_REPORT']['changed_pixels'],0)
            self.assertTrue((folder/'6 stage changes.json').exists())

    def test_failed_import_or_changed_design_cannot_reuse_stale_pixels(self):
        from textile.staircase_v5 import capture_input_context,require_input_context
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);step2=root/'step2.png';draft=root/'draft.png'
            im=Image.new('RGB',(16,12),'white');im.save(step2)
            Image.new('RGB',(16,12),'red').save(draft)
            ns=dict(DESIGN='1862',S2={'1862':step2},sketch=im.copy(),W=16,H=12,
                RED_PS=np.ones((12,16),bool),SOFT_PS=np.ones((12,16),np.float32))
            with self.assertRaisesRegex(ValueError,'not completed'):
                require_input_context(ns)
            ns['V5_ACTIVE_INPUT']=capture_input_context(ns,draft)
            self.assertEqual(require_input_context(ns)['design'],'1862')
            ns['DESIGN']='1071';ns['S2']['1071']=step2
            with self.assertRaisesRegex(ValueError,'changed'):
                require_input_context(ns)
            ns['DESIGN']='1862';ns['V5_ACTIVE_INPUT']=None
            with self.assertRaisesRegex(ValueError,'not completed'):
                require_input_context(ns)
            ns['V5_ACTIVE_INPUT']=capture_input_context(ns,draft)
            Image.new('RGB',(16,12),'pink').save(draft)
            with self.assertRaisesRegex(ValueError,'changed'):
                require_input_context(ns)

    def test_selected_draft_preserves_bytes_validates_size_and_keeps_existing_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'inbox'
            data=io.BytesIO();Image.new('RGB',(16,12),'red').save(data,'PNG');payload=data.getvalue()
            path=stage_ai_draft(payload,folder,'1862',(16,12))
            self.assertEqual(path.name,'1862 GNR 3.png')
            self.assertEqual(path.read_bytes(),payload)
            self.assertEqual(stage_ai_draft(payload,folder,'1862',(16,12)),path)
            other=io.BytesIO();Image.new('RGB',(16,12),'white').save(other,'PNG')
            version=stage_ai_draft(other.getvalue(),folder,'1862',(16,12))
            self.assertNotEqual(path,version)
            self.assertEqual(path.read_bytes(),payload)
            count=len(list(folder.iterdir()))
            with self.assertRaisesRegex(ValueError,'no resizing'):
                stage_ai_draft(payload,folder,'1862',(32,24))
            with self.assertRaisesRegex(ValueError,'readable'):
                stage_ai_draft(b'not an image',folder,'1862',(16,12))
            with self.assertRaisesRegex(ValueError,'characters'):
                stage_ai_draft(payload,folder,'../1862',(16,12))
            self.assertEqual(len(list(folder.iterdir())),count)

    def test_notebook_missing_draft_upload_and_explicit_path(self):
        root=Path(__file__).resolve().parents[1]
        nb=json.loads((root/'notebooks/GNR_Step3_v5_AMB_Staircase.ipynb').read_text())
        cell=next(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code' and '# CELL 3: IMPORT' in ''.join(c['source']))
        start=cell.index('AI_DRAFT_PATH = ""');end=cell.index('DESIGN_DIR =',start)
        handoff=cell[start:end]
        draft=io.BytesIO();Image.new('RGB',(16,12),'red').save(draft,'PNG');payload=draft.getvalue()
        import textile.staircase_v5 as core
        google=types.ModuleType('google');google.__path__=[]
        colab=types.ModuleType('google.colab')
        uploads=[]
        colab.files=types.SimpleNamespace(upload=lambda: uploads.append(True) or {'arbitrary.png':payload})
        with tempfile.TemporaryDirectory() as tmp,patch.dict('sys.modules',{'google':google,'google.colab':colab}):
            folder=Path(tmp)
            ns=dict(Path=Path,V5_CORE=core,inbox_files=lambda:{},DESIGN='1862',INBOX=folder/'inbox',W=16,H=12,
                S3={'1862':folder/'designer_reference_must_not_be_read.bmp'})
            exec(compile(handoff,'input_handoff','exec'),ns)
            self.assertEqual(len(uploads),1)
            self.assertEqual(ns['inbox']['1862'].read_bytes(),payload)
            uploaded=folder/'some_drive_file.png';uploaded.write_bytes(payload)
            direct=handoff.replace('AI_DRAFT_PATH = ""',f'AI_DRAFT_PATH = {str(uploaded)!r}')
            exec(compile(direct,'explicit_path_handoff','exec'),ns)
            self.assertEqual(len(uploads),1)
            # Inbox reuse does not offer another upload or read the designer reference.
            ns['inbox_files']=lambda:{'1862':ns['inbox']['1862']}
            exec(compile(handoff,'existing_inbox','exec'),ns)
            self.assertEqual(len(uploads),1)

    def test_diagram_and_orientations_and_free_flat_lengths(self):
        self.assertIn('passed',self_test()['checks'])
        for fh,fv in ((2,2),(11,7),(20,5)):
            p=build_corner([3,2,2],[3,2,2],fh,fv)
            report=corner_check(p,True,False)
            self.assertEqual(report['status'],'passed')
            self.assertEqual(report['horizontal_toward_M'],[3,2,2,1])
            self.assertEqual(report['vertical_toward_M'],[3,2,2,1])
            self.assertTrue(curve_ok(p[::-1],False,True))

    def test_jumps_growth_multiple_midpoints_and_unresolved(self):
        for h in ([4,2],[2,3,2],[3,3,1],[5,4,3,2]):
            self.assertFalse(curve_ok(unchecked(h,[3,2,2]),True,False))
        self.assertEqual(corner_check(build_corner([2],[2]),True,True)['status'],'unresolved')
        self.assertEqual(corner_check(np.array([[0,0],[4,0]]),True,False)['status'],'unresolved')
        tiny=np.zeros((8,8),bool);tiny[2,2]=True
        self.assertEqual(curve_audit(tiny)['unresolved'],1)
        self.assertEqual(curve_audit(tiny)['passed'],0)

    def test_constructor_and_full_render_repair(self):
        bad=unchecked([3,3,1],[3,2,2])
        result=construct_corner(bad,True,False,max_shift=2,max_repeat=3)
        self.assertIsNotNone(result)
        self.assertTrue(curve_ok(result,True,False))
        np.testing.assert_array_equal(result[[0,-1]],bad[[0,-1]])
        source=specimen(True);before=curve_audit(source)
        self.assertEqual((before['passed'],before['failed']),(3,1))
        out,count,info=repair_curves(source,max_repeat=3,max_edits=5)
        self.assertEqual(count,1)
        self.assertEqual((info['audit']['passed'],info['audit']['failed'],info['audit']['unresolved']),(4,0,0))
        self.assertEqual(int((out!=source).sum()),1)
        good=specimen();unchanged,n,_=repair_curves(good,max_repeat=3)
        self.assertEqual(n,0);np.testing.assert_array_equal(good,unchanged)

    def test_affected_contours_cleanup_guard_and_protected_white(self):
        good=specimen();bad=specimen(True)
        ok,_=transition_guard(good,bad)
        self.assertFalse(ok)
        ok,_=transition_guard(bad,good)
        self.assertTrue(ok)
        white=np.zeros_like(good);white[0,0]=True
        added=good.copy();added[0,0]=True
        ok,reason=transition_guard(good,added,white)
        self.assertFalse(ok);self.assertIn('protected',reason['reason'])
        protected=np.zeros_like(good)
        # Protect ALL currently white pixels: repairs that add red must not override them.
        protected[~bad]=True
        out,_,_=repair_curves(bad,max_repeat=3,max_edits=5,protected_white=protected)
        self.assertFalse((out&protected).any())

    def test_export_uses_original_dimensions_and_review_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);good=specimen();src=root/'step2.png'
            Image.new('RGB',(good.shape[1],good.shape[0]),'white').save(src)
            namespace=dict(PIPE=root,S2={'demo':src})
            with self.assertRaisesRegex(ValueError,'Native size'):
                export_review(np.ones((6,7),bool),'demo',namespace)
            bmp,png,ok=export_review(good,'demo',namespace)
            self.assertTrue(ok)
            with Image.open(bmp) as b,Image.open(src) as s:
                self.assertEqual(b.size,s.size)
            report=json.loads(bmp.with_suffix('.json').read_text())
            self.assertFalse(report['production_ready'])
            self.assertEqual(report['curves']['failed'],0)
            second,_,_=export_review(good,'demo',namespace)
            self.assertNotEqual(bmp,second)
            install_bindings(namespace)
            self.assertTrue(namespace['curve_ok'](build_corner([2],[2]),True,False))
            self.assertTrue(namespace['save_production'](good,'demo')[2])

    def test_notebook_payload_library_and_cell_syntax(self):
        root=Path(__file__).resolve().parents[1]
        nb=json.loads((root/'notebooks/GNR_Step3_v5_AMB_Staircase.ipynb').read_text())
        from IPython.core.interactiveshell import InteractiveShell
        import nbformat
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            nbformat.validate(nbformat.from_dict(nb))
        payloads={}
        for i,c in enumerate(nb['cells']):
            if c['cell_type']!='code':continue
            self.assertFalse(c['outputs']);self.assertIsNone(c['execution_count'])
            text=''.join(c['source'])
            transformed=InteractiveShell.instance().input_transformer_manager.transform_cell(text)
            compile(transformed,f'cell_{i}','exec')
            if '# V5 SETUP' in text:
                for node in ast.walk(ast.parse(text)):
                    if isinstance(node,ast.Assign):
                        for t in node.targets:
                            if isinstance(t,ast.Name) and t.id in ('V5_CORE_SOURCE','V5_COMPAT_SOURCE'):
                                # Extract only the base64 literal, never run setup/pip.
                                strings=[n.value for n in ast.walk(node.value) if isinstance(n,ast.Constant) and isinstance(n.value,str)]
                                payloads[t.id]=zlib.decompress(base64.b64decode(strings[0])).decode()
        self.assertEqual(payloads['V5_CORE_SOURCE'],(root/'textile/staircase_v5.py').read_text())
        with tempfile.TemporaryDirectory() as tmp:
            rootdir=Path(tmp);src=rootdir/'step2.bmp';good=specimen()
            Image.new('RGB',(58,50),'white').save(src)
            ns=dict(PIPE=rootdir,INBOX=rootdir,OUTPUT=rootdir,DATASET=rootdir,S2={'demo':src})
            exec(compile(payloads['V5_COMPAT_SOURCE'],'v5_shared_library','exec'),ns)
            self.assertTrue(ns['curve_ok'](build_corner([3,2,2],[3,2,2]),True,False))
            self.assertEqual(ns['fix_curves'](good)[1],0)
            self.assertTrue(ns['save_production'](good,'demo')[2])
            # Strict native import applies in batch as well as the interactive path.
            with self.assertRaisesRegex(ValueError,'differs'):
                ns['import_export'](src,(20,20))
            self.assertIn('A/M/B corner screening %',ns['measure'](good))
