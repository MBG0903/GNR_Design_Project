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

for dependency in ('cv2','skimage','IPython','nbformat'):
    if importlib.util.find_spec(dependency) is None:
        raise unittest.SkipTest('Optional v5 geometry checks require requirements-staircase.txt')

import cv2
import numpy as np
from PIL import Image
from textile.staircase_v5 import (build_corner,corner_check,curve_ok,curve_audit,construct_corner,
    repair_curves,transition_guard,export_review,install_bindings,self_test)


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
